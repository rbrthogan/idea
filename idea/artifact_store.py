"""Safe, content-addressed storage for locally produced artifacts.

The store is deliberately unaware of artifact producers and evaluators.  Its
only trust boundary is a caller-supplied job output directory: a relative
manifest path may be imported from that directory, after which callers use an
opaque storage key rather than a filesystem path.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import struct
import tempfile
import threading
import zlib
from contextlib import contextmanager
from typing import BinaryIO, Iterator, TextIO

try:  # pragma: no cover - exercised on supported Unix platforms
    import fcntl
except ImportError:  # pragma: no cover - Windows falls back to a process lock
    fcntl = None  # type: ignore[assignment]


DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 250 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 16_777_216

_CHUNK_SIZE = 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_KEY_RE = re.compile(r"art-sha256-([0-9a-f]{64})\Z")
_HEX_PAIR_RE = re.compile(r"[0-9a-f]{2}\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")

_MEDIA_TYPE_BY_EXTENSION = {
    ".json": "application/json",
    ".png": "image/png",
    ".py": "text/x-python",
    ".txt": "text/plain",
}


class ArtifactStoreError(Exception):
    """Base class for artifact-store failures."""


class ArtifactValidationError(ArtifactStoreError, ValueError):
    """The caller supplied an invalid artifact reference or configuration."""


class ArtifactPathError(ArtifactValidationError):
    """A producer output path did not satisfy the import boundary."""


class ArtifactMediaTypeError(ArtifactValidationError):
    """An artifact's extension, declaration, and contents did not agree."""


class ArtifactSizeLimitError(ArtifactValidationError):
    """An artifact would exceed a configured storage limit."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Stored data or store structure was replaced or corrupted."""


class ArtifactNotFoundError(ArtifactStoreError):
    """An otherwise valid storage key is not present in this store."""


ArtifactReference = dict[str, str | int]


class ArtifactStore:
    """A bounded, digest-addressed store for safe local artifact handoff.

    ``root`` should be dedicated to this store.  Imported files are copied into
    it and are never served from their producer-controlled source location.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        self.max_file_bytes = self._positive_limit(
            max_file_bytes, "max_file_bytes"
        )
        self.max_total_bytes = self._positive_limit(
            max_total_bytes, "max_total_bytes"
        )
        if self.max_file_bytes > self.max_total_bytes:
            raise ArtifactValidationError(
                "max_file_bytes cannot be greater than max_total_bytes"
            )

        requested_root = Path(root)
        try:
            requested_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.root = requested_root.resolve(strict=True)
            root_stat = self.root.stat()
        except OSError as exc:
            raise ArtifactValidationError("artifact store root is unavailable") from exc
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ArtifactValidationError("artifact store root must be a directory")
        if hasattr(os, "geteuid") and root_stat.st_uid != os.geteuid():
            raise ArtifactValidationError(
                "artifact store root must be owned by the current user"
            )
        try:
            os.chmod(self.root, 0o700)
        except OSError as exc:
            raise ArtifactValidationError(
                "artifact store root permissions could not be secured"
            ) from exc

        objects_container = self.root / "objects"
        self._make_private_directory(objects_container)
        self._objects_root = objects_container / "sha256"
        self._make_private_directory(self._objects_root)
        self._staging_root = self.root / "staging"
        self._make_private_directory(self._staging_root)

        self._lock_path = self.root / ".artifact-store.lock"
        self._thread_lock = threading.RLock()

    def import_file(
        self,
        job_output_dir: str | os.PathLike[str],
        manifest_path: str | os.PathLike[str],
        media_type: str | None = None,
    ) -> ArtifactReference:
        """Validate and import one producer output file.

        ``manifest_path`` must be a relative path beneath ``job_output_dir``.
        Neither the source path nor the store's absolute path is included in
        the returned JSON-serializable reference.
        """

        parts, extension = self._validate_manifest_path(manifest_path)
        canonical_media_type = self._media_type_for(extension, media_type)
        job_root = self._resolve_job_root(job_output_dir)
        self._assert_directory(self._staging_root)

        temp_fd = -1
        temp_path: Path | None = None
        try:
            temp_fd, temp_name = tempfile.mkstemp(
                prefix=".artifact-import-", dir=self._staging_root
            )
            temp_path = Path(temp_name)
            os.fchmod(temp_fd, 0o600)

            digest = hashlib.sha256()
            byte_count = 0
            with self._open_source(job_root, parts) as (source_fd, initial_stat):
                if initial_stat.st_size > self.max_file_bytes:
                    raise ArtifactSizeLimitError(
                        "artifact exceeds the per-file size limit"
                    )

                with os.fdopen(temp_fd, "wb", closefd=True) as destination:
                    temp_fd = -1
                    while True:
                        chunk = os.read(source_fd, _CHUNK_SIZE)
                        if not chunk:
                            break
                        byte_count += len(chunk)
                        if byte_count > self.max_file_bytes:
                            raise ArtifactSizeLimitError(
                                "artifact exceeds the per-file size limit"
                            )
                        digest.update(chunk)
                        destination.write(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())

                final_stat = os.fstat(source_fd)
                if (
                    final_stat.st_size != initial_stat.st_size
                    or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
                    or byte_count != final_stat.st_size
                ):
                    raise ArtifactPathError("artifact changed while it was imported")

            self._validate_file_contents(temp_path, extension)
            digest_hex = digest.hexdigest()
            self._commit_staged_file(temp_path, digest_hex, byte_count)
            return {
                "storage_key": self._storage_key(digest_hex),
                "sha256": digest_hex,
                "size_bytes": byte_count,
                "media_type": canonical_media_type,
            }
        finally:
            if temp_fd >= 0:
                os.close(temp_fd)
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    # The import result is already safely committed.  A stale
                    # private staging file can be cleaned up by maintenance.
                    pass

    def import_output(
        self,
        job_output_dir: str | os.PathLike[str],
        manifest_path: str | os.PathLike[str],
        media_type: str | None = None,
    ) -> ArtifactReference:
        """Alias for :meth:`import_file` using producer-oriented terminology."""

        return self.import_file(job_output_dir, manifest_path, media_type)

    def resolve(self, storage_key: str) -> Path:
        """Resolve a validated key to an integrity-checked local object path.

        Prefer :meth:`open` when possible: a resolved path can become stale if
        another process deletes the object after this method returns.
        """

        digest = self._digest_from_key(storage_key)
        with self._locked():
            fd, _ = self._open_verified_object(digest)
            os.close(fd)
            return self._object_path(digest)

    def open(
        self,
        storage_key: str,
        mode: str = "rb",
        *,
        encoding: str | None = None,
    ) -> BinaryIO | TextIO:
        """Open an integrity-checked object in a read-only mode."""

        if mode not in {"rb", "r", "rt"}:
            raise ArtifactValidationError("artifacts may only be opened read-only")
        if mode == "rb" and encoding is not None:
            raise ArtifactValidationError(
                "encoding is only valid when opening an artifact as text"
            )

        digest = self._digest_from_key(storage_key)
        with self._locked():
            fd, _ = self._open_verified_object(digest)

        binary = os.fdopen(fd, "rb", closefd=True)
        if mode == "rb":
            return binary
        return io.TextIOWrapper(binary, encoding=encoding or "utf-8")

    def delete(self, storage_key: str) -> bool:
        """Delete exactly one validated object, never a directory tree.

        Returns ``False`` if the valid key is not present.  Corrupt or replaced
        store entries are rejected rather than unlinked.
        """

        digest = self._digest_from_key(storage_key)
        path = self._object_path(digest)
        with self._locked():
            try:
                fd, verified_stat = self._open_verified_object(digest)
            except ArtifactNotFoundError:
                return False
            else:
                os.close(fd)

            try:
                current_stat = path.lstat()
            except FileNotFoundError:
                return False
            if (
                stat.S_ISLNK(current_stat.st_mode)
                or not stat.S_ISREG(current_stat.st_mode)
                or current_stat.st_dev != verified_stat.st_dev
                or current_stat.st_ino != verified_stat.st_ino
            ):
                raise ArtifactIntegrityError(
                    "stored artifact was replaced during deletion"
                )

            try:
                path.unlink()
            except OSError as exc:
                raise ArtifactIntegrityError("stored artifact could not be deleted") from exc

            self._fsync_directory(path.parent)
            for directory in (path.parent, path.parent.parent):
                try:
                    directory.rmdir()
                except OSError:
                    break
            return True

    def total_size_bytes(self) -> int:
        """Return the total bytes occupied by committed artifact objects."""

        with self._locked():
            return self._total_size_locked()

    @staticmethod
    def _positive_limit(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ArtifactValidationError(f"{name} must be a positive integer")
        return value

    def _make_private_directory(self, path: Path) -> None:
        try:
            path.mkdir(mode=0o700, exist_ok=True)
            path_stat = path.lstat()
        except OSError as exc:
            raise ArtifactValidationError("artifact store directory is unavailable") from exc
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
            raise ArtifactValidationError(
                "artifact store directories must not be symbolic links"
            )
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.root)
            os.chmod(path, 0o700)
        except (OSError, ValueError) as exc:
            raise ArtifactValidationError(
                "artifact store directory escapes the configured root"
            ) from exc

    def _assert_directory(self, path: Path) -> None:
        try:
            path_stat = path.lstat()
        except OSError as exc:
            raise ArtifactIntegrityError("artifact store directory is unavailable") from exc
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
            raise ArtifactIntegrityError("artifact store directory was replaced")
        try:
            path.resolve(strict=True).relative_to(self.root)
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError("artifact store directory escapes its root") from exc

    @staticmethod
    def _resolve_job_root(job_output_dir: str | os.PathLike[str]) -> Path:
        try:
            job_root = Path(job_output_dir).resolve(strict=True)
            root_stat = job_root.stat()
        except (OSError, TypeError, ValueError) as exc:
            raise ArtifactPathError("job output directory is unavailable") from exc
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ArtifactPathError("job output root must be a directory")
        return job_root

    @staticmethod
    def _validate_manifest_path(
        manifest_path: str | os.PathLike[str],
    ) -> tuple[tuple[str, ...], str]:
        try:
            raw_path = os.fspath(manifest_path)
        except TypeError as exc:
            raise ArtifactPathError("manifest path must be a relative string") from exc
        if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
            raise ArtifactPathError("manifest path must be a non-empty relative string")

        windows_path = PureWindowsPath(raw_path)
        normalized = raw_path.replace("\\", "/")
        if (
            normalized.startswith("/")
            or windows_path.is_absolute()
            or bool(windows_path.drive)
        ):
            raise ArtifactPathError("absolute manifest paths are not allowed")

        parts = tuple(normalized.split("/"))
        if any(part in {"", ".", ".."} for part in parts):
            raise ArtifactPathError("manifest path traversal is not allowed")
        extension = Path(parts[-1]).suffix.lower()
        if extension not in _MEDIA_TYPE_BY_EXTENSION:
            raise ArtifactMediaTypeError("artifact file extension is not allowed")
        return parts, extension

    @staticmethod
    def _media_type_for(extension: str, declared: str | None) -> str:
        expected = _MEDIA_TYPE_BY_EXTENSION[extension]
        if declared is None:
            return expected
        if not isinstance(declared, str):
            raise ArtifactMediaTypeError("artifact media type must be a string")
        normalized = declared.split(";", 1)[0].strip().lower()
        if normalized != expected:
            raise ArtifactMediaTypeError(
                "artifact media type does not match its file extension"
            )
        return expected

    @contextmanager
    def _open_source(
        self, job_root: Path, parts: tuple[str, ...]
    ) -> Iterator[tuple[int, os.stat_result]]:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        open_fds: list[int] = []
        try:
            root_fd = os.open(job_root, directory_flags)
            open_fds.append(root_fd)
            if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
                raise ArtifactPathError("job output root must be a directory")

            current_fd = root_fd
            for component in parts[:-1]:
                before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                    raise ArtifactPathError(
                        "manifest path contains a non-directory or symbolic link"
                    )
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
                after = os.fstat(next_fd)
                if (
                    not stat.S_ISDIR(after.st_mode)
                    or before.st_dev != after.st_dev
                    or before.st_ino != after.st_ino
                ):
                    os.close(next_fd)
                    raise ArtifactPathError(
                        "manifest path changed while it was opened"
                    )
                open_fds.append(next_fd)
                current_fd = next_fd

            before = os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise ArtifactPathError(
                    "artifact output must be a regular, non-symbolic-link file"
                )
            source_fd = os.open(parts[-1], file_flags, dir_fd=current_fd)
            after = os.fstat(source_fd)
            if (
                not stat.S_ISREG(after.st_mode)
                or before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
            ):
                os.close(source_fd)
                raise ArtifactPathError("artifact changed while it was opened")
            open_fds.append(source_fd)
            yield source_fd, after
        except ArtifactStoreError:
            raise
        except OSError as exc:
            raise ArtifactPathError(
                "artifact is unavailable or crosses a symbolic-link boundary"
            ) from exc
        finally:
            for open_fd in reversed(open_fds):
                try:
                    os.close(open_fd)
                except OSError:
                    pass

    @staticmethod
    def _validate_file_contents(path: Path, extension: str) -> None:
        try:
            with path.open("rb") as artifact:
                contents = artifact.read()
        except OSError as exc:
            raise ArtifactIntegrityError("staged artifact could not be read") from exc

        if extension == ".png":
            ArtifactStore._validate_png(contents)
            return

        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactMediaTypeError("text artifacts must contain valid UTF-8") from exc
        if "\x00" in text:
            raise ArtifactMediaTypeError("text artifacts must not contain NUL bytes")
        if extension == ".json":
            def reject_constant(value: str) -> None:
                raise ValueError(f"non-finite JSON number: {value}")

            try:
                json.loads(text, parse_constant=reject_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ArtifactMediaTypeError(
                    "JSON artifact contents are not valid JSON"
                ) from exc

    @staticmethod
    def _validate_png(contents: bytes) -> None:
        """Validate PNG structure and bound decoded dimensions before storage."""

        if not contents.startswith(_PNG_SIGNATURE):
            raise ArtifactMediaTypeError(
                "PNG artifact contents do not match image/png"
            )

        offset = len(_PNG_SIGNATURE)
        saw_header = False
        saw_image_data = False
        while offset < len(contents):
            if offset + 12 > len(contents):
                raise ArtifactMediaTypeError("PNG artifact is truncated")
            chunk_length = struct.unpack(">I", contents[offset:offset + 4])[0]
            chunk_type = contents[offset + 4:offset + 8]
            chunk_end = offset + 12 + chunk_length
            if chunk_end > len(contents):
                raise ArtifactMediaTypeError("PNG artifact is truncated")
            chunk_data = contents[offset + 8:offset + 8 + chunk_length]
            expected_crc = struct.unpack(">I", contents[chunk_end - 4:chunk_end])[0]
            actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
            if expected_crc != actual_crc:
                raise ArtifactMediaTypeError("PNG artifact has an invalid chunk checksum")

            if not saw_header:
                if chunk_type != b"IHDR" or chunk_length != 13:
                    raise ArtifactMediaTypeError("PNG artifact must begin with IHDR")
                width, height, bit_depth, color_type, compression, filtering, interlace = (
                    struct.unpack(">IIBBBBB", chunk_data)
                )
                valid_depths = {
                    0: {1, 2, 4, 8, 16},
                    2: {8, 16},
                    3: {1, 2, 4, 8},
                    4: {8, 16},
                    6: {8, 16},
                }
                if (
                    width == 0
                    or height == 0
                    or width * height > DEFAULT_MAX_IMAGE_PIXELS
                    or bit_depth not in valid_depths.get(color_type, set())
                    or compression != 0
                    or filtering != 0
                    or interlace not in {0, 1}
                ):
                    raise ArtifactMediaTypeError(
                        "PNG artifact has unsupported or oversized dimensions"
                    )
                saw_header = True
            elif chunk_type == b"IHDR":
                raise ArtifactMediaTypeError("PNG artifact contains multiple IHDR chunks")

            if chunk_type == b"IDAT":
                saw_image_data = True
            if chunk_type == b"IEND":
                if chunk_length != 0 or not saw_image_data or chunk_end != len(contents):
                    raise ArtifactMediaTypeError("PNG artifact has an invalid IEND chunk")
                return
            offset = chunk_end

        raise ArtifactMediaTypeError("PNG artifact is missing IEND")

    def _commit_staged_file(self, staged: Path, digest: str, size: int) -> None:
        with self._locked():
            destination = self._object_path(digest)
            self._ensure_digest_directories(digest)
            try:
                destination.lstat()
            except FileNotFoundError:
                existing = False
            else:
                existing = True

            if existing:
                fd, existing_stat = self._open_verified_object(digest)
                os.close(fd)
                if existing_stat.st_size != size:
                    raise ArtifactIntegrityError(
                        "stored artifact size does not match its digest"
                    )
                return

            total_size = self._total_size_locked()
            if total_size + size > self.max_total_bytes:
                raise ArtifactSizeLimitError(
                    "artifact would exceed the total store size limit"
                )

            try:
                staged_stat = staged.lstat()
                if stat.S_ISLNK(staged_stat.st_mode) or not stat.S_ISREG(
                    staged_stat.st_mode
                ):
                    raise ArtifactIntegrityError("staged artifact was replaced")
                if staged_stat.st_size != size:
                    raise ArtifactIntegrityError("staged artifact size changed")
                staged_flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                staged_fd = os.open(staged, staged_flags)
                try:
                    opened_stat = os.fstat(staged_fd)
                    if (
                        not stat.S_ISREG(opened_stat.st_mode)
                        or opened_stat.st_dev != staged_stat.st_dev
                        or opened_stat.st_ino != staged_stat.st_ino
                    ):
                        raise ArtifactIntegrityError("staged artifact was replaced")
                    staged_digest = hashlib.sha256()
                    while True:
                        chunk = os.read(staged_fd, _CHUNK_SIZE)
                        if not chunk:
                            break
                        staged_digest.update(chunk)
                    if staged_digest.hexdigest() != digest:
                        raise ArtifactIntegrityError("staged artifact digest changed")
                finally:
                    os.close(staged_fd)
                os.replace(staged, destination)
                os.chmod(destination, 0o600)
                self._fsync_directory(destination.parent)
            except ArtifactStoreError:
                raise
            except OSError as exc:
                raise ArtifactIntegrityError("artifact could not be committed") from exc

    def _ensure_digest_directories(self, digest: str) -> None:
        self._assert_directory(self._objects_root)
        for directory in (
            self._objects_root / digest[:2],
            self._objects_root / digest[:2] / digest[2:4],
        ):
            try:
                directory.mkdir(mode=0o700, exist_ok=True)
                directory_stat = directory.lstat()
            except OSError as exc:
                raise ArtifactIntegrityError(
                    "artifact object directory is unavailable"
                ) from exc
            if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(
                directory_stat.st_mode
            ):
                raise ArtifactIntegrityError(
                    "artifact object directory was replaced"
                )
            try:
                directory.resolve(strict=True).relative_to(self._objects_root)
            except (OSError, ValueError) as exc:
                raise ArtifactIntegrityError(
                    "artifact object directory escapes the store"
                ) from exc

    def _assert_digest_directories(self, digest: str) -> None:
        self._assert_directory(self._objects_root)
        for directory in (
            self._objects_root / digest[:2],
            self._objects_root / digest[:2] / digest[2:4],
        ):
            try:
                directory_stat = directory.lstat()
            except FileNotFoundError as exc:
                raise ArtifactNotFoundError("artifact key was not found") from exc
            except OSError as exc:
                raise ArtifactIntegrityError(
                    "artifact object directory is unavailable"
                ) from exc
            if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(
                directory_stat.st_mode
            ):
                raise ArtifactIntegrityError(
                    "artifact object directory was replaced"
                )
            try:
                directory.resolve(strict=True).relative_to(self._objects_root)
            except (OSError, ValueError) as exc:
                raise ArtifactIntegrityError(
                    "artifact object directory escapes the store"
                ) from exc

    def _open_verified_object(
        self, digest: str
    ) -> tuple[int, os.stat_result]:
        self._assert_digest_directories(digest)
        path = self._object_path(digest)
        try:
            before = path.lstat()
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError("artifact key was not found") from exc
        except OSError as exc:
            raise ArtifactIntegrityError("stored artifact is unavailable") from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ArtifactIntegrityError(
                "stored artifact must be a regular, non-symbolic-link file"
            )

        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            fd = os.open(path, flags)
            after = os.fstat(fd)
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError("artifact key was not found") from exc
        except OSError as exc:
            raise ArtifactIntegrityError("stored artifact could not be opened") from exc

        if (
            not stat.S_ISREG(after.st_mode)
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
        ):
            os.close(fd)
            raise ArtifactIntegrityError("stored artifact was replaced while opening")

        actual_digest = hashlib.sha256()
        try:
            while True:
                chunk = os.read(fd, _CHUNK_SIZE)
                if not chunk:
                    break
                actual_digest.update(chunk)
            os.lseek(fd, 0, os.SEEK_SET)
        except OSError as exc:
            os.close(fd)
            raise ArtifactIntegrityError("stored artifact could not be verified") from exc
        if actual_digest.hexdigest() != digest:
            os.close(fd)
            raise ArtifactIntegrityError("stored artifact digest does not match its key")
        return fd, after

    def _total_size_locked(self) -> int:
        self._assert_directory(self._objects_root)
        total = 0
        try:
            first_level = list(os.scandir(self._objects_root))
            for first in first_level:
                if not _HEX_PAIR_RE.fullmatch(first.name) or not first.is_dir(
                    follow_symlinks=False
                ):
                    raise ArtifactIntegrityError(
                        "artifact store contains an unexpected object entry"
                    )
                for second in os.scandir(first.path):
                    if not _HEX_PAIR_RE.fullmatch(second.name) or not second.is_dir(
                        follow_symlinks=False
                    ):
                        raise ArtifactIntegrityError(
                            "artifact store contains an unexpected object entry"
                        )
                    for item in os.scandir(second.path):
                        if (
                            not _DIGEST_RE.fullmatch(item.name)
                            or not item.name.startswith(first.name + second.name)
                            or not item.is_file(follow_symlinks=False)
                        ):
                            raise ArtifactIntegrityError(
                                "artifact store contains an unexpected object entry"
                            )
                        total += item.stat(follow_symlinks=False).st_size
        except ArtifactStoreError:
            raise
        except OSError as exc:
            raise ArtifactIntegrityError("artifact store could not be measured") from exc
        return total

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock_flags = (
            os.O_CREAT
            | os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        with self._thread_lock:
            lock_fd = -1
            try:
                lock_fd = os.open(self._lock_path, lock_flags, 0o600)
                lock_stat = os.fstat(lock_fd)
                if not stat.S_ISREG(lock_stat.st_mode):
                    raise ArtifactIntegrityError(
                        "artifact store lock must be a regular file"
                    )
                if fcntl is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except ArtifactStoreError:
                if lock_fd >= 0:
                    os.close(lock_fd)
                raise
            except OSError as exc:
                if lock_fd >= 0:
                    os.close(lock_fd)
                raise ArtifactIntegrityError("artifact store lock is unavailable") from exc
            try:
                yield
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                os.close(lock_fd)

    @staticmethod
    def _storage_key(digest: str) -> str:
        return f"art-sha256-{digest}"

    @staticmethod
    def _digest_from_key(storage_key: str) -> str:
        if not isinstance(storage_key, str):
            raise ArtifactValidationError("artifact storage key must be a string")
        match = _KEY_RE.fullmatch(storage_key)
        if match is None:
            raise ArtifactValidationError("artifact storage key is invalid")
        return match.group(1)

    def _object_path(self, digest: str) -> Path:
        return self._objects_root / digest[:2] / digest[2:4] / digest

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            directory_fd = os.open(directory, flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not support fsync on directories.  os.replace
            # still guarantees callers never observe a partially copied file.
            pass


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactMediaTypeError",
    "ArtifactNotFoundError",
    "ArtifactPathError",
    "ArtifactReference",
    "ArtifactSizeLimitError",
    "ArtifactStore",
    "ArtifactStoreError",
    "ArtifactValidationError",
    "DEFAULT_MAX_IMAGE_PIXELS",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_TOTAL_BYTES",
]
