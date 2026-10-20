"""Private, content-addressed Cloud Storage artifacts across web/worker instances."""

import hashlib
import io
import re
import tempfile
import threading
from pathlib import Path

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

from idea.artifact_store import (ArtifactStore, ArtifactValidationError,
    ArtifactSizeLimitError, ArtifactIntegrityError, ArtifactNotFoundError)


class CloudArtifactStore(ArtifactStore):
    """Validate imports locally and publish immutable objects to a private bucket.

    A run has one fenced writer. The in-process lock serializes its artifact
    uploads and quota accounting. Reads bind a generation and verify SHA-256;
    no public URLs, bearer tokens, or user-supplied object paths are exposed.
    """

    def __init__(self, root, bucket_name, prefix, *, bucket=None, **limits):
        if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{1,221}[a-z0-9]', bucket_name):
            raise ArtifactValidationError('Invalid artifact bucket name')
        if not re.fullmatch(r'[a-f0-9]{32}/[a-f0-9]{32}', prefix):
            raise ArtifactValidationError('Invalid artifact ownership prefix')
        super().__init__(root, **limits)
        self.bucket = bucket or storage.Client().bucket(bucket_name)
        self.prefix = prefix + '/'
        self._remote_lock = threading.RLock()
        self._remote_sizes = None

    def _blob(self, key):
        digest = self._digest_from_key(key)
        return self.bucket.blob(self.prefix + digest)

    def _read_blob(self, key):
        blob = self._blob(key)
        try:
            blob.reload(timeout=30, retry=None)
            size = int(blob.size)
            if size < 0 or size > self.max_file_bytes:
                raise ArtifactSizeLimitError('Cloud artifact exceeds the per-file limit')
            contents = blob.download_as_bytes(
                start=0, end=self.max_file_bytes, raw_download=True,
                if_generation_match=blob.generation, timeout=30, retry=None,
            )
        except NotFound as exc:
            raise ArtifactNotFoundError('Cloud artifact is unavailable') from exc
        except PreconditionFailed as exc:
            raise ArtifactIntegrityError('Cloud artifact changed during read') from exc
        if len(contents) != size or len(contents) > self.max_file_bytes:
            raise ArtifactIntegrityError('Cloud artifact size mismatch')
        if hashlib.sha256(contents).hexdigest() != self._digest_from_key(key):
            raise ArtifactIntegrityError('Cloud artifact digest mismatch')
        return contents, blob

    def import_file(self, job_output_dir, manifest_path, media_type=None):
        ref = super().import_file(job_output_dir, manifest_path, media_type)
        with self._remote_lock:
            if self._remote_sizes is None:
                sizes = {}
                total = 0
                for blob in self.bucket.list_blobs(prefix=self.prefix, timeout=30, retry=None):
                    total += int(blob.size or 0)
                    sizes[blob.name] = int(blob.size or 0)
                    if total > self.max_total_bytes:
                        raise ArtifactSizeLimitError('Cloud artifact store exceeds its run quota')
                self._remote_sizes = sizes
            blob = self._blob(ref['storage_key'])
            existing = self._remote_sizes.get(blob.name, 0)
            if sum(self._remote_sizes.values()) - existing + ref['size_bytes'] > self.max_total_bytes:
                raise ArtifactSizeLimitError('Cloud artifact would exceed its run quota')
            try:
                with super().open(ref['storage_key'], 'rb') as stream:
                    blob.upload_from_file(stream, size=ref['size_bytes'],
                        content_type=ref['media_type'], if_generation_match=0,
                        timeout=30, retry=None)
            except PreconditionFailed:
                # Duplicate content is safe only if the immutable object is
                # intact. Never overwrite an existing digest-named object.
                self._read_blob(ref['storage_key'])
            self._remote_sizes[blob.name] = ref['size_bytes']
        return ref

    def open(self, storage_key, mode='rb', *, encoding=None):
        if mode not in {'rb', 'r', 'rt'} or (mode == 'rb' and encoding is not None):
            raise ArtifactValidationError('Artifacts may only be opened read-only')
        contents, _ = self._read_blob(storage_key)
        stream = io.BytesIO(contents)
        return stream if mode == 'rb' else io.TextIOWrapper(stream, encoding=encoding or 'utf-8')

    def resolve(self, storage_key):
        contents, _ = self._read_blob(storage_key)
        digest = self._digest_from_key(storage_key)
        with tempfile.NamedTemporaryFile(dir=self._staging_root, delete=False) as stream:
            staged = Path(stream.name)
            stream.write(contents)
        try:
            self._commit_staged_file(staged, digest, len(contents))
            return super().resolve(storage_key)
        finally:
            staged.unlink(missing_ok=True)

    def delete(self, storage_key):
        with self._remote_lock:
            try:
                _, blob = self._read_blob(storage_key)
                blob.delete(if_generation_match=blob.generation, timeout=30, retry=None)
            except ArtifactNotFoundError:
                return False
            except PreconditionFailed as exc:
                raise ArtifactIntegrityError('Cloud artifact changed during deletion') from exc
            if self._remote_sizes is not None:
                self._remote_sizes.pop(blob.name, None)
            super().delete(storage_key)
            return True

    def total_size_bytes(self):
        total = 0
        for blob in self.bucket.list_blobs(prefix=self.prefix, timeout=30, retry=None):
            total += int(blob.size or 0)
            if total > self.max_total_bytes:
                raise ArtifactSizeLimitError('Cloud artifact store exceeds its run quota')
        return total
