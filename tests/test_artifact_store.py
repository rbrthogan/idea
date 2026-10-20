import base64
import hashlib
import json
import os
from pathlib import Path

import pytest

from idea.artifact_store import (
    ArtifactIntegrityError,
    ArtifactMediaTypeError,
    ArtifactNotFoundError,
    ArtifactPathError,
    ArtifactSizeLimitError,
    ArtifactStore,
    ArtifactValidationError,
)


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.mark.parametrize(
    ("relative_path", "contents", "declared_type", "expected_type"),
    [
        ("figure.png", PNG_BYTES, "image/png", "image/png"),
        ("result.json", b'{"score": 0.75}', None, "application/json"),
        ("notes.txt", "useful pattern\n".encode(), "text/plain; charset=utf-8", "text/plain"),
        ("render.py", b"print('hello')\n", "text/x-python", "text/x-python"),
    ],
)
def test_import_returns_opaque_serializable_reference_and_copies_file(
    tmp_path: Path,
    relative_path: str,
    contents: bytes,
    declared_type: str | None,
    expected_type: str,
) -> None:
    job_dir = tmp_path / "job"
    output_dir = job_dir / "nested"
    output_dir.mkdir(parents=True)
    source = output_dir / relative_path
    source.write_bytes(contents)
    store = ArtifactStore(tmp_path / "store")

    reference = store.import_file(
        job_dir, f"nested/{relative_path}", declared_type
    )

    expected_digest = hashlib.sha256(contents).hexdigest()
    assert reference == {
        "storage_key": f"art-sha256-{expected_digest}",
        "sha256": expected_digest,
        "size_bytes": len(contents),
        "media_type": expected_type,
    }
    serialized = json.dumps(reference)
    assert str(tmp_path) not in serialized
    assert relative_path not in serialized
    with store.open(reference["storage_key"]) as artifact:
        assert artifact.read() == contents
    assert list((store.root / "staging").iterdir()) == []


def test_content_deduplication_does_not_consume_total_limit_twice(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    contents = b"same contents"
    (job_dir / "one.txt").write_bytes(contents)
    (job_dir / "two.txt").write_bytes(contents)
    store = ArtifactStore(
        tmp_path / "store",
        max_file_bytes=len(contents),
        max_total_bytes=len(contents),
    )

    first = store.import_file(job_dir, "one.txt")
    second = store.import_output(job_dir, "two.txt")

    assert first["storage_key"] == second["storage_key"]
    assert store.total_size_bytes() == len(contents)
    assert len(list((store.root / "objects" / "sha256").glob("*/*/*"))) == 1


def test_macos_metadata_does_not_block_artifact_imports(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    first_contents = b"first artifact"
    second_contents = b"second artifact"
    (job_dir / "first.txt").write_bytes(first_contents)
    (job_dir / "second.txt").write_bytes(second_contents)
    store = ArtifactStore(tmp_path / "store")
    objects_root = store.root / "objects" / "sha256"
    (objects_root / ".DS_Store").write_bytes(b"a")

    first = store.import_file(job_dir, "first.txt")
    first_digest = str(first["sha256"])
    first_level = objects_root / first_digest[:2]
    second_level = first_level / first_digest[2:4]
    (first_level / ".DS_Store").write_bytes(b"bb")
    (second_level / ".DS_Store").write_bytes(b"ccc")

    second = store.import_file(job_dir, "second.txt")

    assert second["size_bytes"] == len(second_contents)
    assert store.total_size_bytes() == (
        len(first_contents) + len(second_contents) + 1 + 2 + 3
    )


def test_macos_metadata_name_must_still_be_a_regular_file(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    (store.root / "objects" / "sha256" / ".DS_Store").mkdir()

    with pytest.raises(ArtifactIntegrityError):
        store.total_size_bytes()


@pytest.mark.parametrize(
    "manifest_path",
    [
        "",
        ".",
        "./result.txt",
        "../result.txt",
        "nested/../../result.txt",
        "nested//result.txt",
        "/absolute/result.txt",
        r"C:\absolute\result.txt",
        r"nested\..\result.txt",
        r"\\server\share\result.txt",
    ],
)
def test_manifest_path_rejects_absolute_and_traversal_forms(
    tmp_path: Path, manifest_path: str
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    store = ArtifactStore(tmp_path / "store")

    with pytest.raises(ArtifactPathError):
        store.import_file(job_dir, manifest_path)


def test_import_rejects_file_and_parent_symlinks(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (job_dir / "file-link.txt").symlink_to(outside / "secret.txt")
    (job_dir / "directory-link").symlink_to(outside, target_is_directory=True)
    store = ArtifactStore(tmp_path / "store")

    with pytest.raises(ArtifactPathError):
        store.import_file(job_dir, "file-link.txt")
    with pytest.raises(ArtifactPathError):
        store.import_file(job_dir, "directory-link/secret.txt")


def test_import_rejects_non_regular_outputs(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "directory.txt").mkdir()
    store = ArtifactStore(tmp_path / "store")

    with pytest.raises(ArtifactPathError):
        store.import_file(job_dir, "directory.txt")

    if hasattr(os, "mkfifo"):
        os.mkfifo(job_dir / "pipe.txt")
        with pytest.raises(ArtifactPathError):
            store.import_file(job_dir, "pipe.txt")


def test_import_rejects_disallowed_or_mismatched_media(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    store = ArtifactStore(tmp_path / "store")

    (job_dir / "vector.svg").write_text("<svg/>")
    with pytest.raises(ArtifactMediaTypeError):
        store.import_file(job_dir, "vector.svg")

    (job_dir / "notes.txt").write_text("notes")
    with pytest.raises(ArtifactMediaTypeError):
        store.import_file(job_dir, "notes.txt", "application/json")

    (job_dir / "fake.png").write_bytes(b"not a png")
    with pytest.raises(ArtifactMediaTypeError):
        store.import_file(job_dir, "fake.png")

    (job_dir / "truncated.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ArtifactMediaTypeError):
        store.import_file(job_dir, "truncated.png")

    (job_dir / "broken.json").write_text("{not json}")
    with pytest.raises(ArtifactMediaTypeError):
        store.import_file(job_dir, "broken.json")

    (job_dir / "nonfinite.json").write_text('{"score": NaN}')
    with pytest.raises(ArtifactMediaTypeError):
        store.import_file(job_dir, "nonfinite.json")

    (job_dir / "binary.txt").write_bytes(b"\xff\xfe")
    with pytest.raises(ArtifactMediaTypeError):
        store.import_file(job_dir, "binary.txt")

    (job_dir / "nul.py").write_bytes(b"x = 1\x00")
    with pytest.raises(ArtifactMediaTypeError):
        store.import_file(job_dir, "nul.py")

    assert list((store.root / "staging").iterdir()) == []


def test_per_file_and_total_size_limits_are_enforced(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "too-large.txt").write_bytes(b"123456")
    per_file_store = ArtifactStore(
        tmp_path / "per-file-store", max_file_bytes=5, max_total_bytes=20
    )

    with pytest.raises(ArtifactSizeLimitError):
        per_file_store.import_file(job_dir, "too-large.txt")

    (job_dir / "first.txt").write_bytes(b"123456")
    (job_dir / "second.txt").write_bytes(b"abcde")
    total_store = ArtifactStore(
        tmp_path / "total-store", max_file_bytes=6, max_total_bytes=10
    )
    total_store.import_file(job_dir, "first.txt")
    with pytest.raises(ArtifactSizeLimitError):
        total_store.import_file(job_dir, "second.txt")
    assert total_store.total_size_bytes() == 6


@pytest.mark.parametrize(
    "storage_key",
    [
        "",
        "../artifact",
        "/absolute/artifact",
        "0" * 64,
        "art-sha256-" + "A" * 64,
        "art-sha256-" + "0" * 64 + "/extra",
    ],
)
def test_key_operations_reject_non_opaque_keys(
    tmp_path: Path, storage_key: str
) -> None:
    store = ArtifactStore(tmp_path / "store")

    with pytest.raises(ArtifactValidationError):
        store.resolve(storage_key)
    with pytest.raises(ArtifactValidationError):
        store.open(storage_key)
    with pytest.raises(ArtifactValidationError):
        store.delete(storage_key)


def test_valid_missing_key_has_explicit_not_found_semantics(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    missing_key = "art-sha256-" + "0" * 64

    with pytest.raises(ArtifactNotFoundError):
        store.resolve(missing_key)
    with pytest.raises(ArtifactNotFoundError):
        store.open(missing_key)
    assert store.delete(missing_key) is False


def test_delete_is_contained_and_refuses_a_substituted_symlink(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "artifact.txt").write_text("artifact")
    store = ArtifactStore(tmp_path / "store")
    reference = store.import_file(job_dir, "artifact.txt")
    storage_key = str(reference["storage_key"])
    stored_path = store.resolve(storage_key)

    outside = tmp_path / "outside.txt"
    outside.write_text("do not delete")
    stored_path.unlink()
    stored_path.symlink_to(outside)

    with pytest.raises(ArtifactIntegrityError):
        store.open(storage_key)
    with pytest.raises(ArtifactIntegrityError):
        store.delete(storage_key)
    assert outside.read_text() == "do not delete"


def test_delete_removes_only_the_requested_object(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "one.txt").write_text("one")
    (job_dir / "two.txt").write_text("two")
    store = ArtifactStore(tmp_path / "store")
    first = store.import_file(job_dir, "one.txt")
    second = store.import_file(job_dir, "two.txt")

    assert store.delete(str(first["storage_key"])) is True
    assert store.delete(str(first["storage_key"])) is False
    with store.open(str(second["storage_key"]), "r") as artifact:
        assert artifact.read() == "two"


@pytest.mark.parametrize("mode", ["w", "wb", "r+", "a"])
def test_open_rejects_write_capable_modes(tmp_path: Path, mode: str) -> None:
    store = ArtifactStore(tmp_path / "store")

    with pytest.raises(ArtifactValidationError):
        store.open("art-sha256-" + "0" * 64, mode)


def test_resolve_detects_tampered_object_contents(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "artifact.txt").write_text("original")
    store = ArtifactStore(tmp_path / "store")
    reference = store.import_file(job_dir, "artifact.txt")
    storage_key = str(reference["storage_key"])
    stored_path = store.resolve(storage_key)
    stored_path.write_text("tampered")

    with pytest.raises(ArtifactIntegrityError):
        store.resolve(storage_key)
