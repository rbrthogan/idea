"""Acceptance manifests use service fakes; no cloud calls or plaintext records."""

from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location("deployment_acceptance", Path(__file__).parents[1] / "scripts/deployment_acceptance.py")
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)
NOW = datetime(2026, 10, 18, 12, tzinfo=timezone.utc)
KEY = b"offline-fingerprint-key-32-bytes!!"


class FakeClient:
    def __init__(self, store):
        self.store = store
        self.calls = []

    def collection(self, identifier):
        return FakeCollection(self, (identifier,))

    def collections(self, **kwargs):
        self.calls.append(("root.collections", kwargs))
        return [FakeCollection(self, (name,)) for name in sorted({path[0] for path in self.store})]


class FakeCollection:
    def __init__(self, client, path):
        self.client, self.path = client, path

    def document(self, identifier):
        return FakeReference(self.client, self.path + (identifier,))

    def list_documents(self, **kwargs):
        self.client.calls.append(("list_documents", kwargs))
        width = len(self.path)
        identifiers = {path[width] for path in self.client.store if path[:width] == self.path and len(path) > width}
        return [self.document(identifier) for identifier in sorted(identifiers)]


class FakeReference:
    def __init__(self, client, path):
        self.client, self.parts = client, path
        self.path = "/".join(path)

    def get(self, **kwargs):
        self.client.calls.append(("get", kwargs))
        value = self.client.store.get(self.parts)
        return SimpleNamespace(exists=value is not None, read_time=NOW, to_dict=lambda: deepcopy(value))

    def collections(self, **kwargs):
        self.client.calls.append(("document.collections", kwargs))
        width = len(self.parts)
        names = {path[width] for path in self.client.store if path[:width] == self.parts and len(path) > width}
        return [FakeCollection(self.client, self.parts + (name,)) for name in sorted(names)]


def sample_client():
    now_ms = int(NOW.timestamp() * 1000)
    return FakeClient({
        ("users", "private-user", "settings", "main"): {"api_key_encrypted": "PRIVATE-CIPHERTEXT", "email": "private@example.test"},
        ("users", "private-user", "evolutions", "private-run", "chunks", "private-chunk"): {"value": "PRIVATE-STORY"},
        ("users", "private-user", "runs", "active"): {"status": "complete", "active": True, "run_token": "PRIVATE-TOKEN", "lease_expires_at_ms": now_ms + 30_000},
        ("users", "expired-user", "runs", "active"): {"status": "in_progress", "lease_expires_at_ms": now_ms - 1},
        ("system", "run_lock"): {"owner_id": "worker-PRIVATE-OWNER", "lease_expires_at_ms": now_ms + 30_000},
    })


def capture(client):
    return acceptance.capture(client, project="test-project", database="(default)", key=KEY)


def test_capture_includes_missing_parent_children_and_never_outputs_private_values():
    client = sample_client()
    result = capture(client)
    assert result["document_count"] == 5
    assert result["missing_parent_count"] == 3
    assert result["counts"] == {"chunks": 1, "runs": 2, "settings": 1, "system": 1}
    assert result["live_user_work_count"] == 1  # Terminal status still has a live owner.
    assert result["expired_user_work_count"] == 1
    encoded = json.dumps(result)
    for private in ("PRIVATE", "private-user", "expired-user", "private-run", "private@example"):
        assert private not in encoded
    assert all(kwargs.get("read_time") == NOW for _, kwargs in client.calls[1:])
    assert result["not_a_backup"] is True


def test_compare_distinguishes_unchanged_added_removed_and_changed_documents():
    client = sample_client()
    before = capture(client)
    assert acceptance.compare(before, capture(client))["unchanged"] is True
    client.store[("users", "private-user", "settings", "main")]["api_key_encrypted"] = "NEW-PRIVATE-CIPHERTEXT"
    client.store.pop(("users", "expired-user", "runs", "active"))
    client.store[("users", "private-user", "templates", "new-private-template")] = {"prompt": "PRIVATE-TEMPLATE"}
    changes = acceptance.compare(before, capture(client))
    assert changes["unchanged"] is False
    assert [entry["kind"] for entry in changes["added"]] == ["templates"]
    assert [entry["kind"] for entry in changes["removed"]] == ["runs"]
    assert [entry["kind"] for entry in changes["changed"]] == ["settings"]
    assert "PRIVATE" not in json.dumps(changes)


@pytest.mark.parametrize("field", ["project", "database", "key_id", "format_version"])
def test_compare_rejects_unrelated_baselines(field):
    before = capture(sample_client())
    after = deepcopy(before)
    after[field] = "different"
    with pytest.raises(acceptance.AcceptanceError, match=field):
        acceptance.compare(before, after)


def test_document_limit_fails_instead_of_emitting_partial_success():
    with pytest.raises(acceptance.AcceptanceError, match="incomplete"):
        acceptance.capture(sample_client(), project="test-project", database="(default)", key=KEY, max_documents=2)


def test_private_key_is_reused_and_baselines_cannot_be_overwritten(tmp_path):
    path = tmp_path / "acceptance.key"
    first = acceptance.load_key(path)
    assert len(first) == 32
    assert acceptance.load_key(path) == first
    assert path.stat().st_mode & 0o077 == 0
    evidence = tmp_path / "baseline.json"
    acceptance.private_write(evidence, b"original")
    with pytest.raises(FileExistsError):
        acceptance.private_write(evidence, b"replacement")
    assert evidence.read_bytes() == b"original"
    assert evidence.stat().st_mode & 0o077 == 0


def test_production_capture_refuses_emulator_before_client_or_key_access(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FIRESTORE_EMULATOR_HOST", "localhost:8080")
    key = tmp_path / "key"
    output = tmp_path / "baseline.json"
    assert acceptance.main(["capture", "--project", "test-project", "--key-file", str(key), "--output", str(output)]) == 1
    assert "refusing" in capsys.readouterr().err
    assert not key.exists() and not output.exists()


def test_canonical_values_keep_type_distinctions_and_timestamp_precision():
    assert acceptance.fingerprint(KEY, "content", b"secret") != acceptance.fingerprint(KEY, "content", "secret")
    assert acceptance.fingerprint(KEY, "content", 1) != acceptance.fingerprint(KEY, "content", 1.0)
    assert acceptance.fingerprint(KEY, "content", {"b": 1, "a": 2}) == acceptance.fingerprint(KEY, "content", {"a": 2, "b": 1})
    from google.api_core.datetime_helpers import DatetimeWithNanoseconds
    value = DatetimeWithNanoseconds(2026, 10, 18, nanosecond=123456789, tzinfo=timezone.utc)
    assert "123456789" in acceptance.canonical(value)["$timestamp"]
