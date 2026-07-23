import asyncio
import hashlib
import json
from copy import deepcopy

import pytest

from idea import database


class FakeDocSnapshot:
    def __init__(self, doc_id, data):
        self.id = doc_id
        self._data = deepcopy(data) if data is not None else None
        self.exists = data is not None

    def to_dict(self):
        return deepcopy(self._data)


class FakeDocumentRef:
    def __init__(self, root, path):
        self.root = root
        self.path = tuple(path)

    def collection(self, name):
        return FakeCollectionRef(self.root, self.path + (name,))

    def set(self, data, merge=False):
        if merge and self.path in self.root.store:
            next_data = deepcopy(self.root.store[self.path])
            next_data.update(deepcopy(data))
        else:
            next_data = deepcopy(data)
        self.root.store[self.path] = next_data

    def update(self, data):
        next_data = deepcopy(self.root.store.get(self.path, {}))
        next_data.update(deepcopy(data))
        self.root.store[self.path] = next_data

    def get(self, transaction=None):
        return FakeDocSnapshot(self.path[-1], self.root.store.get(self.path))

    def delete(self):
        self.root.store.pop(self.path, None)


class FakeCollectionRef:
    def __init__(self, root, path):
        self.root = root
        self.path = tuple(path)
        self._limit = None

    def document(self, doc_id):
        return FakeDocumentRef(self.root, self.path + (doc_id,))

    def order_by(self, *_args, **_kwargs):
        return self

    def limit(self, value):
        self._limit = value
        return self

    def stream(self):
        docs = []
        doc_path_len = len(self.path) + 1
        for path, data in self.root.store.items():
            if len(path) == doc_path_len and path[: len(self.path)] == self.path:
                docs.append(FakeDocSnapshot(path[-1], data))
        docs.sort(key=lambda doc: doc.to_dict().get("updated_at", ""), reverse=True)
        if self._limit is not None:
            docs = docs[: self._limit]
        return docs


class FakeFirestore:
    def __init__(self):
        self.store = {}

    def collection(self, name):
        return FakeCollectionRef(self, (name,))


class FakeTransaction:
    def __init__(self, root):
        self.root = root

    def set(self, doc_ref, data, merge=False):
        doc_ref.set(data, merge=merge)

    def delete(self, doc_ref):
        doc_ref.delete()


def test_save_evolution_externalizes_large_history_and_reconstructs(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(database, "EVOLUTION_FIELD_CHUNK_SIZE", 80, raising=False)

    history = [
        [{"title": "Seed A", "content": "alpha " * 20}],
        [{"title": "Child A", "content": "beta " * 20}],
    ]
    payload = {
        "evolution_id": "evo-1",
        "name": "Chunked Evolution",
        "status": "complete",
        "current_generation": 1,
        "config": {"generations": 1, "pop_size": 1, "idea_type": "game"},
        "history": history,
        "population": history[-1],
        "token_counts": {"total": 12, "diagnostics": {"events": [{"message": "x" * 90}]}},
    }

    asyncio.run(database.save_evolution("user-1", "evo-1", payload))

    doc_path = ("users", "user-1", "evolutions", "evo-1")
    saved_doc = fake_db.store[doc_path]
    assert "history" not in saved_doc
    assert saved_doc["_externalized_fields"]["history"]["chunk_count"] > 1
    assert saved_doc["_history_summary"] == {
        "generation_count": 2,
        "total_ideas": 2,
        "last_generation_size": 1,
    }

    chunk_prefix = doc_path + ("chunks",)
    assert any(path[: len(chunk_prefix)] == chunk_prefix for path in fake_db.store)

    loaded = asyncio.run(database.get_evolution("user-1", "evo-1"))
    assert loaded["history"] == history
    assert loaded["population"] == history[-1]
    assert loaded["token_counts"] == payload["token_counts"]


def test_delete_evolution_removes_externalized_chunks(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(database, "EVOLUTION_FIELD_CHUNK_SIZE", 80, raising=False)

    payload = {
        "evolution_id": "evo-delete",
        "name": "Delete Me",
        "status": "complete",
        "current_generation": 1,
        "config": {"generations": 1, "pop_size": 1},
        "history": [[{"title": "A", "content": "large " * 30}]],
    }
    asyncio.run(database.save_evolution("user-1", "evo-delete", payload))

    assert asyncio.run(database.delete_evolution("user-1", "evo-delete")) is True

    doc_prefix = ("users", "user-1", "evolutions", "evo-delete")
    assert not any(path[: len(doc_prefix)] == doc_prefix for path in fake_db.store)


def test_save_evolution_removes_stale_chunk_versions_after_success(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(database, "EVOLUTION_FIELD_CHUNK_SIZE", 80, raising=False)
    monkeypatch.setattr(database, "EVOLUTION_CHUNK_GC_GRACE_SECONDS", 0, raising=False)

    payload = {
        "evolution_id": "evo-clean",
        "name": "Clean Chunks",
        "status": "in_progress",
        "current_generation": 1,
        "config": {"generations": 2, "pop_size": 1},
        "history": [[{"title": "A", "content": "first " * 30}]],
    }
    first_revision = asyncio.run(
        database.save_evolution("user-1", "evo-clean", payload)
    )

    doc_path = ("users", "user-1", "evolutions", "evo-clean")
    first_version = fake_db.store[doc_path]["_externalized_fields"]["history"]["chunk_version"]

    payload["history"] = [[{"title": "A", "content": "second " * 30}]]
    payload["status"] = "complete"
    payload["_save_revision"] = first_revision
    asyncio.run(database.save_evolution("user-1", "evo-clean", payload))

    second_version = fake_db.store[doc_path]["_externalized_fields"]["history"]["chunk_version"]
    assert second_version != first_version

    chunk_prefix = doc_path + ("chunks",)
    chunk_docs = [
        data
        for path, data in fake_db.store.items()
        if path[: len(chunk_prefix)] == chunk_prefix
    ]
    assert chunk_docs
    assert {doc["chunk_version"] for doc in chunk_docs} == {second_version}


def test_manifest_publish_rejects_stale_writer(monkeypatch):
    fake_db = FakeFirestore()
    fake_db.transaction = lambda: FakeTransaction(fake_db)
    monkeypatch.setattr(
        database.firestore,
        "transactional",
        lambda function: lambda transaction: function(transaction),
    )

    doc_ref = (
        fake_db.collection("users")
        .document("user-1")
        .collection("evolutions")
        .document("evo-race")
    )
    doc_ref.set({"_save_revision": "base", "name": "Base"})

    winner = {"_save_revision": "winner", "name": "Winner"}
    stale = {"_save_revision": "stale", "name": "Stale"}

    assert database._publish_evolution_manifest(
        fake_db, doc_ref, winner, "base"
    ) is True
    assert database._publish_evolution_manifest(
        fake_db, doc_ref, stale, "base"
    ) is False
    assert doc_ref.get().to_dict() == winner


def test_stale_chunk_gc_preserves_fresh_unpublished_versions(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "EVOLUTION_CHUNK_GC_GRACE_SECONDS", 600, raising=False)
    now_ms = 2_000_000
    monkeypatch.setattr(database, "_now_ms", lambda: now_ms)

    doc_ref = (
        fake_db.collection("users")
        .document("user-1")
        .collection("evolutions")
        .document("evo-gc")
    )
    doc_ref.set(
        {
            "_save_revision": "current",
            "_externalized_fields": {
                "history": {"chunk_version": "current", "chunk_count": 1}
            },
        }
    )
    chunks = doc_ref.collection(database.EVOLUTION_CHUNKS_COLLECTION)
    chunks.document("history__current__0000").set(
        {"chunk_version": "current", "created_at_ms": 1}
    )
    chunks.document("history__old__0000").set(
        {"chunk_version": "old", "created_at_ms": 1}
    )
    chunks.document("history__staged__0000").set(
        {"chunk_version": "staged", "created_at_ms": now_ms}
    )

    database._delete_stale_evolution_chunks(doc_ref)

    remaining_versions = {
        snapshot.to_dict()["chunk_version"] for snapshot in chunks.stream()
    }
    assert remaining_versions == {"current", "staged"}


def _evolution_doc_ref(fake_db, evolution_id):
    return (
        fake_db.collection("users")
        .document("user-1")
        .collection("evolutions")
        .document(evolution_id)
    )


def _chunk_versions(doc_ref):
    return {
        (snapshot.to_dict() or {}).get("chunk_version")
        for snapshot in doc_ref.collection(database.EVOLUTION_CHUNKS_COLLECTION).stream()
    }


def test_sequential_stale_writer_cannot_overwrite_loaded_revision(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(database, "EVOLUTION_FIELD_CHUNK_SIZE", 60, raising=False)

    base = {
        "evolution_id": "evo-cas",
        "name": "CAS",
        "history": [[{"title": "base", "content": "x" * 100}]],
    }
    asyncio.run(database.save_evolution("user-1", "evo-cas", base))
    writer_a = asyncio.run(database.get_evolution("user-1", "evo-cas"))
    writer_b = asyncio.run(database.get_evolution("user-1", "evo-cas"))

    writer_a["history"] = [[{"title": "winner", "content": "a" * 100}]]
    asyncio.run(database.save_evolution("user-1", "evo-cas", writer_a))
    writer_b["history"] = [[{"title": "stale", "content": "b" * 100}]]

    with pytest.raises(database.EvolutionSaveConflict):
        asyncio.run(database.save_evolution("user-1", "evo-cas", writer_b))

    loaded = asyncio.run(database.get_evolution("user-1", "evo-cas"))
    assert loaded["history"][0][0]["title"] == "winner"
    version_states = [
        snapshot.to_dict()["state"]
        for snapshot in _evolution_doc_ref(fake_db, "evo-cas")
        .collection(database.EVOLUTION_CHUNK_VERSIONS_COLLECTION)
        .stream()
    ]
    assert "staged" not in version_states


def test_gc_grace_begins_when_version_is_retired_not_when_created(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(database, "EVOLUTION_FIELD_CHUNK_SIZE", 60, raising=False)
    monkeypatch.setattr(database, "EVOLUTION_CHUNK_GC_GRACE_SECONDS", 600, raising=False)
    clock = {"now": 1_000}
    monkeypatch.setattr(database, "_now_ms", lambda: clock["now"])

    base = {
        "evolution_id": "evo-retire",
        "history": [[{"title": "old", "content": "o" * 100}]],
    }
    first_revision = asyncio.run(
        database.save_evolution("user-1", "evo-retire", base)
    )
    # The first version has been current for far longer than the GC grace.
    clock["now"] = 2_000_000
    updated = dict(base)
    updated["_save_revision"] = first_revision
    updated["history"] = [[{"title": "new", "content": "n" * 100}]]
    second_revision = asyncio.run(
        database.save_evolution("user-1", "evo-retire", updated)
    )

    doc_ref = _evolution_doc_ref(fake_db, "evo-retire")
    assert _chunk_versions(doc_ref) == {first_revision, second_revision}

    clock["now"] += 600_001
    database._delete_stale_evolution_chunks(doc_ref)
    assert _chunk_versions(doc_ref) == {second_revision}


def test_missing_published_chunk_raises_typed_integrity_error(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(database, "EVOLUTION_FIELD_CHUNK_SIZE", 50, raising=False)
    asyncio.run(
        database.save_evolution(
            "user-1",
            "evo-missing",
            {"history": [[{"title": "A", "content": "x" * 100}]]},
        )
    )
    doc_ref = _evolution_doc_ref(fake_db, "evo-missing")
    first_chunk = next(
        iter(doc_ref.collection(database.EVOLUTION_CHUNKS_COLLECTION).stream())
    )
    doc_ref.collection(database.EVOLUTION_CHUNKS_COLLECTION).document(
        first_chunk.id
    ).delete()

    with pytest.raises(database.EvolutionStorageIntegrityError, match="missing chunk"):
        asyncio.run(database.get_evolution("user-1", "evo-missing"))


def test_corrupt_chunk_digest_raises_typed_integrity_error(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(database, "EVOLUTION_FIELD_CHUNK_SIZE", 10_000, raising=False)
    asyncio.run(
        database.save_evolution(
            "user-1",
            "evo-digest",
            {"history": [[{"title": "A", "content": "valid"}]]},
        )
    )
    doc_ref = _evolution_doc_ref(fake_db, "evo-digest")
    chunk = next(iter(doc_ref.collection(database.EVOLUTION_CHUNKS_COLLECTION).stream()))
    corrupt = chunk.to_dict()
    corrupt["value"] = corrupt["value"] + " "
    doc_ref.collection(database.EVOLUTION_CHUNKS_COLLECTION).document(chunk.id).set(corrupt)

    with pytest.raises(database.EvolutionStorageIntegrityError, match="digest mismatch"):
        asyncio.run(database.get_evolution("user-1", "evo-digest"))


def test_malformed_chunk_json_is_not_returned_as_raw_text(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(database, "EVOLUTION_FIELD_CHUNK_SIZE", 10_000, raising=False)
    asyncio.run(
        database.save_evolution(
            "user-1",
            "evo-json",
            {"history": [[{"title": "A", "content": "valid"}]]},
        )
    )
    doc_ref = _evolution_doc_ref(fake_db, "evo-json")
    chunk = next(iter(doc_ref.collection(database.EVOLUTION_CHUNKS_COLLECTION).stream()))
    chunk_data = chunk.to_dict()
    chunk_data["value"] = "{"
    doc_ref.collection(database.EVOLUTION_CHUNKS_COLLECTION).document(chunk.id).set(chunk_data)
    parent = doc_ref.get().to_dict()
    parent["_externalized_fields"]["history"]["payload_sha256"] = hashlib.sha256(
        b"{"
    ).hexdigest()
    doc_ref.set(parent)

    with pytest.raises(database.EvolutionStorageIntegrityError, match="not valid JSON"):
        asyncio.run(database.get_evolution("user-1", "evo-json"))


def test_reader_retries_once_when_manifest_changes_during_reconstruction(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(database, "EVOLUTION_FIELD_CHUNK_SIZE", 60, raising=False)

    base = {"history": [[{"title": "old", "content": "o" * 100}]]}
    first_revision = asyncio.run(
        database.save_evolution("user-1", "evo-reader", base)
    )
    old_parent = _evolution_doc_ref(fake_db, "evo-reader").get().to_dict()
    updated = dict(base)
    updated["_save_revision"] = first_revision
    updated["history"] = [[{"title": "new", "content": "n" * 100}]]
    asyncio.run(database.save_evolution("user-1", "evo-reader", updated))

    doc_path = ("users", "user-1", "evolutions", "evo-reader")
    new_parent = deepcopy(fake_db.store[doc_path])
    old_version_path = doc_path + (
        database.EVOLUTION_CHUNK_VERSIONS_COLLECTION,
        first_revision,
    )
    old_current_version = deepcopy(fake_db.store[old_version_path])
    old_current_version["state"] = "current"
    fake_db.store[doc_path] = old_parent
    fake_db.store[old_version_path] = old_current_version

    original_get = FakeDocumentRef.get
    parent_reads = {"count": 0}

    def switching_get(self, transaction=None):
        if self.path == doc_path and parent_reads["count"] == 0:
            parent_reads["count"] += 1
            snapshot = FakeDocSnapshot(self.path[-1], old_parent)
            fake_db.store[doc_path] = deepcopy(new_parent)
            fake_db.store[old_version_path]["state"] = "retired"
            return snapshot
        return original_get(self, transaction=transaction)

    monkeypatch.setattr(FakeDocumentRef, "get", switching_get)
    loaded = asyncio.run(database.get_evolution("user-1", "evo-reader"))
    assert loaded["history"][0][0]["title"] == "new"


def test_delete_tombstone_prevents_stale_save_resurrection(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(database, "EVOLUTION_FIELD_CHUNK_SIZE", 60, raising=False)
    asyncio.run(
        database.save_evolution(
            "user-1",
            "evo-delete-race",
            {"history": [[{"title": "A", "content": "x" * 100}]]},
        )
    )
    stale = asyncio.run(database.get_evolution("user-1", "evo-delete-race"))

    assert asyncio.run(
        database.delete_evolution("user-1", "evo-delete-race")
    ) is True
    with pytest.raises(database.EvolutionSaveConflict):
        asyncio.run(
            database.save_evolution("user-1", "evo-delete-race", stale)
        )

    assert asyncio.run(database.get_evolution("user-1", "evo-delete-race")) is None
    assert _chunk_versions(_evolution_doc_ref(fake_db, "evo-delete-race")) == set()
    tombstone_path = (
        "users",
        "user-1",
        database.EVOLUTION_TOMBSTONES_COLLECTION,
        "evo-delete-race",
    )
    assert tombstone_path in fake_db.store


def test_post_publish_gc_failure_does_not_report_save_failure(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(
        database,
        "_delete_stale_evolution_chunks",
        lambda _doc_ref: (_ for _ in ()).throw(RuntimeError("gc failed")),
    )

    revision = asyncio.run(
        database.save_evolution("user-1", "evo-gc-error", {"history": [[]]})
    )
    assert isinstance(revision, str) and revision
    assert asyncio.run(database.get_evolution("user-1", "evo-gc-error")) is not None


def test_ambiguous_publish_error_never_deletes_possibly_current_chunks(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    original_publish = database._publish_evolution_manifest

    def publish_then_lose_response(*args, **kwargs):
        assert original_publish(*args, **kwargs) is True
        raise RuntimeError("connection lost after commit")

    monkeypatch.setattr(
        database,
        "_publish_evolution_manifest",
        publish_then_lose_response,
    )

    with pytest.raises(RuntimeError, match="connection lost"):
        asyncio.run(
            database.save_evolution(
                "user-1",
                "evo-ambiguous",
                {"history": [[{"title": "A", "content": "kept"}]]},
            )
        )

    loaded = asyncio.run(database.get_evolution("user-1", "evo-ambiguous"))
    assert loaded["history"][0][0]["content"] == "kept"
