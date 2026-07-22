import asyncio
from copy import deepcopy

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

    def get(self):
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

    payload = {
        "evolution_id": "evo-clean",
        "name": "Clean Chunks",
        "status": "in_progress",
        "current_generation": 1,
        "config": {"generations": 2, "pop_size": 1},
        "history": [[{"title": "A", "content": "first " * 30}]],
    }
    asyncio.run(database.save_evolution("user-1", "evo-clean", payload))

    doc_path = ("users", "user-1", "evolutions", "evo-clean")
    first_version = fake_db.store[doc_path]["_externalized_fields"]["history"]["chunk_version"]

    payload["history"] = [[{"title": "A", "content": "second " * 30}]]
    payload["status"] = "complete"
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
