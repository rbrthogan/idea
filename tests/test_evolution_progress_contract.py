import asyncio
from types import SimpleNamespace

from idea.evolution_progress import ProgressEmitter


class DummyEngine:
    def __init__(self):
        self.generations = 3
        self.history = [[{"id": "a"}]]
        self.contexts = ["ctx"]
        self.specific_prompts = ["p0"]
        self.breeding_prompts = [["bp0"]]
        self.diversity_history = [{"enabled": True, "diversity_score": 0.42}]

    def get_total_token_count(self):
        return {"total": 10, "cost": {"total_cost": 0.001}}


def test_progress_base_payload_shape():
    emitter = ProgressEmitter(DummyEngine())
    payload = emitter.base(current_generation=1, is_running=True, include_core_state=True)

    assert payload["current_generation"] == 1
    assert payload["total_generations"] == 3
    assert payload["is_running"] is True
    assert "history" in payload
    assert "contexts" in payload
    assert "specific_prompts" in payload
    assert "breeding_prompts" in payload
    assert "diversity_history" in payload


def test_progress_with_tokens_adds_token_counts():
    emitter = ProgressEmitter(DummyEngine())
    payload = emitter.base(current_generation=0, is_running=True)
    payload = emitter.with_tokens(payload)

    assert "token_counts" in payload
    assert payload["token_counts"]["total"] == 10


def test_progress_emit_backfills_diversity_history():
    emitter = ProgressEmitter(DummyEngine())
    updates = []

    async def callback(data):
        updates.append(data)

    asyncio.run(emitter.emit(callback, {"is_running": False}))

    assert len(updates) == 1
    assert "diversity_history" in updates[0]
    assert updates[0]["is_running"] is False
