"""Offline redeploy boundaries: database selection and graceful rating recovery."""

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from idea import database, viewer
from idea.auth import UserInfo
from idea.evolution_persistence import EvolutionRepository
from idea.user_state import UserEvolutionState
from tests.test_evolution_database_chunking import FakeFirestore, FakeTransaction

_get_db = database.get_db


@pytest.mark.parametrize("configured,expected", [(None, "(default)"), ("restored-20261018", "restored-20261018")])
def test_firestore_database_selection_is_shared_and_lazy(monkeypatch, configured, expected):
    monkeypatch.setattr(database, "_db", None)
    if configured is None:
        monkeypatch.delenv("IDEA_FIRESTORE_DATABASE", raising=False)
    else:
        monkeypatch.setenv("IDEA_FIRESTORE_DATABASE", configured)
    client = Mock()
    monkeypatch.setattr(database.firestore, "Client", client)
    assert _get_db() is _get_db()
    client.assert_called_once_with(database=expected)


def test_blank_database_configuration_fails_instead_of_writing_to_default(monkeypatch):
    monkeypatch.setattr(database, "_db", None)
    monkeypatch.setenv("IDEA_FIRESTORE_DATABASE", "  ")
    client = Mock()
    monkeypatch.setattr(database.firestore, "Client", client)
    with pytest.raises(ValueError, match="must not be empty"):
        _get_db()
    client.assert_not_called()


@pytest.mark.parametrize("outcome", ["saved", "newer_manual_edit", "skip_save", "uncommitted"])
def test_graceful_autorating_cancellation_preserves_committed_round_without_overwriting_newer_edits(monkeypatch, outcome):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    state = UserEvolutionState()
    started = threading.Event()

    class Evaluator:
        def evaluate_population(self, _critic, ideas, _idea_type, **kwargs):
            ideas[0]["ratings"]["auto"] = 1216
            ideas[1]["ratings"]["auto"] = 1184
            kwargs["details"].append({
                "round": 1, "ratings_committed": outcome != "uncommitted",
                "pairs": [{"a_idx": 0, "b_idx": 1, "winner": "A", "applied_this_run": True,
                           "new_elo_a": 1216, "new_elo_b": 1184}],
            })
            kwargs["progress_callback"](1, 2)
            started.set()
            deadline = time.monotonic() + 3
            while not kwargs["should_stop"]() and time.monotonic() < deadline:
                time.sleep(.005)
            return SimpleNamespace(batch=SimpleNamespace(records=[]))

    async def load(evolution_id, user):
        return None, await database.get_evolution(user.uid, evolution_id), "firestore"

    monkeypatch.setattr(viewer.user_states, "get", AsyncMock(return_value=state))
    monkeypatch.setattr(viewer, "_load_rating_evolution", load)
    monkeypatch.setattr(viewer, "_live_or_persisted_artifact_store", lambda *a, **kw: None)
    monkeypatch.setattr(viewer, "_autorating_pairwise_evaluator", lambda *a, **kw: Evaluator())
    monkeypatch.setattr(viewer, "_resolve_user_provider_key", AsyncMock(return_value="offline-test-key"))
    monkeypatch.setattr(viewer, "Critic", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(viewer, "_get_autorating_costs", lambda _: {})
    monkeypatch.setattr(viewer, "close_engine_clients", lambda _: None)

    async def check():
        initial = {"config": {"idea_type": "drabble"}, "history": [[
            {"id": "a", "idea": {"title": "A", "content": "Alpha"}},
            {"id": "b", "idea": {"title": "B", "content": "Beta"}},
        ]]}
        original_revision = await database.save_evolution("u", "e", initial)
        request = SimpleNamespace(json=AsyncMock(return_value={
            "evolutionId": "e", "numRounds": 2, "skipSave": outcome == "skip_save",
        }))
        task = asyncio.create_task(viewer.auto_rate(request, UserInfo("u")))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            if outcome == "newer_manual_edit":
                newer = await database.get_evolution("u", "e")
                newer["history"][0][0]["ratings"] = {"human": 4}
                newer_revision = await database.save_evolution("u", "e", newer)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        stored = await database.get_evolution("u", "e")
        first = stored["history"][0][0]
        assert state.autorating_status["is_running"] is False
        assert state.autorating_task is None
        if outcome == "saved":
            assert first["ratings"]["auto"] == 1216
            assert first["match_count"] == 1
            assert stored["_save_revision"] != original_revision
            assert state.autorating_status["status"] == "stopped"
            assert "Completed rounds were saved" in state.autorating_status["status_message"]
        elif outcome == "newer_manual_edit":
            assert stored["_save_revision"] == newer_revision
            assert first["ratings"] == {"human": 4}
            assert state.autorating_status["status"] == "error"
            assert "could not be confirmed saved" in state.autorating_status["status_message"]
        else:
            assert stored["_save_revision"] == original_revision
            assert "ratings" not in first
            assert state.autorating_status["status_message"] == "Auto-rating stopped"

    asyncio.run(check())


def test_legacy_checkpoint_resume_clones_snapshot_under_new_claim_without_overwriting_originals(monkeypatch):
    fake_db = FakeFirestore()
    fake_db.transaction = lambda: FakeTransaction(fake_db)
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(database.firestore, "transactional", lambda fn: lambda tx: fn(tx))
    monkeypatch.setattr(viewer, "GLOBAL_MAX_ACTIVE_RUNS", 0)
    monkeypatch.setattr("idea.llm.LLMWrapper._setup_provider", lambda _: None)
    state = UserEvolutionState()
    monkeypatch.setattr(viewer.user_states, "get", AsyncMock(return_value=state))
    credential = AsyncMock(return_value="offline-key")
    monkeypatch.setattr(viewer, "_resolve_user_provider_key", credential)
    monkeypatch.setattr(viewer, "_heartbeat_loop", AsyncMock())

    async def checkpoint_only(current, **kwargs):
        assert kwargs == {"mode": "resume"}
        await current.engine.save_checkpoint(status="queued")

    monkeypatch.setattr(viewer, "_launch_run", checkpoint_only)

    async def check():
        existing = {"name": "Newer existing evolution", "history": [], "config": {"idea_type": "drabble"}}
        old_revision = await database.save_evolution("u", "existing-evolution", existing)
        legacy = {
            "evolution_id": "existing-evolution", "checkpoint_id": "legacy-checkpoint",
            "_save_revision": old_revision, "run_token": "obsolete-token", "status": "paused",
            "config": json.dumps({"idea_type": "drabble", "model_type": "gpt-6-luna", "pop_size": 2, "generations": 2}),
            "history": json.dumps([[{"idea": {"title": "Saved", "content": "Original body"}}]]),
            "population": json.dumps([{"idea": {"title": "Saved", "content": "Original body"}}]),
            "current_generation": 0,
        }
        await database.save_checkpoint("u", "legacy-checkpoint", legacy)
        original = await database.get_checkpoint("u", "legacy-checkpoint")
        response = await viewer.resume_evolution(SimpleNamespace(json=AsyncMock(return_value={
            "checkpoint_id": "legacy-checkpoint",
        })), UserInfo("u"))
        assert response.status_code == 200, response.body
        payload = json.loads(response.body)
        new_id = payload["evolution_id"]
        assert new_id not in {"existing-evolution", "legacy-checkpoint"}
        assert payload["checkpoint_id"] == "legacy-checkpoint"  # Request/source identifier.
        assert state.engine.checkpoint_id != "legacy-checkpoint"
        credential.assert_awaited_once()
        assert credential.await_args.args[1] == "openai"
        saved = await database.get_evolution("u", new_id)
        assert saved["source_checkpoint_id"] == "legacy-checkpoint"
        assert saved["history"][0][0]["idea"]["content"] == "Original body"
        assert saved["_save_revision"] != old_revision
        assert state.engine._run_token == state.run_token
        assert state.run_token != "obsolete-token"
        assert await database.get_checkpoint("u", "legacy-checkpoint") == original
        assert (await database.get_evolution("u", "existing-evolution"))["_save_revision"] == old_revision

    asyncio.run(check())


def test_legacy_archived_checkpoint_is_rejected_before_credentials_or_restoration(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(viewer.user_states, "get", AsyncMock(return_value=UserEvolutionState()))
    credential = AsyncMock()
    load = AsyncMock()
    monkeypatch.setattr(viewer, "_resolve_user_provider_key", credential)
    monkeypatch.setattr(viewer.EvolutionEngine, "load_evolution_for_user", load)

    async def check():
        await database.save_checkpoint("u", "legacy-prime", {"config": json.dumps({"idea_type": "prime_visualization"})})
        response = await viewer.resume_evolution(SimpleNamespace(json=AsyncMock(return_value={
            "checkpoint_id": "legacy-prime",
        })), UserInfo("u"))
        assert response.status_code == 410
        credential.assert_not_awaited()
        load.assert_not_awaited()

    asyncio.run(check())


def test_unified_snapshot_takes_precedence_over_same_id_legacy_checkpoint(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)

    async def check():
        revision = await database.save_evolution("u", "same-id", {"name": "Unified", "history": []})
        await database.save_checkpoint("u", "same-id", {"name": "Legacy"})
        loaded = await database.get_evolution_for_restore("u", "same-id")
        assert loaded["name"] == "Unified"
        assert loaded["_save_revision"] == revision
        assert "_legacy_checkpoint_id" not in loaded

    asyncio.run(check())


def test_mixed_legacy_checkpoint_listing_keeps_valid_records_and_source_document_ids(monkeypatch, caplog):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(viewer.user_states, "get", AsyncMock(return_value=UserEvolutionState()))
    monkeypatch.setattr(viewer, "_local_user_evolution_directory", lambda *a, **kw: None)
    monkeypatch.setattr(viewer, "_allow_unscoped_legacy_local_files", lambda: False)

    async def check():
        for identifier, config in [("valid-json", json.dumps({"idea_type": "drabble", "generations": 3})),
                                   ("valid-object", {"idea_type": "airesearch", "generations": 4}),
                                   ("corrupt-json", '{"private_detail": broken'),
                                   ("wrong-shape", "[]")]:
            await database.save_checkpoint("u", identifier, {
                "checkpoint_id": "embedded-alias", "config": config, "current_generation": 1,
                "history": "[]", "status": "paused",
            })
        listing = await EvolutionRepository.list_checkpoints_for_user("u")
        assert {entry["id"] for entry in listing} == {"valid-json", "valid-object"}
        assert all(isinstance(entry["config"], dict) for entry in listing)
        response = await viewer.get_unified_history(UserInfo("u"))
        rows = json.loads(response.body)["items"]
        assert {row["id"] for row in rows} == {"valid-json", "valid-object"}
        assert all(row["can_resume"] and not row["can_rate"] for row in rows)
        assert all("create an experiment you can rate" in row["can_rate_reason"] for row in rows)
        with pytest.raises(database.EvolutionStorageIntegrityError):
            await database.get_checkpoint("u", "corrupt-json")
        assert "private_detail" not in caplog.text

    asyncio.run(check())


def test_legacy_checkpoint_view_is_read_only_and_uses_source_identity(monkeypatch):
    fake_db = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: fake_db)
    monkeypatch.setattr(viewer.user_states, "get", AsyncMock(return_value=UserEvolutionState()))

    async def check():
        await database.save_checkpoint("u", "legacy", {
            "evolution_id": "other-evolution", "config": json.dumps({"idea_type": "drabble"}),
            "history": json.dumps([[{"idea": {"title": "Saved", "content": "Original body"}}]]),
        })
        before = dict(fake_db.store)
        response = await viewer.get_evolution("legacy", UserInfo("u"))
        payload = json.loads(response.body)
        assert payload["id"] == "legacy"
        assert payload["data"]["evolution_id"] == "legacy"
        assert payload["data"]["source_checkpoint_id"] == "legacy"
        assert payload["data"]["config"]["idea_type"] == "drabble"
        assert payload["data"]["history"][0][0]["idea"]["content"] == "Original body"
        assert fake_db.store == before

    asyncio.run(check())


def test_user_data_survives_recreated_client_and_wrong_key_does_not_erase_credentials(monkeypatch):
    current = FakeFirestore()
    monkeypatch.setattr(database, "get_db", lambda: current)
    original_master = "offline-test-master-key-with-at-least-32-bytes"
    monkeypatch.setenv("ENCRYPTION_KEY", original_master)
    monkeypatch.delenv("ENCRYPTION_SALT", raising=False)

    async def check():
        nonlocal current
        await database.save_user_api_key("u", "fake-gemini", "google_generative_ai")
        await database.save_user_api_key("u", "fake-openai", "openai")
        await database.save_user_execution_preference("u", "remote")
        await database.save_user_template("u", "custom", {"name": "My template", "requirements": ["Keep this brief."]})
        before = await database.get_user_settings("u")
        recreated = FakeFirestore()
        recreated.store = current.store
        current = recreated
        assert await database.get_user_api_key("u") == "fake-gemini"
        assert await database.get_user_api_key("u", "openai") == "fake-openai"
        assert (await database.get_user_settings("u"))["execution_preference"] == "remote"
        assert (await database.get_user_template("u", "custom"))["requirements"] == ["Keep this brief."]
        monkeypatch.setenv("ENCRYPTION_KEY", "changed-offline-master-key-also-at-least-32-bytes")
        with pytest.raises(database.APIKeyDecryptionError):
            await database.get_user_api_key("u")
        assert await database.get_user_settings("u") == before
        monkeypatch.setenv("ENCRYPTION_KEY", original_master)
        monkeypatch.setenv("ENCRYPTION_SALT", "changed-salt")
        with pytest.raises(database.APIKeyDecryptionError):
            await database.get_user_api_key("u")
        assert await database.get_user_settings("u") == before
        monkeypatch.delenv("ENCRYPTION_SALT")
        assert await database.get_user_api_key("u") == "fake-gemini"

    asyncio.run(check())
