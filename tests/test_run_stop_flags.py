import asyncio
from unittest.mock import patch

from idea.user_state import UserEvolutionState
from idea.viewer import _claim_run_slot, _finalize_run_state, _should_honor_remote_stop


def test_should_honor_remote_stop_requires_stopping_status():
    assert _should_honor_remote_stop(None) is False
    assert _should_honor_remote_stop({"stop_requested": False, "status": "stopping"}) is False
    assert _should_honor_remote_stop({"stop_requested": True, "status": "in_progress"}) is False
    assert _should_honor_remote_stop({"stop_requested": True, "status": "stopping"}) is True


def test_claim_run_slot_resets_stop_requested_flag():
    captured = {}

    async def mock_claim_active_run(user_id, run_data, lease_seconds, owner_id):
        captured["run_data"] = dict(run_data)
        return {"ok": True, "data": run_data}

    with patch("idea.viewer.db.claim_active_run", side_effect=mock_claim_active_run):
        ok, existing, scope = asyncio.run(
            _claim_run_slot(
                user_id="u1",
                evolution_id="e1",
                evolution_name="name",
                total_generations=3,
                start_time="2026-02-09T00:00:00",
                tournament_count=1.0,
                full_tournament_rounds=4,
                target_tournament_rounds=4,
            )
        )

    assert ok is True
    assert existing is not None
    assert scope is None
    assert captured["run_data"]["stop_requested"] is False


def test_finalize_run_state_clears_stop_requested_flag():
    state = UserEvolutionState()
    state.run_owner_id = "owner-1"

    updates_seen = []

    async def mock_refresh(*args, **kwargs):
        return None

    async def mock_update_active_run(user_id, updates, lease_seconds, owner_id):
        updates_seen.append({
            "user_id": user_id,
            "updates": dict(updates),
            "lease_seconds": lease_seconds,
            "owner_id": owner_id,
        })

    with patch("idea.viewer._refresh_run_state", side_effect=mock_refresh), patch(
        "idea.viewer.db.update_active_run", side_effect=mock_update_active_run
    ):
        asyncio.run(_finalize_run_state("u1", state, {"is_running": False}))

    assert len(updates_seen) == 1
    assert updates_seen[0]["updates"]["is_running"] is False
    assert updates_seen[0]["updates"]["active"] is False
    assert updates_seen[0]["updates"]["stop_requested"] is False
