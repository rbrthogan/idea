import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.auth import UserInfo
from idea.user_state import UserEvolutionState, user_states as real_user_states
from idea.viewer import app, require_auth


def _live_state() -> UserEvolutionState:
    state = UserEvolutionState()
    live_history = [[{"id": f"idea-{gen}", "title": f"Gen {gen}", "content": "x"}] for gen in range(6)]
    state.engine = SimpleNamespace(
        evolution_id="evo-live",
        evolution_name="Live Complete Evolution",
        history=live_history,
        current_generation=5,
        generations=5,
        contexts=["ctx"],
        specific_prompts=["prompt"],
        breeding_prompts=[],
        tournament_history=[],
        diversity_history=[{"generation": 0, "diversity_score": 0.1}],
        island_diversity_history=[],
        migration_history=[],
        get_total_token_count=lambda: {"total": 99, "cost": {"total_cost": 1.23}},
    )
    state.latest_data = live_history
    state.history_version = 6
    state.status = {
        "evolution_id": "evo-live",
        "evolution_name": "Live Complete Evolution",
        "current_generation": 5,
        "total_generations": 5,
        "is_running": False,
        "history": live_history,
        "history_version": 6,
        "token_counts": {"total": 99, "cost": {"total_cost": 1.23}},
    }
    return state


def test_history_prefers_live_complete_state_over_stale_saved_summary():
    client = TestClient(app)
    mock_user = UserInfo(uid="history-user", email="history@example.com", is_admin=False)
    mock_state = _live_state()

    async def mock_require_auth():
        return mock_user

    async def mock_get_state(_uid):
        return mock_state

    async def mock_list_evolutions(_uid):
        return [
            {
                "id": "evo-live",
                "name": "Live Complete Evolution",
                "status": "in_progress",
                "updated_at": "2026-04-30T15:46:00",
                "generation": 3,
                "total_generations": 5,
                "pop_size": 1,
                "idea_type": "game",
                "model_type": "gemini",
                "total_ideas": 4,
            }
        ]

    async def mock_list_checkpoints(_uid):
        return []

    app.dependency_overrides[require_auth] = mock_require_auth
    with patch.object(real_user_states, "get", mock_get_state), patch(
        "idea.viewer.EvolutionEngine.list_evolutions_for_user", side_effect=mock_list_evolutions
    ), patch(
        "idea.viewer.EvolutionEngine.list_checkpoints_for_user", side_effect=mock_list_checkpoints
    ):
        resp = client.get("/api/history")

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    item = next(item for item in resp.json()["items"] if item["id"] == "evo-live")
    assert item["status"] == "complete"
    assert item["generations"] == 5
    assert item["total_generations"] == 5
    assert item["total_ideas"] == 6
    assert item["can_continue"] is True
    assert item["can_resume"] is False


def test_get_evolution_prefers_live_history_when_saved_doc_is_stale():
    client = TestClient(app)
    mock_user = UserInfo(uid="history-user", email="history@example.com", is_admin=False)
    mock_state = _live_state()
    saved_data = {
        "evolution_id": "evo-live",
        "name": "Live Complete Evolution",
        "status": "in_progress",
        "current_generation": 3,
        "config": {"generations": 5},
        "history": [[{"id": f"old-{gen}", "title": f"Old {gen}", "content": "x"}] for gen in range(4)],
    }

    async def mock_require_auth():
        return mock_user

    async def mock_get_state(_uid):
        return mock_state

    async def mock_get_evolution(_uid, _evolution_id):
        return saved_data

    async def mock_get_active_run(_uid):
        return None

    app.dependency_overrides[require_auth] = mock_require_auth
    with patch.object(real_user_states, "get", mock_get_state), patch(
        "idea.viewer.db.get_evolution", side_effect=mock_get_evolution
    ), patch("idea.viewer.db.get_active_run", side_effect=mock_get_active_run):
        resp = client.get("/api/evolution/evo-live")

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status"] == "complete"
    assert data["current_generation"] == 5
    assert len(data["history"]) == 6
    assert data["token_counts"]["total"] == 99


def test_repair_evolution_from_client_updates_when_client_history_is_newer():
    client = TestClient(app)
    mock_user = UserInfo(uid="history-user", email="history@example.com", is_admin=False)
    existing_data = {
        "evolution_id": "evo-live",
        "name": "Live Complete Evolution",
        "status": "in_progress",
        "current_generation": 3,
        "config": {"generations": 5, "pop_size": 1, "idea_type": "game"},
        "history": [[{"id": f"old-{gen}", "title": f"Old {gen}", "content": "x"}] for gen in range(4)],
    }
    saved_payloads = []

    async def mock_require_auth():
        return mock_user

    async def mock_get_evolution(_uid, _evolution_id):
        return existing_data

    async def mock_save_evolution(_uid, _evolution_id, payload):
        saved_payloads.append(payload)

    client_history = [
        [{"id": f"idea-{gen}", "title": f"Gen {gen}", "content": "x"}]
        for gen in range(6)
    ]

    app.dependency_overrides[require_auth] = mock_require_auth
    with patch("idea.viewer.db.get_evolution", side_effect=mock_get_evolution), patch(
        "idea.viewer.db.save_evolution", side_effect=mock_save_evolution
    ):
        resp = client.post(
            "/api/evolution/evo-live/repair",
            json={
                "data": {
                    "evolution_id": "evo-live",
                    "evolution_name": "Live Complete Evolution",
                    "history": client_history,
                    "current_generation": 5,
                    "total_generations": 5,
                    "is_running": False,
                    "token_counts": {"total": 99},
                }
            },
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["status"] == "repaired"
    assert len(saved_payloads) == 1
    saved = saved_payloads[0]
    assert saved["status"] == "complete"
    assert saved["current_generation"] == 5
    assert saved["history"] == client_history
    assert saved["population"] == client_history[-1]
    assert saved["config"]["generations"] == 5


def test_repair_evolution_from_client_does_not_downgrade_newer_saved_history():
    client = TestClient(app)
    mock_user = UserInfo(uid="history-user", email="history@example.com", is_admin=False)
    existing_data = {
        "evolution_id": "evo-live",
        "status": "complete",
        "current_generation": 5,
        "config": {"generations": 5, "pop_size": 1},
        "history": [[{"id": f"saved-{gen}", "title": f"Saved {gen}", "content": "x"}] for gen in range(6)],
    }
    saved_payloads = []

    async def mock_require_auth():
        return mock_user

    async def mock_get_evolution(_uid, _evolution_id):
        return existing_data

    async def mock_save_evolution(_uid, _evolution_id, payload):
        saved_payloads.append(payload)

    app.dependency_overrides[require_auth] = mock_require_auth
    with patch("idea.viewer.db.get_evolution", side_effect=mock_get_evolution), patch(
        "idea.viewer.db.save_evolution", side_effect=mock_save_evolution
    ):
        resp = client.post(
            "/api/evolution/evo-live/repair",
            json={
                "data": {
                    "history": [[{"id": "old", "title": "Old", "content": "x"}]],
                    "current_generation": 0,
                    "total_generations": 5,
                }
            },
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["status"] == "skipped"
    assert saved_payloads == []
