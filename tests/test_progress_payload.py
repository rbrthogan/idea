import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.viewer import app, require_auth
from idea.auth import UserInfo
from idea.user_state import UserEvolutionState, user_states as real_user_states


def _make_state() -> UserEvolutionState:
    state = UserEvolutionState()
    state.engine = SimpleNamespace(tournament_history=[])
    state.latest_data = [[{"id": "a1", "title": "A", "content": "B"}]]
    state.status = {
        "current_generation": 1,
        "total_generations": 2,
        "is_running": True,
        "history": state.latest_data,
        "progress": 50,
    }
    return state


def test_progress_omits_history_when_version_unchanged():
    client = TestClient(app)
    mock_user = UserInfo(uid="progress_user", email="p@example.com", is_admin=False)
    mock_state = _make_state()
    mock_state.history_version = 2
    mock_state.last_sent_history_version = 2

    async def mock_require_auth():
        return mock_user

    async def mock_get_state(user_id):
        return mock_state

    app.dependency_overrides[require_auth] = mock_require_auth
    with patch.object(real_user_states, "get", mock_get_state):
        resp = client.get("/api/progress")
        assert resp.status_code == 200
        data = resp.json()
        assert data["history_available"] is True
        assert data["history_changed"] is False
        assert "history" not in data

    app.dependency_overrides.clear()


def test_progress_includes_history_when_changed_or_requested():
    client = TestClient(app)
    mock_user = UserInfo(uid="progress_user_2", email="p2@example.com", is_admin=False)
    mock_state = _make_state()
    mock_state.history_version = 3
    mock_state.last_sent_history_version = 2

    async def mock_require_auth():
        return mock_user

    async def mock_get_state(user_id):
        return mock_state

    app.dependency_overrides[require_auth] = mock_require_auth
    with patch.object(real_user_states, "get", mock_get_state):
        resp = client.get("/api/progress")
        assert resp.status_code == 200
        data = resp.json()
        assert data["history_changed"] is True
        assert "history" in data
        assert data["history"] == mock_state.latest_data

        # Explicit includeHistory should still include full history
        resp2 = client.get("/api/progress?includeHistory=1")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert "history" in data2
        assert data2["history"] == mock_state.latest_data

    app.dependency_overrides.clear()


def test_progress_clears_stale_running_state_when_no_engine_and_no_active_run():
    client = TestClient(app)
    mock_user = UserInfo(uid="progress_user_3", email="p3@example.com", is_admin=False)
    mock_state = UserEvolutionState()
    mock_state.engine = None
    mock_state.status = {
        "current_generation": 1,
        "total_generations": 2,
        "is_running": True,
        "progress": 67,
    }

    async def mock_require_auth():
        return mock_user

    async def mock_get_state(_user_id):
        return mock_state

    async def mock_get_active_run(_uid):
        return None

    app.dependency_overrides[require_auth] = mock_require_auth
    with patch.object(real_user_states, "get", mock_get_state), patch(
        "idea.viewer.db.get_active_run", side_effect=mock_get_active_run
    ):
        resp = client.get("/api/progress")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_running"] is False
        assert data["is_stopped"] is True
        assert mock_state.status["is_running"] is False

    app.dependency_overrides.clear()
