import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.viewer import app, require_auth
from idea.auth import UserInfo
from idea.user_state import UserEvolutionState, user_states as real_user_states


def test_stop_without_engine_sets_stopping_without_lease_refresh():
    client = TestClient(app)
    mock_user = UserInfo(uid="stop_user", email="s@example.com", is_admin=False)
    mock_state = UserEvolutionState()
    mock_state.engine = None

    async def mock_require_auth():
        return mock_user

    async def mock_get_state(_user_id):
        return mock_state

    active_run = {
        "status": "in_progress",
        "owner_id": "owner-stop-1",
        "is_running": True,
        "active": True,
    }
    update_calls = []

    async def mock_get_active_run(_uid):
        return active_run

    async def mock_update_active_run(user_id, updates, lease_seconds=None, owner_id=None):
        update_calls.append(
            {
                "user_id": user_id,
                "updates": dict(updates),
                "lease_seconds": lease_seconds,
                "owner_id": owner_id,
            }
        )

    app.dependency_overrides[require_auth] = mock_require_auth
    with patch.object(real_user_states, "get", mock_get_state), patch(
        "idea.viewer.db.get_active_run", side_effect=mock_get_active_run
    ), patch("idea.viewer.db.update_active_run", side_effect=mock_update_active_run):
        resp = client.post("/api/stop-evolution")

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert len(update_calls) == 1
    assert update_calls[0]["lease_seconds"] is None
    assert update_calls[0]["owner_id"] == "owner-stop-1"
    assert update_calls[0]["updates"]["status"] == "stopping"
    assert update_calls[0]["updates"]["stop_requested"] is True


def test_stop_without_engine_returns_400_when_no_active_run():
    client = TestClient(app)
    mock_user = UserInfo(uid="stop_user_2", email="s2@example.com", is_admin=False)
    mock_state = UserEvolutionState()
    mock_state.engine = None

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
        resp = client.post("/api/stop-evolution")

    app.dependency_overrides.clear()

    assert resp.status_code == 400
    data = resp.json()
    assert data["status"] == "error"
