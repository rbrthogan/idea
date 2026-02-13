import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.viewer import app, require_auth
from idea.auth import UserInfo
from idea.user_state import UserEvolutionState, user_states as real_user_states


def test_force_stop_clears_stale_active_run_when_engine_missing():
    client = TestClient(app)
    mock_user = UserInfo(uid="force_stop_user", email="f@example.com", is_admin=False)
    mock_state = UserEvolutionState()
    mock_state.engine = None

    async def mock_require_auth():
        return mock_user

    async def mock_get_state(_user_id):
        return mock_state

    active_run = {
        "status": "stopping",
        "owner_id": "owner-1",
        "checkpoint_id": "ckpt-123",
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
        resp = client.post("/api/force-stop-evolution")

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["checkpoint_id"] == "ckpt-123"

    assert len(update_calls) == 1
    call = update_calls[0]
    assert call["user_id"] == "force_stop_user"
    assert call["lease_seconds"] == 0
    assert call["owner_id"] == "owner-1"
    assert call["updates"]["status"] == "force_stopped"
    assert call["updates"]["is_running"] is False
    assert call["updates"]["is_stopped"] is True
    assert call["updates"]["is_resumable"] is True
    assert call["updates"]["stop_requested"] is False


def test_force_stop_returns_400_when_no_engine_and_no_active_run():
    client = TestClient(app)
    mock_user = UserInfo(uid="force_stop_user2", email="f2@example.com", is_admin=False)
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
        resp = client.post("/api/force-stop-evolution")

    app.dependency_overrides.clear()

    assert resp.status_code == 400
    data = resp.json()
    assert data["status"] == "error"


def test_force_stop_allows_inconsistent_active_flags_when_status_not_active():
    client = TestClient(app)
    mock_user = UserInfo(uid="force_stop_user3", email="f3@example.com", is_admin=False)
    mock_state = UserEvolutionState()
    mock_state.engine = None

    async def mock_require_auth():
        return mock_user

    async def mock_get_state(_user_id):
        return mock_state

    active_run = {
        "status": "paused",
        "is_running": True,
        "active": True,
        "owner_id": "owner-3",
        "checkpoint_id": "ckpt-789",
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
        resp = client.post("/api/force-stop-evolution")

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["checkpoint_id"] == "ckpt-789"

    assert len(update_calls) == 1
    assert update_calls[0]["lease_seconds"] == 0
    assert update_calls[0]["updates"]["status"] == "force_stopped"
