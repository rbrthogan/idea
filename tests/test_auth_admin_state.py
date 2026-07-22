import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request

from idea import auth
from idea.admin import get_stats, get_user_evolutions, list_users
from idea.auth import UserInfo, get_current_user, require_admin, require_auth
from idea.user_state import UserEvolutionState, UserStateManager


def response_json(response):
    return json.loads(response.body.decode("utf-8"))


def local_request(host: str = "localhost:8000", client_host: str = "127.0.0.1") -> Request:
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"host", host.encode())],
        "client": (client_host, 12345),
    })


def test_user_info_serializes_all_fields():
    user = UserInfo(
        uid="user-1",
        email="user@example.com",
        display_name="Test User",
        is_admin=True,
    )

    assert user.to_dict() == {
        "uid": "user-1",
        "email": "user@example.com",
        "display_name": "Test User",
        "is_admin": True,
    }


def test_auth_dependencies_reject_missing_or_non_admin_user():
    with pytest.raises(HTTPException) as missing_auth:
        asyncio.run(require_auth(credentials=None))
    assert missing_auth.value.status_code == 401

    with pytest.raises(HTTPException) as non_admin:
        asyncio.run(require_admin(UserInfo(uid="user-1", is_admin=False)))
    assert non_admin.value.status_code == 403

    admin = UserInfo(uid="admin-1", is_admin=True)
    assert asyncio.run(require_admin(admin)) is admin


def test_get_current_user_returns_none_without_credentials():
    assert asyncio.run(get_current_user(credentials=None)) is None


def test_dev_auth_supplies_local_user_with_ephemeral_bearer_token(monkeypatch):
    monkeypatch.setenv("IDEA_DEV_AUTH", "1")
    monkeypatch.setenv("IDEA_DEV_AUTH_TOKEN", "test-local-token")
    monkeypatch.setenv("IDEA_DEV_AUTH_UID", "codex-local")
    monkeypatch.setenv("IDEA_DEV_AUTH_EMAIL", "codex@example.local")
    monkeypatch.setenv("IDEA_DEV_AUTH_NAME", "Codex Local")
    monkeypatch.setenv("IDEA_DEV_AUTH_ADMIN", "1")

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="test-local-token",
    )
    user = asyncio.run(
        get_current_user(credentials=credentials, request=local_request())
    )

    assert user is not None
    assert user.to_dict() == {
        "uid": "codex-local",
        "email": "codex@example.local",
        "display_name": "Codex Local",
        "is_admin": True,
    }


def test_dev_auth_rejects_missing_bearer_token_and_defaults_admin_off(monkeypatch):
    monkeypatch.setenv("IDEA_DEV_AUTH", "1")
    monkeypatch.delenv("IDEA_DEV_AUTH_ADMIN", raising=False)

    assert asyncio.run(
        get_current_user(credentials=None, request=local_request())
    ) is None
    assert auth.get_dev_user_info().is_admin is False


def test_dev_auth_is_rejected_for_non_local_requests(monkeypatch):
    monkeypatch.setenv("IDEA_DEV_AUTH", "1")

    user = asyncio.run(
        get_current_user(
            credentials=None,
            request=local_request(host="idea.example.com", client_host="203.0.113.9"),
        )
    )

    assert user is None


def test_dev_auth_rejects_external_host_behind_local_proxy(monkeypatch):
    monkeypatch.setenv("IDEA_DEV_AUTH", "1")
    monkeypatch.setenv("IDEA_DEV_AUTH_TOKEN", "local-token")
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="local-token",
    )

    user = asyncio.run(
        get_current_user(
            credentials=credentials,
            request=local_request(host="idea.example.com", client_host="127.0.0.1"),
        )
    )

    assert user is None


def test_require_auth_accepts_local_dev_bearer_token(monkeypatch):
    monkeypatch.setenv("IDEA_DEV_AUTH", "1")
    monkeypatch.setenv("IDEA_DEV_AUTH_TOKEN", "local-token")
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="local-token",
    )

    user = asyncio.run(require_auth(credentials=credentials, request=local_request()))

    assert user.uid == "local-dev-user"


def test_get_current_user_decodes_firebase_token(monkeypatch):
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="token-123",
    )
    decoded_token = {
        "uid": "firebase-user",
        "email": "admin@example.com",
        "name": "Admin User",
    }

    monkeypatch.setattr(auth, "get_firebase_app", lambda: object())
    monkeypatch.setattr(auth.auth, "verify_id_token", lambda token: decoded_token)
    monkeypatch.setattr(auth, "ADMIN_EMAILS", ["admin@example.com"])

    user = asyncio.run(get_current_user(credentials))

    assert user is not None
    assert user.to_dict() == {
        "uid": "firebase-user",
        "email": "admin@example.com",
        "display_name": "Admin User",
        "is_admin": True,
    }


def test_get_current_user_returns_none_for_invalid_token(monkeypatch):
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="bad-token",
    )

    monkeypatch.setattr(auth, "get_firebase_app", lambda: object())

    def raise_invalid_token(_token):
        raise ValueError("bad token")

    monkeypatch.setattr(auth.auth, "verify_id_token", raise_invalid_token)

    assert asyncio.run(get_current_user(credentials)) is None


def test_user_state_reset_methods_clear_runtime_payloads():
    state = UserEvolutionState()
    state.queue.put_nowait({"stale": True})
    state.status = {
        "current_generation": 5,
        "total_generations": 10,
        "is_running": True,
        "history": [[{"id": "idea"}]],
    }
    state.history_version = 7
    state.last_sent_history_version = 6
    state.autorating_status["is_running"] = True

    state.reset_status()
    state.reset_autorating_status()

    assert state.status == {
        "current_generation": 0,
        "total_generations": 0,
        "is_running": False,
        "history": [],
    }
    assert state.history_version == 0
    assert state.last_sent_history_version == -1
    assert state.queue.empty()
    assert state.autorating_status["is_running"] is False
    assert state.autorating_status["status"] == "idle"


def test_user_state_manager_reuses_state_and_cleans_only_idle_stopped_users():
    async def run():
        manager = UserStateManager(idle_timeout_hours=1)

        active = await manager.get("active")
        idle_stopped = await manager.get("idle-stopped")
        idle_running = await manager.get("idle-running")

        assert await manager.get("active") is active

        old_timestamp = datetime.utcnow() - timedelta(hours=2)
        idle_stopped.last_activity = old_timestamp
        idle_running.last_activity = old_timestamp
        idle_running.status["is_running"] = True

        assert await manager.get_total_count() == 3
        assert await manager.get_active_count() == 1

        removed_count = await manager.cleanup_idle()

        assert removed_count == 1
        assert await manager.get_total_count() == 2
        assert "idle-stopped" not in manager._states
        assert "idle-running" in manager._states

    asyncio.run(run())


def test_admin_list_users_success_and_error():
    async def run():
        users = [{"uid": "u1"}, {"uid": "u2"}]
        with patch("idea.admin.db.get_all_users_summary", new=AsyncMock(return_value=users)):
            payload = response_json(await list_users(UserInfo(uid="admin", is_admin=True)))

        assert payload == {
            "status": "success",
            "users": users,
            "total": 2,
        }

        with patch(
            "idea.admin.db.get_all_users_summary",
            new=AsyncMock(side_effect=RuntimeError("db down")),
        ):
            response = await list_users(UserInfo(uid="admin", is_admin=True))

        assert response.status_code == 500
        assert response_json(response) == {
            "status": "error",
            "message": "db down",
        }

    asyncio.run(run())


def test_admin_stats_aggregates_user_summary():
    async def run():
        users = [
            {"evolution_count": 2, "template_count": 1, "has_api_key": True},
            {"evolution_count": 3, "template_count": 4, "has_api_key": False},
        ]

        with patch("idea.admin.db.get_all_users_summary", new=AsyncMock(return_value=users)):
            payload = response_json(await get_stats(UserInfo(uid="admin", is_admin=True)))

        assert payload == {
            "status": "success",
            "stats": {
                "total_users": 2,
                "users_with_api_key": 1,
                "total_evolutions": 5,
                "total_templates": 5,
            },
        }

    asyncio.run(run())


def test_admin_user_evolutions_success_and_error():
    async def run():
        evolutions = [{"id": "evo-1"}]
        with patch("idea.admin.db.list_evolutions", new=AsyncMock(return_value=evolutions)):
            payload = response_json(
                await get_user_evolutions("target-user", UserInfo(uid="admin", is_admin=True))
            )

        assert payload == {
            "status": "success",
            "evolutions": evolutions,
            "user_id": "target-user",
        }

        with patch(
            "idea.admin.db.list_evolutions",
            new=AsyncMock(side_effect=RuntimeError("nope")),
        ):
            response = await get_user_evolutions(
                "target-user",
                UserInfo(uid="admin", is_admin=True),
            )

        assert response.status_code == 500
        assert response_json(response) == {
            "status": "error",
            "message": "nope",
        }

    asyncio.run(run())
