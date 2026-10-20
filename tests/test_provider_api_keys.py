from unittest.mock import AsyncMock, patch
import pytest

from fastapi.testclient import TestClient

from idea.auth import UserInfo, require_auth
from idea.viewer import app
from idea import database


async def _authenticated_user():
    return UserInfo(uid="user-1", email="user@example.com", is_admin=False)


@pytest.mark.parametrize("other_key", [None, "synthetic-openai-key"])
def test_unreadable_key_status_is_provider_scoped_actionable_and_never_mutates_storage(other_key):
    async def read(_uid, provider):
        if provider == "google_generative_ai":
            raise database.APIKeyDecryptionError("PRIVATE-UID cannot be decrypted")
        return other_key

    save, delete = AsyncMock(), AsyncMock()
    app.dependency_overrides[require_auth] = _authenticated_user
    try:
        with patch("idea.viewer.db.get_user_api_key", new=AsyncMock(side_effect=read)), \
             patch("idea.viewer.db.save_user_api_key", new=save), \
             patch("idea.viewer.db.delete_user_api_key", new=delete):
            client = TestClient(app)
            status = client.get("/api/settings/status")
            user_status = client.get("/api/user/status")
    finally:
        app.dependency_overrides.pop(require_auth, None)
    assert status.status_code == user_status.status_code == 200
    gemini = status.json()["api_keys"]["google_generative_ai"]
    assert gemini["configured"] is False
    assert gemini["source"] == "unreadable"
    assert gemini["error_code"] == "stored_key_unreadable"
    assert gemini["masked_key"] is None
    assert "replacement" in gemini["message"] and "has not been changed" in gemini["message"]
    assert status.json()["api_keys"]["openai"]["configured"] is bool(other_key)
    assert user_status.json()["has_api_key"] is bool(other_key)
    assert user_status.json()["key_errors"] == {"google_generative_ai": "stored_key_unreadable"}
    assert "PRIVATE-UID" not in status.text + user_status.text
    assert (other_key or "unused-sentinel") not in status.text + user_status.text
    save.assert_not_awaited()
    delete.assert_not_awaited()


def test_unavailable_settings_status_does_not_expose_internal_error():
    app.dependency_overrides[require_auth] = _authenticated_user
    try:
        with patch("idea.viewer.db.get_user_api_key", new=AsyncMock(side_effect=RuntimeError("PRIVATE-UID PRIVATE-DETAIL"))):
            response = TestClient(app).get("/api/settings/status")
    finally:
        app.dependency_overrides.pop(require_auth, None)
    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["message"]
    assert "PRIVATE" not in response.text


def test_settings_exposes_independent_masked_provider_key_statuses():
    async def provider_key(_user_id, provider):
        return "sk-openai-secret" if provider == "openai" else None

    app.dependency_overrides[require_auth] = _authenticated_user
    try:
        with patch(
            "idea.viewer.db.get_user_api_key",
            new=AsyncMock(side_effect=provider_key),
        ), patch(
            "idea.viewer.db.get_user_settings",
            new=AsyncMock(return_value={}),
        ):
            response = TestClient(app).get("/api/settings/status")
    finally:
        app.dependency_overrides.pop(require_auth, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_keys"]["google_generative_ai"] == {
        "configured": False,
        "masked_key": None,
        "source": "missing",
    }
    assert payload["api_keys"]["openai"] == {
        "configured": True,
        "masked_key": "sk-o...cret",
        "source": "stored",
    }
    assert "sk-openai-secret" not in response.text


def test_openai_key_save_and_delete_are_provider_scoped():
    save_key = AsyncMock()
    delete_key = AsyncMock()
    app.dependency_overrides[require_auth] = _authenticated_user
    try:
        with patch("idea.viewer.db.save_user_api_key", new=save_key), patch(
            "idea.viewer.db.delete_user_api_key",
            new=delete_key,
        ):
            client = TestClient(app)
            saved = client.post(
                "/api/settings/api-key",
                json={"provider": "openai", "api_key": "sk-openai-secret"},
            )
            deleted = client.delete("/api/settings/api-key?provider=openai")
    finally:
        app.dependency_overrides.pop(require_auth, None)

    assert saved.status_code == 200
    assert saved.json()["message"] == "OpenAI API key saved successfully"
    save_key.assert_awaited_once_with("user-1", "sk-openai-secret", "openai")
    assert deleted.status_code == 200
    delete_key.assert_awaited_once_with("user-1", "openai")


def test_providerless_key_save_is_rejected_instead_of_falling_back_to_gemini():
    save_key = AsyncMock()
    app.dependency_overrides[require_auth] = _authenticated_user
    try:
        with patch("idea.viewer.db.save_user_api_key", new=save_key):
            response = TestClient(app).post(
                "/api/settings/api-key",
                json={"api_key": "sk-misfiled-by-stale-client"},
            )
    finally:
        app.dependency_overrides.pop(require_auth, None)

    assert response.status_code == 400
    assert "provider is required" in response.json()["message"]
    save_key.assert_not_awaited()


def test_settings_ui_offers_openai_and_sends_selected_provider():
    from pathlib import Path
    import re

    root = Path(__file__).parent.parent
    viewer_html = (root / "idea/static/html/viewer.html").read_text()
    viewer_js = (root / "idea/static/js/viewer.js").read_text()

    assert 'id="apiKeyProviderSelect"' in viewer_html
    assert '<option value="openai">OpenAI</option>' in viewer_html
    assert re.search(r'/static/js/viewer\.js\?v=[^"\s]+', viewer_html)
    assert "JSON.stringify({ api_key: key, provider })" in viewer_js
    assert "api_keys?.[provider]" in viewer_js
