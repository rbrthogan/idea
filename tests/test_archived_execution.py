"""Release boundary for the archived Prime/generated-code experiment.

All services are faked: these tests never make provider or container calls.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from idea import viewer
from idea.auth import UserInfo
from idea.user_state import UserEvolutionState


USER = UserInfo(uid="archive-test", email="archive@example.test")
ARTIFACT_EVALUATION = {
    "allow_generated_code_execution": True,
    "components": [{"id": "render", "kind": "llm_code_artifact", "enabled": True}],
}


class Body:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


@pytest.mark.parametrize("body", [
    {"ideaType": "prime_visualization"},
    {"ideaType": "drabble", "evaluation": ARTIFACT_EVALUATION},
    {"ideaType": "drabble", "evaluation_config": ARTIFACT_EVALUATION},
])
def test_start_blocks_archived_execution_before_credentials_or_engine(body, monkeypatch):
    monkeypatch.setenv("IDEA_ENABLE_GEMINI_CODE_ARTIFACTS", "true")
    credential = AsyncMock()
    with patch.object(viewer.user_states, "get", AsyncMock(return_value=UserEvolutionState())), \
         patch.object(viewer, "_resolve_user_provider_key", credential), \
         patch.object(viewer, "EvolutionEngine") as engine:
        response = asyncio.run(viewer.start_evolution(Body(body), USER))
    assert response.status_code == 410
    assert json.loads(response.body)["code"] == "feature_archived"
    credential.assert_not_awaited()
    engine.assert_not_called()


def test_custom_artifact_template_is_blocked_without_changing_saved_template():
    custom = {"name": "Custom renderer", "evaluation": {"artifact_generation": True}}
    credential = AsyncMock()
    with patch.object(viewer.user_states, "get", AsyncMock(return_value=UserEvolutionState())), \
         patch.object(viewer.db, "get_user_template", AsyncMock(return_value=custom)), \
         patch.object(viewer, "_resolve_user_provider_key", credential):
        response = asyncio.run(viewer.start_evolution(Body({"ideaType": "custom-renderer"}), USER))
    assert response.status_code == 410
    credential.assert_not_awaited()
    assert custom["evaluation"]["artifact_generation"] is True


@pytest.mark.parametrize("handler,body", [
    (viewer.resume_evolution, {"checkpoint_id": "saved-run"}),
    (viewer.continue_evolution, {"evolution_id": "saved-run", "additional_generations": 1}),
    (viewer.continue_saved_evolution, {"evolution_id": "saved-run", "additional_generations": 1}),
])
@pytest.mark.parametrize("snapshot", [
    {"config": {"idea_type": "prime_visualization"}},
    {"config": {"idea_type": "drabble", "evaluation": ARTIFACT_EVALUATION}},
    {"config": {"idea_type": "custom", "evaluation_config": ARTIFACT_EVALUATION}},
    {"config": {"idea_type": "custom"}, "template_data": {"evaluation": {"artifact_generation": True}}},
])
def test_saved_archived_runs_are_read_only_before_restoration(handler, body, snapshot):
    credential = AsyncMock()
    settings = AsyncMock()
    load = AsyncMock()
    with patch.object(viewer.user_states, "get", AsyncMock(return_value=UserEvolutionState())), \
         patch.object(viewer.db, "get_evolution", AsyncMock(return_value=snapshot)), \
         patch.object(viewer.db, "get_user_settings", settings), \
         patch.object(viewer, "_resolve_user_provider_key", credential), \
         patch.object(viewer.EvolutionEngine, "load_evolution_for_user", load):
        response = asyncio.run(handler(Body(body), USER))
    assert response.status_code == 410
    assert "Existing results remain available" in json.loads(response.body)["message"]
    credential.assert_not_awaited()
    settings.assert_not_awaited()
    load.assert_not_awaited()


def test_template_catalogs_hide_archived_custom_templates_and_reserved_id():
    async def authenticated_user():
        return USER

    templates = [
        {"id": "prime_visualization", "name": "Old Prime copy"},
        {"id": "custom-renderer", "name": "Render", "evaluation": {"artifact_generation": True}},
        {"id": "custom-story", "name": "Story", "evaluation": {"artifact_generation": False}},
    ]
    viewer.app.dependency_overrides[viewer.require_auth] = authenticated_user
    try:
        with patch.object(viewer.db, "list_user_templates", AsyncMock(return_value=templates)):
            client = TestClient(viewer.app)
            types = client.get("/api/template-types")
            catalog = client.get("/api/templates/")
    finally:
        viewer.app.dependency_overrides.pop(viewer.require_auth, None)
    assert types.status_code == catalog.status_code == 200
    type_ids = {entry["id"] for entry in types.json()["templates"]}
    catalog_ids = set(catalog.json()["templates"])
    for ids in (type_ids, catalog_ids):
        assert "prime_visualization" not in ids
        assert "custom-renderer" not in ids
        assert {"drabble", "custom-story"}.issubset(ids)


def test_archived_history_has_no_resume_continue_or_reuse_actions():
    item = {"status": "error", "idea_type": "prime_visualization", "current_generation": 1,
            "total_generations": 2, "can_resume": True, "can_continue": True, "can_use_config": True}
    viewer._apply_history_capabilities(item)
    assert item["execution_archived"] is True
    assert item["can_resume"] is item["can_continue"] is item["can_use_config"] is False


def test_archived_auto_rating_is_explicitly_rejected_and_clears_running_state():
    state = UserEvolutionState()
    saved = {"config": {"idea_type": "prime_visualization"}}
    credential = AsyncMock()
    # A caller cannot bypass the archive by supplying a different template ID.
    request = Body({"evolutionId": "saved-run", "ideaType": "drabble"})
    with patch.object(viewer.user_states, "get", AsyncMock(return_value=state)), \
         patch.object(viewer.db, "ensure_evolution_editable", AsyncMock()), \
         patch.object(viewer, "_load_rating_evolution", AsyncMock(return_value=(None, saved, "firestore"))), \
         patch.object(viewer, "_resolve_user_provider_key", credential), \
         patch.object(viewer, "Critic") as critic:
        response = asyncio.run(viewer.auto_rate(request, USER))
    assert response.status_code == 410
    assert json.loads(response.body)["code"] == "feature_archived"
    assert state.autorating_status["is_running"] is False
    assert state.autorating_task is None
    credential.assert_not_awaited()
    critic.assert_not_called()
