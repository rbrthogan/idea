"""Template authoring model choices and lossless guided/advanced editing."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from idea import template_manager as manager
from idea import viewer
from idea.auth import UserInfo
from idea.config import DEFAULT_MODEL, TEMPLATE_GENERATION_DEFAULTS_BY_PROVIDER


USER = UserInfo(uid="template-author", email="author@example.test")


@pytest.fixture
def client():
    async def user():
        return USER
    viewer.app.dependency_overrides[viewer.require_auth] = user
    try:
        yield TestClient(viewer.app)
    finally:
        viewer.app.dependency_overrides.pop(viewer.require_auth, None)


def test_model_catalog_publishes_separate_strong_template_defaults(client):
    data = client.get("/api/models").json()
    assert data["default"] == DEFAULT_MODEL
    choices = data["template_generation"]
    assert choices["default_model"] == "gemini-4-argon"
    assert choices["defaults_by_provider"] == TEMPLATE_GENERATION_DEFAULTS_BY_PROVIDER
    catalog = {model["id"]: model for model in data["models"]}
    for provider, model_id in choices["defaults_by_provider"].items():
        assert catalog[model_id]["provider"] == provider


@pytest.mark.parametrize("selection,expected_model,expected_provider", [
    ({}, "gemini-4-argon", "google_generative_ai"),
    ({"provider": "openai"}, "gpt-6-astra", "openai"),
    ({"model_id": "gpt-6-astra"}, "gpt-6-astra", "openai"),
    ({"provider": "google_generative_ai", "model_id": "gemini-3.5-flash-lite"}, "gemini-3.5-flash-lite", "google_generative_ai"),
])
def test_generation_routes_selected_model_and_matching_credential(client, selection, expected_model, expected_provider):
    kwargs_seen = {}
    class FakeLLM:
        def __init__(self, **kwargs):
            kwargs_seen.update(kwargs)
        def generate_text(self, prompt):
            return "Name: Test\nDescription: A draft for review"
    credential = AsyncMock(return_value="synthetic-test-key")
    with patch.object(manager, "resolve_user_provider_key", credential), \
         patch.object(manager, "LLMWrapper", FakeLLM):
        result = client.post("/api/templates/generate", json={"idea_type_suggestion": "Story ideas", **selection})
    assert result.status_code == 200
    assert kwargs_seen["model_name"] == expected_model
    assert kwargs_seen["provider"] == expected_provider
    assert kwargs_seen["api_key"] == "synthetic-test-key"
    assert kwargs_seen["thinking_level"] == "high"
    assert credential.await_args.args[1] == expected_provider
    assert result.json()["generation"]["model_id"] == expected_model


@pytest.mark.parametrize("body", [
    {"idea_type_suggestion": "  "},
    {"idea_type_suggestion": "x" * 12001},
    {"idea_type_suggestion": "Stories", "model_id": "unknown-model"},
    {"idea_type_suggestion": "Stories", "provider": "unknown-provider"},
    {"idea_type_suggestion": "Stories", "provider": "openai", "model_id": "gemini-4-argon"},
    {"idea_type_suggestion": "Stories", "unexpected_setting": "ignored?"},
    ["not-an-object"],
])
def test_invalid_generation_input_never_reaches_credentials_or_provider(client, body):
    credential = AsyncMock()
    with patch.object(manager, "resolve_user_provider_key", credential), \
         patch.object(manager, "LLMWrapper") as llm:
        response = client.post("/api/templates/generate", json=body)
    assert response.status_code == 422
    assert response.json()["message"]
    credential.assert_not_awaited()
    llm.assert_not_called()


def test_multipart_model_choice_and_openai_attachment_validation(client):
    credential = AsyncMock()
    with patch.object(manager, "resolve_user_provider_key", credential), \
         patch.object(manager, "LLMWrapper") as llm:
        response = client.post("/api/templates/generate",
            data={"idea_type_suggestion": "Follow-up research", "model_id": "gpt-6-astra", "provider": "openai"},
            files={"context_file": ("paper.pdf", b"%PDF synthetic", "application/pdf")})
    assert response.status_code == 422
    assert "supported with Gemini" in response.json()["message"]
    credential.assert_not_awaited()
    llm.assert_not_called()


def test_missing_selected_provider_key_is_actionable(client):
    with patch.object(manager, "resolve_user_provider_key", AsyncMock(return_value=None)), \
         patch.object(manager, "LLMWrapper") as llm:
        response = client.post("/api/templates/generate", json={"idea_type_suggestion": "Stories", "provider": "openai"})
    assert response.status_code == 400
    assert "OpenAI API key" in response.json()["message"]
    llm.assert_not_called()


def full_template():
    template = manager.get_template_starter()
    template["name"] = "Advanced stories"
    template["prompts"].update({
        "oracle_constraints": "Preserve theme and emotional intent. {requirements}",
        "example_idea_prompts": "A dream told through linked objects.",
        "comparison_prompt": "Compare {idea_a} with {idea_b}.",
    })
    template["comparison_prompt"] = "Legacy comparison instruction"
    return template


def flat_fields(template):
    return {
        "name": template["name"], "description": template["description"],
        "author": template["author"], "item_type": template["metadata"]["item_type"],
        "context_prompt": template["prompts"]["context"],
        "specific_prompt": template["prompts"]["specific_prompt"],
        "format_prompt": template["prompts"]["format"],
        "critique_prompt": template["prompts"]["critique"],
        "refine_prompt": template["prompts"]["refine"],
        "breed_prompt": template["prompts"]["breed"],
        "genotype_encode_prompt": template["prompts"]["genotype_encode"],
        "comparison_criteria": template["comparison_criteria"],
    }


def test_create_copy_preserves_advanced_prompts_from_full_editor():
    original = full_template()
    snapshot = deepcopy(original)
    request = manager.TemplateCreateRequest(**flat_fields(original), template_data=original)
    saved = AsyncMock()
    with patch.object(manager, "_generate_unique_template_id", AsyncMock(return_value="advanced_stories")), \
         patch.object(manager.db, "save_user_template", saved):
        response = asyncio.run(manager.create_template(request, USER))
    assert response.status_code == 200
    payload = saved.await_args.args[2]
    assert payload["prompts"] == original["prompts"]
    assert payload["comparison_prompt"] == original["comparison_prompt"]
    assert original == snapshot


@pytest.mark.parametrize("full_editor", [False, True])
def test_edit_preserves_advanced_prompts_and_saved_snapshot(full_editor):
    original = full_template()
    snapshot = deepcopy(original)
    edited = deepcopy(original)
    if full_editor:
        edited["prompts"]["oracle_constraints"] = "Revised advanced direction."
    request = manager.TemplateUpdateRequest(name="Renamed stories", template_data=edited if full_editor else None)
    saved = AsyncMock()
    with patch.object(manager.db, "get_user_template", AsyncMock(return_value=original)), \
         patch.object(manager.db, "save_user_template", saved):
        response = asyncio.run(manager.update_template("advanced_stories", request, USER))
    assert response.status_code == 200
    payload = saved.await_args.args[2]
    assert payload["name"] == "Renamed stories"
    assert payload["prompts"] == edited["prompts"]
    assert payload["comparison_prompt"] == original["comparison_prompt"]
    assert original == snapshot


@pytest.mark.parametrize("replacement", ["", "A prompt without the required variable"])
def test_invalid_editor_prompt_returns_field_specific_error_without_saving(replacement):
    original = full_template()
    request = manager.TemplateUpdateRequest(format_prompt=replacement)
    saved = AsyncMock()
    with patch.object(manager.db, "get_user_template", AsyncMock(return_value=original)), \
         patch.object(manager.db, "save_user_template", saved):
        response = asyncio.run(manager.update_template("advanced_stories", request, USER))
    assert response.status_code == 422
    assert "format prompt" in json.loads(response.body)["message"]
    saved.assert_not_awaited()


def test_processing_timeout_deletes_uploaded_context_without_generating():
    files = SimpleNamespace(upload=lambda **_: SimpleNamespace(name="files/test"), delete=lambda **_: None)
    with patch.object(manager, "_wait_for_template_context_file", side_effect=RuntimeError("still processing")), \
         patch.object(files, "delete") as delete:
        with pytest.raises(RuntimeError, match="still processing"):
            manager._upload_template_context_file(SimpleNamespace(files=files),
                manager.TemplateContextFile(filename="paper.pdf", content_type="application/pdf", data=b"%PDF test"))
    delete.assert_called_once_with(name="files/test")
