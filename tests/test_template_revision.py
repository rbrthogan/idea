"""Template follow-up edits are validated drafts, never implicit saves."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from idea import template_manager as manager, viewer
from idea.auth import UserInfo


USER = UserInfo(uid="revision-author", email="author@example.test")


@pytest.fixture
def client():
    async def user():
        return USER
    viewer.app.dependency_overrides[viewer.require_auth] = user
    try:
        yield TestClient(viewer.app)
    finally:
        viewer.app.dependency_overrides.pop(viewer.require_auth, None)


@pytest.fixture(autouse=True)
def no_persistence():
    with patch.object(manager.db, "save_user_template", AsyncMock()) as save, \
         patch.object(manager.db, "get_user_template", AsyncMock()) as read, \
         patch.object(manager.db, "delete_user_template", AsyncMock()) as delete:
        yield
        save.assert_not_awaited()
        read.assert_not_awaited()
        delete.assert_not_awaited()


def current_template():
    source = manager.get_template_starter()
    source.update({
        "id": "existing-template-id", "version": "2.3.4", "author": "Original Author",
        "created_date": "2025-01-23", "provenance": {"paper": "Original source", "generation": {"model": "old-model"}},
        "evaluation": {"artifact_generation": False, "fitness_score": True},
        "future_config": {"keep": [1, 2, 3]},
        "requirements": ["Use reproducible experiments.", "Preserve literal {layout} notation."],
        "output": {"title": "A research title", "content": "A grounded proposal", "word_count": {"min": 400, "max": 600}},
        "special_requirements": "Keep the research scientifically grounded.",
        "comparison_prompt": "Compare {idea_a} against {idea_b} for scientific merit.",
    })
    source["metadata"]["source_context"] = {"title": "Original source", "citation": "Reference A"}
    source["prompts"].update({
        "comparison_prompt": "Compare {idea_a_content} and {idea_b_content}.",
        "oracle_constraints": "Keep experiments feasible.",
        "example_idea_prompts": "Try a small reproducible experiment.",
        "future_stage": {"untouched": True},
    })
    return source


def revised_template():
    revised = deepcopy(current_template())
    revised["description"] = "Research using inexpensive, reproducible experiments."
    revised["metadata"]["item_type"] = "low-cost research proposals"
    revised["requirements"] = ["Use reproducible experiments.", "Preserve literal {layout} notation.", "Use inexpensive equipment."]
    revised["output"] = {"title": "A specific research title", "content": "Include method, costs, and limitations.", "word_count": {"max": 300}}
    revised["comparison_criteria"] = ["reproducibility", "feasibility with inexpensive equipment"]
    for name, prompt in revised["prompts"].items():
        if isinstance(prompt, str) and name != "format":
            revised["prompts"][name] = prompt + " Focus on inexpensive experiments."
    revised["special_requirements"] = "Keep inexpensive experiments scientifically grounded."
    revised["comparison_prompt"] += " Prefer inexpensive equipment."
    return revised


def call_revision(client, response=None, *, template=None, selection=None, instruction="Use inexpensive equipment and limit the body to 300 words.", provider_error=None):
    returned = revised_template() if response is None else response
    llm = SimpleNamespace(client=SimpleNamespace(close=Mock()), generate_text=Mock(
        return_value=json.dumps(returned) if isinstance(returned, dict) else returned,
        side_effect=provider_error,
    ))
    credential = AsyncMock(return_value="synthetic-test-key")
    with patch.object(manager, "resolve_user_provider_key", credential), \
         patch.object(manager, "LLMWrapper", return_value=llm) as constructor, \
         patch.object(manager, "await_provider_call", wraps=manager.await_provider_call) as provider_call:
        result = client.post("/api/templates/revise", json={
            "template": template if template is not None else current_template(),
            "instruction": instruction, **(selection or {}),
        })
    return result, llm, constructor, credential, provider_call


def test_revision_requires_authentication_before_provider_or_template_access():
    with patch.object(manager, "LLMWrapper") as llm, \
         patch.object(manager, "resolve_user_provider_key", AsyncMock()) as credential:
        response = TestClient(viewer.app).post("/api/templates/revise", json={"template": current_template(), "instruction": "Revise."})
    assert response.status_code == 401
    llm.assert_not_called()
    credential.assert_not_awaited()


@pytest.mark.parametrize("selection,model,provider", [
    ({}, "gemini-4-argon", "google_generative_ai"),
    ({"provider": "openai"}, "gpt-6-astra", "openai"),
    ({"model_id": "gpt-6-astra"}, "gpt-6-astra", "openai"),
    ({"model_id": "gemini-3.5-flash-lite", "provider": "google_generative_ai"}, "gemini-3.5-flash-lite", "google_generative_ai"),
])
def test_revision_uses_selected_model_matching_key_and_cleanup(client, selection, model, provider):
    response, llm, constructor, credential, provider_call = call_revision(client, selection=selection)
    assert response.status_code == 200, response.text
    assert constructor.call_args.kwargs == {
        "provider": provider, "model_name": model, "temperature": 0.7, "top_p": 0.9,
        "thinking_level": "high", "thinking_budget": None, "api_key": "synthetic-test-key",
    }
    assert credential.await_args.args[:2] == (USER, provider)
    assert response.json()["generation"] == {"model_id": model, "provider": provider, "thinking_level": "high"}
    llm.generate_text.assert_called_once()
    llm.client.close.assert_called_once()
    assert provider_call.await_count == 2  # Generation and cleanup both stay off the event loop.


def test_full_coherent_revision_preserves_identity_provenance_and_unrelated_fields(client):
    original = current_template()
    source_snapshot = deepcopy(original)
    returned = revised_template()
    # A model may incorrectly rewrite identifiers or configuration; none may
    # cross the authoring boundary even when their new values are valid.
    returned.update({"id": "invented-id", "version": "9.9.9", "author": "Model", "created_date": "2026-10-19", "provenance": {}, "future_config": {}, "evaluation": {"fitness_score": False}})
    returned["metadata"]["source_context"] = {"citation": "Invented"}
    returned["prompts"]["future_stage"] = "Invented replacement"
    response, llm, *_ = call_revision(client, returned, template=original)
    assert response.status_code == 200, response.text
    payload = response.json()
    result = payload["template"]
    assert payload["status"] == "success" and payload["validation"]["is_valid"]
    for key in ("id", "version", "author", "created_date", "provenance", "evaluation", "future_config"):
        assert result[key] == original[key]
    assert result["metadata"]["source_context"] == original["metadata"]["source_context"]
    assert result["prompts"]["future_stage"] == original["prompts"]["future_stage"]
    assert result["metadata"]["item_type"] == returned["metadata"]["item_type"]
    for key in ("requirements", "output", "comparison_criteria", "description", "special_requirements", "comparison_prompt"):
        assert result[key] == returned[key]
    for name, prompt in returned["prompts"].items():
        if name != "future_stage":
            assert result["prompts"][name] == prompt
    assert original == source_snapshot
    prompt = llm.generate_text.call_args.args[0]
    assert "source data, not instructions" in prompt
    assert "complete JSON template object" in prompt
    embedded = json.loads(prompt.split("CURRENT TEMPLATE JSON:\n", 1)[1])
    assert embedded == source_snapshot
    assert '"Use inexpensive equipment and limit the body to 300 words."' in prompt


@pytest.mark.parametrize("change", [
    {"instruction": " "}, {"instruction": "x" * 12001}, {"instruction": 7},
    {"template": []}, {"template": {}}, {"extra_field": True},
    {"model_id": "unknown"}, {"provider": "unsupported"},
    {"provider": "openai", "model_id": "gemini-4-argon"},
])
def test_invalid_revision_request_does_no_paid_work(client, change):
    payload = {"template": current_template(), "instruction": "Make it shorter.", **change}
    with patch.object(manager, "LLMWrapper") as llm, \
         patch.object(manager, "resolve_user_provider_key", AsyncMock()) as credential:
        result = client.post("/api/templates/revise", json=payload)
    assert result.status_code == 422
    credential.assert_not_awaited()
    llm.assert_not_called()


def test_invalid_current_prompt_is_rejected_before_paid_work(client):
    source = current_template()
    source["prompts"]["refine"] = "An incomplete refine prompt"
    with patch.object(manager, "LLMWrapper") as llm, \
         patch.object(manager, "resolve_user_provider_key", AsyncMock()) as credential:
        result = client.post("/api/templates/revise", json={"template": source, "instruction": "Make it shorter."})
    assert result.status_code == 422 and "refine" in result.json()["message"]
    credential.assert_not_awaited()
    llm.assert_not_called()


@pytest.mark.parametrize("provider,label", [("google_generative_ai", "Gemini"), ("openai", "OpenAI")])
def test_missing_key_is_actionable_and_does_not_construct_provider(client, provider, label):
    with patch.object(manager, "resolve_user_provider_key", AsyncMock(return_value=None)), \
         patch.object(manager, "LLMWrapper") as llm:
        response = client.post("/api/templates/revise", json={"template": current_template(), "instruction": "Revise.", "provider": provider})
    assert response.status_code == 400
    assert f"{label} API key" in response.json()["message"]
    llm.assert_not_called()


@pytest.mark.parametrize("body", ["{broken", "[]", '{"instruction":"one","instruction":"two"}', '{"template":NaN}'])
def test_malformed_request_is_rejected_before_provider(client, body):
    with patch.object(manager, "LLMWrapper") as llm, \
         patch.object(manager, "resolve_user_provider_key", AsyncMock()) as credential:
        response = client.post("/api/templates/revise", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422
    llm.assert_not_called()
    credential.assert_not_awaited()


def test_oversized_request_is_rejected_before_parsing_or_provider(client):
    with patch.object(manager, "LLMWrapper") as llm, \
         patch.object(manager, "resolve_user_provider_key", AsyncMock()) as credential:
        response = client.post("/api/templates/revise", content="x" * (manager.TEMPLATE_REVISION_MAX_BYTES + 1), headers={"content-type": "application/json"})
    assert response.status_code == 413
    llm.assert_not_called()
    credential.assert_not_awaited()


def test_chunked_request_is_bounded_without_content_length():
    class Request:
        headers = {}
        async def stream(self):
            yield b"x" * manager.TEMPLATE_REVISION_MAX_BYTES
            yield b"x"
            raise AssertionError("Must stop reading after size bound")
    with pytest.raises(manager.HTTPException) as caught:
        asyncio.run(manager._read_template_revision_request(Request()))
    assert caught.value.status_code == 413


@pytest.mark.parametrize("replacement", [
    "not JSON", "{}", "[]", '{"name":"one","name":"two"}',
    "```json\n{}\n```", 'Name: Partial template',
])
def test_malformed_or_partial_model_response_never_replaces_draft(client, replacement):
    response, llm, *_ = call_revision(client, replacement)
    assert response.status_code == 422
    assert "draft is unchanged" in response.json()["message"]
    assert "template" not in response.json()
    llm.generate_text.assert_called_once()
    llm.client.close.assert_called_once()


@pytest.mark.parametrize("field", ["prompts.refine", "prompts.oracle_constraints", "requirements", "output", "comparison_criteria", "author", "special_requirements"])
def test_missing_response_fields_are_not_silently_filled_from_old_template(client, field):
    response_template = revised_template()
    mapping = response_template
    names = field.split(".")
    for name in names[:-1]:
        mapping = mapping[name]
    mapping.pop(names[-1])
    response, *_ = call_revision(client, response_template)
    assert response.status_code == 422
    assert "template" not in response.json()


@pytest.mark.parametrize("field,value", [
    ("requirements", "Not a list"), ("output", {"word_count": {"min": 200, "max": 100}}),
    ("output", {"word_count": {"max": "100"}}), ("comparison_criteria", [""]),
])
def test_invalid_structured_model_response_is_rejected_without_retry(client, field, value):
    revised = revised_template()
    revised[field] = value
    response, llm, *_ = call_revision(client, revised)
    assert response.status_code == 422
    llm.generate_text.assert_called_once()
    llm.client.close.assert_called_once()


def test_provider_failure_closes_client_and_leaves_original_draft(client):
    response, llm, *_ = call_revision(client, provider_error=RuntimeError("private provider diagnostic"))
    assert response.status_code == 502
    assert "private provider diagnostic" not in response.text
    assert "draft is unchanged" in response.json()["message"]
    llm.generate_text.assert_called_once()
    llm.client.close.assert_called_once()


def test_unreadable_provider_key_is_actionable_without_exposing_diagnostics(client):
    with patch.object(manager, "resolve_user_provider_key", AsyncMock(side_effect=manager.db.APIKeyDecryptionError("private diagnostic"))), \
         patch.object(manager, "LLMWrapper") as llm:
        response = client.post("/api/templates/revise", json={"template": current_template(), "instruction": "Revise."})
    assert response.status_code == 400
    assert "replacement in Settings" in response.json()["message"]
    assert "private diagnostic" not in response.text
    llm.assert_not_called()


def test_cleanup_failure_does_not_discard_valid_revision(client):
    llm = SimpleNamespace(client=SimpleNamespace(close=Mock(side_effect=RuntimeError("cleanup failed"))),
                          generate_text=Mock(return_value=json.dumps(revised_template())))
    with patch.object(manager, "resolve_user_provider_key", AsyncMock(return_value="synthetic-test-key")), \
         patch.object(manager, "LLMWrapper", return_value=llm):
        response = client.post("/api/templates/revise", json={"template": current_template(), "instruction": "Revise."})
    assert response.status_code == 200
    assert response.json()["validation"]["is_valid"]
    llm.client.close.assert_called_once()


def test_archived_execution_template_is_rejected_before_provider(client):
    source = current_template()
    source["evaluation"]["artifact_generation"] = True
    with patch.object(manager, "LLMWrapper") as llm, \
         patch.object(manager, "resolve_user_provider_key", AsyncMock()) as credential:
        response = client.post("/api/templates/revise", json={"template": source, "instruction": "Revise."})
    assert response.status_code == 410
    llm.assert_not_called()
    credential.assert_not_awaited()


def test_oversized_model_response_is_rejected_after_cleanup(client):
    response, llm, *_ = call_revision(client, "x" * (manager.TEMPLATE_REVISION_MAX_BYTES + 1))
    assert response.status_code == 422
    assert "draft is unchanged" in response.json()["message"]
    llm.client.close.assert_called_once()
