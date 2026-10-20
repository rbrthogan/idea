"""Structured requirements/output survive authoring without paid provider calls."""

import asyncio
from copy import deepcopy
import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
import yaml

from idea import template_manager as manager
from idea.auth import UserInfo


USER = UserInfo(uid="structured-template-author")


def sample_template():
    template = manager.get_template_starter()
    template["requirements"] = ["Use everyday ingredients.", "Avoid unnecessary waste."]
    template["output"] = {
        "title": "A useful meal name.",
        "content": "Include ingredients, steps, and a leftovers plan.",
        "word_count": {"min": 120, "max": 200},
    }
    return template


@pytest.mark.parametrize("encoding", ["json", "fenced_json", "yaml", "fenced_yaml"])
def test_structured_generated_contract_roundtrips(encoding):
    source = sample_template()
    body = json.dumps(source) if "json" in encoding else yaml.safe_dump(source)
    if encoding.startswith("fenced_"):
        body = f"```{encoding.removeprefix('fenced_')}\n{body}\n```"
    parsed = manager.parse_generated_template(body, "Meal ideas")
    assert parsed["requirements"] == source["requirements"]
    assert parsed["output"] == source["output"]
    assert parsed["prompts"] == source["prompts"]
    assert manager._template_validation_report(parsed)["is_valid"]


def test_invalid_structured_fields_are_preserved_for_validation():
    source = sample_template()
    source["requirements"] = "Not a list"
    source["output"]["word_count"] = {"max": "200"}
    parsed = manager.parse_generated_template(json.dumps(source), "Meal ideas")
    assert parsed["requirements"] == "Not a list"
    assert parsed["output"]["word_count"] == {"max": "200"}
    assert not manager._template_validation_report(parsed)["is_valid"]


def test_legacy_human_labelled_draft_keeps_additional_guidance():
    parsed = manager.parse_generated_template(
        "Name: Legacy draft\nDescription: Older response\nSpecial Requirements: Keep this guidance.\n"
        "Item Type: ideas\nContext Prompt: Generate concepts.\n"
        "Direction Builder Prompt: Use {context_pool}.\nFormat Prompt: Preserve {input_text}.\n"
        "Critique Prompt: Review {idea}.\nRefine Prompt: Improve {idea} with {critique}.\n"
        "Breed Prompt: Combine {ideas}.\nGenotype Encode Prompt: Encode {idea_content}.\n"
        "Comparison Criteria: clarity, originality",
        "Legacy ideas",
    )
    assert parsed["name"] == "Legacy draft"
    assert parsed["special_requirements"] == "Keep this guidance."
    assert parsed["requirements"] == []
    assert "output" not in parsed
    assert manager._template_validation_report(parsed)["is_valid"]


def test_guided_update_preserves_source_and_legacy_text():
    source = sample_template()
    legacy = "  Original mixed guidance.\nTitle: keep legacy instructions intact.\n"
    source["special_requirements"] = legacy
    original = deepcopy(source)
    request = manager.TemplateUpdateRequest(
        name="Updated meals", requirements=["Use one pan."],
        output={"title": "A descriptive title", "content": "Two concise sections", "word_count": {"max": 150}},
    )
    saved = AsyncMock()
    with patch.object(manager.db, "get_user_template", AsyncMock(return_value=source)), \
         patch.object(manager.db, "save_user_template", saved):
        response = asyncio.run(manager.update_template("my_meals", request, USER))
    assert response.status_code == 200
    result = saved.await_args.args[2]
    assert result["requirements"] == ["Use one pan."]
    assert result["output"]["word_count"] == {"max": 150}
    assert result["special_requirements"] == legacy
    assert source == original


@pytest.mark.parametrize("fields", [{}, {"requirements": [], "output": None}])
def test_omitted_fields_preserve_and_explicit_empty_fields_clear(fields):
    source = sample_template()
    payload = manager._coerce_template_payload("my_meals", source)
    manager._apply_template_fields(payload, manager.TemplateUpdateRequest(**fields))
    assert payload["requirements"] == (fields["requirements"] if fields else source["requirements"])
    assert payload["output"] == (None if fields else source["output"])
    manager._validate_template_for_save(payload)


@pytest.mark.parametrize("fields", [
    {"requirements": None}, {"requirements": "A rule"}, {"requirements": [7]},
    {"output": []}, {"output": {"content": 7}},
    {"output": {"word_count": {"max": "100"}}},
    {"output": {"word_count": {"max": True}}},
    {"output": {"word_count": {"min": 300, "max": 100}}},
    {"output": {"word_count": {}}}, {"output": {"unknown": "ignored?"}},
])
def test_guided_api_rejects_invalid_contract_fields(fields):
    with pytest.raises(ValidationError):
        manager.TemplateUpdateRequest(**fields)


@pytest.mark.parametrize("field,value", [
    ("requirements", None), ("requirements", [" "]),
    ("output", {"word_count": {"max": "100"}}),
    ("output", {"word_count": {"max": 0}}),
])
def test_full_source_invalid_contract_is_not_silently_dropped(field, value):
    source = sample_template()
    invalid = deepcopy(source)
    invalid[field] = value
    saved = AsyncMock()
    with patch.object(manager.db, "get_user_template", AsyncMock(return_value=source)), \
         patch.object(manager.db, "save_user_template", saved):
        response = asyncio.run(manager.update_template(
            "my_meals", manager.TemplateUpdateRequest(template_data=invalid), USER,
        ))
    assert response.status_code == 422
    saved.assert_not_awaited()


def test_generation_prompt_teaches_contract_separation_and_json():
    prompt = manager.create_template_generation_prompt("Meals", {"meals": sample_template()})
    assert "Return one JSON object only" in prompt
    assert "Put content goals" in prompt
    assert "output.word_count" in prompt
    assert '"requirements": [' in prompt
    assert '"output": {' in prompt
    assert "fixed title/content transport format" in prompt
    assert "special_requirements is a legacy field" in prompt


def test_shipped_templates_use_explicit_contract_without_transport_duplication():
    for template_id in ("airesearch", "drabble", "game_design", "arcade_game_design"):
        source = yaml.safe_load((manager.TEMPLATES_DIR / f"{template_id}.yaml").read_text())
        manager._validate_template_for_save(source)
        assert source["requirements"]
        assert source["output"]["title"] and source["output"]["content"]
        assert "special_requirements" not in source
        assert "{requirements}" not in json.dumps(source["prompts"])
        assert "{requirements_digest}" not in json.dumps(source["prompts"])
        assert "Title:" not in json.dumps(source)
        if template_id == "drabble":
            assert source["output"]["word_count"] == {"min": 100, "max": 100}
