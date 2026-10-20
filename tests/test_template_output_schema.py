"""Structured output contracts survive loading and reject ambiguous bounds."""

from copy import deepcopy

import pytest

from idea.prompts.loader import get_prompts_from_dict
from idea.prompts.validation import TemplateValidator


def template_data():
    return {
        "name": "A short story",
        "description": "A complete miniature narrative",
        "version": "1.0.0",
        "author": "Test",
        "created_date": "2026-10-18",
        "metadata": {"item_type": "stories"},
        "requirements": [" Give the protagonist a consequential choice. "],
        "output": {
            "title": "A short evocative title",
            "content": "A complete narrative arc",
            "word_count": {"min": 100, "max": 100},
        },
        "prompts": {
            "context": "Generate concepts.",
            "specific_prompt": "Use {context_pool} to choose a direction.",
            "format": "Extract {input_text}.",
            "critique": "Review {idea}.",
            "refine": "Revise {idea} using {critique}.",
            "breed": "Combine {ideas}.",
            "genotype_encode": "Encode {idea_content}.",
        },
        "comparison_criteria": ["Emotional impact"],
    }


@pytest.mark.parametrize("bounds", [
    {}, {"min": 0}, {"max": -1}, {"min": 101, "max": 100},
    {"max": True}, {"max": 100.5}, {"max": "100"}, {"exact": 100},
])
def test_invalid_word_count_is_rejected_before_runtime(bounds):
    data = template_data()
    data["output"]["word_count"] = bounds
    with pytest.raises(ValueError):
        TemplateValidator.validate_dict(data)


@pytest.mark.parametrize("bounds", [None, {"min": 100}, {"max": 100}, {"min": 50, "max": 100}])
def test_optional_and_one_sided_word_limits_round_trip(bounds):
    data = template_data()
    data["output"]["word_count"] = bounds
    validated = TemplateValidator.validate_dict(data)
    snapshot = validated.model_dump(mode="json")
    restored = get_prompts_from_dict(snapshot).template
    assert restored == validated
    assert restored.requirements == ["Give the protagonist a consequential choice."]


@pytest.mark.parametrize("rules", [None, "one rule", [""], ["  "], [100], [{"rule": "ambiguous"}]])
def test_requirements_must_be_a_list_of_nonempty_text(rules):
    data = template_data()
    data["requirements"] = rules
    with pytest.raises(ValueError):
        TemplateValidator.validate_dict(data)


def test_loading_legacy_guidance_does_not_split_or_rewrite_the_source():
    data = template_data()
    data.pop("requirements")
    data.pop("output")
    legacy = "Output constraints:\n- Keep this exact text.\n  Title: [name]\n  Content: [body]"
    data["special_requirements"] = legacy
    before = deepcopy(data)
    wrapper = get_prompts_from_dict(data)
    assert data == before
    assert wrapper.template.special_requirements == legacy
    assert wrapper.template.requirements == []
    assert wrapper.template.output is None


def test_undefined_requirements_placeholder_is_reported():
    data = template_data()
    data.pop("requirements")
    data.pop("output")
    data["prompts"]["refine"] += " {requirements} {requirements_digest}"
    warnings = TemplateValidator.check_prompt_interpolation(TemplateValidator.validate_dict(data))
    assert "refine references requirements but none are defined" in warnings


@pytest.mark.parametrize("name,slot", [("specific_prompt", "context_pool"), ("format", "input_text"), ("critique", "idea"), ("refine", "idea"), ("breed", "ideas"), ("genotype_encode", "idea_content")])
def test_save_rejects_unknown_prompt_slots_and_escaped_required_slots(name, slot):
    from idea.template_manager import _validate_template_for_save
    data = template_data()
    data["prompts"][name] += " {typo}"
    with pytest.raises(ValueError, match=r"unsupported variable \{typo\}"):
        _validate_template_for_save(data)
    data = template_data()
    data["prompts"][name] = data["prompts"][name].replace("{" + slot + "}", "{{" + slot + "}}")
    with pytest.raises(ValueError, match="must include"):
        _validate_template_for_save(data)


def test_save_allows_literal_notation_in_guidance_and_escaped_prompt_text():
    from idea.template_manager import _validate_template_for_save
    data = template_data()
    data["requirements"] = ["Preserve {name} and {idea} in the example."]
    data["output"]["content"] += " Use {sections}."
    data["prompts"]["context"] += " Include notation {x}."
    data["prompts"]["oracle_constraints"] = "Explore {x}."
    data["prompts"]["critique"] += " Literal notation: {{x}}."
    data["prompts"]["comparison_prompt"] = "Compare {idea_a} against {idea_b}."
    _validate_template_for_save(data)
