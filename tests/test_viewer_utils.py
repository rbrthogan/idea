import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.viewer import idea_to_dict, _normalize_rating_fields
from idea.models import Idea


def test_idea_to_dict_from_model():
    idea = Idea(title="Hello", content="World")
    result = idea_to_dict(idea)
    assert result["title"] == "Hello"
    assert result["content"] == "World"
    assert result["parent_ids"] == []


def test_idea_to_dict_from_dict_with_model():
    input_data = {
        "id": "12345",
        "idea": Idea(title="Hello", content="World"),
        "parent_ids": [1, 2],
        "match_count": 5,
        "auto_match_count": 3,
        "manual_match_count": 2
    }
    result = idea_to_dict(input_data)
    assert result["id"] == "12345"
    assert result["title"] == "Hello"
    assert result["content"] == "World"
    assert result["parent_ids"] == [1, 2]
    assert result["match_count"] == 5
    assert result["auto_match_count"] == 3
    assert result["manual_match_count"] == 2


def test_idea_to_dict_from_dict_with_string():
    input_data = {
        "id": "12345",
        "idea": "Hello World",
        "parent_ids": [1, 2],
    }
    result = idea_to_dict(input_data)
    assert result["id"] == "12345"
    assert result["title"] == "Untitled"
    assert result["content"] == "Hello World"
    assert result["parent_ids"] == [1, 2]


def test_idea_to_dict_from_dict_with_dict():
    input_data = {
        "id": "12345",
        "idea": {"title": "Hello", "content": "World"},
        "parent_ids": [1, 2],
    }
    result = idea_to_dict(input_data)
    assert result["id"] == "12345"
    assert result["title"] == "Hello"
    assert result["content"] == "World"
    assert result["parent_ids"] == [1, 2]


def test_idea_to_dict_from_string():
    result = idea_to_dict("hello")
    assert result["title"] == "Untitled"
    assert result["content"] == "hello"
    assert result["parent_ids"] == []


def test_idea_to_dict_preserves_oracle_metadata():
    """Test that Oracle metadata is properly preserved"""
    input_data = {
        "id": "oracle-test-id",
        "idea": Idea(title="Oracle Generated Idea", content="This is an oracle idea"),
        "parent_ids": [],
        "oracle_generated": True,
        "oracle_analysis": "This idea was generated to address diversity gaps in the population."
    }
    result = idea_to_dict(input_data)
    assert result["id"] == "oracle-test-id"
    assert result["title"] == "Oracle Generated Idea"
    assert result["content"] == "This is an oracle idea"
    assert result["parent_ids"] == []
    assert result["oracle_generated"] is True
    assert result["oracle_analysis"] == "This idea was generated to address diversity gaps in the population."


def test_idea_to_dict_preserves_breeding_genotypes():
    input_data = {
        "id": "genotype-test-id",
        "idea": Idea(title="Parent Idea", content="This idea was selected"),
        "parent_ids": [],
        "breeding_genotypes": [
            {
                "genotype": "Gothic fantasy; dusty attic",
                "source_generation": 0,
                "child_generation": 1,
                "child_id": "child-1",
            }
        ],
    }

    result = idea_to_dict(input_data)

    assert result["breeding_genotypes"] == input_data["breeding_genotypes"]


def test_idea_to_dict_preserves_island_metadata():
    input_data = {
        "id": "island-test-id",
        "idea": Idea(title="Island Idea", content="This idea migrated"),
        "parent_ids": [],
        "island_id": 2,
        "origin_island_id": 0,
        "migrated_from_id": "source-id",
        "migration_generation": 3,
        "migration_event": True,
    }

    result = idea_to_dict(input_data)

    assert result["island_id"] == 2
    assert result["origin_island_id"] == 0
    assert result["migrated_from_id"] == "source-id"
    assert result["migration_generation"] == 3
    assert result["migration_event"] is True


def test_idea_to_dict_preserves_tree_metric_metadata():
    input_data = {
        "id": "metric-test-id",
        "idea": Idea(title="Metric Idea", content="This idea has scores"),
        "parent_ids": [],
        "selection_metrics": {
            "evaluation_generation": 2,
            "elo": 1532.5,
            "diversity": 0.84,
            "fitness": 0.62,
        },
        "survival_metrics": {
            "survived_to_generation": 3,
            "survival_score": 0.51,
        },
    }

    result = idea_to_dict(input_data)

    assert result["selection_metrics"]["elo"] == 1532.5
    assert result["selection_metrics"]["diversity"] == 0.84
    assert result["selection_metrics"]["fitness"] == 0.62
    assert result["survival_metrics"]["survived_to_generation"] == 3


def test_idea_to_dict_without_oracle_metadata():
    """Test that non-Oracle ideas don't have Oracle metadata"""
    input_data = {
        "id": "regular-test-id",
        "idea": Idea(title="Regular Idea", content="This is a regular idea"),
        "parent_ids": []
    }
    result = idea_to_dict(input_data)
    assert result["id"] == "regular-test-id"
    assert result["title"] == "Regular Idea"
    assert result["content"] == "This is a regular idea"
    assert result["parent_ids"] == []
    assert "oracle_generated" not in result
    assert "oracle_analysis" not in result


def test_normalize_rating_fields_uses_nested_idea_payload():
    idea = {
        "id": "nested-idea-1",
        "idea": {
            "title": "Nested Title",
            "content": "Nested content body",
        },
    }

    _normalize_rating_fields(idea, "fallback-id")

    assert idea["title"] == "Nested Title"
    assert idea["content"] == "Nested content body"
    assert idea["ratings"] == {"auto": 1500, "manual": 1500}


def test_normalize_rating_fields_parses_json_encoded_content():
    idea = {
        "content": json.dumps({
            "title": "Decoded Title",
            "content": "Decoded content body",
        }),
    }

    _normalize_rating_fields(idea, "fallback-json")

    assert idea["id"] == "fallback-json"
    assert idea["title"] == "Decoded Title"
    assert idea["content"] == "Decoded content body"


def test_prompt_modal_renders_markdown_content():
    viewer_js = Path(__file__).parent.parent / "idea" / "static" / "js" / "viewer.js"
    script = viewer_js.read_text()
    show_prompt_modal = script[script.index("function showPromptModal"):]
    show_prompt_modal = show_prompt_modal[:show_prompt_modal.index("// Show the modal")]

    assert "renderMarkdown(displayContent)" in show_prompt_modal
    assert ".map(item => `<div class=\"prompt-item\">${item.trim()}</div>`)" not in show_prompt_modal
