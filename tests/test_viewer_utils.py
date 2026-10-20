import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.viewer import (
    _attach_live_engine_progress_fields,
    _compact_token_counts_for_run_state,
    _derive_run_status,
    _get_autorating_costs,
    _is_run_stale,
    _merge_status_update,
    _normalize_rating_fields,
    _should_honor_remote_stop,
    _update_autorating_state,
    idea_to_dict,
)
from idea.models import Idea
from idea.user_state import UserEvolutionState


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


@pytest.mark.parametrize("content", [
    Idea(title="Parent", content="Rules"),
    {"title": "Parent", "content": "Rules"},
    "Rules",
])
def test_live_candidate_serialization_preserves_gene_observation(content):
    observation = {
        "version": 1, "status": "ok", "genotype": "delayed consequences",
        "genes": [{"text": "delayed consequences", "gene_id": "gene-1"}],
        "mapping_status": "ok", "mapping_revision": 2,
    }
    candidate = {"id": "parent", "idea": content, "gene_observation": observation}

    published = json.loads(json.dumps(idea_to_dict(candidate)))

    assert published["gene_observation"] == observation
    assert published["content"] == "Rules"


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
        "generation_metrics": {
            "2": {
                "evaluation_generation": 2,
                "elo": 1532.5,
                "diversity": 0.84,
                "fitness": 0.62,
            }
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
    assert result["generation_metrics"]["2"]["fitness"] == 0.62
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


def test_run_status_derivation_handles_terminal_and_active_states():
    assert _derive_run_status({"error": "boom"}) == "error"
    assert _derive_run_status({"is_running": False, "is_stopped": True}) == "paused"
    assert _derive_run_status({"is_running": False, "is_resumable": True}) == "paused"
    assert _derive_run_status({"is_running": False}) == "complete"
    assert _derive_run_status({"is_resuming": True}) == "resuming"
    assert _derive_run_status({"is_continuing": True}) == "continuing"
    assert _derive_run_status({"is_running": True}) == "in_progress"


def test_run_stale_and_remote_stop_helpers_are_conservative():
    assert _is_run_stale({}, now_ms=1000) is False
    assert _is_run_stale({"status": "complete", "lease_expires_at_ms": 0}, now_ms=1000) is False
    assert _is_run_stale({"status": "in_progress", "lease_expires_at_ms": 999}, now_ms=1000) is True
    assert _is_run_stale({"status": "in_progress", "lease_expires_at_ms": 1001}, now_ms=1000) is False

    assert _should_honor_remote_stop(None) is False
    assert _should_honor_remote_stop({"stop_requested": False, "status": "stopping"}) is False
    assert _should_honor_remote_stop({"stop_requested": True, "status": "in_progress"}) is False
    assert _should_honor_remote_stop({"stop_requested": True, "status": "stopping"}) is True


def test_merge_status_update_clears_transient_flags_when_absent():
    state = UserEvolutionState()
    state.status.update({
        "oracle_update": True,
        "elite_selection_update": True,
        "checkpoint_saved": True,
        "progress": 10,
    })

    _merge_status_update(state, {"progress": 20, "oracle_update": True})

    assert state.status["progress"] == 20
    assert state.status["oracle_update"] is True
    assert "elite_selection_update" not in state.status
    assert "checkpoint_saved" not in state.status


def test_compact_token_counts_keeps_totals_but_truncates_diagnostic_events():
    token_counts = {
        "total": 100,
        "diagnostics": {
            "agents": {"critic": 2},
            "totals": {"events": 3},
            "events": [{"detail": "large"}],
        },
    }

    compact = _compact_token_counts_for_run_state(token_counts)

    assert compact == {
        "total": 100,
        "diagnostics": {
            "agents": {"critic": 2},
            "totals": {"events": 3},
            "events_truncated": True,
        },
    }
    assert _compact_token_counts_for_run_state("raw") == "raw"


def test_attach_live_engine_progress_fields_copies_engine_snapshots():
    state = UserEvolutionState()
    state.engine = SimpleNamespace(
        tournament_history=[{"generation": 1}],
        diversity_history=[{"diversity_score": 0.5}],
        island_diversity_history=[{"island_id": 0}],
        migration_history=[{"generation": 2}],
        get_total_token_count=lambda: {"total": 42},
    )

    _attach_live_engine_progress_fields(state)

    assert state.status["tournament_history"] == [{"generation": 1}]
    assert state.status["diversity_history"] == [{"diversity_score": 0.5}]
    assert state.status["island_diversity_history"] == [{"island_id": 0}]
    assert state.status["migration_history"] == [{"generation": 2}]
    assert state.status["token_counts"] == {"total": 42}

    state.engine.diversity_history.append({"diversity_score": 0.9})
    assert state.status["diversity_history"] == [{"diversity_score": 0.5}]


def test_update_autorating_state_merges_and_versions_payload(monkeypatch):
    state = UserEvolutionState()
    monkeypatch.setattr("idea.viewer._now_ms", lambda: 123456)

    _update_autorating_state(state, {"status": "running", "progress": 25})

    assert state.autorating_status["status"] == "running"
    assert state.autorating_status["progress"] == 25
    assert state.autorating_status["version"] == 1
    assert state.autorating_status["updated_at"] == 123456

    _update_autorating_state(None, {"status": "ignored"})


def test_autorating_costs_include_thinking_and_cached_inputs():
    critic = SimpleNamespace(
        get_token_usage_summary=lambda: {
            "total": 213_000,
            "input": 210_000,
            "output": 1_000,
            "thoughts": 2_000,
            "cached_input": 50_000,
            "uncached_input": 160_000,
            "tool_input": 0,
            "model": "gemini-2.5-pro",
            "cost": 0.4575,
            "input_cost": 0.4125,
            "uncached_input_cost": 0.4,
            "cached_input_cost": 0.0125,
            "output_cost": 0.015,
            "thought_cost": 0.03,
        }
    )

    result = _get_autorating_costs(critic)

    assert result["critic"]["thoughts"] == 2_000
    assert result["critic"]["cached_input"] == 50_000
    assert result["total_thoughts"] == 2_000
    assert result["total_cached_input"] == 50_000
    assert result["cost"]["thought_cost"] == 0.03
    assert result["cost"]["cached_input_cost"] == 0.0125
    assert result["cost"]["total_cost"] == 0.4575


def test_prompt_modal_renders_markdown_content():
    viewer_js = Path(__file__).parent.parent / "idea" / "static" / "js" / "viewer.js"
    script = viewer_js.read_text()
    show_prompt_modal = script[script.index("function showPromptModal"):]
    show_prompt_modal = show_prompt_modal[:show_prompt_modal.index("// Show the modal")]

    assert "renderMarkdown(displayContent)" in show_prompt_modal
    assert ".map(item => `<div class=\"prompt-item\">${item.trim()}</div>`)" not in show_prompt_modal
