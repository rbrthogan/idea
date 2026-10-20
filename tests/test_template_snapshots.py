"""Built-in prompt updates reach new runs without rewriting saved briefs."""
import asyncio
import copy
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from idea import viewer
from idea.auth import UserInfo
from idea.evolution import EvolutionEngine
from idea.evolution_persistence import EvolutionRepository, EvolutionSerializer
from idea.prompts import loader
from idea.user_state import UserEvolutionState


@pytest.fixture
def template_file(tmp_path, monkeypatch):
    data = yaml.safe_load(Path("idea/prompts/templates/game_design.yaml").read_text())
    data["version"] = "1.0.0"
    data["special_requirements"] = "- Preserve an artistic emotional signature.\n- Keep interaction playable."
    data["prompts"]["context"] = "OLD_CONTEXT"
    data["prompts"]["oracle_constraints"] = "OLD_ORACLE\n{requirements}"
    data["prompts"]["example_idea_prompts"] = "OLD_EXAMPLE: a playable poem"
    path = tmp_path / "game_design.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(loader, "_get_yaml_template_path", lambda _: path)
    for name in ("_prompt_cache", "_yaml_wrapper_cache", "_custom_template_cache", "_custom_template_wrapper_cache"):
        monkeypatch.setattr(loader, name, {})
    return path, data


def test_cached_builtin_notices_yaml_update_and_custom_cache_still_has_priority(template_file):
    path, data = template_file
    old = loader.get_prompts("game_design")
    assert old.version == "1.0.0"
    updated = copy.deepcopy(data)
    updated["version"] = "2.0.0"
    updated["prompts"]["context"] = "NEW_CONTEXT_AND_LONGER_CONTENT"
    path.write_text(yaml.safe_dump(updated))
    current = loader.get_prompts("game_design")
    assert current.version == "2.0.0"
    assert current.CONTEXT_PROMPT.startswith(updated["prompts"]["context"])
    assert old.version == "1.0.0"  # Existing wrapper references remain stable.
    custom = copy.deepcopy(data)
    custom["version"] = "3.0.0"
    loader.register_custom_template("custom_game", custom)
    assert loader.get_prompts("custom_game").version == "3.0.0"


def test_template_specific_oracle_brief_and_examples_reach_analysis(template_file):
    _, data = template_file
    snapshot = loader.get_builtin_template_snapshot("game_design")
    engine = new_engine(snapshot)
    prompt = engine.oracle._build_analysis_prompt([], "game_design")
    assert "OLD_ORACLE" in prompt
    assert "OLD_EXAMPLE: a playable poem" in prompt
    assert "Preserve an artistic emotional signature." in prompt
    assert "{requirements}" not in prompt
    assert "GENERATE a new diverse idea" in prompt  # Keep the core Oracle protocol.
    assert snapshot["prompts"]["example_idea_prompts"] == data["prompts"]["example_idea_prompts"]


def test_oracle_optional_fields_keep_generic_defaults_for_other_templates(template_file):
    _, data = template_file
    legacy = copy.deepcopy(data)
    legacy["prompts"].pop("oracle_constraints")
    legacy["prompts"].pop("example_idea_prompts")
    with patch("idea.prompts.oracle_prompts.ORACLE_CONSTRAINTS", "GENERIC_CONSTRAINTS"):
        generic = loader.get_prompts_from_dict(legacy)
        explicit = loader.get_prompts_from_dict(data)
    assert generic.ORACLE_CONSTRAINTS == "GENERIC_CONSTRAINTS"
    assert generic.EXAMPLE_IDEA_PROMPTS == ""
    assert "OLD_ORACLE" in explicit.ORACLE_CONSTRAINTS
    assert "GENERIC_CONSTRAINTS" not in explicit.ORACLE_CONSTRAINTS


def new_engine(snapshot=None):
    with patch("idea.llm.LLMWrapper._setup_provider"):
        return EvolutionEngine(pop_size=2, generations=1, idea_type="game_design", template_data=snapshot)


def test_structured_output_and_legacy_guidance_survive_checkpoint_restore(template_file):
    _, data = template_file
    snapshot = copy.deepcopy(data)
    snapshot["requirements"] = ["Keep the rules playable with two people."]
    snapshot["output"] = {
        "title": "An evocative name",
        "content": "One concise playable concept",
        "word_count": {"min": 80, "max": 150},
    }
    original = copy.deepcopy(snapshot)
    engine = new_engine(snapshot)
    engine.initialize_evolution(name="Structured game", evolution_id="structured")
    state = EvolutionSerializer.to_checkpoint_state(engine)
    with patch("idea.llm.LLMWrapper._setup_provider"):
        restored = EvolutionRepository.restore_from_state(EvolutionEngine, state)
    assert restored.template_data == original
    for agent in (restored.ideator, restored.formatter, restored.critic, restored.breeder, restored.oracle):
        template = agent._resolve_prompts("game_design").template
        assert template.requirements == original["requirements"]
        assert template.output.model_dump() == original["output"]
        assert template.special_requirements == original["special_requirements"]


def test_saved_snapshot_and_breeder_helper_remain_authoritative_after_builtin_update(template_file):
    path, data = template_file
    snapshot = loader.get_builtin_template_snapshot("game_design")
    engine = new_engine(snapshot)
    engine.initialize_evolution(name="Historical game", evolution_id="e")
    state = EvolutionSerializer.to_checkpoint_state(engine)
    updated = copy.deepcopy(data)
    updated["version"] = "2.0.0"
    updated["prompts"]["context"] = "NEW_CONTEXT"
    updated["prompts"]["oracle_constraints"] = "NEW_ORACLE"
    path.write_text(yaml.safe_dump(updated))
    assert loader.get_prompts("game_design").version == "2.0.0"
    with patch("idea.llm.LLMWrapper._setup_provider"):
        restored = EvolutionRepository.restore_from_state(EvolutionEngine, state)
        helper = restored.breeder._get_thread_ideator()
    assert restored.template_data == snapshot
    assert restored.ideator._resolve_prompts("game_design").CONTEXT_PROMPT.startswith("OLD_CONTEXT")
    assert helper._resolve_prompts("game_design").CONTEXT_PROMPT.startswith("OLD_CONTEXT")
    assert "OLD_ORACLE" in restored.oracle._build_analysis_prompt([], "game_design")


def test_legacy_unsnapshotted_restore_does_not_claim_current_file_as_original(template_file):
    engine = new_engine()
    engine.initialize_evolution(name="Legacy game", evolution_id="legacy")
    state = EvolutionSerializer.to_checkpoint_state(engine)
    assert "template_data" not in state
    with patch("idea.llm.LLMWrapper._setup_provider"):
        restored = EvolutionRepository.restore_from_state(EvolutionEngine, state)
    assert restored.template_data is None
    assert "template_data" not in EvolutionSerializer.to_checkpoint_state(restored)


@pytest.mark.parametrize("idea_type", ["game_design", "drabble", "airesearch", "arcade_game_design"])
def test_new_web_start_snapshots_every_builtin_instead_of_same_id_custom_template(template_file, idea_type):
    _, data = template_file
    state = UserEvolutionState()
    captured = []
    original_constructor = EvolutionEngine
    def construct(**kwargs):
        engine = original_constructor(**kwargs)
        captured.append(engine)
        return engine
    class Body:
        async def json(self):
            return {"ideaType": idea_type, "popSize": 2, "generations": 1,
                    "evolutionName": "New game"}
    custom = AsyncMock(return_value={"version": "0.5.0"})
    with patch("idea.llm.LLMWrapper._setup_provider"), \
         patch.object(viewer, "EvolutionEngine", side_effect=construct), \
         patch.object(viewer.user_states, "get", AsyncMock(return_value=state)), \
         patch.object(viewer, "_resolve_user_provider_key", AsyncMock(return_value="fake-test-key")), \
         patch.object(viewer, "_user_execution_mode", AsyncMock(return_value=None)), \
         patch.object(viewer, "list_available_templates", return_value={idea_type: {"name": "Built-in", "evaluation": {}}}), \
         patch.object(viewer.db, "get_user_template", custom), \
         patch.object(viewer, "_claim_run_slot", AsyncMock(return_value=(False, {}, "user"))):
        response = asyncio.run(viewer.start_evolution(Body(), UserInfo(uid="owner")))
    assert response.status_code == 409  # No worker or paid provider call is launched.
    assert len(captured) == 1
    custom.assert_not_awaited()
    snapshot = captured[0].template_data
    assert snapshot["version"] == "1.0.0"
    assert snapshot["prompts"]["context"] == data["prompts"]["context"]
    assert EvolutionSerializer.to_checkpoint_state(captured[0])["template_data"] == snapshot
