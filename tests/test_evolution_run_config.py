import inspect
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field, ValidationError

from idea.artifact_store import ArtifactStore
from idea.evolution import EvolutionEngine
from idea.evolution_persistence import EvolutionRepository, EvolutionSerializer
from idea.evolution_types import EvolutionConfig, EvolutionRunConfig
from idea.viewer import _engine_config_payload


class _EvaluationConfig(BaseModel):
    schema_version: int = 1
    enabled: bool = True
    metadata: dict = Field(default_factory=dict)


def _checkpoint_engine(
    config: EvolutionRunConfig,
    *,
    template_data=None,
):
    values = config.to_persistence_dict()
    evaluation_config = values.pop("evaluation")
    return SimpleNamespace(
        **values,
        evaluation_config=evaluation_config,
        checkpoint_id="checkpoint-1",
        stop_requested=False,
        current_generation=1,
        population=[],
        history=[],
        contexts=[],
        specific_prompts=[],
        breeding_prompts=[],
        tournament_history=[],
        diversity_history=[],
        island_diversity_history=[],
        migration_history=[],
        comparison_cache={},
        avg_idea_cost=0.0,
        avg_tournament_cost=0.0,
        fitness_elo_stats={"count": 0, "mean": 0.0, "m2": 0.0},
        fitness_diversity_stats={"count": 0, "mean": 0.0, "m2": 0.0},
        fitness_metric_stats={},
        get_total_token_count=lambda: {"total": 0},
        template_data=template_data,
    )


class _CapturingEngine:
    def __init__(self, **kwargs):
        self.constructor_kwargs = kwargs
        self.generations = kwargs["generations"]

    def _sync_typed_state_from_attrs(self):
        self.typed_state_synchronized = True


def test_run_config_defaults_are_stable_and_complete():
    config = EvolutionRunConfig()

    assert config.to_persistence_dict() == {
        "idea_type": "airesearch",
        "pop_size": 5,
        "generations": 3,
        "model_type": "gemini-3.5-flash-lite",
        "creative_temp": 1.5,
        "top_p": 0.9,
        "random_seed": None,
        "tournament_rounds": 1,
        "tournament_count": 0.25,
        "full_tournament_rounds": 4,
        "thinking_budget": None,
        "thinking_level": None,
        "max_budget": None,
        "mutation_rate": 0.3,
        "independent_birth_rate": 0.5,
        "single_parent_birth_rate": 0.25,
        "diversity_representation": "criterion_summaries",
        "novelty_neighbors": 3,
        "seed_context_pool_size": None,
        "refine_all_ideas": False,
        "survivor_refinement_rate": 0.1,
        "island_count": 2,
        "migration_rate": 0.05,
        "migration_novelty_protection_quantile": 0.8,
        "replacement_rate": 0.5,
        "fitness_alpha": 0.7,
        "age_decay_rate": 0.25,
        "age_decay_floor": 0.35,
        "context_novelty_threshold": 0.8,
        "context_novelty_max_attempts": 2,
        "evaluation": None,
    }
    assert EvolutionConfig is EvolutionRunConfig


def test_run_config_covers_every_engine_constructor_run_setting():
    constructor_parameters = set(inspect.signature(EvolutionEngine.__init__).parameters)
    constructor_run_settings = (
        constructor_parameters - EvolutionRunConfig.ENGINE_CONTEXT_PARAMETERS
    )

    assert constructor_run_settings == EvolutionRunConfig.engine_parameter_names()


def test_run_config_coerces_and_bounds_current_engine_settings():
    config = EvolutionRunConfig(
        idea_type="  prime_visualization ",
        pop_size="3",
        generations="0",
        model_type=" gemini-2.5-flash ",
        creative_temp="9",
        top_p="-0.4",
        random_seed="42",
        tournament_rounds="0",
        tournament_count=None,
        full_tournament_rounds="0",
        thinking_budget="-1",
        thinking_level=" HIGH ",
        max_budget="-3.5",
        mutation_rate="1.7",
        seed_context_pool_size="0",
        refine_all_ideas="yes",
        survivor_refinement_rate="4",
        island_count="10",
        migration_rate="not-a-number",
        migration_novelty_protection_quantile="-1",
        replacement_rate="2",
        fitness_alpha="-2",
        age_decay_rate="-0.5",
        age_decay_floor="2",
        context_novelty_threshold="1.4",
        context_novelty_max_attempts="-3",
        evaluation={"enabled": False},
    )

    assert config.idea_type == "prime_visualization"
    assert config.pop_size == 3
    assert config.generations == 1
    assert config.model_type == "gemini-2.5-flash"
    assert config.creative_temp == 2.0
    assert config.top_p == 0.0
    assert config.random_seed == 42
    assert config.tournament_rounds == 1
    assert config.full_tournament_rounds == 2
    assert config.tournament_count == 0.5
    assert config.thinking_budget == -1
    assert config.thinking_level == "high"
    assert config.max_budget == 0.0
    assert config.mutation_rate == 1.0
    assert config.seed_context_pool_size == 1
    assert config.refine_all_ideas is True
    assert config.survivor_refinement_rate == 1.0
    assert config.island_count == 1  # Small populations must not create singleton islands.
    assert config.migration_rate == 0.05
    assert config.migration_novelty_protection_quantile == 0.0
    assert config.replacement_rate == 1.0
    assert config.fitness_alpha == 0.0
    assert config.age_decay_rate == 0.0
    assert config.age_decay_floor == 1.0
    assert config.context_novelty_threshold == 1.0
    assert config.context_novelty_max_attempts == 0
    assert config.evaluation == {"enabled": False}


def test_run_config_parses_legacy_snake_case_storage_with_new_defaults():
    config = EvolutionRunConfig.from_stored_mapping(
        {
            "idea_type": "drabble",
            "pop_size": "3",
            "generations": "2",
            "model_type": "gemini-2.5-flash-lite",
            "creative_temp": "1.2",
            "top_p": "0.85",
            "tournament_size": 5,
            "tournament_comparisons": 35,
            "thinking_budget": "-1",
            "max_budget": None,
            "mutation_rate": "0.3",
            "evaluation_config": {"enabled": False},
        }
    )

    assert config.idea_type == "drabble"
    assert config.pop_size == 3
    assert config.generations == 2
    assert config.full_tournament_rounds == 2
    assert config.tournament_count == 0.5
    assert config.random_seed is None
    assert config.context_novelty_threshold == 0.8
    assert config.context_novelty_max_attempts == 2
    assert config.evaluation == {"enabled": False}
    assert "tournament_size" not in config.to_persistence_dict()
    assert "tournament_comparisons" not in config.to_persistence_dict()


def test_canonical_persistence_round_trip_and_engine_kwargs_mapping():
    evaluation = _EvaluationConfig(
        enabled=True,
        metadata={"source": "test", "weights": [0.8, 0.2]},
    )
    config = EvolutionRunConfig(
        idea_type="prime_visualization",
        pop_size=8,
        generations=4,
        random_seed=1234,
        tournament_rounds=3,
        full_tournament_rounds=7,
        context_novelty_threshold=0.72,
        context_novelty_max_attempts=5,
        evaluation=evaluation,
    )

    persisted = config.to_persistence_dict()
    json.dumps(persisted, allow_nan=False)
    assert set(persisted) == set(EvolutionRunConfig.model_fields)
    assert "evaluation_config" not in persisted
    assert persisted["evaluation"] == evaluation.model_dump(mode="json")
    assert EvolutionRunConfig.from_stored_mapping(persisted) == config

    kwargs = config.engine_kwargs()
    assert set(kwargs) == EvolutionRunConfig.engine_parameter_names()
    assert "evaluation" not in kwargs
    assert kwargs["evaluation_config"] == persisted["evaluation"]
    assert kwargs["random_seed"] == 1234
    assert kwargs["context_novelty_threshold"] == 0.72
    assert kwargs["context_novelty_max_attempts"] == 5


def test_engine_helpers_filter_context_and_snapshot_live_settings():
    from_kwargs = EvolutionRunConfig.from_engine_kwargs(
        {
            "idea_type": "drabble",
            "pop_size": 2,
            "generations": 1,
            "random_seed": 9,
            "evaluation_config": _EvaluationConfig(enabled=False),
            "api_key": "must-not-enter-config",
            "user_id": "user-1",
            "template_data": {"name": "snapshot"},
            "evaluation_runners": {"metric": object()},
        }
    )

    assert from_kwargs.idea_type == "drabble"
    assert from_kwargs.random_seed == 9
    assert from_kwargs.evaluation == {
        "schema_version": 1,
        "enabled": False,
        "metadata": {},
    }

    engine_values = from_kwargs.to_persistence_dict()
    evaluation = engine_values.pop("evaluation")
    engine = SimpleNamespace(**engine_values, evaluation_config=evaluation)

    assert EvolutionRunConfig.from_engine(engine) == from_kwargs


def test_engine_typed_snapshot_uses_the_canonical_run_config():
    engine = EvolutionEngine(
        pop_size=2,
        generations=1,
        random_seed=4242,
        context_novelty_threshold=0.61,
        context_novelty_max_attempts=4,
        evaluation_config={
            "enabled": False,
            "components": [],
            "fitness": {"enabled": False, "metrics": []},
        },
    )

    assert isinstance(engine.config_state, EvolutionRunConfig)
    assert engine.config_state == EvolutionRunConfig.from_engine(engine)
    assert engine.config_state.random_seed == 4242
    assert engine.config_state.context_novelty_threshold == 0.61
    assert engine.config_state.context_novelty_max_attempts == 4


def test_run_config_rejects_unknown_direct_fields_and_invalid_seed_range():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvolutionRunConfig(unknown_future_setting=True)

    with pytest.raises(ValidationError):
        EvolutionRunConfig(random_seed=-1)

    with pytest.raises(ValidationError):
        EvolutionRunConfig(random_seed=2**64)

    with pytest.raises(ValidationError, match="evaluation must be"):
        EvolutionRunConfig(evaluation=["not", "a", "mapping"])


def test_checkpoint_and_live_payload_share_canonical_config_fields():
    config = EvolutionRunConfig(
        idea_type="prime_visualization",
        pop_size=6,
        generations=4,
        random_seed=9_876,
        context_novelty_threshold=0.63,
        context_novelty_max_attempts=7,
        evaluation={"enabled": True, "components": []},
    )
    engine = _checkpoint_engine(config)

    checkpoint_config = EvolutionSerializer.to_checkpoint_state(engine)["config"]
    live_config = _engine_config_payload(engine)

    assert checkpoint_config == live_config == config.to_persistence_dict()
    assert checkpoint_config["random_seed"] == 9_876
    assert checkpoint_config["context_novelty_threshold"] == 0.63
    assert checkpoint_config["context_novelty_max_attempts"] == 7


def test_restore_round_trips_run_config_and_preserves_constructor_context():
    config = EvolutionRunConfig(
        idea_type="prime_visualization",
        pop_size=4,
        generations=2,
        random_seed=314_159,
        context_novelty_threshold=0.55,
        context_novelty_max_attempts=6,
        evaluation={"enabled": True, "components": [{"type": "script"}]},
    )
    template_data = {"id": "custom-primes", "prompt": "Render primes"}
    evaluation_runners = {"script": object()}
    state = {
        "config": config.to_persistence_dict(),
        "template_data": template_data,
        "user_id": "stored-user",
    }

    engine = EvolutionRepository.restore_from_state(
        _CapturingEngine,
        state,
        api_key="encoded-key",
        user_id="override-user",
        evaluation_runners=evaluation_runners,
    )

    constructor_kwargs = engine.constructor_kwargs
    expected_run_kwargs = config.engine_kwargs()
    assert {
        name: constructor_kwargs[name]
        for name in EvolutionRunConfig.engine_parameter_names()
    } == expected_run_kwargs
    assert constructor_kwargs["random_seed"] == 314_159
    assert constructor_kwargs["context_novelty_threshold"] == 0.55
    assert constructor_kwargs["context_novelty_max_attempts"] == 6
    assert constructor_kwargs["api_key"] == "encoded-key"
    assert constructor_kwargs["user_id"] == "override-user"
    assert constructor_kwargs["template_data"] is template_data
    assert constructor_kwargs["evaluation_runners"] is evaluation_runners
    assert engine.typed_state_synchronized is True


def test_restore_applies_stable_defaults_to_legacy_config():
    state = {
        "config": {
            "idea_type": "drabble",
            "pop_size": "3",
            "generations": "2",
            "tournament_size": 9,
        },
        "user_id": "stored-user",
    }

    engine = EvolutionRepository.restore_from_state(_CapturingEngine, state)
    constructor_kwargs = engine.constructor_kwargs

    assert constructor_kwargs["idea_type"] == "drabble"
    assert constructor_kwargs["pop_size"] == 3
    assert constructor_kwargs["generations"] == 2
    assert constructor_kwargs["random_seed"] is None
    assert constructor_kwargs["mutation_rate"] == 0.2
    assert constructor_kwargs["context_novelty_threshold"] == 0.8
    assert constructor_kwargs["context_novelty_max_attempts"] == 2
    assert constructor_kwargs["evaluation_config"] is None
    assert constructor_kwargs["user_id"] == "stored-user"
    assert "tournament_size" not in constructor_kwargs


def test_engine_resume_reinjects_artifact_store_without_serializing_it(tmp_path):
    runner = object()
    artifact_store = ArtifactStore(tmp_path / "artifact-store")
    evaluation = {
        "allow_generated_code_execution": True,
        "components": [
            {
                "id": "render",
                "kind": "llm_code_artifact",
                "execution_profile": "gemini-code-execution-v1",
                "provides": [
                    "artifact.primary",
                    "artifact.source",
                    "artifact.metadata",
                ],
            }
        ],
        "fitness": {"enabled": False, "metrics": []},
    }

    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        source = EvolutionEngine(
            pop_size=1,
            generations=1,
            user_id="user-1",
            evaluation_config=evaluation,
            evaluation_runners={"render": runner},
            artifact_store=artifact_store,
        )
    state = EvolutionSerializer.to_checkpoint_state(source)

    assert "artifact_store" not in state["config"]
    assert str(artifact_store.root) not in json.dumps(state)

    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        restored = EvolutionEngine._restore_from_state(
            state,
            user_id="user-1",
            evaluation_runners={"render": runner},
            artifact_store=artifact_store,
        )

    assert restored.artifact_store is artifact_store
    assert restored.unsupported_evaluation_components == []
    assert len(restored.artifact_producers) == 1
    assert restored.artifact_producers[0][1] is runner
