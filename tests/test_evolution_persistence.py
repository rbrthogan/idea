import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from idea import database
from idea.evolution import EvolutionEngine
from idea.evolution_persistence import EvolutionRepository, EvolutionSerializer
from idea.models import Idea


class _NumericRunner:
    async def run(self, candidate, **_kwargs):
        from idea.evaluation import ScriptResultManifest

        return ScriptResultManifest(
            status="ok",
            metrics={"score": float(candidate["score"])},
        )


def test_serializer_round_trip_for_idea_payload():
    raw = {
        "id": uuid.uuid4(),
        "idea": Idea(title="T", content="C"),
        "parent_ids": [uuid.uuid4()],
        "oracle_generated": True,
        "oracle_analysis": "analysis",
    }

    serialized = EvolutionSerializer.serialize_idea(raw)
    assert isinstance(serialized["id"], str)
    assert isinstance(serialized["parent_ids"][0], str)
    assert isinstance(serialized["idea"], dict)

    restored = EvolutionSerializer.deserialize_idea(serialized)
    assert isinstance(restored["id"], uuid.UUID)
    assert isinstance(restored["parent_ids"][0], uuid.UUID)
    assert isinstance(restored["idea"], Idea)
    assert restored["idea"].title == "T"
    assert restored["oracle_generated"] is True


def test_repository_restore_from_state():
    state = {
        "config": {
            "idea_type": "airesearch",
            "pop_size": 6,
            "generations": 3,
            "model_type": "gemini-2.0-flash",
            "creative_temp": 1.1,
            "top_p": 0.9,
            "tournament_rounds": 2,
            "tournament_count": 0.5,
            "full_tournament_rounds": 4,
            "thinking_budget": None,
            "thinking_level": "low",
            "max_budget": None,
            "mutation_rate": 0.2,
            "seed_context_pool_size": 3,
            "refine_all_ideas": False,
            "survivor_refinement_rate": 0.42,
            "island_count": 3,
            "migration_rate": 0.05,
            "migration_novelty_protection_quantile": 0.75,
            "age_decay_floor": 0.22,
        },
        "evolution_id": "evo-1",
        "_save_revision": "revision-1",
        "name": "Test Evolution",
        "created_at": "2025-01-01T00:00:00",
        "updated_at": "2025-01-01T00:10:00",
        "checkpoint_id": "ckpt-1",
        "current_generation": 1,
        "population": [
            {
                "id": str(uuid.uuid4()),
                "idea": {"title": "A", "content": "B"},
                "parent_ids": [str(uuid.uuid4())],
            }
        ],
        "history": [
            [
                {
                    "id": str(uuid.uuid4()),
                    "idea": {"title": "H", "content": "I"},
                    "parent_ids": [],
                }
            ]
        ],
        "contexts": ["ctx"],
        "specific_prompts": ["sp"],
        "breeding_prompts": [["bp"]],
        "tournament_history": [{"generation": 1, "rounds": []}],
        "diversity_history": [{"enabled": True, "diversity_score": 0.5}],
        "island_diversity_history": [{"generation": 1, "islands": []}],
        "migration_history": [{"generation": 1, "from_island_id": 0, "to_island_id": 1}],
        "comparison_cache": {
            "model|template|pair": {
                "winner_key": "idea-a",
                "comparison_count": 1,
            }
        },
        "avg_idea_cost": 0.1,
        "avg_tournament_cost": 0.2,
        "token_counts": {
            "total": 10,
            "total_input": 8,
            "total_output": 2,
            "cost": {"total_cost": 0.01, "input_cost": 0.004, "output_cost": 0.006},
        },
        "template_data": {"name": "Custom"},
    }

    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        engine = EvolutionRepository.restore_from_state(
            EvolutionEngine,
            state,
            api_key=None,
            user_id="user-1",
        )

    assert engine.evolution_id == "evo-1"
    assert engine.persistence_revision == "revision-1"
    assert engine.evolution_name == "Test Evolution"
    assert engine.current_generation == 1
    assert engine.contexts == ["ctx"]
    assert engine.specific_prompts == ["sp"]
    assert engine.breeding_prompts == [["bp"]]
    assert engine.seed_context_pool_size == 3
    assert engine.thinking_level == "low"
    assert engine.refine_all_ideas is False
    assert engine.survivor_refinement_rate == 0.42
    assert engine.island_count == 3
    assert engine.migration_rate == 0.05
    assert engine.migration_novelty_protection_quantile == 0.75
    assert engine.age_decay_floor == 0.22
    assert engine.island_diversity_history == [{"generation": 1, "islands": []}]
    assert engine.migration_history == [{"generation": 1, "from_island_id": 0, "to_island_id": 1}]
    assert engine.comparison_cache == state["comparison_cache"]
    assert engine._persisted_token_counts == state["token_counts"]
    assert isinstance(engine.population[0]["idea"], Idea)
    assert engine.population[0]["idea"].title == "A"


@pytest.mark.parametrize("checkpoint_id", ["../secret", "/tmp/secret", r"..\\secret"])
def test_local_checkpoint_helpers_reject_path_components(checkpoint_id):
    with pytest.raises(ValueError, match="path-free"):
        EvolutionRepository.load_checkpoint(EvolutionEngine, checkpoint_id)
    with pytest.raises(ValueError, match="path-free"):
        EvolutionRepository.delete_checkpoint(checkpoint_id)


def test_serializer_persists_comparison_cache():
    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        engine = EvolutionEngine(
            idea_type="airesearch",
            pop_size=2,
            generations=1,
            api_key=None,
            user_id="user-1",
        )

    engine.comparison_cache = {
        "signature|pair": {
            "winner_key": "idea-a",
            "comparison_count": 1,
        }
    }

    state = EvolutionSerializer.to_checkpoint_state(engine)

    assert state["comparison_cache"] == engine.comparison_cache


def test_serializer_persists_age_decay_floor():
    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        engine = EvolutionEngine(
            idea_type="airesearch",
            pop_size=2,
            generations=1,
            age_decay_floor=0.18,
            api_key=None,
            user_id="user-1",
        )

    state = EvolutionSerializer.to_checkpoint_state(engine)

    assert state["config"]["age_decay_floor"] == 0.18


def test_serializer_persists_resolved_evaluation_suite():
    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        engine = EvolutionEngine(
            idea_type="airesearch",
            pop_size=2,
            generations=1,
            fitness_alpha=0.6,
            api_key=None,
            user_id="user-1",
        )

    state = EvolutionSerializer.to_checkpoint_state(engine)

    assert state["config"]["evaluation"]["schema_version"] == 1
    assert [
        metric["weight"]
        for metric in state["config"]["evaluation"]["fitness"]["metrics"]
    ] == [0.6, 0.4]


def test_computed_evaluator_checkpoint_restores_with_injected_runner_and_stats():
    evaluation = {
        "components": [
            {
                "id": "numeric",
                "kind": "computed",
                "runner_ref": "python-script-v1",
                "metric_names": ["score"],
            }
        ],
        "fitness": {
            "metrics": [
                {
                    "component_id": "numeric",
                    "metric_name": "score",
                    "normalization": "running_zscore",
                }
            ]
        },
    }
    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        engine = EvolutionEngine(
            idea_type="airesearch",
            pop_size=2,
            generations=1,
            user_id="user-1",
            evaluation_config=evaluation,
            evaluation_runners={"numeric": _NumericRunner()},
        )

    asyncio.run(
        engine._score_population_fitness(
            [
                {"id": "idea-a", "idea": "A", "score": 10},
                {"id": "idea-b", "idea": "B", "score": 20},
            ],
            {},
        )
    )
    state = EvolutionSerializer.to_checkpoint_state(engine)

    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        restored = EvolutionRepository.restore_from_state(
            EvolutionEngine,
            state,
            user_id="user-1",
            evaluation_runners={"numeric": _NumericRunner()},
        )

    assert restored.unsupported_evaluation_components == []
    assert restored.fitness_metric_stats["numeric.score"]["count"] == 2
    fitness = asyncio.run(
        restored._score_population_fitness(
            [
                {"id": "idea-c", "idea": "C", "score": 30},
                {"id": "idea-d", "idea": "D", "score": 40},
            ],
            {},
        )
    )
    assert fitness[0]["fitness"] < fitness[1]["fitness"]
    assert restored.fitness_metric_stats["numeric.score"]["count"] == 4


def test_list_evolutions_includes_saved_config_metadata():
    saved_config = {
        "idea_type": "game_design",
        "pop_size": 4,
        "generations": 2,
        "model_type": "gemini-3.1-pro-preview",
        "thinking_level": "high",
        "creative_temp": 1.4,
        "top_p": 0.9,
        "tournament_count": 1.5,
        "mutation_rate": 0.35,
        "seed_context_pool_size": 5,
        "replacement_rate": 0.6,
        "fitness_alpha": 0.8,
        "age_decay_rate": 0.3,
        "age_decay_floor": 0.2,
        "refine_all_ideas": False,
        "survivor_refinement_rate": 0.15,
        "island_count": 3,
        "migration_rate": 0.07,
        "migration_novelty_protection_quantile": 0.85,
    }

    async def fake_list_evolutions(_user_id):
        return [
            {
                "evolution_id": "evo-1",
                "name": "Games 3.1 Pro High",
                "status": "complete",
                "config": saved_config,
                "_history_summary": {"generation_count": 2, "total_ideas": 8},
                "_population_summary": {"population_size": 4},
            }
        ]

    with patch("idea.evolution_persistence.db.list_evolutions", side_effect=fake_list_evolutions):
        result = asyncio.run(EvolutionRepository.list_evolutions_for_user("user-1"))

    assert result[0]["config"] == saved_config


def test_repository_save_uses_loaded_revision_and_advances_it():
    engine = SimpleNamespace(
        user_id="user-1",
        evolution_id="evo-1",
        evolution_name="Name",
        created_at="created",
        updated_at="updated",
        persistence_revision="revision-1",
    )
    save = AsyncMock(return_value="revision-2")

    with patch.object(EvolutionSerializer, "to_checkpoint_state", return_value={}), patch(
        "idea.evolution_persistence.db.save_evolution", save
    ):
        result = asyncio.run(EvolutionRepository.save_checkpoint(engine))

    assert result == "evo-1"
    assert engine.persistence_revision == "revision-2"
    assert save.await_args.kwargs["expected_revision"] == "revision-1"


def test_repository_does_not_swallow_stale_save_conflict():
    engine = SimpleNamespace(
        user_id="user-1",
        evolution_id="evo-1",
        evolution_name="Name",
        created_at="created",
        updated_at="updated",
        persistence_revision="revision-1",
    )
    save = AsyncMock(side_effect=database.EvolutionSaveConflict("stale"))

    with patch.object(EvolutionSerializer, "to_checkpoint_state", return_value={}), patch(
        "idea.evolution_persistence.db.save_evolution", save
    ), pytest.raises(database.EvolutionSaveConflict):
        asyncio.run(EvolutionRepository.save_checkpoint(engine))

    assert engine.persistence_revision == "revision-1"


def test_draft_checkpoint_roundtrip_preserves_candidate_types_and_provenance():
    import json
    with patch('idea.llm.LLMWrapper._setup_provider', return_value=None):
        engine = EvolutionEngine(pop_size=2, generations=1, user_id='u')
    candidate = {'id': uuid.uuid4(), 'idea': Idea(title='Saved child', content='Finished prose'),
                 'parent_ids': [uuid.uuid4()], 'parent_genotypes': [{'genotype': 'space; western'}]}
    engine.generation_draft = {'version': 1, 'generation': 1, 'islands': {'0': {
        'island_id': None, 'capacity': 2, 'survivors': [candidate], 'fallback': [],
        'plans': [{'operator': 'crossover', 'parents': [candidate]}],
        'results': [{'candidate': candidate, 'prompt': 'saved prompt'}], 'tournament_rounds': []}}}
    state = json.loads(json.dumps(EvolutionSerializer.to_checkpoint_state(engine)))
    with patch('idea.llm.LLMWrapper._setup_provider', return_value=None):
        restored = EvolutionEngine(pop_size=2, generations=1, user_id='u')
    EvolutionSerializer.restore_runtime_state(restored, state)
    result = restored.generation_draft['islands']['0']['results'][0]
    assert result['candidate'] == candidate
    assert result['prompt'] == 'saved prompt'
    assert restored.population == [] and restored.current_generation == 0
    restored.generation_draft['islands']['0']['results'][0]['prompt'] = 'edited'
    assert state['generation_draft']['islands']['0']['results'][0]['prompt'] == 'saved prompt'
