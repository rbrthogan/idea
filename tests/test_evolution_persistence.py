import uuid
from unittest.mock import patch

from idea.evolution import EvolutionEngine
from idea.evolution_persistence import EvolutionRepository, EvolutionSerializer
from idea.models import Idea


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
            "pop_size": 2,
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
        },
        "evolution_id": "evo-1",
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
        "avg_idea_cost": 0.1,
        "avg_tournament_cost": 0.2,
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
    assert engine.evolution_name == "Test Evolution"
    assert engine.current_generation == 1
    assert engine.contexts == ["ctx"]
    assert engine.specific_prompts == ["sp"]
    assert engine.breeding_prompts == [["bp"]]
    assert engine.seed_context_pool_size == 3
    assert engine.thinking_level == "low"
    assert isinstance(engine.population[0]["idea"], Idea)
    assert engine.population[0]["idea"].title == "A"
