import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.evolution import EvolutionEngine
from idea.config import model_prices_per_million_tokens
from idea.config import LLM_MODELS


class DummyAgent:
    def __init__(self, model_name, input_tokens, output_tokens):
        self.model_name = model_name
        self.input_token_count = input_tokens
        self.output_token_count = output_tokens
        self.total_token_count = input_tokens + output_tokens


def test_get_total_token_count():
    engine = EvolutionEngine(pop_size=1, generations=1)
    engine.ideator = DummyAgent("gemini-2.0-flash-lite", 100, 200)
    engine.formatter = DummyAgent("gemini-2.0-flash", 10, 20)
    engine.critic = DummyAgent("gemini-2.0-flash-lite", 30, 40)
    engine.breeder = DummyAgent("gemini-2.0-flash", 50, 60)

    result = engine.get_total_token_count()

    assert result["total_input"] == 100 + 10 + 30 + 50
    assert result["total_output"] == 200 + 20 + 40 + 60
    assert result["total"] == result["total_input"] + result["total_output"]

    def calc_cost(model, inp, out):
        pricing = model_prices_per_million_tokens.get(model)
        return (
            pricing["input"] * inp / 1_000_000
            + pricing["output"] * out / 1_000_000
        )

    expected_total_cost = sum(
        [
            calc_cost("gemini-2.0-flash-lite", 100, 200),
            calc_cost("gemini-2.0-flash", 10, 20),
            calc_cost("gemini-2.0-flash-lite", 30, 40),
            calc_cost("gemini-2.0-flash", 50, 60),
        ]
    )

    assert abs(result["cost"]["total_cost"] - expected_total_cost) < 1e-9

    # Ensure estimates exist for all configured models and are calculated using
    # the overall token totals
    total_inp = result["total_input"]
    total_out = result["total_output"]
    for model in LLM_MODELS:
        model_id = model["id"]
        assert model_id in result["estimates"]
        pricing = model_prices_per_million_tokens.get(model_id)
        expected = (
            pricing["input"] * total_inp / 1_000_000
            + pricing["output"] * total_out / 1_000_000
        )
        assert abs(result["estimates"][model_id]["cost"] - expected) < 1e-9

    # Diagnostics payload should include event buckets even when agents do not expose events.
    assert "diagnostics" in result
    assert "events" in result["diagnostics"]
    for agent in ["ideator", "formatter", "critic", "breeder", "oracle"]:
        assert agent in result["diagnostics"]["events"]


def test_get_total_token_count_includes_auxiliary_components():
    engine = EvolutionEngine(pop_size=1, generations=1)
    engine.ideator = DummyAgent("gemini-2.0-flash-lite", 0, 0)
    engine.formatter = DummyAgent("gemini-2.0-flash", 0, 0)
    engine.critic = DummyAgent("gemini-2.0-flash-lite", 0, 0)
    engine.breeder = DummyAgent("gemini-2.0-flash", 0, 0)
    engine.oracle = DummyAgent("gemini-2.0-flash", 0, 0)
    engine.genotype_encoder = DummyAgent("gemini-2.0-flash", 5, 7)
    engine.diversity_calculator = SimpleNamespace(
        get_token_usage_summary=lambda: {
            "total": 20,
            "input": 20,
            "output": 0,
            "model": "gemini-embedding-2",
            "cost": 0.000004,
            "input_cost": 0.000004,
            "output_cost": 0.0,
            "estimated_input": 20,
            "requests": 2,
        },
        get_diagnostics=lambda: {},
        get_diagnostic_events=lambda: [],
    )

    result = engine.get_total_token_count()

    assert result["genotype_encoder"]["total"] == 12
    assert result["diversity_embeddings"]["total"] == 20
    assert result["diversity_embeddings"]["estimated_input"] == 20
    assert result["total_input"] == 25
    assert result["total_output"] == 7
    assert result["total"] == 32
    assert result["models"]["genotype_encoder"] == "gemini-2.0-flash"
    assert result["models"]["diversity_embeddings"] == "gemini-embedding-2"


def test_island_sizes_split_total_population_evenly():
    engine = EvolutionEngine(pop_size=7, generations=1, island_count=3)

    assert engine._compute_island_sizes() == [3, 2, 2]
    assert engine._initial_island_ids() == [0, 0, 0, 1, 1, 2, 2]


def test_default_config_uses_stochastic_refine_and_two_islands():
    engine = EvolutionEngine(pop_size=5, generations=1)

    assert engine.refine_all_ideas is False
    assert engine.island_count == 2
    assert engine._compute_island_sizes() == [3, 2]
    assert engine._initial_island_ids() == [0, 0, 0, 1, 1]


def test_population_diversity_scores_use_global_history_centroid():
    engine = EvolutionEngine(pop_size=2, generations=1)
    engine.diversity_calculator = SimpleNamespace(is_enabled=lambda: True)
    engine.idea_embeddings = {
        "history": np.array([0.0, 0.0]),
        "current-a": np.array([10.0, 0.0]),
        "current-b": np.array([11.0, 0.0]),
    }
    engine.embedding_id_to_index = {
        "history": 0,
        "current-a": 1,
        "current-b": 2,
    }
    engine.all_embeddings = [
        engine.idea_embeddings["history"],
        engine.idea_embeddings["current-a"],
        engine.idea_embeddings["current-b"],
    ]

    scores = asyncio.run(
        engine._calculate_population_diversity_scores(
            [
                {"id": "current-a", "idea": "A"},
                {"id": "current-b", "idea": "B"},
            ]
        )
    )

    assert scores == {0: 3.0, 1: 4.0}


def test_diversity_analytics_tracks_island_and_inter_island_metrics():
    engine = EvolutionEngine(pop_size=4, generations=1, island_count=2)
    history = [
        [
            {"id": "g0-a", "idea": "A", "island_id": 0},
            {"id": "g0-b", "idea": "B", "island_id": 1},
        ],
        [
            {"id": "g1-a", "idea": "A1", "island_id": 0},
            {"id": "g1-b", "idea": "B1", "island_id": 0},
            {"id": "g1-c", "idea": "C1", "island_id": 1},
            {"id": "g1-d", "idea": "D1", "island_id": 1},
        ],
    ]
    embeddings = [
        np.array([0.0]),
        np.array([20.0]),
        np.array([0.0]),
        np.array([2.0]),
        np.array([10.0]),
        np.array([14.0]),
    ]

    analytics = engine._build_diversity_analytics(
        history,
        embeddings,
        {
            "enabled": True,
            "diversity_score": 7.0,
            "inter_generation_diversity": 5.0,
            "generation_diversities": [
                {"generation": 0, "diversity_score": 20.0},
                {"generation": 1, "diversity_score": 8.3333333333},
            ],
        },
    )

    assert analytics["current_generation"] == 1
    assert round(analytics["current_generation_diversity"], 6) == round(50 / 6, 6)
    assert analytics["cumulative_diversity"] == 7.0
    assert analytics["inter_generation_diversity"] == 5.0
    assert analytics["island_count"] == 2
    assert analytics["islands"][0]["island_id"] == 0
    assert analytics["islands"][0]["diversity_score"] == 2.0
    assert analytics["islands"][1]["diversity_score"] == 4.0
    assert analytics["mean_island_diversity"] == 3.0
    assert analytics["inter_island_diversity"] == 11.0
    assert round(analytics["island_separation_ratio"], 6) == round(11.0 / 3.0, 6)
