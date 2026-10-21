import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.evolution import EvolutionEngine
from idea.config import model_prices_per_million_tokens
from idea.config import LLM_MODELS


class DummyAgent:
    def __init__(
        self,
        model_name,
        input_tokens,
        output_tokens,
        thought_tokens=0,
        cached_input_tokens=0,
    ):
        self.model_name = model_name
        self.input_token_count = input_tokens
        self.output_token_count = output_tokens
        self.thought_token_count = thought_tokens
        self.cached_input_token_count = cached_input_tokens
        self.uncached_input_token_count = max(input_tokens - cached_input_tokens, 0)
        self.total_token_count = input_tokens + output_tokens + thought_tokens


def test_openai_model_routes_every_generation_agent_and_diversity_to_openai():
    engine = EvolutionEngine(
        pop_size=1,
        generations=1,
        model_type="gpt-5.6-luna",
        api_key="test-key",
        thinking_level="max",
    )

    assert engine.provider == "openai"
    for agent in (
        engine.ideator,
        engine.formatter,
        engine.critic,
        engine.breeder,
        engine.gene_manager,
        engine.oracle,
    ):
        assert agent.provider == "openai"
        assert agent.model_name == "gpt-5.6-luna"
    assert engine.diversity_calculator.provider == "openai"
    assert engine.diversity_calculator.embedding_model == "text-embedding-3-small"


@pytest.mark.parametrize("model", LLM_MODELS, ids=lambda model: model["id"])
def test_gene_analysis_follows_selected_model_and_counts_its_usage(model):
    engine = EvolutionEngine(
        pop_size=1, generations=1, model_type=model["id"], api_key="test-key",
    )
    assert engine.gene_manager.model_name == engine.ideator.model_name == model["id"]
    assert engine.gene_manager.provider == engine.ideator.provider == model["provider"]
    assert engine.gene_manager.api_key == engine.ideator.api_key == "test-key"
    assert engine.gene_manager.thinking_level == engine.ideator.thinking_level
    assert engine.gene_manager.thinking_budget == engine.ideator.thinking_budget
    assert engine.breeder.gene_encoder is engine.gene_manager
    baseline = engine.get_total_token_count()
    engine.gene_manager = DummyAgent(model["id"], 1000, 100, thought_tokens=50)
    usage = engine.get_total_token_count()
    assert usage["total"] - baseline["total"] == 1150
    prices = model_prices_per_million_tokens[model["id"]]
    expected = (prices["input"] * 1000 + prices["output"] * 150) / 1_000_000
    assert usage["cost"]["total_cost"] - baseline["cost"]["total_cost"] == pytest.approx(expected)


@pytest.mark.parametrize("model,settings", [
    ("gemini-3.5-flash-lite", {"thinking_level": "minimal"}),
    ("gemini-3.8-flash", {"thinking_level": "low"}),
    ("gemini-4-argon", {"thinking_level": "high"}),
    ("gpt-6-luna", {"thinking_level": "none"}),
    ("gpt-6.1-sol", {"thinking_level": "low"}),
    ("gpt-6-astra", {"thinking_level": "high"}),
    ("gemini-2.5-flash", {"thinking_budget": 0}),
    ("gemini-2.5-flash", {"thinking_budget": 8192}),
])
def test_gene_analysis_respects_selected_thinking_settings(model, settings):
    engine = EvolutionEngine(pop_size=1, generations=1, model_type=model, **settings)
    assert engine.gene_manager.model_name == engine.ideator.model_name == model
    assert engine.gene_manager.thinking_level == engine.ideator.thinking_level == settings.get("thinking_level")
    assert engine.gene_manager.thinking_budget == engine.ideator.thinking_budget == settings.get("thinking_budget")


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


def test_restored_token_usage_remains_in_budget_totals():
    engine = EvolutionEngine(pop_size=1, generations=1, max_budget=0.01)
    for name in ("ideator", "formatter", "critic", "breeder", "oracle"):
        setattr(engine, name, DummyAgent("gemini-2.0-flash", 0, 0))
    engine._persisted_token_counts = {
        "total": 3_000,
        "total_input": 2_000,
        "total_output": 1_000,
        "total_thoughts": 0,
        "total_cached_input": 0,
        "total_uncached_input": 2_000,
        "total_tool_input": 0,
        "cost": {
            "input_cost": 0.005,
            "uncached_input_cost": 0.005,
            "cached_input_cost": 0.0,
            "output_cost": 0.015,
            "thought_cost": 0.0,
            "total_cost": 0.02,
        },
    }

    result = engine.get_total_token_count()

    assert result["restored_usage"]["total"] == 3_000
    assert result["total"] == 3_000
    assert result["cost"]["total_cost"] == pytest.approx(0.02)
    assert engine.check_budget() is True


def test_get_total_token_count_includes_thinking_cached_and_tiered_costs():
    engine = EvolutionEngine(pop_size=1, generations=1)
    engine.ideator = DummyAgent(
        "gemini-2.5-pro",
        input_tokens=210_000,
        output_tokens=1_000,
        thought_tokens=2_000,
        cached_input_tokens=50_000,
    )
    engine.formatter = DummyAgent("gemini-2.0-flash", 0, 0)
    engine.critic = DummyAgent("gemini-2.0-flash-lite", 0, 0)
    engine.breeder = DummyAgent("gemini-2.0-flash", 0, 0)
    engine.oracle = DummyAgent("gemini-2.0-flash", 0, 0)

    result = engine.get_total_token_count()

    assert result["ideator"]["thoughts"] == 2_000
    assert result["ideator"]["cached_input"] == 50_000
    assert result["ideator"]["uncached_input"] == 160_000
    assert result["total"] == 213_000
    assert result["total_input"] == 210_000
    assert result["total_output"] == 1_000
    assert result["total_thoughts"] == 2_000
    assert result["total_cached_input"] == 50_000

    expected_cost = (
        160_000 * 2.50 / 1_000_000
        + 50_000 * 0.25 / 1_000_000
        + 1_000 * 15.00 / 1_000_000
        + 2_000 * 15.00 / 1_000_000
    )
    assert result["cost"]["input_cost"] == expected_cost - (
        1_000 * 15.00 / 1_000_000
        + 2_000 * 15.00 / 1_000_000
    )
    assert result["cost"]["thought_cost"] == 2_000 * 15.00 / 1_000_000
    assert result["cost"]["total_cost"] == expected_cost
    # Counterfactual estimates follow the current picker; the actual legacy
    # agent still retains its own model's tiered cost above.
    assert result["estimates"]["gpt-6.1-sol"]["cost"] == pytest.approx(
        160_000 * 2.00 / 1_000_000
        + 50_000 * 0.10 / 1_000_000
        + 3_000 * 10.00 / 1_000_000
    )


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


def test_population_novelty_ignores_historical_cache_and_collinear_duplicates():
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

    assert scores == {0: 0.0, 1: 0.0}


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


def test_default_evaluation_suite_preserves_legacy_hybrid_fitness():
    engine = EvolutionEngine(
        pop_size=2,
        generations=1,
        fitness_alpha=0.75,
    )
    population = [
        {"id": "idea-a", "idea": "A"},
        {"id": "idea-b", "idea": "B"},
    ]

    async def diversity_scores(_population):
        return {0: 0.1, 1: 0.9}

    engine._calculate_population_diversity_scores = diversity_scores
    fitness = asyncio.run(
        engine._score_population_fitness(
            population,
            {0: 1400.0, 1: 1600.0},
        )
    )

    for row in fitness.values():
        assert row["fitness"] == (
            pytest.approx(0.75 * row["elo_norm"] + 0.25 * row["diversity_norm"])
        )
        assert "quality_elo.quality_elo" in row["components"]
        assert "diversity.diversity" in row["components"]

    assert population[0]["evaluations"]["quality_elo"]["metrics"] == {
        "quality_elo": 1400.0
    }
    assert population[0]["evaluations"]["diversity"]["metrics"] == {
        "diversity": 0.1
    }
    assert engine.last_evaluation_batch is not None


def test_disabled_evaluation_suite_uses_uniform_selection_fitness():
    engine = EvolutionEngine(
        pop_size=2,
        generations=1,
        evaluation_config={
            "enabled": False,
            "components": [],
            "fitness": {"enabled": False, "metrics": []},
        },
    )
    population = [
        {"id": "idea-a", "idea": "A"},
        {"id": "idea-b", "idea": "B"},
    ]

    fitness = asyncio.run(
        engine._score_population_fitness(population, {0: 1500.0, 1: 1500.0})
    )

    assert [fitness[index]["fitness"] for index in range(2)] == [0.5, 0.5]
    assert all("evaluations" not in candidate for candidate in population)


def test_computed_runner_can_drive_selection_without_llm_judging():
    class Runner:
        async def run(self, candidate, **_kwargs):
            from idea.evaluation import ScriptResultManifest

            return ScriptResultManifest(
                status="ok",
                metrics={"score": float(candidate["score"])},
            )

    engine = EvolutionEngine(
        pop_size=2,
        generations=1,
        evaluation_config={
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
                        "normalization": "bounds",
                        "bounds": [0, 100],
                    }
                ]
            },
        },
        evaluation_runners={"numeric": Runner()},
    )
    population = [
        {"id": "idea-a", "idea": "A", "score": 25},
        {"id": "idea-b", "idea": "B", "score": 90},
    ]

    fitness = asyncio.run(
        engine._score_population_fitness(population, {0: 1500.0, 1: 1500.0})
    )

    assert fitness[0]["fitness"] == pytest.approx(0.25)
    assert fitness[1]["fitness"] == pytest.approx(0.90)
    assert "numeric.score" in fitness[0]["components"]
    assert population[0]["evaluations"]["numeric"]["metrics"] == {"score": 25.0}


def test_explicit_bounds_normalization_is_not_overridden_by_legacy_running_stats():
    engine = EvolutionEngine(
        pop_size=2,
        generations=1,
        evaluation_config={
            "components": [
                {
                    "id": "quality",
                    "kind": "llm_pairwise",
                    "metric_name": "score",
                }
            ],
            "fitness": {
                "metrics": [
                    {
                        "component_id": "quality",
                        "metric_name": "score",
                        "normalization": "bounds",
                        "bounds": [0, 100],
                    }
                ]
            },
        },
    )
    population = [
        {"id": "idea-a", "idea": "A"},
        {"id": "idea-b", "idea": "B"},
    ]

    fitness = asyncio.run(
        engine._score_population_fitness(population, {0: 0.0, 1: 100.0})
    )

    assert fitness[0]["fitness"] == pytest.approx(0.0)
    assert fitness[1]["fitness"] == pytest.approx(1.0)
    assert fitness[0]["elo_norm"] == pytest.approx(0.0)
    assert fitness[1]["elo_norm"] == pytest.approx(1.0)
    assert population[0]["evaluations"]["quality"]["normalized_metrics"] == {}


def test_computed_metric_supports_persistable_running_normalization_state():
    class Runner:
        async def run(self, candidate, **_kwargs):
            from idea.evaluation import ScriptResultManifest

            return ScriptResultManifest(
                status="ok",
                metrics={"score": float(candidate["score"])},
            )

    engine = EvolutionEngine(
        pop_size=2,
        generations=1,
        evaluation_config={
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
        },
        evaluation_runners={"numeric": Runner()},
    )
    population = [
        {"id": "idea-a", "idea": "A", "score": 0},
        {"id": "idea-b", "idea": "B", "score": 100},
    ]

    fitness = asyncio.run(
        engine._score_population_fitness(population, {0: 1500.0, 1: 1500.0})
    )

    assert fitness[0]["fitness"] < 0.5
    assert fitness[1]["fitness"] > 0.5
    assert engine.fitness_metric_stats["numeric.score"]["count"] == 2
    assert population[0]["evaluations"]["numeric"]["normalized_metrics"]["score"] < 0.5


def test_computed_artifacts_enrich_candidates_before_pairwise_judging():
    from idea.evaluation import ScriptResultManifest
    from idea.evolution_orchestrator import EvolutionOrchestrator

    class Runner:
        async def run(self, _candidate, **_kwargs):
            return ScriptResultManifest(
                status="ok",
                metrics={"structure_signal": 0.6},
                artifacts=[
                    {
                        "path": "visualization.png",
                        "role": "primary",
                        "media_type": "image/png",
                    }
                ],
            )

    engine = EvolutionEngine(
        pop_size=2,
        generations=1,
        evaluation_config={
            "components": [
                {
                    "id": "render-and-score",
                    "kind": "computed",
                    "runner_ref": "isolated-prime-v1",
                    "metric_names": ["structure_signal"],
                },
                {
                    "id": "quality",
                    "kind": "llm_pairwise",
                },
            ],
            "fitness": {
                "metrics": [
                    {
                        "component_id": "quality",
                        "metric_name": "quality_elo",
                    }
                ]
            },
        },
        evaluation_runners={"render-and-score": Runner()},
    )
    population = [
        {"id": "idea-a", "idea": "A"},
        {"id": "idea-b", "idea": "B"},
    ]

    def tournament_ranks(ideas, _idea_type, **kwargs):
        assert all("render-and-score" in idea["artifact_outputs"] for idea in ideas)
        kwargs["details"].append({"round": 1, "pairs": []})
        return {index: 1500.0 for index, _idea in enumerate(ideas)}

    engine.critic.get_tournament_ranks = tournament_ranks

    async def scenario():
        orchestrator = EvolutionOrchestrator(engine, lambda _payload: None)
        await orchestrator._run_tournament_and_attach_tree_metrics(
            population,
            generation_index=0,
            target_generation=1,
            rounds=1,
            full_rounds=1,
        )

    asyncio.run(scenario())

    assert population[0]["evaluations"]["render-and-score"]["metrics"] == {
        "structure_signal": 0.6
    }
