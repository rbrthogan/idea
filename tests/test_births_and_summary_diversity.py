"""Search operators and summary diversity with explicit offline provider fakes."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from idea.births import allocate_birth_counts, distribute_births
from idea.diversity_summaries import combine_criterion_embeddings
from idea.evolution import EvolutionEngine
from idea.evolution_orchestrator import EvolutionOrchestrator
from idea.evolution_types import EvolutionRunConfig
from idea.evolution_types import GenerationWorkState
from idea.llm import Ideator
from idea.run_requests import parse_run_request
from tests.test_evolution_orchestrator import _FakeEngine


@pytest.mark.parametrize("total", range(21))
def test_birth_counts_conserve_generation_and_round_by_at_most_one(total):
    counts = allocate_birth_counts(total, .5, .25)
    assert sum(counts.values()) == total
    for kind, rate in (("independent", .5), ("single_parent", .25), ("crossover", .25)):
        assert abs(counts[kind]-total*rate) < 1


def test_birth_counts_are_global_and_small_pool_ties_rotate():
    plans = distribute_births([4, 4, 4], .5, .25)
    assert all(plan.count("independent") == 2 and
               plan.count("single_parent") == plan.count("crossover") == 1 for plan in plans)
    small = [distribute_births([3, 3], .5, .25, gen) for gen in range(3)]
    assert all(sum(plan.count("independent") for plan in plans) == 3 for plans in small)
    assert {sum(plan.count("crossover") for plan in plans) for plans in small} == {1, 2}
    assert distribute_births([0, 1, 2], 1, 0) == [[], ["independent"], ["independent", "independent"]]


def test_invalid_birth_mix_rejected_by_engine_and_request_config():
    with pytest.raises(ValueError, match="sum to at most 1"):
        EvolutionRunConfig(independent_birth_rate=.8, single_parent_birth_rate=.3)
    with pytest.raises(ValueError, match="sum to at most 1"):
        parse_run_request({"independentBirthRate": .8, "singleParentBirthRate": .3})


def test_all_birth_types_have_actual_parentage_and_generation():
    engine = _FakeEngine()
    engine.refine_all_ideas = False
    engine.breeder.breed = Mock(side_effect=lambda parents, _: {
        "id": f"child-{len(parents)}", "idea": "new content",
        "parent_ids": [str(parent["id"]) for parent in parents], "specific_prompt": "variant"})
    orchestrator = EvolutionOrchestrator(engine, lambda _: asyncio.sleep(0))
    orchestrator._generate_single_seed = AsyncMock(return_value=(
        "fresh context", {"id": "fresh", "idea": "fresh content", "parent_ids": []}, "fresh prompt"))

    async def run():
        tasks = orchestrator._build_birth_tasks(
            engine.population, {0: .8, 1: .6}, ["independent", "single_parent", "crossover"], 2, 0)
        return [await task() for task in tasks]

    children = [child for child, _ in asyncio.run(run())]
    assert [child["birth_type"] for child in children] == ["independent", "single_parent", "crossover"]
    assert [len(child["parent_ids"]) for child in children] == [0, 1, 2]
    assert all(child["birth_generation"] == 3 and child["island_id"] == 0 for child in children)
    assert engine.breeder.breed.call_count == 2
    assert len(set(children[2]["parent_ids"])) == 2


@pytest.mark.parametrize("size,expected", [(6, 1), (8, 2), (12, 3)])
def test_tournament_fraction_is_scaled_to_local_size(size, expected):
    engine = _FakeEngine()
    engine.tournament_count = .25
    engine.tournament_rounds = 99
    orchestrator = EvolutionOrchestrator(engine, lambda _: asyncio.sleep(0))
    assert orchestrator._local_tournament_rounds(size) == expected
    engine.tournament_count = 1
    assert orchestrator._local_tournament_rounds(size) == size-1


@pytest.mark.parametrize("rate", [0, .2, .5, 1])
def test_mutation_replaces_without_growing_and_full_rate_removes_parent_genes(rate):
    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        ideator = Ideator()
    ideator._random_sample = lambda values, k: list(values)[:k]
    ideator._random_shuffle = lambda _: None
    inherited = [f"parent-{idx}" for idx in range(10)]
    fresh = [f"fresh-{idx}" for idx in range(10)]
    result = ideator.generate_context_from_parents(
        [";".join(inherited)], rate, ",".join(fresh)).split(", ")
    assert len(result) == len(inherited)
    assert sum(value in fresh for value in result) == int(10*rate+.5)
    if rate == 1:
        assert set(result).isdisjoint(inherited)


def summary_engine(response='{"summaries": ["route with limited fuel", "rotate acoustic surfaces"]}'):
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.diversity_representation = "criterion_summaries"
    engine.novelty_neighbors = 3
    engine.population = []
    engine.stop_requested = False
    engine.check_budget = lambda: False
    engine.idea_embeddings = {}
    engine.all_embeddings = []
    engine.embedding_id_to_index = {}
    engine.formatter = SimpleNamespace(generate_text=Mock(return_value=response),
                                      provider="fake", model_name="offline", temperature=0)
    engine._get_template_prompts = lambda: SimpleNamespace(COMPARISON_CRITERIA=["decisions", "interaction"])
    engine.diversity_calculator = SimpleNamespace(
        _get_idea_text=lambda candidate: candidate["idea"],
        _get_embeddings_batch=AsyncMock(side_effect=lambda texts: [
            np.array([1., 0.]) if "fuel" in text else np.array([0., 1.]) for text in texts]),
        is_enabled=lambda: True,
    )
    return engine


def test_criterion_vector_distance_is_equal_mean_without_cross_criterion_cancellation():
    first = [np.array([2., 0.]), np.array([0., 7.])]
    second = [np.array([0., 3.]), np.array([0., 1.])]
    a, b = combine_criterion_embeddings(first), combine_criterion_embeddings(second)
    assert 1-np.dot(a, b) == pytest.approx(.5)
    assert np.linalg.norm(a) == pytest.approx(1)
    # Averaging raw vectors would mix unrelated dimensions/criterion magnitudes.
    assert combine_criterion_embeddings([first[0], None]) is None


def test_summary_extraction_deduplicates_and_saved_summaries_reuse_on_resume():
    engine = summary_engine()
    candidates = [{"id": "a", "idea": "same detailed rules"}, {"id": "b", "idea": "same detailed rules"}]
    async def run():
        vectors = await engine._get_or_compute_embeddings_for_ideas(candidates)
        await engine._get_or_compute_embeddings_for_ideas(candidates)
        return vectors
    vectors = asyncio.run(run())
    assert all(vector.shape == (4,) for vector in vectors)
    engine.formatter.generate_text.assert_called_once()
    engine.diversity_calculator._get_embeddings_batch.assert_awaited_once()
    texts = engine.diversity_calculator._get_embeddings_batch.await_args.args[0]
    assert "same detailed rules" not in texts
    assert candidates[0]["diversity_summary"]["criteria"] == ["decisions", "interaction"]
    restored = summary_engine()
    restored.formatter.generate_text.side_effect = AssertionError("Saved summary should be reused")
    asyncio.run(restored._get_or_compute_embeddings_for_ideas(copy.deepcopy(candidates)))
    restored.formatter.generate_text.assert_not_called()


def test_changed_content_or_criteria_invalidates_extraction_cache():
    engine = summary_engine()
    first = {"id": "a", "idea": "original"}
    asyncio.run(engine._get_or_compute_embeddings_for_ideas([first]))
    changed = copy.deepcopy(first)
    changed.update(id="changed", idea="different rules")
    asyncio.run(engine._get_or_compute_embeddings_for_ideas([changed]))
    engine._get_template_prompts = lambda: SimpleNamespace(COMPARISON_CRITERIA=["new criterion", "interaction"])
    criteria_changed = {**first, "id": "new-criteria"}
    asyncio.run(engine._get_or_compute_embeddings_for_ideas([criteria_changed]))
    assert engine.formatter.generate_text.call_count == 3


def test_malformed_summary_fails_without_full_text_fallback_or_repeated_charge():
    engine = summary_engine('{"summaries": ["too few"]}')
    candidate = {"id": "broken", "idea": "do not embed this whole pitch"}
    async def run():
        assert await engine._get_or_compute_embeddings_for_ideas([candidate]) == [None]
        assert await engine._get_or_compute_embeddings_for_ideas([candidate]) == [None]
    asyncio.run(run())
    engine.formatter.generate_text.assert_called_once()
    engine.diversity_calculator._get_embeddings_batch.assert_not_awaited()


def test_new_novelty_includes_other_islands_but_not_implicit_history():
    engine = summary_engine()
    engine.novelty_neighbors = 1
    vectors = {"a": np.array([1., 0.]), "b": np.array([0., 1.]), "clone": np.array([1., 0.])}
    engine._get_or_compute_embeddings_for_ideas = AsyncMock(side_effect=lambda candidates: [vectors[c["id"]] for c in candidates])
    engine.population = [{"id": idx} for idx in vectors]
    scores = asyncio.run(engine._calculate_population_diversity_scores(engine.population[:2]))
    assert scores[0] == 0
    assert scores[1] == 1


def test_legacy_resume_keeps_old_diversity_space_and_new_birth_defaults():
    config = EvolutionRunConfig.from_stored_mapping({"pop_size": 12})
    assert config.diversity_representation == "full_text"
    assert config.novelty_neighbors == 5
    assert config.independent_birth_rate == .5
    assert EvolutionRunConfig().diversity_representation == "criterion_summaries"


@pytest.mark.parametrize("islands,rounds", [(1, 6), (2, 3), (3, 2)])
def test_complete_generation_uses_new_defaults_in_every_execution_path(islands, rounds):
    import hashlib
    import uuid
    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        engine = EvolutionEngine(pop_size=24, generations=1, island_count=islands,
                                 tournament_count=.25, tournament_rounds=99,
                                 survivor_refinement_rate=0, random_seed=42)
    engine.oracle = None
    engine.migration_rate = 0
    engine.save_checkpoint = AsyncMock(return_value=None)
    engine.population = [{"id": str(uuid.uuid4()), "idea": f"initial rules {idx}",
                          "parent_ids": [], "birth_generation": 0,
                          "island_id": idx % islands} for idx in range(24)]
    engine.history = [engine.population.copy()]
    engine.breeding_prompts = [[None]*24]
    engine.ideator.generate_context = Mock(return_value="fresh ideas")
    engine.ideator.generate_idea_from_context = Mock(side_effect=lambda *_: (
        f"fresh rules {uuid.uuid4()}", "fresh prompt"))
    engine.breeder.breed = Mock(side_effect=lambda parents, _: {
        "id": str(uuid.uuid4()), "idea": f"child rules {uuid.uuid4()}",
        "parent_ids": [parent["id"] for parent in parents]})
    engine.formatter.format_idea = Mock(side_effect=lambda candidate, _: candidate)

    def summarize(prompt, **_kwargs):
        criteria = json.loads(prompt.split('Criteria: ', 1)[1].split('\nCandidate: ', 1)[0])
        text = prompt.split('\nCandidate: ', 1)[1]
        return json.dumps({"summaries": [f"criterion {idx} rules {text}" for idx in range(len(criteria))]})

    engine.formatter.generate_text = Mock(side_effect=summarize)
    engine.diversity_calculator.is_enabled = lambda: True
    engine.diversity_calculator._get_embeddings_batch = AsyncMock(side_effect=lambda texts: [
        np.array([byte+1. for byte in hashlib.sha256(text.encode()).digest()[:4]]) for text in texts])
    engine.critic.get_tournament_ranks = Mock(side_effect=lambda candidates, *_args, **_kwargs: {
        idx: 1500+idx for idx in range(len(candidates))})

    async def run():
        orchestrator = EvolutionOrchestrator(engine, lambda _: asyncio.sleep(0))
        await orchestrator._evolve_generations(
            start_generation=0, total_steps=100, step_offset=0,
            work_state=GenerationWorkState(generation_index=0, start_generation=0,
                total_generations=1, total_steps=100, steps_per_generation=100), resuming=False)

    asyncio.run(run())
    children = [candidate for candidate in engine.population if candidate["birth_generation"] == 1]
    assert len(engine.population) == 24
    assert len(children) == 12
    assert [child["birth_type"] for child in children].count("independent") == 6
    assert [child["birth_type"] for child in children].count("single_parent") == 3
    assert [child["birth_type"] for child in children].count("crossover") == 3
    assert all(len(child["parent_ids"]) == {"independent": 0, "single_parent": 1, "crossover": 2}[child["birth_type"]]
               for child in children)
    assert all(call.kwargs["rounds"] == rounds for call in engine.critic.get_tournament_ranks.call_args_list)
    assert engine.current_generation == 1
    assert engine.diversity_history[-1]["metric_version"] == 3
    assert all(candidate.get("diversity_summary") for candidate in engine.population)
