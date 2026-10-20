"""Synthetic evidence and provider contracts; never make paid requests."""

import asyncio
import copy
import threading
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

import numpy as np
import pytest

from idea.diversity import DiversityCalculator
from idea.evolution import EvolutionEngine
from idea.evolution_orchestrator import EvolutionOrchestrator
from idea.llm import Breeder, Critic, GenerationCancelledError
from idea.provider_execution import await_provider_call
from tests.test_evolution_orchestrator import _FakeEngine


def critic():
    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        return Critic()


def test_cached_observation_applies_once_across_generations_and_resume():
    judge = critic()
    ideas = [{"id": "a", "idea": "alpha"}, {"id": "b", "idea": "beta"}]
    cache = {}
    with patch.object(judge, "compare_ideas", return_value="A") as compare:
        first = judge.get_tournament_ranks(ideas, "airesearch", comparison_cache=cache)
        for _ in range(4):
            assert judge.get_tournament_ranks(ideas, "airesearch", comparison_cache=cache) == first
        assert compare.call_count == 1
    restored = critic()
    with patch.object(restored, "compare_ideas", side_effect=AssertionError("paid replay")):
        assert restored.get_tournament_ranks(copy.deepcopy(ideas), "airesearch",
                                              comparison_cache=copy.deepcopy(cache)) == first
    assert all(entry["applied"] and entry["evidence_version"] == 2 for entry in cache.values())
    assert ideas[0]["rating_evaluations"]["comparison_count"] == 1


def test_flat_candidates_reuse_judgments_after_evaluation_reset_and_restart():
    import json
    from idea.evaluation.components import LLMPairwiseEvaluator
    from idea.evaluation.models import LLMPairwiseComponentConfig

    judge = critic()
    evaluator = LLMPairwiseEvaluator(LLMPairwiseComponentConfig(id="quality_elo"))
    ideas = [{"id": "a", "title": "A", "content": "alpha"},
             {"id": "b", "title": "B", "content": "beta"}]
    cache = {}

    def evaluate(current_judge, candidates, generation):
        details = []
        return evaluator.evaluate_population(
            current_judge, candidates, "airesearch", rounds=1,
            progress_callback=None, details=details, full_tournament_rounds=1,
            should_stop=lambda: False, comparison_cache=cache,
            evaluation_generation=generation,
        )

    with patch.object(judge, "compare_ideas", return_value="A") as compare:
        first = evaluate(judge, ideas, 0)
        assert first.scores == {0: 1516, 1: 1484}
        assert all(idea["evaluations"] for idea in ideas)
        assert evaluate(judge, ideas, 1).scores == first.scores
        for idea in ideas:
            idea.update(ratings={"auto": 1500, "human": 5}, elo=1500,
                        evaluations={}, selection_metrics={"fitness": 0.99},
                        birth_generation=9, survived_to_generation=12,
                        rating_evaluations={}, extension={"display_nonce": "changed"})
        reset = evaluate(judge, ideas, 12)
        assert reset.scores == first.scores
        assert reset.details[0]["pairs"][0]["applied_this_run"] is False
        assert compare.call_count == 1

    cache = json.loads(json.dumps(cache))
    restored_judge = critic()
    restarted_ideas = copy.deepcopy(ideas)
    with patch.object(restored_judge, "compare_ideas", side_effect=AssertionError("paid replay")):
        assert evaluate(restored_judge, restarted_ideas, 13).scores == first.scores
        nested = [{"id": idea["id"], "idea": {"title": idea["title"], "content": idea["content"]}}
                  for idea in restarted_ideas]
        assert evaluate(restored_judge, nested, 14).scores == first.scores
    changed = copy.deepcopy(restarted_ideas)
    changed[0]["content"] = "a different mechanism"
    with patch.object(restored_judge, "compare_ideas", return_value="A") as compare:
        fresh = evaluate(restored_judge, changed, 15)
        assert fresh.details[0]["pairs"][0]["applied_this_run"] is True
        compare.assert_called_once()
    assert len(cache) == 2


def test_flat_cache_identity_tracks_judged_text_and_required_artifacts_only():
    from hashlib import sha256
    import json

    judge = critic()
    flat = {"id": "a", "title": "A", "content": "alpha", "ratings": {"auto": 1700}}
    nested = {"id": "a", "idea": {"title": "A", "content": "alpha"}, "ratings": {"auto": 1700}}
    legacy_payload = {"idea": nested["idea"], "artifacts": []}
    legacy_digest = sha256(json.dumps(legacy_payload, sort_keys=True, default=str).encode()).hexdigest()
    assert judge._idea_cache_key(nested) == f"a:{legacy_digest}"
    assert judge._idea_cache_key(flat) == judge._idea_cache_key(nested)
    changed_title = {**flat, "title": "A revised title"}
    assert judge._idea_cache_key(changed_title) != judge._idea_cache_key(flat)
    artifact = {"id": "render", "role": "visualization", "media_type": "image/png", "sha256": "a" * 64}
    with_artifact = {**flat, "artifact_refs": [artifact]}
    assert judge._idea_cache_key(with_artifact) == judge._idea_cache_key(flat)
    judge._required_artifact_roles = ("visualization",)
    changed_artifact = {**flat, "artifact_refs": [{**artifact, "sha256": "b" * 64}]}
    assert judge._idea_cache_key(with_artifact) != judge._idea_cache_key(flat)
    assert judge._idea_cache_key(changed_artifact) != judge._idea_cache_key(with_artifact)


@pytest.mark.parametrize("legacy", [False, True])
def test_round_commit_metadata_separates_applied_evidence_from_cached_outcomes(legacy):
    judge = critic()
    ideas = [{"id": "a", "idea": "alpha"}, {"id": "b", "idea": "beta"}]
    cache, first_details, progress = {}, [], []

    def record_progress(*_):
        progress.append(copy.deepcopy(first_details))

    with patch.object(judge, "compare_ideas", return_value="A"):
        expected = judge.get_tournament_ranks(ideas, "airesearch", comparison_cache=cache,
                                              details=first_details, progress_callback=record_progress)
    assert progress[0][0]["ratings_committed"] is False
    assert progress[0][0]["pairs"][0]["applied_this_run"] is False
    assert progress[-1][0]["ratings_committed"] is True
    pair = first_details[0]["pairs"][0]
    assert pair["applied_this_run"] is True
    assert (pair["new_elo_a"], pair["new_elo_b"]) == (1516, 1484)

    if legacy:
        for entry in cache.values():
            for field in ("applied", "evidence_version", "application_sequence", "post_ratings"):
                entry.pop(field, None)
    reused_details = []
    with patch.object(judge, "compare_ideas", side_effect=AssertionError("paid replay")):
        assert judge.get_tournament_ranks(ideas, "airesearch", comparison_cache=cache,
                                          details=reused_details) == expected
    assert reused_details[0]["ratings_committed"] is True
    reused = reused_details[0]["pairs"][0]
    assert reused["cached"] is True
    assert reused["applied_this_run"] is False
    assert "new_elo_a" not in reused and "new_elo_b" not in reused


def test_identical_content_commits_round_without_claiming_new_evidence():
    judge = critic()
    ideas = [{"id": "a", "idea": "alpha"}, {"id": "alias", "idea": "alpha"}]
    details = []
    with patch.object(judge, "compare_ideas", side_effect=AssertionError("identical content")):
        assert judge.get_tournament_ranks(ideas, "airesearch", details=details) == {0: 1500, 1: 1500}
    assert details[0]["ratings_committed"] is True
    assert details[0]["pairs"][0]["applied_this_run"] is False


def test_cancelled_round_retains_unapplied_evidence_for_resume():
    judge = critic()
    ideas = [{"id": "a", "idea": "alpha"}, {"id": "b", "idea": "beta"}]
    cache, interrupted_details = {}, []
    stopped = False

    def compare(*_):
        nonlocal stopped
        stopped = True
        return "A"

    with patch.object(judge, "compare_ideas", side_effect=compare):
        ranks = judge.get_tournament_ranks(ideas, "airesearch", comparison_cache=cache,
                                          should_stop=lambda: stopped, details=interrupted_details)
    assert ranks == {0: 1500, 1: 1500}
    assert not next(iter(cache.values()))["applied"]
    assert interrupted_details[0]["ratings_committed"] is False
    assert interrupted_details[0]["pairs"][0]["applied_this_run"] is False
    resumed_details = []
    with patch.object(judge, "compare_ideas", side_effect=AssertionError("repeat call")):
        ranks = judge.get_tournament_ranks(ideas, "airesearch", comparison_cache=cache,
                                          details=resumed_details)
    assert ranks == {0: 1516, 1: 1484}
    assert resumed_details[0]["ratings_committed"] is True
    assert resumed_details[0]["pairs"][0]["applied_this_run"] is True
    assert resumed_details[0]["pairs"][0]["cached"] is True


def test_content_aliases_reuse_one_observation_and_current_global_rating():
    judge = critic()
    originals = [{"id": "a", "idea": "alpha"}, {"id": "b", "idea": "beta"}]
    cache = {}
    with patch.object(judge, "compare_ideas", return_value="A") as compare:
        expected = judge.get_tournament_ranks(originals, "airesearch", comparison_cache=cache)
        aliases = [{"id": "migrant-a", "idea": "alpha"}, {"id": "migrant-b", "idea": "beta"}]
        assert judge.get_tournament_ranks(aliases, "airesearch", comparison_cache=cache) == expected
        assert compare.call_count == 1


def test_legacy_instance_keyed_evidence_migrates_without_replaying():
    from hashlib import sha256
    judge = critic()
    originals = [{"id": "a", "idea": "alpha"}, {"id": "b", "idea": "beta"}]
    cache = {}
    with patch.object(judge, "compare_ideas", return_value="A"):
        expected = judge.get_tournament_ranks(originals, "airesearch", comparison_cache=cache)
    entry = next(iter(cache.values()))
    for field in ("applied", "evidence_version", "application_sequence", "post_ratings"):
        entry.pop(field, None)
    first, second = sorted((entry["idea_a_key"], entry["idea_b_key"]))
    digest = sha256(f"{first}\n{second}".encode()).hexdigest()
    legacy = {f"{entry['signature']}:{digest}": entry}
    with patch.object(judge, "compare_ideas", side_effect=AssertionError("legacy replay")):
        assert judge.get_tournament_ranks(originals, "airesearch", comparison_cache=legacy) == expected


def test_explicit_judge_version_invalidates_comparison_namespace():
    judge = critic()
    legacy = judge._tournament_comparison_signature("airesearch")
    judge.judge_component_id = "quality_elo"
    judge.judge_component_version = "1"
    assert judge._tournament_comparison_signature("airesearch") == legacy
    judge.judge_component_version = "2"
    assert judge._tournament_comparison_signature("airesearch") != legacy


def test_two_content_aliases_accumulate_distinct_opponents_in_one_shared_rating():
    judge = critic()
    ideas = [{"id": "a", "idea": "alpha"}, {"id": "clone-a", "idea": "alpha"},
             {"id": "b", "idea": "beta"}, {"id": "c", "idea": "gamma"}]
    with patch.object(judge, "_generate_swiss_round_pairs", return_value=([(0, 2), (1, 3)], None)), \
         patch.object(judge, "compare_ideas", return_value="A"):
        ranks = judge.get_tournament_ranks(ideas, "airesearch", comparison_cache={})
    expected, _ = judge._elo_update(1516, 1500, "A")
    assert ranks[0] == ranks[1] == expected


def test_cancellation_forbids_provider_calls_and_new_retry_attempts():
    judge = critic()
    judge.cancel_event = threading.Event()
    judge.cancel_event.set()
    with patch.object(judge, "_generate_content") as provider:
        with pytest.raises(GenerationCancelledError):
            judge.generate_text("prompt")
        provider.assert_not_called()
    judge.cancel_event.clear()

    def transient_failure(*_):
        judge.cancel_event.set()
        raise TimeoutError("transient request")

    with patch.object(judge, "_generate_content", side_effect=transient_failure) as provider, \
         patch.object(judge.generate_text.retry, "wait", return_value=0):
        with pytest.raises(GenerationCancelledError):
            judge.generate_text("prompt")
        assert provider.call_count == 1


def test_changed_refinement_has_fresh_identity_and_no_parent_evidence():
    engine = _FakeEngine()
    engine.refine_all_ideas = False
    engine.survivor_refinement_rate = 1.0
    engine.critic.refine = lambda *_: {"idea": "a materially different proposal"}
    parent = {"id": "parent", "idea": "old", "ratings": {"auto": 1850, "human": 8},
              "elo": 1850, "artifact_refs": [{"id": "old-artifact"}],
              "evaluations": {"quality_elo": {"candidate_id": "parent"}}}
    original = copy.deepcopy(parent)
    orchestrator = EvolutionOrchestrator(engine, lambda _: asyncio.sleep(0))
    child = asyncio.run(orchestrator._maybe_refine_survivor(parent, target_generation=3))
    assert child["id"] != parent["id"]
    assert child["parent_ids"] == ["parent"]
    assert child["birth_generation"] == 3
    assert not child.get("evaluations") and not child.get("artifact_refs")
    assert child["ratings"] == {}
    assert Critic._initial_tournament_rating(None, child) == 1500
    assert parent == original


def test_small_populations_have_no_singleton_islands():
    assert EvolutionEngine._coerce_island_count(2, pop_size=2) == 1
    assert EvolutionEngine._coerce_island_count(2, pop_size=3) == 1
    assert EvolutionEngine._coerce_island_count(3, pop_size=6) == 3


@pytest.mark.parametrize("stopped,budget", [(False, 0), (True, None)])
def test_zero_budget_or_preexisting_stop_performs_no_provider_work(stopped, budget):
    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        engine = EvolutionEngine(pop_size=2, generations=1, max_budget=budget)
    engine.stop_requested = stopped
    updates = []

    async def progress(payload):
        updates.append(payload)

    orchestrator = EvolutionOrchestrator(engine, progress)
    with patch.object(engine.ideator, "generate_context") as context, \
         patch.object(engine, "_prepare_population_for_judging", new=AsyncMock()) as artifacts, \
         patch.object(engine.oracle, "analyze_and_diversify") as oracle, \
         patch.object(engine.diversity_calculator, "_get_embeddings_batch", new=AsyncMock()) as embeddings:
        asyncio.run(orchestrator.run(start_generation=0, mode="new"))
        context.assert_not_called()
        artifacts.assert_not_called()
        oracle.assert_not_called()
        embeddings.assert_not_called()
    assert engine.stop_requested and engine.is_stopped
    assert updates[-1]["is_stopped"] and not updates[-1]["is_running"]


def test_cancelled_provider_waiter_drains_thread_before_checkpoint_accounting():
    started = threading.Event()
    release = threading.Event()
    accounting = []

    def provider():
        started.set()
        assert release.wait(timeout=2)
        accounting.append("paid usage recorded")

    async def run():
        task = asyncio.create_task(await_provider_call(provider))
        await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not accounting
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert accounting == ["paid usage recorded"]

    asyncio.run(run())


@pytest.mark.parametrize("population_size", [2, 3, 5, 6, 8, 10])
@pytest.mark.parametrize("children", [1, 2, 3, 5, 10])
def test_parent_quotas_fill_children_without_exceeding_contributions(population_size, children):
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.breeder = SimpleNamespace(parent_count=2)
    engine.random_choice = np.random.default_rng(73).choice
    quotas = engine._allocate_parent_slots({idx: population_size-idx for idx in range(population_size)}, children)
    allocated = dict(quotas)
    used = {idx: 0 for idx in allocated}
    for _ in range(children):
        parents = engine._select_parents_from_slots(quotas, list(range(population_size)))
        assert len(parents) == len(set(parents)) == 2
        for parent in parents:
            used[parent] += 1
    assert used == allocated
    assert sum(quotas.values()) == 0


def test_six_candidate_island_does_not_skip_third_ranked_parent():
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.breeder = SimpleNamespace(parent_count=2)
    quotas = engine._allocate_parent_slots({idx: 6-idx for idx in range(6)}, 3)
    # With six contributions available, every candidate can breed once.
    # Rounding a top-three cap down to zero used to exclude rank three while
    # admitting all lower-ranked candidates and giving rank one two slots.
    assert all(quotas.get(idx, 0) > 0 for idx in range(3))
    assert quotas[2] >= max(quotas.get(idx, 0) for idx in range(3, 6))
    assert sum(quotas.values()) == 6


def test_novelty_scope_is_explicit_and_duplicate_peers_are_not_novel():
    engine = EvolutionEngine.__new__(EvolutionEngine)
    vectors = {"a": np.array([1., 0.]), "b": np.array([1., 0.]),
               "c": np.array([0., 1.]), "old": np.array([-1., 0.])}
    engine.diversity_calculator = SimpleNamespace(is_enabled=lambda: True)
    engine.idea_embeddings = vectors
    engine.all_embeddings = list(vectors.values())
    engine.embedding_id_to_index = {key: idx for idx, key in enumerate(vectors)}
    current = [{"id": key} for key in ("a", "b", "c")]
    scores = asyncio.run(engine._calculate_population_diversity_scores(current))
    assert scores == {0: .5, 1: .5, 2: 1.}
    assert asyncio.run(engine._calculate_population_diversity_scores(current)) == scores
    archived = asyncio.run(engine._calculate_population_diversity_scores(current,
                                                    novelty_archive=[{"id": "old"}]))
    assert archived != scores


def test_migration_is_simultaneous_ring_and_preserves_content_age():
    engine = _FakeEngine()
    engine.pop_size = 6
    engine.island_count = 3
    engine.migration_rate = 1.0
    engine.population = [{"id": f"{island}-{idx}", "idea": f"{island}-{idx}",
                          "island_id": island, "birth_generation": 1}
                         for island in range(3) for idx in range(2)]
    engine.history = [copy.deepcopy(engine.population)]
    engine.breeding_prompts = [[None]*6]
    original_ids = {idea["id"] for idea in engine.population}
    orchestrator = EvolutionOrchestrator(engine, lambda _: asyncio.sleep(0))
    asyncio.run(orchestrator._apply_island_migration(gen=8))
    assert {(event["from_island_id"], event["to_island_id"])
            for event in engine.migration_history} == {(0, 1), (1, 2), (2, 0)}
    migrants = [idea for idea in engine.population if idea.get("migration_event")]
    assert len(migrants) == 3
    assert all(idea["migrated_from_id"] in original_ids for idea in migrants)
    assert all(idea["birth_generation"] == 1 for idea in migrants)


def test_quality_archive_copies_verified_evidence_and_is_bounded():
    engine = _FakeEngine()
    engine.quality_evaluator = SimpleNamespace(config=SimpleNamespace(id="quality"))
    orchestrator = EvolutionOrchestrator(engine, lambda _: asyncio.sleep(0))
    population = [{"id": str(idx), "idea": "content", "evaluations": {
        "quality": {"status": "ok"}}} for idx in range(60)]
    orchestrator._archive_evaluated_candidates(population, {idx: {"elo": idx} for idx in range(60)})
    assert len(engine.quality_archive) == 50
    assert engine.quality_archive[0]["id"] == "59"
    population[-1]["evaluations"]["quality"]["status"] = "failed"
    assert engine.quality_archive[0]["evaluations"]["quality"]["status"] == "ok"


def test_oracle_enters_directly_without_an_incumbent_duel():
    engine = _FakeEngine()
    engine.quality_evaluator = SimpleNamespace(config=SimpleNamespace(id="quality"))
    orchestrator = EvolutionOrchestrator(engine, lambda _: asyncio.sleep(0))
    original = engine.population[0]["id"]

    def incumbent_wins(population, **kwargs):
        raise AssertionError("Oracle admission must not invoke the quality judge")

    orchestrator._get_tournament_ranks = incumbent_wins
    asyncio.run(orchestrator._apply_oracle_if_enabled(gen=0))
    assert engine.population[0]["id"] != original
    assert engine.population[0]["oracle_generated"]
    assert engine.population[0]["birth_type"] == "oracle"
    assert len(engine.population) == 2
    assert engine.breeding_prompts[-1][0] == "oracle prompt"


def test_genotype_cache_is_content_and_prompt_versioned():
    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        breeder = Breeder()
    prompts = SimpleNamespace(GENOTYPE_ENCODE_PROMPT="Encode {idea_content}")
    with patch.object(breeder, "_resolve_prompts", return_value=prompts), \
         patch.object(breeder, "generate_text", return_value="Genotype: mechanisms, constraints") as call:
        assert breeder.encode_to_genotype({"idea": "parent"}, "custom") == "mechanisms, constraints"
        breeder.encode_to_genotype({"idea": "parent"}, "custom")
        breeder.encode_to_genotype({"idea": "changed"}, "custom")
        prompts.GENOTYPE_ENCODE_PROMPT = "New {idea_content}"
        breeder.encode_to_genotype({"idea": "parent"}, "custom")
    assert call.call_count == 3


def test_batched_embeddings_yield_event_loop_and_track_one_request():
    started = threading.Event()
    release = threading.Event()
    captured = []

    class Models:
        def embed_content(self, **kwargs):
            captured.append([content.parts[0].text for content in kwargs["contents"]])
            started.set()
            assert release.wait(timeout=2)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[1., idx+1.])
                 for idx, _ in enumerate(kwargs["contents"])],
                 usage_metadata=SimpleNamespace(prompt_token_count=12))

    calculator = DiversityCalculator()
    calculator.client = SimpleNamespace(models=Models())

    async def run():
        task = asyncio.create_task(calculator._get_embeddings_batch(["a", "b", "a"]))
        await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1)
        # Reaching here while the provider is blocked proves loop responsiveness.
        release.set()
        return await task

    embeddings = asyncio.run(run())
    assert captured == [["a", "b"]]
    assert np.array_equal(embeddings[0], embeddings[2])
    assert calculator.get_token_usage_summary()["requests"] == 1
    assert calculator.get_token_usage_summary()["input"] == 12


def test_embedding_model_is_pinned_and_rejects_dimension_changes():
    calls = []
    class Models:
        def embed_content(self, **kwargs):
            calls.append(kwargs["model"])
            return SimpleNamespace(embedding=SimpleNamespace(values=[1., 2.] if len(calls) == 1 else [1.]))
    calculator = DiversityCalculator()
    calculator.client = SimpleNamespace(models=Models())
    assert asyncio.run(calculator._get_embedding("first")) is not None
    assert asyncio.run(calculator._get_embedding("second")) is None
    assert calls == [calculator.embedding_model]*2


def test_embedding_text_excludes_candidate_identity_and_rating_metadata():
    calculator = DiversityCalculator()
    assert calculator._get_idea_text({"id": "secret-id", "idea": "actual idea", "elo": 1800}) == "actual idea"
    assert calculator._get_idea_text({"id": "other", "idea": {"title": "Title", "content": "Body"},
                                      "fitness": .9}) == "Title\n\nBody"


def test_blocked_pairwise_distances_match_naive_calculation():
    vectors = np.random.default_rng(13).normal(size=(270, 3))
    calculator = DiversityCalculator()
    naive = np.linalg.norm(vectors[:, None, :]-vectors[None, :, :], axis=2)
    expected = naive[np.triu_indices(len(vectors), k=1)].mean()
    assert calculator._calculate_mean_square_distance(list(vectors)) == pytest.approx(expected)


def test_large_history_analytics_bound_calls_vectors_and_preserve_scalar_measurements():
    from hashlib import sha256
    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        engine = EvolutionEngine(pop_size=20, generations=500, island_count=2,
                                 diversity_representation="full_text")
    requests = []

    class Models:
        def embed_content(self, **kwargs):
            texts = kwargs["contents"]
            if isinstance(texts, str):
                texts = [texts]
            texts = [text if isinstance(text, str) else text.parts[0].text for text in texts]
            requests.append(list(texts))
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[
                byte/255 for byte in sha256(text.encode()).digest()[:3]]) for text in texts])

    engine.diversity_calculator.client = SimpleNamespace(models=Models())
    engine.history = [[{"id": f"{generation}-{idx}", "idea": f"idea {generation}-{idx}",
                        "island_id": idx//10} for idx in range(20)] for generation in range(500)]
    engine.population = engine.history[-1].copy()
    engine.diversity_history = [{"generation_diversities": [
        {"generation": 0, "diversity_score": .432, "valid_embeddings": 20, "idea_count": 20}]}]
    sample, count = engine._bounded_history_sample()
    engine.history[0].reverse()
    reordered_sample, _ = engine._bounded_history_sample()
    assert [candidate["id"] for candidate in sample] == [candidate["id"] for candidate in reordered_sample]
    assert count == 10000 and len(sample) == 256
    data = asyncio.run(engine._calculate_and_store_diversity())
    assert "error" not in data
    assert sum(map(len, requests)) <= 276
    assert len(requests) <= 9
    assert len(engine.idea_embeddings) <= 276
    assert len(engine.diversity_calculator._embedding_cache) <= 276
    assert data["history_sampling"] == {
        "method": "bottom_sha256_unique_candidate_v1", "sample_size": 256,
        "total_size": 10000, "total_history_slots": 10000, "truncated": True}
    assert data["generation_diversities"][0]["diversity_score"] == .432
    assert data["generation_diversities"][-1]["generation"] == 499
    assert len(engine.history) == 500 and sum(map(len, engine.history)) == 10000
    first_request_count = len(requests)
    asyncio.run(engine._calculate_and_store_diversity())
    assert len(requests) == first_request_count
