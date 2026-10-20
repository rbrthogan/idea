"""Gene observation is durable evidence, never an evolution selection signal."""
import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from idea.evolution_orchestrator import EvolutionOrchestrator
from idea.evolution_persistence import EvolutionSerializer
from idea.genes import empty_gene_catalogue
from tests.test_evolution_orchestrator import _FakeEngine
from tests.test_genes import breeder_with_output, matching


def engine_for(population, breeder):
    return SimpleNamespace(
        breeder=breeder, critic=SimpleNamespace(refine=lambda candidate, _: {
            **candidate, "idea": {"title": "Refined", "content": "Changed rules"}}),
        formatter=SimpleNamespace(format_idea=lambda candidate, _: candidate),
        stop_requested=False, check_budget=lambda: False, idea_type="game_design",
        population=population, history=[population.copy()], current_generation=0,
        generations=1, diversity_history=[], island_diversity_history=[],
        migration_history=[], get_total_token_count=lambda: {"total": 0},
    )


def test_observation_uses_final_prose_and_invalidates_previous_revision():
    source = {"id": "original", "idea": {"title": "Old", "content": "Old rules"},
              "gene_observation": {"status": "ok", "genotype": "old gene"},
              "evaluations": {"judge": {"status": "ok"}}, "elo": 1700}
    seen = []

    def observe(candidate, _):
        assert "gene_observation" not in candidate
        seen.append(candidate["idea"]["content"])
        candidate["gene_observation"] = {"status": "ok", "genotype": "new gene"}

    async def progress(_):
        pass

    engine = engine_for([source], SimpleNamespace(observe_genes=observe))
    orchestrator = EvolutionOrchestrator(engine, progress)
    result = asyncio.run(orchestrator._prepare_idea_for_population(source, refine=True))
    assert seen == []  # Preparation cannot delay committing a completed birth.
    assert "gene_observation" not in result
    engine.population = [result]
    asyncio.run(orchestrator._observe_candidate_genes(result))
    assert seen == ["Changed rules"]
    assert result["gene_observation"]["genotype"] == "new gene"
    assert source["gene_observation"]["genotype"] == "old gene"
    assert "elo" not in result
    assert result["parent_ids"] == ["original"]


def test_failed_observation_does_not_discard_or_rescore_candidate():
    def fail(*_):
        raise RuntimeError("provider problem")

    async def progress(_):
        pass

    candidate = {"id": "a", "idea": "usable idea", "elo": 1550, "fitness": .7}
    engine = engine_for([candidate], SimpleNamespace(observe_genes=fail))
    orchestrator = EvolutionOrchestrator(engine, progress)
    result = asyncio.run(orchestrator._prepare_idea_for_population(candidate, refine=False))
    engine.population = [result]
    asyncio.run(orchestrator._observe_candidate_genes(result))
    assert result["idea"] == "usable idea"
    assert result["gene_observation"]["status"] == "failed"
    assert result["elo"] == 1550 and result["fitness"] == .7
    assert "provider problem" not in result["gene_observation"]["error"]


def test_population_mapping_is_serialized_after_observation_and_emits_history():
    events = []
    candidates = [{"id": str(i), "idea": f"Idea {i}", "fitness": .4} for i in range(2)]

    def observe(candidate, _):
        events.append(candidate["id"])
        candidate["gene_observation"] = {"status": "ok", "genes": [{"text": "a"}]}

    def reconcile(population, catalogue, breeder, idea_type, generation):
        assert events == ["0", "1"]
        assert population is candidates
        return {**catalogue, "revision": 1}

    messages = []

    async def progress(payload):
        messages.append(payload)

    engine = engine_for(candidates, SimpleNamespace(observe_genes=observe))
    orchestrator = EvolutionOrchestrator(engine, progress)
    with patch("idea.evolution_orchestrator.reconcile_gene_catalogue", side_effect=reconcile):
        asyncio.run(orchestrator._observe_population_genes(0))
    assert [c["fitness"] for c in candidates] == [.4, .4]
    assert messages[-1]["gene_catalogue"]["revision"] == 1
    assert messages[-1]["history"] == engine.history


def test_budget_stops_observation_before_a_provider_call():
    seen = []
    candidate = {"id": "a", "idea": "Ready"}
    engine = engine_for([candidate], SimpleNamespace(observe_genes=lambda *_: seen.append(True)))
    engine.check_budget = lambda: True

    async def progress(_):
        pass

    asyncio.run(EvolutionOrchestrator(engine, progress)._observe_population_genes(0))
    assert engine.stop_requested
    assert not seen
    assert "gene_observation" not in candidate


def test_catalogue_and_candidate_observation_survive_checkpoint_round_trip():
    observation = {"version": 1, "status": "ok", "fingerprint": "content-config",
                   "genotype": "delayed consequences", "genes": [{"text": "delayed consequences", "gene_id": "g1"}]}
    catalogue = {**empty_gene_catalogue(), "revision": 2,
                 "genes": [{"id": "g1", "label": "delayed consequences", "aliases": []}]}
    candidate = {"id": "candidate-a", "idea": {"title": "A", "content": "Rules"}, "gene_observation": observation}
    candidate["gene_projection"] = {
        "version": 1, "status": "ok", "revision": 1, "raw_fingerprint": "content-config",
        "genotype": "delayed consequences", "genes": [{"text": "delayed consequences", "gene_id": "g1",
            "source_terms": ["delayed consequences"], "decisions": []}],
    }
    catalogue["consolidation"] = {"revision": 1, "status": "ok", "rules": {}}
    catalogue["consolidation_revisions"] = [copy.deepcopy(catalogue["consolidation"])]
    restored = SimpleNamespace()
    EvolutionSerializer.restore_runtime_state(restored, {
        "population": [candidate], "history": [[candidate]], "gene_catalogue": catalogue,
    })
    restored.stop_requested = False
    restored.template_data = None
    restored.get_total_token_count = lambda: {"total": 0}
    state = EvolutionSerializer.to_checkpoint_state(restored)
    assert state["gene_catalogue"] == catalogue
    assert state["population"][0]["gene_observation"] == observation
    assert state["population"][0]["gene_projection"] == candidate["gene_projection"]
    assert state["history"][0][0]["gene_projection"] == candidate["gene_projection"]
    legacy = SimpleNamespace()
    EvolutionSerializer.restore_runtime_state(legacy, {"population": [candidate]})
    assert legacy.gene_catalogue == empty_gene_catalogue()


def test_resume_repairs_restored_history_without_reextracting_or_reseeding():
    breeder = breeder_with_output("hidden bids; shared clock")
    candidates = [
        {"id": "survivor", "idea": {"title": "A", "content": "First design"}, "elo": 1600},
        {"id": "child", "idea": {"title": "B", "content": "Second design"}, "elo": 1500},
    ]
    for candidate in candidates:
        breeder.observe_genes(candidate, "game_design")
        candidate["gene_observation"]["mapping_status"] = "failed"
    saved_genotypes = [candidate["gene_observation"]["genotype"] for candidate in candidates]
    historical = copy.deepcopy(candidates)
    historical[0]["elo"] = 1550  # Historical ratings must not follow current ratings.
    engine = _FakeEngine()
    EvolutionSerializer.restore_runtime_state(engine, {
        "current_generation": 1, "population": candidates,
        "history": [[copy.deepcopy(candidates[0])], historical],
        "gene_catalogue": empty_gene_catalogue(),
    })
    engine.generations = 3
    engine.idea_type = "game_design"
    engine.breeder = breeder
    assert engine.history[-1][1] is not engine.population[1]
    breeder.generate_text.reset_mock()
    breeder.generate_text.return_value = matching((None, [0]), (None, [1]))
    messages = []

    async def progress(payload):
        messages.append(copy.deepcopy(payload))

    orchestrator = EvolutionOrchestrator(engine, progress)
    orchestrator._complete_initial_seeding = AsyncMock(side_effect=AssertionError("Must not reseed"))
    orchestrator._seed_initial_population = AsyncMock(side_effect=AssertionError("Must not reseed"))
    resumed = []

    async def remaining_generations(**kwargs):
        resumed.append(kwargs)
        assert all(candidate["gene_observation"]["mapping_status"] == "ok"
                   for candidate in engine.population)
        # The repaired child then leaves the active population. Its historical
        # copy is now the only surviving evidence and must remain fully mapped.
        engine.population = [engine.population[0]]

    orchestrator._evolve_generations = remaining_generations
    asyncio.run(orchestrator._run_resume(start_generation=engine.current_generation))
    assert len(resumed) == 1
    assert resumed[0]["start_generation"] == 1 and resumed[0]["resuming"]
    assert engine.generations == 3 and len(engine.history) == 2
    orchestrator._complete_initial_seeding.assert_not_called()
    orchestrator._seed_initial_population.assert_not_called()
    breeder.generate_text.assert_called_once()
    assert breeder.generate_text.call_args.args[0].startswith("Annotate freely extracted design genes")
    assert [candidate["gene_observation"]["genotype"] for candidate in engine.history[-1]] == saved_genotypes
    assert all(candidate["gene_observation"]["mapping_status"] == "ok"
               for population in engine.history for candidate in population)
    assert engine.history[-1][0]["elo"] == 1550
    assert engine.population[0]["elo"] == 1600
    assert messages[-1]["is_resuming"]


def test_history_gene_repair_respects_identity_phenotype_and_extraction_settings():
    breeder = breeder_with_output("hidden bids; shared clock")
    current = {"id": "same", "idea": {"title": "Title", "content": "Design"}}
    breeder.observe_genes(current, "game_design")
    current["gene_observation"]["mapping_status"] = "ok"
    different_id = copy.deepcopy(current)
    different_id["id"] = "another"
    changed_prose = copy.deepcopy(current)
    changed_prose["idea"]["content"] = "Revised design"
    changed_settings = copy.deepcopy(current)
    changed_settings["gene_observation"]["fingerprint"] = "other-extraction-settings"
    historical = [different_id, changed_prose, changed_settings]
    for candidate in historical:
        candidate["gene_observation"]["mapping_status"] = "failed"
    before = copy.deepcopy(historical)
    missing = {"id": "same", "idea": copy.deepcopy(current["idea"]), "elo": 1700}
    engine = engine_for([current], breeder)
    engine.history = [historical, [missing]]
    orchestrator = EvolutionOrchestrator(engine, AsyncMock())
    orchestrator._sync_gene_observation_to_history(current)
    assert historical == before
    assert missing["gene_observation"] == current["gene_observation"]
    assert missing["gene_observation"] is not current["gene_observation"]
    assert missing["elo"] == 1700

    successful = copy.deepcopy(missing["gene_observation"])
    current["gene_observation"]["mapping_status"] = "failed"
    orchestrator._sync_gene_observation_to_history(current)
    assert missing["gene_observation"] == successful
    current["gene_observation"]["status"] = "failed"
    orchestrator._sync_gene_observation_to_history(current)
    assert missing["gene_observation"] == successful


@pytest.mark.parametrize("island_count", [1, 2], ids=["normal", "islands"])
def test_stop_during_first_post_birth_observation_saves_committed_generation(island_count):
    engine = _FakeEngine()
    engine.oracle = None
    engine.island_count = island_count
    engine.migration_rate = 0.0
    engine.pop_size = 2 * island_count
    engine.population = [
        {"id": f"seed-{index}", "idea": f"Seed {index}", "parent_ids": [],
         "birth_generation": 0, "island_id": index // 2}
        for index in range(engine.pop_size)
    ]
    engine.history = [engine.population.copy()]
    engine.contexts = ["context"] * engine.pop_size
    engine.specific_prompts = ["seed direction"] * engine.pop_size
    engine.breeding_prompts = [[None] * engine.pop_size]
    engine.random_shuffle = lambda _values: None
    engine.reset_stop_state = lambda: None
    initial_ids = {candidate["id"] for candidate in engine.population}
    checkpoints = []
    updates = []
    interrupted_candidates = []
    children_at_interruption = []

    def observe(candidate, _idea_type):
        if engine.current_generation == 1:
            # Simulate a user pause while the first post-birth extraction is in
            # flight. All completed births must already belong to saved state.
            interrupted_candidates.append(candidate["id"])
            children_at_interruption.extend(
                idea["id"] for idea in engine.population if idea["id"] not in initial_ids
            )
            assert engine.history[-1] is engine.population
            assert engine.diversity_history  # Pausing annotation cannot skip completed analytics.
            engine.stop_requested = True
            raise RuntimeError("Annotation interrupted after prose was committed")
        candidate["gene_observation"] = {
            "version": 1, "status": "ok", "fingerprint": f"seed-{candidate['id']}",
            "genotype": "shared seed mechanism",
            "genes": [{"text": "shared seed mechanism", "gene_id": None}],
        }

    async def save_checkpoint(status="in_progress"):
        checkpoints.append(copy.deepcopy({
            "status": status, "current_generation": engine.current_generation,
            "population": engine.population, "history": engine.history,
            "gene_catalogue": engine.gene_catalogue,
        }))
        return "paused-checkpoint"

    async def progress(payload):
        updates.append(copy.deepcopy(payload))

    engine.breeder.observe_genes = observe
    engine.save_checkpoint = save_checkpoint
    orchestrator = EvolutionOrchestrator(engine, progress)
    asyncio.run(orchestrator.run(start_generation=0, mode="resume"))

    assert len(interrupted_candidates) == 1
    assert len(children_at_interruption) == island_count
    assert engine.current_generation == 1
    assert engine.stop_requested and engine.is_stopped
    assert len(engine.history) == 2
    assert len(engine.population) == engine.pop_size
    assert set(children_at_interruption).issubset({idea["id"] for idea in engine.population})
    assert checkpoints and checkpoints[-1]["status"] == "paused"
    assert all(checkpoint["status"] != "complete" for checkpoint in checkpoints)
    paused = checkpoints[-1]
    assert paused["current_generation"] == 1
    assert len(paused["history"]) == 2
    assert paused["population"] == paused["history"][-1]
    assert set(children_at_interruption).issubset({idea["id"] for idea in paused["population"]})
    assert {idea["id"] for idea in paused["history"][0]} == initial_ids
    assert paused["gene_catalogue"]["genes"]  # Previously mapped evidence survives the pause.
    terminal = updates[-1]
    assert terminal["is_stopped"] and terminal["is_resumable"]
    assert terminal["durably_saved"] and terminal["current_generation"] == 1
    assert terminal["history"] == paused["history"]
    assert terminal["gene_catalogue"] == paused["gene_catalogue"]
