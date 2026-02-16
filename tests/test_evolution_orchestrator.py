import asyncio
import random
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from idea.evolution import EvolutionEngine
from idea.evolution_orchestrator import EvolutionOrchestrator
from idea.evolution_types import GenerationWorkState


class _FakeAgent:
    def __init__(self):
        self.model_name = "gemini-2.0-flash"
        self.total_token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0

    def refine(self, idea, _idea_type):
        return idea

    def format_idea(self, idea, _idea_type):
        return idea


class _FakeCritic(_FakeAgent):
    def get_tournament_ranks(self, population, _idea_type, _rounds, callback, rounds_details, _full):
        if callback:
            callback(1, 1)
        rounds_details.append({"round": 1, "pairs": []})
        return {idx: 1500 - idx for idx in range(len(population))}


class _FakeBreeder(_FakeAgent):
    parent_count = 2

    def breed(self, _parents, _idea_type):
        return {
            "id": uuid.uuid4(),
            "idea": "child",
            "parent_ids": [],
            "specific_prompt": "child prompt",
        }


class _FakeOracle:
    def __init__(self):
        self.total_token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0
        self.model_name = "gemini-2.0-flash"

    def analyze_and_diversify(self, _history, _idea_type):
        return {
            "action": "replace",
            "idea_prompt": "oracle prompt",
            "oracle_analysis": "oracle analysis",
        }


class _FakeEngine:
    def __init__(self):
        self.generations = 1
        self.pop_size = 2
        self.idea_type = "airesearch"
        self.tournament_rounds = 1
        self.full_tournament_rounds = 1
        self.tournament_count = 1.0
        self.max_budget = None
        self.stop_requested = False
        self.is_stopped = False
        self.current_generation = 0
        self.checkpoint_id = "ckpt"
        self.avg_idea_cost = 0.0
        self.avg_tournament_cost = 0.0

        self.population = [
            {"id": uuid.uuid4(), "idea": "A", "parent_ids": [], "birth_generation": 0},
            {"id": uuid.uuid4(), "idea": "B", "parent_ids": [], "birth_generation": 0},
        ]
        self.history = [self.population.copy()]
        self.contexts = ["ctx0", "ctx1"]
        self.specific_prompts = ["sp0", "sp1"]
        self.breeding_prompts = [["bp0", "bp1"]]
        self.tournament_history = []
        self.diversity_history = []

        self.ideator = SimpleNamespace(generate_text=lambda _p: "oracle idea")
        self.critic = _FakeCritic()
        self.formatter = _FakeAgent()
        self.breeder = _FakeBreeder()
        self.oracle = _FakeOracle()

        self._stop_on_breeding = False

    def _set_tournament_history(self, generation, rounds):
        self.tournament_history.append({"generation": generation, "rounds": rounds})

    def random_shuffle(self, values):
        random.shuffle(values)

    def random_choice(self, a, *, size=None, replace=True, p=None):
        if size is None:
            if isinstance(a, int):
                return random.randrange(a)
            return random.choice(list(a))

        options = list(a)
        if not options or size <= 0:
            return np.array([])
        if replace:
            return np.array([random.choice(options) for _ in range(size)])
        size = min(size, len(options))
        return np.array(random.sample(options, size))

    def _allocate_parent_slots(self, _ranks, _ideas_to_breed):
        return {}

    def _compute_replacement_count(self, population_size):
        if population_size <= 1:
            return 0
        return 1

    async def _score_population_fitness(self, population, ranks):
        return {
            idx: {
                "elo": float(ranks.get(idx, 1500)),
                "diversity": 0.1 * idx,
                "elo_norm": 0.5,
                "diversity_norm": 0.5,
                "fitness": 0.5 + (0.1 * (len(population) - idx)),
            }
            for idx in range(len(population))
        }

    def _score_survival_with_age_decay(self, _population, fitness_map, target_generation=None):
        result = {}
        for idx, row in fitness_map.items():
            result[idx] = {
                **row,
                "age": 0.0,
                "age_decay": 1.0,
                "survival_score": row["fitness"],
            }
        return result

    def _select_survivor_indices(self, survival_scores, survivor_count):
        ordered = sorted(
            survival_scores.keys(),
            key=lambda idx: survival_scores[idx]["survival_score"],
            reverse=True,
        )
        return ordered[:survivor_count]

    def _get_birth_generation(self, idea):
        return idea.get("birth_generation", 0) if isinstance(idea, dict) else 0

    def _select_parents_from_slots(self, _slots, available_indices):
        return available_indices[:2]

    async def _run_batch_with_progress(self, tasks, progress_callback, base_progress_info, start_step, total_steps, description_template):
        if self._stop_on_breeding:
            self.stop_requested = True
            return [None for _ in tasks]

        results = []
        for idx, task in enumerate(tasks, start=1):
            results.append(await task())
            await progress_callback(
                {
                    **base_progress_info,
                    "progress": (start_step + idx) / max(1, total_steps) * 100,
                    "status_message": description_template.format(completed=idx, total=len(tasks)),
                }
            )
        return results

    async def _calculate_and_store_diversity(self):
        self.diversity_history.append({"enabled": True, "diversity_score": 0.3})
        return self.diversity_history[-1]

    async def _find_least_interesting_idea_idx(self, _population):
        return 0

    async def _find_most_diverse_idea_idx(self, _population):
        return 1

    async def _remove_embedding(self, _idea_id):
        return None

    def _get_template_prompts(self):
        return SimpleNamespace(template=SimpleNamespace(special_requirements=""))

    def get_total_token_count(self):
        return {
            "total": 0,
            "total_input": 0,
            "total_output": 0,
            "cost": {
                "total_cost": 0.0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "currency": "USD",
            },
        }

    def check_budget(self):
        return False

    async def save_checkpoint(self, status="in_progress"):
        self.last_checkpoint_status = status
        return "ckpt"


def test_engine_run_and_resume_delegate_to_single_orchestrator():
    calls = []

    class _DummyOrchestrator:
        async def run(self, **kwargs):
            calls.append(kwargs)

    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        engine = EvolutionEngine(pop_size=1, generations=1)

    with patch.object(EvolutionEngine, "_new_orchestrator", return_value=_DummyOrchestrator()):
        asyncio.run(engine.run_evolution_with_updates(lambda _data: asyncio.sleep(0)))
        asyncio.run(engine.resume_evolution_with_updates(lambda _data: asyncio.sleep(0), additional_generations=2))

    assert calls[0]["mode"] == "new"
    assert calls[0]["start_generation"] == 0
    assert calls[1]["mode"] == "resume"
    assert calls[1]["additional_generations"] == 2


def test_oracle_replacement_updates_population_and_prompts():
    engine = _FakeEngine()
    updates = []

    async def callback(data):
        updates.append(data)

    orchestrator = EvolutionOrchestrator(engine, callback)
    asyncio.run(orchestrator._apply_oracle_if_enabled(gen=0))

    assert engine.population[0]["oracle_generated"] is True
    assert engine.population[0]["oracle_analysis"] == "oracle analysis"
    assert engine.breeding_prompts[-1][0] == "oracle prompt"
    assert any(update.get("oracle_update") for update in updates)


def test_elite_selection_marks_source_and_returns_prompt():
    engine = _FakeEngine()
    updates = []

    async def callback(data):
        updates.append(data)

    orchestrator = EvolutionOrchestrator(engine, callback)
    elite_idea, elite_prompt = asyncio.run(orchestrator._select_elite_for_next_generation(0))

    assert elite_idea is not None
    assert elite_prompt == "bp1"
    assert engine.population[1]["elite_selected_source"] is True
    assert engine.population[1]["elite_target_generation"] == 1
    assert any(update.get("elite_selection_update") for update in updates)


def test_stop_during_breeding_emits_stopped_update():
    engine = _FakeEngine()
    engine._stop_on_breeding = True
    updates = []

    async def callback(data):
        updates.append(data)

    orchestrator = EvolutionOrchestrator(engine, callback)
    state = GenerationWorkState(
        generation_index=0,
        start_generation=0,
        total_generations=1,
        total_steps=4,
        steps_per_generation=4,
    )

    asyncio.run(
        orchestrator._evolve_generations(
            start_generation=0,
            total_steps=4,
            step_offset=0,
            work_state=state,
            resuming=False,
        )
    )

    assert any(update.get("is_stopped") for update in updates)
    assert any("breeding" in update.get("stop_message", "") for update in updates)


def test_seed_novelty_guardrail_regenerates_similar_context():
    engine = _FakeEngine()
    engine.context_novelty_threshold = 0.5
    engine.context_novelty_max_attempts = 2
    orchestrator = EvolutionOrchestrator(engine, lambda _data: asyncio.sleep(0))

    async def generate_novel_seed():
        return (
            "x, y, z",
            {"id": uuid.uuid4(), "idea": "novel", "parent_ids": [], "birth_generation": 0},
            "prompt",
        )

    orchestrator._generate_single_seed = generate_novel_seed

    seed_result = (
        "a, b, c",
        {"id": uuid.uuid4(), "idea": "dup", "parent_ids": [], "birth_generation": 0},
        "prompt",
    )
    existing = [{"a", "b", "c"}]

    context_pool, _idea, _prompt = asyncio.run(
        orchestrator._ensure_seed_novelty(seed_result, existing)
    )

    assert context_pool == "x, y, z"
