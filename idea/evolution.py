from typing import List, Dict, Any, Callable, Awaitable, Optional
import random
import secrets
import threading
import numpy as np
import asyncio
import uuid
import math
from pathlib import Path
from datetime import datetime
from idea.models import Idea
from idea.config import (
    DEFAULT_CREATIVE_TEMP,
    DEFAULT_TOP_P,
    PROGRESS_JITTER_MAX_SECONDS,
    estimate_generation_token_cost,
)
from idea.llm import Ideator, Formatter, Critic, Breeder, Oracle
from idea.prompts.loader import list_available_templates, get_prompts, get_prompts_from_dict
from idea.diversity import DiversityCalculator
from idea.evaluation import (
    ComponentKind,
    ComputedEvaluator,
    ArtifactStoreLoader,
    EmbeddingDiversityEvaluator,
    EvaluationBatch,
    EvaluationRecord,
    EvaluationStatus,
    FitnessAggregator,
    LLMCodeArtifactProducer,
    LLMPairwiseEvaluator,
    MetricNormalization,
    attach_evaluation_record,
    candidate_identifier,
    create_builtin_computed_runner,
    resolve_evaluation_suite,
)
from idea import database as db  # noqa: F401 - compatibility seam for tests and external patches

# Legacy filesystem storage directories (kept for backwards compatibility with local dev)
# These are not used on Cloud Run - all data goes to Firestore
EVOLUTIONS_DIR = Path("data/evolutions")
CHECKPOINT_DIR = Path("data/checkpoints")


def get_default_template_id():
    """Get the first available template ID as default"""
    try:
        templates = list_available_templates()
        # Find the first template without errors
        for template_id, template_info in templates.items():
            if 'error' not in template_info:
                return template_id
        # Fallback to airesearch if nothing else works
        return 'airesearch'
    except Exception:
        return 'airesearch'


class EvolutionEngine:
    def __init__(
        self,
        idea_type=None,
        pop_size: int = 5,
        generations: int = 3,
        model_type: str = "gemini-2.0-flash",
        creative_temp: float = DEFAULT_CREATIVE_TEMP,
        top_p: float = DEFAULT_TOP_P,
        random_seed: Optional[int] = None,
        tournament_rounds: int = 1,
        tournament_count: Optional[float] = None,
        full_tournament_rounds: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        thinking_level: Optional[str] = None,
        max_budget: Optional[float] = None,
        mutation_rate: float = 0.2,
        seed_context_pool_size: Optional[int] = None,
        refine_all_ideas: bool = False,
        survivor_refinement_rate: float = 0.1,
        island_count: int = 2,
        migration_rate: float = 0.05,
        migration_novelty_protection_quantile: float = 0.8,
        replacement_rate: float = 0.5,
        fitness_alpha: float = 0.7,
        age_decay_rate: float = 0.25,
        age_decay_floor: float = 0.35,
        context_novelty_threshold: float = 0.8,
        context_novelty_max_attempts: int = 2,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,  # Required for Firestore storage
        template_data: Optional[Dict[str, Any]] = None,  # Custom template data from Firestore
        evaluation_config: Optional[Dict[str, Any]] = None,
        evaluation_runners: Optional[Dict[str, Any]] = None,
        artifact_store: Optional[Any] = None,
    ):
        self.user_id = user_id  # User ID for scoped storage
        self.idea_type = idea_type or get_default_template_id()
        self.template_data = template_data  # Store custom template data for LLM agents
        self._template_prompt_wrapper = None
        self.pop_size = pop_size
        self.generations = generations
        self.island_count = self._coerce_island_count(island_count, pop_size=pop_size)
        self.migration_rate = self._coerce_unit_float(migration_rate, default=0.05)
        self.migration_novelty_protection_quantile = self._coerce_unit_float(
            migration_novelty_protection_quantile,
            default=0.8,
        )
        if full_tournament_rounds is None or full_tournament_rounds <= 0:
            full_tournament_rounds = max(1, pop_size - 1)
        if tournament_rounds is None or tournament_rounds <= 0:
            tournament_rounds = 1
        self.full_tournament_rounds = int(full_tournament_rounds)
        self.tournament_rounds = int(tournament_rounds)
        if tournament_count is None:
            self.tournament_count = self.tournament_rounds / self.full_tournament_rounds
        else:
            self.tournament_count = float(tournament_count)
        self.thinking_budget = thinking_budget
        self.thinking_level = (
            str(thinking_level).strip().lower() if thinking_level is not None else None
        )
        self.max_budget = max_budget
        self.mutation_rate = mutation_rate
        try:
            self.seed_context_pool_size = (
                max(1, int(seed_context_pool_size))
                if seed_context_pool_size is not None
                else None
            )
        except (TypeError, ValueError):
            self.seed_context_pool_size = None
        self.replacement_rate = max(0.0, min(1.0, float(replacement_rate)))
        self.fitness_alpha = max(0.0, min(1.0, float(fitness_alpha)))
        self.age_decay_rate = max(0.0, float(age_decay_rate))
        self.age_decay_floor = max(0.0, min(1.0, float(age_decay_floor)))
        self.context_novelty_threshold = max(0.0, min(1.0, float(context_novelty_threshold)))
        self.context_novelty_max_attempts = max(0, int(context_novelty_max_attempts))
        self.api_key = api_key
        self.refine_all_ideas = self._coerce_bool(refine_all_ideas, default=False)
        try:
            self.survivor_refinement_rate = max(0.0, min(1.0, float(survivor_refinement_rate)))
        except (TypeError, ValueError):
            self.survivor_refinement_rate = 0.1
        self.random_seed = int(random_seed) if random_seed is not None else int(secrets.randbits(64))
        self._py_rng = random.Random(self.random_seed)
        self._np_rng = np.random.default_rng(self.random_seed)
        self._py_rng_lock = threading.Lock()
        self._np_rng_lock = threading.Lock()
        self.population: List[Idea] = []

        # Initialize LLM components with appropriate temperatures
        print(
            "Initializing agents with "
            f"creative_temp={creative_temp}, top_p={top_p}, "
            f"thinking_budget={thinking_budget}, thinking_level={self.thinking_level}"
        )

        self.ideator = Ideator(
            provider="google_generative_ai",
            model_name=model_type,
            temperature=creative_temp,
            top_p=top_p,
            thinking_budget=thinking_budget,
            thinking_level=self.thinking_level,
            api_key=api_key,
            random_seed=self._random_randbits(64),
            seed_context_pool_size=self.seed_context_pool_size,
        )

        # Always use 2.5 Flash for formatting as it has better instruction following for structured output
        # than 2.0 Flash or older models
        self.formatter = Formatter(
            provider="google_generative_ai",
            model_name="gemini-2.5-flash",
            api_key=api_key,
            random_seed=self._random_randbits(64),
        )

        critic_model_name = "gemini-2.5-flash" if model_type == "gemini-2.5-pro" else model_type
        self.critic = Critic(
            provider="google_generative_ai",
            model_name=critic_model_name,
            temperature=creative_temp,
            top_p=top_p,
            thinking_budget=thinking_budget,
            thinking_level=self.thinking_level,
            api_key=api_key,
            random_seed=self._random_randbits(64),
        )
        self.breeder = Breeder(
            provider="google_generative_ai",
            model_name=model_type,
            temperature=creative_temp,
            top_p=top_p,
            thinking_budget=thinking_budget,
            thinking_level=self.thinking_level,
            mutation_rate=mutation_rate,
            seed_context_pool_size=self.seed_context_pool_size,
            api_key=api_key,
            random_seed=self._random_randbits(64),
        )

        self.oracle = Oracle(
            provider="google_generative_ai",
            model_name=model_type,
            temperature=creative_temp,
            top_p=top_p,
            thinking_budget=thinking_budget,
            thinking_level=self.thinking_level,
            api_key=api_key,
            random_seed=self._random_randbits(64),
        )

        # Keep custom template data scoped to this engine instance (no global cross-user cache coupling).
        if self.template_data:
            self.ideator.register_custom_template(self.idea_type, self.template_data)
            self.formatter.register_custom_template(self.idea_type, self.template_data)
            self.critic.register_custom_template(self.idea_type, self.template_data)
            self.breeder.register_custom_template(self.idea_type, self.template_data)
            self.oracle.register_custom_template(self.idea_type, self.template_data)

        self.history = []  # List[List[Idea]]
        self.contexts = []  # List of contexts for the initial population
        self.specific_prompts = []  # Direction prompts generated from sampled contexts.
        self.breeding_prompts = []  # List of lists: breeding prompts for each generation (empty for gen 0)
        self.tournament_history = []  # List of tournament details per generation

        # Initialize diversity calculator
        self.diversity_calculator = DiversityCalculator(api_key=api_key)
        self.diversity_history = []  # List of diversity metrics for each generation
        self.island_diversity_history = []  # List of per-island diversity snapshots
        self.migration_history = []  # List of cross-island migration events

        # Add stop flag for graceful interruption
        self.stop_requested = False
        self.is_stopped = False

        # Dedicated embedding storage for efficient centroid computation
        # Maps idea ID to its embedding vector for fast lookup and updates
        self.idea_embeddings = {}  # Dict[str, np.ndarray]
        # Maintain ordered list of all valid embeddings for quick centroid calculation
        self.all_embeddings = []  # List[np.ndarray]
        # Track which ideas have embeddings to handle removals efficiently
        self.embedding_id_to_index = {}  # Dict[str, int] - maps idea ID to index in all_embeddings

        # Cost tracking for better estimation
        self.avg_idea_cost = 0.0
        self.avg_tournament_cost = 0.0
        self._persisted_token_counts: Optional[Dict[str, Any]] = None

        # Concurrency control
        self.concurrency_limit = 10
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)

        # Store configuration for checkpointing/resuming
        self.model_type = model_type
        self.creative_temp = creative_temp
        self.top_p = top_p
        self.mutation_rate = mutation_rate
        self.refine_all_ideas = self._coerce_bool(refine_all_ideas, default=False)
        self.replacement_rate = max(0.0, min(1.0, float(replacement_rate)))
        self.fitness_alpha = max(0.0, min(1.0, float(fitness_alpha)))
        self.age_decay_rate = max(0.0, float(age_decay_rate))
        self.age_decay_floor = max(0.0, min(1.0, float(age_decay_floor)))

        # Running normalization state (persisted) to keep fitness scores comparable across generations.
        self.fitness_elo_stats = {"count": 0, "mean": 0.0, "m2": 0.0}
        self.fitness_diversity_stats = {"count": 0, "mean": 0.0, "m2": 0.0}
        # Stateful normalization is keyed by the evaluator metric rather than
        # hard-wired to Elo/diversity.  The two legacy aliases remain the same
        # dictionary objects so older checkpoints and UI payloads stay valid.
        self.fitness_metric_stats: Dict[str, Dict[str, float]] = {
            "quality_elo.quality_elo": self.fitness_elo_stats,
            "diversity.diversity": self.fitness_diversity_stats,
        }
        self.comparison_cache: Dict[str, Any] = {}
        self._configure_evaluation(
            evaluation_config,
            evaluation_runners,
            artifact_store=artifact_store,
        )

        # Evolution identity and tracking
        self.evolution_id = None  # Unique ID for this evolution (UUID)
        self.evolution_name = None  # Human-readable name
        self.created_at = None  # ISO timestamp when evolution started
        self.updated_at = None  # ISO timestamp of last update
        # Optimistic-concurrency token for the Firestore manifest.  This is
        # injected when a persisted run is loaded and advanced only after a
        # successful save; it is not part of the user-facing run config.
        self.persistence_revision = None

        # Legacy checkpoint tracking (for backwards compatibility)
        self.checkpoint_id = None  # Set when evolution starts
        self.current_generation = 0  # Tracks which generation we're on
        self.checkpoint_callback = None  # Callback for saving checkpoints
        self._sync_typed_state_from_attrs()

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
            return default
        return bool(value)

    def _configure_evaluation(
        self,
        config: Optional[Dict[str, Any]],
        runners: Optional[Dict[str, Any]] = None,
        *,
        artifact_store: Optional[Any] = None,
    ) -> None:
        """Resolve evaluator configuration and instantiate supported adapters."""

        self.evaluation_config = resolve_evaluation_suite(
            config,
            fitness_alpha=self.fitness_alpha,
        )
        self.fitness_aggregator = FitnessAggregator(self.evaluation_config)
        self.quality_evaluator = None
        self.diversity_evaluator = None
        self.computed_evaluators = []
        self.artifact_producers = []
        self.pre_judge_evaluators = []
        self.unsupported_evaluation_components = []
        self.last_evaluation_batch = None
        self.pre_judge_evaluation_batch = None
        self._pre_judge_candidate_ids = None

        self.artifact_store = artifact_store
        artifact_loader = (
            ArtifactStoreLoader(artifact_store)
            if artifact_store is not None
            else None
        )

        if not self.evaluation_config.enabled:
            return

        runners = runners or {}
        for component in self.evaluation_config.components:
            if not component.enabled:
                continue
            if (
                component.kind == ComponentKind.LLM_PAIRWISE
                and self.quality_evaluator is None
            ):
                self.quality_evaluator = LLMPairwiseEvaluator(
                    component,
                    artifact_loader=artifact_loader,
                )
            elif (
                component.kind == ComponentKind.EMBEDDING_DIVERSITY
                and self.diversity_evaluator is None
            ):
                self.diversity_evaluator = EmbeddingDiversityEvaluator(component)
            elif component.kind == ComponentKind.COMPUTED:
                runner = runners.get(component.id)
                if runner is None:
                    runner = runners.get(component.runner_ref)
                if runner is None:
                    runner = create_builtin_computed_runner(component.runner_ref)
                if runner is None:
                    self.unsupported_evaluation_components.append(component.id)
                else:
                    entry = (ComputedEvaluator(component), runner)
                    self.computed_evaluators.append(entry)
                    self.pre_judge_evaluators.append(entry)
            elif component.kind == ComponentKind.LLM_CODE_ARTIFACT:
                runner = runners.get(component.id)
                if runner is None:
                    runner = runners.get(component.execution_profile)
                if runner is None:
                    runner = runners.get("llm_code_artifact")
                if runner is None or artifact_store is None:
                    self.unsupported_evaluation_components.append(component.id)
                else:
                    entry = (
                        LLMCodeArtifactProducer(component, artifact_store),
                        runner,
                    )
                    self.artifact_producers.append(entry)
                    self.pre_judge_evaluators.append(entry)

    @staticmethod
    def _coerce_unit_float(value: Any, *, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_island_count(value: Any, *, pop_size: int) -> int:
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = 1
        max_count = max(1, int(pop_size or 1))
        return max(1, min(max_count, count))

    def _random_randbits(self, bits: int) -> int:
        with self._py_rng_lock:
            return self._py_rng.getrandbits(bits)

    def random_uniform(self, a: float, b: float) -> float:
        with self._py_rng_lock:
            return self._py_rng.uniform(a, b)

    def random_shuffle(self, values: List[Any]) -> None:
        with self._py_rng_lock:
            self._py_rng.shuffle(values)

    def random_choice(
        self,
        a: Any,
        *,
        size: Optional[int] = None,
        replace: bool = True,
        p: Optional[Any] = None,
    ) -> Any:
        with self._np_rng_lock:
            return self._np_rng.choice(a, size=size, replace=replace, p=p)

    async def _run_batch_with_progress(
        self,
        tasks: List[Callable[[], Awaitable[Any]]],
        progress_callback: Callable[[Dict[str, Any]], Awaitable[None]],
        base_progress_info: Dict[str, Any],
        start_step: int,
        total_steps: int,
        description_template: str
    ) -> List[Any]:
        """
        Run a batch of async tasks with concurrency control and progress updates.

        Args:
            tasks: List of async functions to run (not coroutines yet, factories)
            progress_callback: Callback for progress updates
            base_progress_info: Base dictionary for progress updates (generation, etc)
            start_step: Starting step number for progress calculation
            total_steps: Total steps for progress calculation
            description_template: Template string for status message (e.g. "Processing item {completed}/{total}")

        Returns:
            List of results in the same order as tasks
        """
        results = [None] * len(tasks)
        completed_count = 0

        async def wrapped_task(index, task_func):
            async with self.semaphore:
                try:
                    # Check for stop before starting
                    if self.stop_requested:
                        return None

                    # Small stagger to prevent "thundering herd" completion where all parallel tasks
                    # finish at the exact same millisecond, causing the UI to jump from 0% to 100% instantly.
                    # This makes the progress bar feel smoother.
                    if len(tasks) > 1 and PROGRESS_JITTER_MAX_SECONDS > 0:
                        await asyncio.sleep(self.random_uniform(0, PROGRESS_JITTER_MAX_SECONDS))

                    result = await task_func()
                    return index, result
                except Exception as e:
                    print(f"Error in task {index}: {e}")
                    return index, None

        # Create coroutines
        coroutines = [wrapped_task(i, task) for i, task in enumerate(tasks)]

        # Run as completed to update progress
        for future in asyncio.as_completed(coroutines):
            if self.stop_requested:
                break

            index, result = await future
            results[index] = result
            completed_count += 1

            # Calculate progress
            current_step = start_step + completed_count
            progress_percent = (current_step / total_steps) * 100

            # Update status
            status_message = description_template.format(
                completed=completed_count,
                total=len(tasks)
            )

            update_data = base_progress_info.copy()
            update_data.update({
                "progress": progress_percent,
                "status_message": status_message
            })

            await progress_callback(update_data)

        return results

    def generate_contexts(self):
        """Generate contexts for the initial population"""
        self.contexts = []
        for _ in range(self.pop_size):
            context = self.ideator.generate_context(self.idea_type)
            self.contexts.append(context)
        return self.contexts

    def stop_evolution(self):
        """Request the evolution to stop gracefully"""
        print("Stop requested - evolution will halt at the next safe point")
        self.stop_requested = True
        self._sync_typed_state_from_attrs()

    def reset_stop_state(self):
        """Reset the stop state for a new evolution"""
        self.stop_requested = False
        self.is_stopped = False
        self._sync_typed_state_from_attrs()

    def _get_template_prompts(self):
        """Resolve prompt wrapper for current template, preferring the engine-scoped custom template snapshot."""
        if self.template_data:
            if self._template_prompt_wrapper is None:
                self._template_prompt_wrapper = get_prompts_from_dict(self.template_data)
            return self._template_prompt_wrapper
        return get_prompts(self.idea_type)


    def _sync_typed_state_from_attrs(self):
        """Keep typed internal snapshots aligned while preserving legacy attribute access."""
        from idea.evolution_types import (
            EvolutionIdentity,
            EvolutionRunConfig,
            EvolutionRuntimeState,
        )

        self.config_state = EvolutionRunConfig.from_engine(self)
        self.identity_state = EvolutionIdentity(
            evolution_id=self.evolution_id,
            evolution_name=self.evolution_name,
            created_at=self.created_at,
            updated_at=self.updated_at,
            checkpoint_id=self.checkpoint_id,
        )
        self.runtime_state = EvolutionRuntimeState(
            current_generation=self.current_generation,
            population=self.population,
            history=self.history,
            contexts=self.contexts,
            specific_prompts=self.specific_prompts,
            breeding_prompts=self.breeding_prompts,
            tournament_history=self.tournament_history,
            diversity_history=self.diversity_history,
            island_diversity_history=self.island_diversity_history,
            migration_history=self.migration_history,
            comparison_cache=self.comparison_cache,
            avg_idea_cost=self.avg_idea_cost,
            avg_tournament_cost=self.avg_tournament_cost,
            stop_requested=self.stop_requested,
            is_stopped=self.is_stopped,
        )

    def _new_orchestrator(self, progress_callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        from idea.evolution_orchestrator import EvolutionOrchestrator

        return EvolutionOrchestrator(self, progress_callback)

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """Serialize current evolution state for checkpointing."""
        from idea.evolution_persistence import EvolutionSerializer

        self._sync_typed_state_from_attrs()
        return EvolutionSerializer.to_checkpoint_state(self)

    def set_name(self, name: str):
        """Set the human-readable name for this evolution."""
        self.evolution_name = name
        self.updated_at = datetime.now().isoformat()
        self._sync_typed_state_from_attrs()

    def generate_default_name(self) -> str:
        """Generate a default name based on template and date."""
        if self.template_data and self.template_data.get("name"):
            template_name = self.template_data.get("name")
        else:
            try:
                templates = list_available_templates()
                template_info = templates.get(self.idea_type, {})
                template_name = template_info.get('name', self.idea_type)
            except Exception:
                template_name = self.idea_type or 'Evolution'

        date_str = datetime.now().strftime('%b %d, %Y')
        return f"{template_name} - {date_str}"

    def initialize_evolution(
        self,
        name: str = None,
        *,
        evolution_id: Optional[str] = None,
    ):
        """Initialize a new evolution with ID, name, and timestamps."""
        self.evolution_id = str(evolution_id or uuid.uuid4())
        self.evolution_name = name or self.generate_default_name()
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.persistence_revision = None
        self.checkpoint_id = self.evolution_id[:18].replace('-', '')
        self._sync_typed_state_from_attrs()

    async def save_checkpoint(self, status: str = 'in_progress') -> Optional[str]:
        from idea.evolution_persistence import EvolutionRepository

        self._sync_typed_state_from_attrs()
        checkpoint_id = await EvolutionRepository.save_checkpoint(self, status=status)
        self._sync_typed_state_from_attrs()
        return checkpoint_id

    @classmethod
    async def list_evolutions_for_user(cls, user_id: str) -> List[Dict[str, Any]]:
        from idea.evolution_persistence import EvolutionRepository

        return await EvolutionRepository.list_evolutions_for_user(user_id)

    @classmethod
    async def list_checkpoints_for_user(cls, user_id: str) -> List[Dict[str, Any]]:
        from idea.evolution_persistence import EvolutionRepository

        return await EvolutionRepository.list_checkpoints_for_user(user_id)

    @classmethod
    async def load_evolution_for_user(
        cls,
        user_id: str,
        evolution_id: str,
        api_key: Optional[str] = None,
        evaluation_runners: Optional[Dict[str, Any]] = None,
        artifact_store: Optional[Any] = None,
    ) -> Optional['EvolutionEngine']:
        from idea.evolution_persistence import EvolutionRepository

        return await EvolutionRepository.load_evolution_for_user(
            cls,
            user_id,
            evolution_id,
            api_key=api_key,
            evaluation_runners=evaluation_runners,
            artifact_store=artifact_store,
        )

    @classmethod
    async def rename_evolution_for_user(cls, user_id: str, evolution_id: str, new_name: str) -> bool:
        from idea.evolution_persistence import EvolutionRepository

        return await EvolutionRepository.rename_evolution_for_user(user_id, evolution_id, new_name)

    @classmethod
    async def delete_evolution_for_user(cls, user_id: str, evolution_id: str) -> bool:
        from idea.evolution_persistence import EvolutionRepository

        return await EvolutionRepository.delete_evolution_for_user(user_id, evolution_id)

    @classmethod
    def _load_from_file(
        cls,
        file_path: Path,
        api_key: Optional[str] = None,
        evaluation_runners: Optional[Dict[str, Any]] = None,
        artifact_store: Optional[Any] = None,
    ) -> Optional['EvolutionEngine']:
        from idea.evolution_persistence import EvolutionRepository

        return EvolutionRepository.load_from_file(
            cls,
            file_path,
            api_key=api_key,
            evaluation_runners=evaluation_runners,
            artifact_store=artifact_store,
        )

    @classmethod
    def _restore_from_state(
        cls,
        state: Dict[str, Any],
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        evaluation_runners: Optional[Dict[str, Any]] = None,
        artifact_store: Optional[Any] = None,
    ) -> 'EvolutionEngine':
        from idea.evolution_persistence import EvolutionRepository

        return EvolutionRepository.restore_from_state(
            cls,
            state,
            api_key=api_key,
            user_id=user_id,
            evaluation_runners=evaluation_runners,
            artifact_store=artifact_store,
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_id: str,
        api_key: Optional[str] = None,
        evaluation_runners: Optional[Dict[str, Any]] = None,
        artifact_store: Optional[Any] = None,
    ) -> Optional['EvolutionEngine']:
        from idea.evolution_persistence import EvolutionRepository

        return EvolutionRepository.load_checkpoint(
            cls,
            checkpoint_id,
            api_key=api_key,
            evaluation_runners=evaluation_runners,
            artifact_store=artifact_store,
        )

    @classmethod
    def delete_checkpoint(cls, checkpoint_id: str) -> bool:
        from idea.evolution_persistence import EvolutionRepository

        return EvolutionRepository.delete_checkpoint(checkpoint_id)

    async def run_evolution_with_updates(self, progress_callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Run a new evolution using the canonical orchestrator loop."""
        orchestrator = self._new_orchestrator(progress_callback)
        await orchestrator.run(start_generation=0, mode="new")
        self._sync_typed_state_from_attrs()

    async def resume_evolution_with_updates(
        self,
        progress_callback: Callable[[Dict[str, Any]], Awaitable[None]],
        additional_generations: int = 0,
    ):
        """Resume/continue evolution using the canonical orchestrator loop."""
        orchestrator = self._new_orchestrator(progress_callback)
        await orchestrator.run(
            start_generation=self.current_generation,
            mode="resume",
            additional_generations=additional_generations,
        )
        self._sync_typed_state_from_attrs()

    @staticmethod
    def _valid_embedding_list(embeddings: List[Optional[np.ndarray]]) -> List[np.ndarray]:
        return [embedding for embedding in embeddings if embedding is not None]

    @staticmethod
    def _embedding_mean_distance(embeddings: List[np.ndarray]) -> float:
        valid_embeddings = EvolutionEngine._valid_embedding_list(embeddings)
        if len(valid_embeddings) < 2:
            return 0.0
        matrix = np.array(valid_embeddings)
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            return 0.0
        deltas = matrix[:, None, :] - matrix[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        upper = np.triu_indices(matrix.shape[0], k=1)
        return float(np.mean(distances[upper])) if upper[0].size else 0.0

    @staticmethod
    def _embedding_centroid(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
        valid_embeddings = EvolutionEngine._valid_embedding_list(embeddings)
        if not valid_embeddings:
            return None
        return np.mean(valid_embeddings, axis=0)

    def _build_diversity_analytics(
        self,
        history: List[List[Any]],
        precomputed_embeddings: Optional[List[Optional[np.ndarray]]],
        diversity_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        current_generation_index = max(0, len(history or []) - 1)
        current_generation = history[current_generation_index] if history else []
        embeddings_by_generation: List[List[Optional[np.ndarray]]] = []
        embedding_index = 0

        for generation in history or []:
            generation_embeddings: List[Optional[np.ndarray]] = []
            for _idea in generation:
                embedding = (
                    precomputed_embeddings[embedding_index]
                    if precomputed_embeddings is not None
                    and embedding_index < len(precomputed_embeddings)
                    else None
                )
                generation_embeddings.append(embedding)
                embedding_index += 1
            embeddings_by_generation.append(generation_embeddings)

        current_embeddings = (
            embeddings_by_generation[current_generation_index]
            if embeddings_by_generation
            else []
        )
        valid_current_embeddings = self._valid_embedding_list(current_embeddings)
        current_generation_diversity = self._embedding_mean_distance(valid_current_embeddings)
        if len(valid_current_embeddings) < 2:
            for generation_data in diversity_data.get("generation_diversities", []):
                if generation_data.get("generation") == current_generation_index:
                    current_generation_diversity = float(
                        generation_data.get("diversity_score", 0.0) or 0.0
                    )
                    break

        island_rows: List[Dict[str, Any]] = []
        island_centroids: List[np.ndarray] = []
        island_count = max(1, int(getattr(self, "island_count", 2) or 2))
        fallback_ids = self._initial_island_ids(len(current_generation))

        for island_id in range(island_count):
            island_embeddings = [
                current_embeddings[idx]
                for idx, idea in enumerate(current_generation)
                if idx < len(current_embeddings)
                and self._get_idea_island_id(
                    idea,
                    fallback=fallback_ids[idx] if idx < len(fallback_ids) else 0,
                )
                == island_id
            ]
            valid_island_embeddings = self._valid_embedding_list(island_embeddings)
            centroid = self._embedding_centroid(valid_island_embeddings)
            if centroid is not None:
                island_centroids.append(centroid)

            island_rows.append(
                {
                    "island_id": island_id,
                    "diversity_score": self._embedding_mean_distance(valid_island_embeddings),
                    "idea_count": len(island_embeddings),
                    "valid_embeddings": len(valid_island_embeddings),
                    "centroid_available": centroid is not None,
                }
            )

        island_diversity_values = [
            row["diversity_score"]
            for row in island_rows
            if row.get("valid_embeddings", 0) >= 2
        ]
        mean_island_diversity = (
            float(sum(island_diversity_values) / len(island_diversity_values))
            if island_diversity_values
            else 0.0
        )
        inter_island_diversity = (
            self._embedding_mean_distance(island_centroids)
            if len(island_centroids) >= 2
            else None
        )
        island_separation_ratio = (
            inter_island_diversity / mean_island_diversity
            if inter_island_diversity is not None and mean_island_diversity > 0
            else None
        )

        return {
            "current_generation": current_generation_index,
            "current_generation_diversity": current_generation_diversity,
            "cumulative_diversity": float(diversity_data.get("diversity_score", 0.0) or 0.0),
            "inter_generation_diversity": diversity_data.get("inter_generation_diversity"),
            "island_count": island_count,
            "islands": island_rows,
            "mean_island_diversity": mean_island_diversity,
            "inter_island_diversity": inter_island_diversity,
            "island_separation_ratio": island_separation_ratio,
        }

    async def _calculate_and_store_diversity(self) -> Dict[str, Any]:
        """
        Calculate diversity metrics for the current population history and store them.
        Also populates our embedding storage for efficient centroid computation.

        Returns:
            Dictionary containing diversity metrics
        """
        try:
            if not self.history:
                return {"enabled": False, "reason": "No history available"}

            print("🔍 Calculating population diversity...")

            # Flatten all ideas from all generations to populate embedding storage
            all_ideas = []
            for generation in self.history:
                all_ideas.extend(generation)

            precomputed_embeddings = None
            # Ensure we have embeddings for all ideas (this will populate our storage)
            if all_ideas:
                precomputed_embeddings = await self._get_or_compute_embeddings_for_ideas(all_ideas)
                print(f"📦 Embedding storage now contains {len(self.all_embeddings)} embeddings")

            # Calculate diversity using the standard diversity calculator
            diversity_data = await self.diversity_calculator.calculate_diversity(
                self.history,
                precomputed_embeddings=precomputed_embeddings,
            )
            diversity_analytics = self._build_diversity_analytics(
                self.history,
                precomputed_embeddings,
                diversity_data,
            )
            diversity_data["analytics"] = diversity_analytics
            diversity_data["current_generation"] = diversity_analytics["current_generation"]
            diversity_data["current_generation_diversity"] = diversity_analytics[
                "current_generation_diversity"
            ]
            diversity_data["mean_island_diversity"] = diversity_analytics[
                "mean_island_diversity"
            ]
            diversity_data["inter_island_diversity"] = diversity_analytics[
                "inter_island_diversity"
            ]
            diversity_data["island_separation_ratio"] = diversity_analytics[
                "island_separation_ratio"
            ]
            diversity_data["current_island_diversities"] = diversity_analytics["islands"]
            if self.island_count > 1:
                island_diversities = []
                for island_id in range(self.island_count):
                    island_history: List[List[Any]] = []
                    for generation in self.history:
                        fallback_ids = self._initial_island_ids(len(generation))
                        island_generation = [
                            idea
                            for idx, idea in enumerate(generation)
                            if self._get_idea_island_id(
                                idea,
                                fallback=fallback_ids[idx] if idx < len(fallback_ids) else 0,
                            )
                            == island_id
                        ]
                        island_history.append(island_generation)

                    flat_island_ideas = [
                        idea for generation in island_history for idea in generation
                    ]
                    island_embeddings = None
                    if flat_island_ideas:
                        island_embeddings = await self._get_or_compute_embeddings_for_ideas(
                            flat_island_ideas
                        )
                    island_data = await self.diversity_calculator.calculate_diversity(
                        island_history,
                        precomputed_embeddings=island_embeddings,
                    )
                    island_data["island_id"] = island_id
                    island_data["idea_count"] = len(flat_island_ideas)
                    island_diversities.append(island_data)

                diversity_data["island_diversities"] = island_diversities
                self.island_diversity_history.append(
                    {
                        "generation": len(self.history) - 1,
                        "inter_island_diversity": diversity_analytics[
                            "inter_island_diversity"
                        ],
                        "mean_island_diversity": diversity_analytics[
                            "mean_island_diversity"
                        ],
                        "island_separation_ratio": diversity_analytics[
                            "island_separation_ratio"
                        ],
                        "islands": island_diversities,
                        "current_islands": diversity_analytics["islands"],
                    }
                )
            self.diversity_history.append(diversity_data)

            # Print diversity summary to logs
            self.diversity_calculator.print_diversity_summary(diversity_data)

            return diversity_data

        except Exception as e:
            print(f"Warning: Diversity calculation failed: {e}")
            return {"enabled": True, "error": str(e)}

    def _set_tournament_history(self, generation: int, rounds: List[Dict[str, Any]]) -> None:
        """
        Store Swiss tournament details for a generation.

        Args:
            generation: Generation index the tournament ran on
            rounds: List of round details from the tournament
        """
        entry = {
            "generation": generation,
            "rounds": rounds
        }

        for idx, existing in enumerate(self.tournament_history):
            if existing.get("generation") == generation:
                self.tournament_history[idx] = entry
                break
        else:
            self.tournament_history.append(entry)
        self.tournament_history.sort(key=lambda e: e.get("generation", 0))

    async def _find_least_interesting_idea_idx(self, current_generation: List[str]) -> int:
        """
        Find the index of the least interesting idea in the current generation using embedding-based centroid distance.

        The "interesting-ness" score is computed as the distance of each idea to the POPULATION centroid
        (computed from ALL historical ideas across ALL generations).
        The idea with the lowest score (closest to centroid) is considered least interesting.

        Args:
            current_generation: List of ideas in the current generation

        Returns:
            Index of the least interesting idea (0-based)
        """
        try:
            if not self.diversity_calculator.is_enabled():
                print("Diversity calculator not enabled, defaulting to first idea for replacement")
                return 0

            if len(current_generation) <= 1:
                return 0

            print(f"🎯 Calculating embedding-based interesting-ness scores for {len(current_generation)} ideas...")
            print(f"🎯 Population has {len(self.all_embeddings)} total embeddings for centroid calculation")

            # Get or compute embeddings for current generation ideas
            embeddings = await self._get_or_compute_embeddings_for_ideas(current_generation)

            # Filter out failed embeddings and track their indices
            valid_embeddings = []
            valid_indices = []
            for idx, embedding in enumerate(embeddings):
                if embedding is not None:
                    valid_embeddings.append(embedding)
                    valid_indices.append(idx)

            if len(valid_embeddings) == 0:
                print("No valid embeddings for current generation, defaulting to first idea")
                return 0

            # Compute population centroid from ALL historical embeddings
            population_centroid = await self._compute_population_centroid()
            if population_centroid is None:
                print("No population centroid available, defaulting to first idea")
                return 0

            # Calculate distance from each current generation idea to the population centroid
            import numpy as np
            distances = []
            for i, embedding in enumerate(valid_embeddings):
                distance = np.sqrt(np.sum((embedding - population_centroid) ** 2))
                distances.append(distance)

            # Find the index of the idea with minimum distance (least interesting)
            min_distance_idx = np.argmin(distances)
            least_interesting_original_idx = valid_indices[min_distance_idx]

            # Extract title for logging
            least_interesting_idea = current_generation[least_interesting_original_idx]
            title = "Unknown"
            if isinstance(least_interesting_idea, dict) and "idea" in least_interesting_idea:
                idea_obj = least_interesting_idea["idea"]
                if hasattr(idea_obj, 'title'):
                    title = idea_obj.title

            print(f"🎯 Least interesting idea (closest to population centroid): '{title}' at index {least_interesting_original_idx}")
            print(f"   Distance to population centroid: {distances[min_distance_idx]:.4f}")
            print(f"   Population centroid computed from {len(self.all_embeddings)} historical embeddings")

            return least_interesting_original_idx

        except Exception as e:
            print(f"Error calculating interesting-ness scores: {e}")
            print("Defaulting to first idea for replacement")
            return 0

    async def _find_most_diverse_idea_idx(self, current_generation: List[str]) -> int:
        """
        Find the index of the most diverse idea in the current generation using embedding-based centroid distance.

        The diversity score is computed as the distance of each idea to the POPULATION centroid
        (computed from ALL historical ideas across ALL generations).
        The idea with the highest score (farthest from centroid) is considered most diverse.

        Args:
            current_generation: List of ideas in the current generation

        Returns:
            Index of the most diverse idea (0-based)
        """
        try:
            if not self.diversity_calculator.is_enabled():
                print("Diversity calculator not enabled, defaulting to first idea for creative selection")
                return 0

            if len(current_generation) <= 1:
                return 0

            print(f"🌟 Calculating embedding-based diversity scores for {len(current_generation)} ideas...")
            print(f"🌟 Population has {len(self.all_embeddings)} total embeddings for centroid calculation")

            # Get or compute embeddings for current generation ideas
            embeddings = await self._get_or_compute_embeddings_for_ideas(current_generation)

            # Filter out failed embeddings and track their indices
            valid_embeddings = []
            valid_indices = []
            for idx, embedding in enumerate(embeddings):
                if embedding is not None:
                    valid_embeddings.append(embedding)
                    valid_indices.append(idx)

            if len(valid_embeddings) == 0:
                print("No valid embeddings for current generation, defaulting to first idea")
                return 0

            # Compute population centroid from ALL historical embeddings
            population_centroid = await self._compute_population_centroid()
            if population_centroid is None:
                print("No population centroid available, defaulting to first idea")
                return 0

            # Calculate distance from each current generation idea to the population centroid
            import numpy as np
            distances = []
            for i, embedding in enumerate(valid_embeddings):
                distance = np.sqrt(np.sum((embedding - population_centroid) ** 2))
                distances.append(distance)

            # Find the index of the idea with maximum distance (most diverse)
            max_distance_idx = np.argmax(distances)
            most_diverse_original_idx = valid_indices[max_distance_idx]

            # Extract title for logging
            most_diverse_idea = current_generation[most_diverse_original_idx]
            title = "Unknown"
            if isinstance(most_diverse_idea, dict) and "idea" in most_diverse_idea:
                idea_obj = most_diverse_idea["idea"]
                if hasattr(idea_obj, 'title'):
                    title = idea_obj.title

            print(f"🌟 Most diverse idea (farthest from population centroid): '{title}' at index {most_diverse_original_idx}")
            print(f"   Distance to population centroid: {distances[max_distance_idx]:.4f}")
            print(f"   Population centroid computed from {len(self.all_embeddings)} historical embeddings")

            return most_diverse_original_idx

        except Exception as e:
            print(f"Error calculating diversity scores: {e}")
            print("Defaulting to first idea for creative selection")
            return 0

    def check_budget(self) -> bool:
        """
        Check if the max budget has been exceeded.

        Returns:
            True if budget exceeded, False otherwise
        """
        if self.max_budget is None:
            return False

        token_counts = self.get_total_token_count()
        current_cost = token_counts['cost']['total_cost']

        return current_cost >= self.max_budget

    def get_ideas_by_generation(self, generation_index: int) -> List[Dict]:
        """
        Get all ideas from a specific generation

        Args:
            generation_index: The index of the generation to retrieve

        Returns:
            List of idea dictionaries with 'id' and 'idea' keys
        """
        if generation_index < 0 or generation_index >= len(self.history):
            return []
        return self.history[generation_index]

    def get_total_token_count(self):
        """Get the total token count from all LLM components with cost calculation"""
        required_components = {"ideator", "formatter", "critic", "breeder", "oracle"}

        def safe_int(value: Any) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        def safe_float(value: Any) -> float:
            try:
                return float(value or 0.0)
            except (TypeError, ValueError):
                return 0.0

        def summarize_component(component_name: str, component: Any) -> Optional[Dict[str, Any]]:
            if component is None:
                return None

            custom_summary = getattr(component, "get_token_usage_summary", None)
            if callable(custom_summary):
                summary = custom_summary() or {}
                component_input = safe_int(summary.get("input"))
                component_output = safe_int(summary.get("output"))
                component_thoughts = safe_int(summary.get("thoughts"))
                component_cached_input = safe_int(summary.get("cached_input"))
                component_tool_input = safe_int(summary.get("tool_input"))
                component_uncached_input = (
                    safe_int(summary.get("uncached_input"))
                    if "uncached_input" in summary
                    else max(component_input + component_tool_input - component_cached_input, 0)
                )
                component_total = safe_int(summary.get("total")) or (
                    component_input + component_output + component_thoughts + component_tool_input
                )
                component_cost = safe_float(summary.get("cost"))
                component_input_cost = safe_float(summary.get("input_cost"))
                if "input_cost" not in summary and component_cost:
                    component_input_cost = component_cost
                component_cached_input_cost = safe_float(summary.get("cached_input_cost"))
                component_uncached_input_cost = safe_float(summary.get("uncached_input_cost"))
                component_output_cost = safe_float(summary.get("output_cost"))
                component_thought_cost = safe_float(summary.get("thought_cost"))
                component_model = summary.get("model") or getattr(component, "model_name", "unknown")
                if component_name not in required_components and component_total == 0 and component_cost == 0:
                    return None
                data = {
                    "total": component_total,
                    "input": component_input,
                    "output": component_output,
                    "thoughts": component_thoughts,
                    "cached_input": component_cached_input,
                    "uncached_input": component_uncached_input,
                    "tool_input": component_tool_input,
                    "model": component_model,
                    "cost": component_cost,
                    "_input_cost": component_input_cost,
                    "_cached_input_cost": component_cached_input_cost,
                    "_uncached_input_cost": component_uncached_input_cost,
                    "_output_cost": component_output_cost,
                    "_thought_cost": component_thought_cost,
                }
                for optional_key in (
                    "estimated_input",
                    "requests",
                    "large_prompt_requests",
                    "per_model",
                ):
                    if optional_key in summary:
                        data[optional_key] = summary[optional_key]
                return data

            component_input = safe_int(getattr(component, "input_token_count", 0))
            component_output = safe_int(getattr(component, "output_token_count", 0))
            component_thoughts = safe_int(
                getattr(component, "thought_token_count", getattr(component, "thinking_token_count", 0))
            )
            component_cached_input = safe_int(getattr(component, "cached_input_token_count", 0))
            component_tool_input = safe_int(getattr(component, "tool_input_token_count", 0))
            component_total = safe_int(
                getattr(
                    component,
                    "total_token_count",
                    component_input + component_output + component_thoughts + component_tool_input,
                )
            )
            if component_total == 0:
                component_total = component_input + component_output + component_thoughts + component_tool_input
            component_model = getattr(component, "model_name", "gemini-2.0-flash")
            usage_cost = estimate_generation_token_cost(
                component_model,
                input_tokens=component_input,
                output_tokens=component_output,
                thought_tokens=component_thoughts,
                cached_input_tokens=component_cached_input,
                tool_input_tokens=component_tool_input,
            )
            component_uncached_input = safe_int(
                getattr(component, "uncached_input_token_count", usage_cost["uncached_input_tokens"])
            )
            component_cached_input = safe_int(usage_cost["cached_input_tokens"])
            component_input_cost = usage_cost["input_cost"]
            component_uncached_input_cost = usage_cost["uncached_input_cost"]
            component_cached_input_cost = usage_cost["cached_input_cost"]
            component_output_cost = usage_cost["output_cost"]
            component_thought_cost = usage_cost["thought_cost"]
            component_cost = usage_cost["total_cost"]
            if component_name not in required_components and component_total == 0 and component_cost == 0:
                return None
            return {
                "total": component_total,
                "input": component_input,
                "output": component_output,
                "thoughts": component_thoughts,
                "cached_input": component_cached_input,
                "uncached_input": component_uncached_input,
                "tool_input": component_tool_input,
                "model": component_model,
                "cost": component_cost,
                "_input_cost": component_input_cost,
                "_cached_input_cost": component_cached_input_cost,
                "_uncached_input_cost": component_uncached_input_cost,
                "_output_cost": component_output_cost,
                "_thought_cost": component_thought_cost,
            }

        component_specs = [
            ("ideator", getattr(self, "ideator", None)),
            ("formatter", getattr(self, "formatter", None)),
            ("critic", getattr(self, "critic", None)),
            ("breeder", getattr(self, "breeder", None)),
            ("genotype_encoder", getattr(self, "genotype_encoder", None)),
            ("diversity_embeddings", getattr(self, "diversity_calculator", None)),
            ("oracle", getattr(self, "oracle", None)),
        ]
        seen_usage_components = {id(component) for _, component in component_specs}
        for evaluator, runner in getattr(self, "pre_judge_evaluators", []):
            if id(runner) in seen_usage_components:
                continue
            seen_usage_components.add(id(runner))
            kind = getattr(getattr(evaluator, "config", None), "kind", None)
            prefix = "artifact" if kind == ComponentKind.LLM_CODE_ARTIFACT else "computed"
            component_specs.append((f"{prefix}_{evaluator.config.id}", runner))

        token_data: Dict[str, Any] = {}
        models: Dict[str, str] = {}
        diagnostics_by_agent: Dict[str, Any] = {}
        diagnostic_events_by_agent: Dict[str, Any] = {}
        total_input = 0
        total_output = 0
        total_thoughts = 0
        total_cached_input = 0
        total_uncached_input = 0
        total_tool_input = 0
        total_provider_tokens = 0
        total_input_cost = 0.0
        total_uncached_input_cost = 0.0
        total_cached_input_cost = 0.0
        total_output_cost = 0.0
        total_thought_cost = 0.0

        for component_name, component in component_specs:
            summary = summarize_component(component_name, component)
            if summary is None:
                continue

            total_provider_tokens += safe_int(summary.get("total"))
            total_input += safe_int(summary.get("input"))
            total_output += safe_int(summary.get("output"))
            total_thoughts += safe_int(summary.get("thoughts"))
            total_cached_input += safe_int(summary.get("cached_input"))
            total_uncached_input += safe_int(summary.get("uncached_input"))
            total_tool_input += safe_int(summary.get("tool_input"))
            total_input_cost += safe_float(summary.pop("_input_cost", 0.0))
            total_uncached_input_cost += safe_float(summary.pop("_uncached_input_cost", 0.0))
            total_cached_input_cost += safe_float(summary.pop("_cached_input_cost", 0.0))
            total_output_cost += safe_float(summary.pop("_output_cost", 0.0))
            total_thought_cost += safe_float(summary.pop("_thought_cost", 0.0))
            models[component_name] = summary.get("model", "unknown")
            token_data[component_name] = summary
            diagnostics_by_agent[component_name] = getattr(component, "get_diagnostics", lambda: {})()
            diagnostic_events_by_agent[component_name] = getattr(
                component, "get_diagnostic_events", lambda: []
            )()

        persisted = getattr(self, "_persisted_token_counts", None)
        if isinstance(persisted, dict):
            persisted_cost = persisted.get("cost")
            if not isinstance(persisted_cost, dict):
                persisted_cost = {}
            prior_total = safe_int(persisted.get("total"))
            prior_input = safe_int(persisted.get("total_input"))
            prior_output = safe_int(persisted.get("total_output"))
            prior_thoughts = safe_int(persisted.get("total_thoughts"))
            prior_cached = safe_int(persisted.get("total_cached_input"))
            prior_uncached = safe_int(
                persisted.get("total_uncached_input", max(prior_input - prior_cached, 0))
            )
            prior_tool = safe_int(persisted.get("total_tool_input"))
            prior_input_cost = safe_float(persisted_cost.get("input_cost"))
            prior_output_cost = safe_float(persisted_cost.get("output_cost"))
            prior_thought_cost = safe_float(persisted_cost.get("thought_cost"))
            prior_total_cost = safe_float(persisted_cost.get("total_cost"))
            if (
                prior_input_cost + prior_output_cost + prior_thought_cost == 0
                and prior_total_cost > 0
            ):
                prior_input_cost = prior_total_cost
            if prior_total > 0 or prior_total_cost > 0:
                total_provider_tokens += prior_total
                total_input += prior_input
                total_output += prior_output
                total_thoughts += prior_thoughts
                total_cached_input += prior_cached
                total_uncached_input += prior_uncached
                total_tool_input += prior_tool
                total_input_cost += prior_input_cost
                total_uncached_input_cost += safe_float(
                    persisted_cost.get("uncached_input_cost")
                )
                total_cached_input_cost += safe_float(
                    persisted_cost.get("cached_input_cost")
                )
                total_output_cost += prior_output_cost
                total_thought_cost += prior_thought_cost
                token_data["restored_usage"] = {
                    "total": prior_total,
                    "input": prior_input,
                    "output": prior_output,
                    "thoughts": prior_thoughts,
                    "cached_input": prior_cached,
                    "uncached_input": prior_uncached,
                    "tool_input": prior_tool,
                    "model": "checkpoint aggregate",
                    "cost": prior_total_cost,
                }
                models["restored_usage"] = "checkpoint aggregate"

        total = total_provider_tokens or (total_input + total_output + total_thoughts + total_tool_input)
        total_cost = total_input_cost + total_output_cost + total_thought_cost

        token_data.update(
            {
                "total": total,
                "total_input": total_input,
                "total_output": total_output,
                "total_thoughts": total_thoughts,
                "total_cached_input": total_cached_input,
                "total_uncached_input": total_uncached_input,
                "total_tool_input": total_tool_input,
                "total_billable_output": total_output + total_thoughts,
                "cost": {
                    "input_cost": total_input_cost,
                    "uncached_input_cost": total_uncached_input_cost,
                    "cached_input_cost": total_cached_input_cost,
                    "output_cost": total_output_cost,
                    "thought_cost": total_thought_cost,
                    "billable_output_cost": total_output_cost + total_thought_cost,
                    "total_cost": total_cost,
                    "currency": "USD",
                },
                "models": models,
            }
        )

        diagnostic_totals: Dict[str, int] = {}
        for stats in diagnostics_by_agent.values():
            for key, value in stats.items():
                diagnostic_totals[key] = diagnostic_totals.get(key, 0) + int(value)
        token_data["diagnostics"] = {
            "agents": diagnostics_by_agent,
            "totals": diagnostic_totals,
            "events": diagnostic_events_by_agent,
        }

        # Calculate estimated total cost for each available model using the
        # overall token counts. This gives users a rough idea of what the
        # evolution would have cost if a different model had been selected.
        from idea.config import LLM_MODELS

        estimates = {}
        for model in LLM_MODELS:
            model_id = model['id']
            model_name = model.get('name', model_id)
            cost_estimate = estimate_generation_token_cost(
                model_id,
                input_tokens=total_input,
                output_tokens=total_output,
                thought_tokens=total_thoughts,
                cached_input_tokens=total_cached_input,
                tool_input_tokens=total_tool_input,
            )
            estimates[model_id] = {'name': model_name, 'cost': cost_estimate["total_cost"]}

        token_data['estimates'] = estimates
        return token_data

    async def _store_embedding(self, idea_id: str, embedding: np.ndarray):
        """
        Store an embedding for an idea.

        Args:
            idea_id: Unique ID of the idea
            embedding: The embedding vector for the idea
        """
        if idea_id not in self.idea_embeddings:
            # New embedding - add to all storage structures
            self.idea_embeddings[idea_id] = embedding
            index = len(self.all_embeddings)
            self.all_embeddings.append(embedding)
            self.embedding_id_to_index[idea_id] = index
        else:
            # Update existing embedding
            index = self.embedding_id_to_index[idea_id]
            self.all_embeddings[index] = embedding
            self.idea_embeddings[idea_id] = embedding

    async def _remove_embedding(self, idea_id: str):
        """
        Remove an embedding for an idea.

        Args:
            idea_id: Unique ID of the idea to remove
        """
        if idea_id not in self.idea_embeddings:
            return

        # Get index of the embedding to remove
        remove_index = self.embedding_id_to_index[idea_id]

        # Remove from all storage structures
        del self.idea_embeddings[idea_id]
        self.all_embeddings.pop(remove_index)
        del self.embedding_id_to_index[idea_id]

        # Update indices for all embeddings after the removed one
        for other_id, other_index in self.embedding_id_to_index.items():
            if other_index > remove_index:
                self.embedding_id_to_index[other_id] = other_index - 1

    async def _compute_population_centroid(self) -> Optional[np.ndarray]:
        """
        Compute the centroid of all stored embeddings.

        Returns:
            Population centroid as numpy array, or None if no embeddings available
        """
        if not self.all_embeddings:
            return None

        import numpy as np
        return np.mean(self.all_embeddings, axis=0)

    async def _get_or_compute_embeddings_for_ideas(self, ideas: List[str]) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for ideas, computing them if not already stored.

        Args:
            ideas: List of idea dictionaries

        Returns:
            List of embeddings (or None for failed embeddings)
        """
        embeddings = []
        new_embeddings_needed = []
        new_embedding_indices = []

        # Check which embeddings we already have
        for i, idea in enumerate(ideas):
            idea_id = str(idea.get("id", "")) if isinstance(idea, dict) else ""
            if idea_id and idea_id in self.idea_embeddings:
                embeddings.append(self.idea_embeddings[idea_id])
            else:
                embeddings.append(None)  # Placeholder
                new_embeddings_needed.append(idea)
                new_embedding_indices.append(i)

        # Compute missing embeddings
        if new_embeddings_needed:
            print(f"Computing {len(new_embeddings_needed)} new embeddings...")
            texts = [self.diversity_calculator._get_idea_text(idea) for idea in new_embeddings_needed]
            new_embeddings = await self.diversity_calculator._get_embeddings_batch(texts)

            # Store and assign new embeddings
            for i, (idea, embedding) in enumerate(zip(new_embeddings_needed, new_embeddings)):
                original_index = new_embedding_indices[i]
                embeddings[original_index] = embedding

                # Store the embedding if it's valid
                if embedding is not None:
                    idea_id = str(idea.get("id", "")) if isinstance(idea, dict) else ""
                    if idea_id:
                        await self._store_embedding(idea_id, embedding)

        return embeddings

    @staticmethod
    def _update_running_stats(stats: Dict[str, float], values: List[float]) -> None:
        """Update Welford running stats for cross-generation metric normalization."""
        for value in values:
            count = stats["count"] + 1
            delta = value - stats["mean"]
            mean = stats["mean"] + (delta / count)
            delta2 = value - mean
            m2 = stats["m2"] + (delta * delta2)
            stats["count"] = count
            stats["mean"] = mean
            stats["m2"] = m2

    @staticmethod
    def _normalize_with_running_stats(value: float, stats: Dict[str, float]) -> float:
        """
        Normalize a value into [0, 1] using running z-score statistics.

        We clamp z-scores to [-3, 3] to prevent outliers from dominating.
        """
        count = int(stats.get("count", 0))
        m2 = float(stats.get("m2", 0.0))
        mean = float(stats.get("mean", 0.0))
        if count < 2 or m2 <= 0:
            return 0.5

        variance = m2 / max(1, count - 1)
        std = math.sqrt(max(variance, 1e-12))
        z = (value - mean) / std
        z = max(-3.0, min(3.0, z))
        return (z + 3.0) / 6.0

    @staticmethod
    def _empty_running_stats() -> Dict[str, float]:
        return {"count": 0, "mean": 0.0, "m2": 0.0}

    def _running_stats_for_metric(self, metric_key: str) -> Dict[str, float]:
        """Return persisted state for one configured evaluator metric."""

        stats = self.fitness_metric_stats.get(metric_key)
        if not isinstance(stats, dict):
            stats = self._empty_running_stats()
            self.fitness_metric_stats[metric_key] = stats
        return stats

    def _apply_stateful_metric_normalization(
        self,
        records: List[EvaluationRecord],
    ) -> None:
        """Populate running-z-score values for every configured metric.

        Normalization belongs to the fitness configuration, not to a specific
        evaluator implementation. This lets any injected computed metric use
        the same persisted normalization behavior as legacy Elo and diversity.
        """

        successful_statuses = {EvaluationStatus.OK, EvaluationStatus.CACHED}
        for metric in self.evaluation_config.fitness.metrics:
            if metric.normalization != MetricNormalization.RUNNING_ZSCORE:
                continue

            matching_records = [
                record
                for record in records
                if record.component_id == metric.component_id
                and record.status in successful_statuses
                and metric.metric_name in record.metrics
            ]
            if not matching_records:
                continue

            stats = self._running_stats_for_metric(metric.key)
            self._update_running_stats(
                stats,
                [float(record.metrics[metric.metric_name]) for record in matching_records],
            )
            for record in matching_records:
                normalized = dict(record.normalized_metrics)
                normalized[metric.metric_name] = self._normalize_with_running_stats(
                    float(record.metrics[metric.metric_name]),
                    stats,
                )
                record.normalized_metrics = normalized

    def _get_birth_generation(self, idea: Any) -> int:
        """Read persisted birth generation metadata (defaults to 0 for legacy ideas)."""
        if not isinstance(idea, dict):
            return 0
        raw = idea.get("birth_generation", 0)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    def _compute_replacement_count(self, population_size: int) -> int:
        """Compute how many child slots to create for the next generation."""
        if population_size <= 0:
            return 0

        replace_count = int(round(population_size * self.replacement_rate))
        if self.replacement_rate > 0 and replace_count == 0:
            replace_count = 1

        if self.replacement_rate < 1.0 and replace_count >= population_size:
            replace_count = population_size - 1

        return max(0, min(population_size, replace_count))

    def _compute_island_sizes(self, population_size: Optional[int] = None) -> List[int]:
        """Split the total population into stable island sizes."""
        size = self.pop_size if population_size is None else int(population_size or 0)
        if size <= 0:
            return []

        count = max(1, min(int(getattr(self, "island_count", 2) or 2), size))
        base_size = size // count
        remainder = size % count
        return [
            base_size + (1 if island_id < remainder else 0)
            for island_id in range(count)
        ]

    def _initial_island_ids(self, population_size: Optional[int] = None) -> List[int]:
        """Return island assignments in deterministic population order."""
        island_ids: List[int] = []
        for island_id, island_size in enumerate(self._compute_island_sizes(population_size)):
            island_ids.extend([island_id] * island_size)
        return island_ids

    def _get_idea_island_id(self, idea: Any, *, fallback: int = 0) -> int:
        """Read normalized island metadata from an idea payload."""
        count = max(1, int(getattr(self, "island_count", 2) or 2))
        if isinstance(idea, dict):
            try:
                return max(0, min(count - 1, int(idea.get("island_id", fallback))))
            except (TypeError, ValueError):
                pass
        return max(0, min(count - 1, int(fallback or 0)))

    def _set_idea_island_id(self, idea: Any, island_id: int) -> Any:
        """Attach island metadata when the idea payload is mutable."""
        if isinstance(idea, dict):
            idea["island_id"] = self._get_idea_island_id(
                {"island_id": island_id},
                fallback=0,
            )
        return idea

    def _population_island_entries(
        self,
        population: Optional[List[Any]] = None,
        *,
        assign_missing: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """Group population entries by island while preserving global indices."""
        current_population = self.population if population is None else population
        island_count = max(1, min(int(getattr(self, "island_count", 2) or 2), max(1, len(current_population))))
        fallback_ids = self._initial_island_ids(len(current_population))
        groups: List[List[Dict[str, Any]]] = [[] for _ in range(island_count)]

        for global_idx, idea in enumerate(current_population):
            fallback = fallback_ids[global_idx] if global_idx < len(fallback_ids) else 0
            island_id = self._get_idea_island_id(idea, fallback=fallback)
            if assign_missing:
                self._set_idea_island_id(idea, island_id)
            groups[island_id].append({"global_index": global_idx, "idea": idea})

        return groups

    async def _calculate_population_diversity_scores(
        self, population: List[Any]
    ) -> Dict[int, float]:
        """
        Compute per-idea diversity as distance from the current population centroid.

        Returns:
            Dict mapping population index -> raw diversity distance.
        """
        if not population:
            return {}

        default_scores = {idx: 0.0 for idx in range(len(population))}
        if not self.diversity_calculator.is_enabled():
            return default_scores

        embeddings = await self._get_or_compute_embeddings_for_ideas(population)

        valid_embeddings = []
        valid_indices = []
        for idx, embedding in enumerate(embeddings):
            if embedding is not None:
                valid_embeddings.append(embedding)
                valid_indices.append(idx)

        if not valid_embeddings:
            return default_scores

        centroid = await self._compute_population_centroid()
        if centroid is None:
            centroid = np.mean(valid_embeddings, axis=0)
        diversity_scores = default_scores.copy()
        for emb_idx, pop_idx in enumerate(valid_indices):
            embedding = valid_embeddings[emb_idx]
            distance = float(np.sqrt(np.sum((embedding - centroid) ** 2)))
            diversity_scores[pop_idx] = distance

        return diversity_scores

    async def _prepare_population_for_judging(
        self,
        population: List[Any],
        *,
        evaluation_generation: Optional[int] = None,
    ) -> EvaluationBatch:
        """Run configured producer/computed components before the pairwise judge.

        Components execute in their declared suite order.  This guarantees,
        for example, that an artifact producer precedes a computed scorer that
        requires its output, and that the same enriched candidate object then
        reaches the existing pairwise judge. Numeric records are retained for
        subsequent fitness aggregation instead of recomputed.
        """

        if self.unsupported_evaluation_components:
            raise RuntimeError(
                "evaluation execution backends are not configured for: "
                + ", ".join(self.unsupported_evaluation_components)
            )

        generation = (
            max(0, int(evaluation_generation))
            if evaluation_generation is not None
            else max(0, int(getattr(self, "current_generation", 0) or 0))
        )
        records: List[EvaluationRecord] = []
        artifacts_by_id = {}
        for evaluator, runner in self.pre_judge_evaluators:
            result = await evaluator.evaluate_population(
                population,
                runner,
                evaluation_generation=generation,
                should_stop=lambda: bool(self.stop_requested) or self.check_budget(),
            )
            records.extend(result.batch.records)
            for artifact in result.batch.artifacts:
                existing = artifacts_by_id.get(artifact.id)
                if existing is not None and existing != artifact:
                    raise RuntimeError(
                        f"pre-judge artifact id collision for {artifact.id!r}"
                    )
                artifacts_by_id[artifact.id] = artifact

        batch = EvaluationBatch(
            generation=generation,
            records=records,
            artifacts=list(artifacts_by_id.values()),
            metadata={"phase": "pre_judge"},
        )
        self.pre_judge_evaluation_batch = batch
        self._pre_judge_candidate_ids = tuple(
            candidate_identifier(candidate, index)
            for index, candidate in enumerate(population)
        )
        return batch

    async def _score_population_fitness(
        self, population: List[Any], ranks: Dict[int, float]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Evaluate and aggregate fitness while preserving legacy score aliases.

        The default suite remains equivalent to the historical formula:
        ``alpha * elo_norm + (1 - alpha) * diversity_norm``. Typed
        evaluation records make the inputs and weighting extensible without
        changing the survivor/parent-selection contract in one step.
        """
        if not population:
            return {}

        candidate_ids = [
            candidate_identifier(candidate, index)
            for index, candidate in enumerate(population)
        ]
        generation = max(0, int(getattr(self, "current_generation", 0) or 0))
        if (
            isinstance(self.pre_judge_evaluation_batch, EvaluationBatch)
            and self.pre_judge_evaluation_batch.generation is not None
            and self._pre_judge_candidate_ids == tuple(candidate_ids)
        ):
            generation = self.pre_judge_evaluation_batch.generation
        records: List[EvaluationRecord] = []

        if self.unsupported_evaluation_components:
            raise RuntimeError(
                "evaluation execution backends are not configured for: "
                + ", ".join(self.unsupported_evaluation_components)
            )

        prepared_batch = self.pre_judge_evaluation_batch
        prepared_ids = self._pre_judge_candidate_ids
        if self.pre_judge_evaluators and not (
            isinstance(prepared_batch, EvaluationBatch)
            and prepared_batch.generation == generation
            and prepared_ids == tuple(candidate_ids)
        ):
            prepared_batch = await self._prepare_population_for_judging(
                population,
                evaluation_generation=generation,
            )
        if isinstance(prepared_batch, EvaluationBatch):
            records.extend(prepared_batch.records)
        prepared_artifacts = (
            list(prepared_batch.artifacts)
            if isinstance(prepared_batch, EvaluationBatch)
            else []
        )
        self.pre_judge_evaluation_batch = None
        self._pre_judge_candidate_ids = None

        prior_batch = self.last_evaluation_batch
        quality_scores: Dict[int, float] = {}
        quality_records: Dict[str, EvaluationRecord] = {}
        if self.quality_evaluator is not None:
            if isinstance(prior_batch, EvaluationBatch):
                quality_records = {
                    record.candidate_id: record
                    for record in prior_batch.records
                    if record.component_id == self.quality_evaluator.config.id
                }
            for index in range(len(population)):
                try:
                    score = float(ranks.get(index))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(score):
                    quality_scores[index] = score

        diversity_scores: Dict[int, float] = {}
        diversity_records: Dict[int, EvaluationRecord] = {}
        if self.diversity_evaluator is not None:
            diversity_result = await self.diversity_evaluator.evaluate_population(
                population,
                self._calculate_population_diversity_scores,
                evaluation_generation=generation,
            )
            diversity_scores = diversity_result.scores
            diversity_records = {
                index: record
                for index, record in enumerate(diversity_result.batch.records)
            }

        if self.quality_evaluator is not None:
            quality_config = self.quality_evaluator.config
            for index, candidate in enumerate(population):
                if index in quality_scores:
                    record = EvaluationRecord(
                        candidate_id=candidate_ids[index],
                        component_id=quality_config.id,
                        component_kind=quality_config.kind,
                        component_version=quality_config.component_version,
                        scope=quality_config.scope,
                        generation=generation,
                        status=EvaluationStatus.OK,
                        metrics={quality_config.metric_name: quality_scores[index]},
                        metadata={
                            "kind": quality_config.kind.value,
                            "generation": generation,
                        },
                    )
                else:
                    prior_record = quality_records.get(candidate_ids[index])
                    record = prior_record or EvaluationRecord(
                        candidate_id=candidate_ids[index],
                        component_id=quality_config.id,
                        component_kind=quality_config.kind,
                        component_version=quality_config.component_version,
                        scope=quality_config.scope,
                        generation=generation,
                        status=EvaluationStatus.FAILED,
                        error="pairwise evaluator did not return a finite rating",
                        metadata={
                            "kind": quality_config.kind.value,
                            "generation": generation,
                        },
                    )
                records.append(record)

        if self.diversity_evaluator is not None:
            diversity_config = self.diversity_evaluator.config
            for index, candidate in enumerate(population):
                if index in diversity_scores:
                    record = EvaluationRecord(
                        candidate_id=candidate_ids[index],
                        component_id=diversity_config.id,
                        component_kind=diversity_config.kind,
                        component_version=diversity_config.component_version,
                        scope=diversity_config.scope,
                        generation=generation,
                        status=EvaluationStatus.OK,
                        metrics={diversity_config.metric_name: diversity_scores[index]},
                        metadata={
                            "kind": diversity_config.kind.value,
                            "generation": generation,
                        },
                    )
                else:
                    record = diversity_records.get(index) or EvaluationRecord(
                        candidate_id=candidate_ids[index],
                        component_id=diversity_config.id,
                        component_kind=diversity_config.kind,
                        component_version=diversity_config.component_version,
                        scope=diversity_config.scope,
                        generation=generation,
                        status=EvaluationStatus.FAILED,
                        error="diversity evaluator did not return a finite score",
                        metadata={
                            "kind": diversity_config.kind.value,
                            "generation": generation,
                        },
                    )
                records.append(record)

        self._apply_stateful_metric_normalization(records)
        records_by_candidate: Dict[str, List[EvaluationRecord]] = {}
        for record in records:
            records_by_candidate.setdefault(record.candidate_id, []).append(record)
        for index, candidate in enumerate(population):
            for record in records_by_candidate.get(candidate_ids[index], []):
                attach_evaluation_record(candidate, record)

        batch = EvaluationBatch(
            generation=generation,
            records=records,
            artifacts=prepared_artifacts,
            metadata={"suite": self.evaluation_config.model_dump(mode="json")},
        )
        self.last_evaluation_batch = batch
        aggregated = self.fitness_aggregator.aggregate(
            batch,
            candidate_ids=candidate_ids,
        )

        records_by_key = {
            (record.candidate_id, record.component_id): record
            for record in records
        }
        fitness_map: Dict[int, Dict[str, Any]] = {}
        for index, candidate_id in enumerate(candidate_ids):
            fitness_score = aggregated[candidate_id]
            try:
                elo = float(quality_scores.get(index, ranks.get(index, 1500.0)))
            except (TypeError, ValueError):
                elo = 1500.0
            diversity = float(diversity_scores.get(index, 0.0))

            elo_norm = 0.5
            if self.quality_evaluator is not None:
                config = self.quality_evaluator.config
                metric_key = f"{config.id}.{config.metric_name}"
                record = records_by_key.get((candidate_id, config.id))
                contribution = fitness_score.components.get(metric_key)
                if record and config.metric_name in record.normalized_metrics:
                    elo_norm = float(record.normalized_metrics[config.metric_name])
                elif contribution and contribution.score_value is not None:
                    elo_norm = float(contribution.score_value)

            diversity_norm = 0.5
            if self.diversity_evaluator is not None:
                config = self.diversity_evaluator.config
                metric_key = f"{config.id}.{config.metric_name}"
                record = records_by_key.get((candidate_id, config.id))
                contribution = fitness_score.components.get(metric_key)
                if record and config.metric_name in record.normalized_metrics:
                    diversity_norm = float(
                        record.normalized_metrics[config.metric_name]
                    )
                elif contribution and contribution.score_value is not None:
                    diversity_norm = float(contribution.score_value)
            fitness = (
                float(fitness_score.score)
                if fitness_score.score is not None
                else 0.5
            )
            fitness_map[index] = {
                "elo": elo,
                "diversity": diversity,
                "elo_norm": elo_norm,
                "diversity_norm": diversity_norm,
                "fitness": fitness,
                "components": {
                    key: value.model_dump(mode="json")
                    for key, value in fitness_score.components.items()
                },
            }

        return fitness_map

    def _score_survival_with_age_decay(
        self, population: List[Any], fitness_map: Dict[int, Dict[str, Any]], target_generation: int
    ) -> Dict[int, Dict[str, float]]:
        """
        Combine hybrid fitness with age decay to produce survivor selection weights.
        """
        survival_scores: Dict[int, Dict[str, float]] = {}
        for idx, data in fitness_map.items():
            if idx >= len(population):
                continue

            birth_generation = self._get_birth_generation(population[idx])
            age = max(0, int(target_generation) - birth_generation)
            decay = self.age_decay_floor + (
                (1.0 - self.age_decay_floor) * math.exp(-self.age_decay_rate * age)
            )
            score = float(data.get("fitness", 0.0)) * decay
            survival_scores[idx] = {
                **data,
                "age": float(age),
                "age_decay": decay,
                "survival_score": score,
            }
        return survival_scores

    def _select_survivor_indices(
        self, survival_scores: Dict[int, Dict[str, float]], survivor_count: int
    ) -> List[int]:
        """Weighted sampling without replacement for survivor selection."""
        if survivor_count <= 0 or not survival_scores:
            return []

        available = sorted(survival_scores.keys())
        if not available:
            return []

        weights = np.array(
            [max(0.0, float(survival_scores[idx].get("survival_score", 0.0))) for idx in available],
            dtype=float,
        )
        if float(weights.sum()) <= 0:
            weights = np.ones_like(weights, dtype=float)

        selected: List[int] = []
        draw_count = min(survivor_count, len(available))
        for _ in range(draw_count):
            probabilities = weights / float(weights.sum())
            chosen_pos = int(self.random_choice(len(available), p=probabilities))
            selected.append(available.pop(chosen_pos))
            weights = np.delete(weights, chosen_pos)
            if weights.size > 0 and float(weights.sum()) <= 0:
                weights = np.ones_like(weights, dtype=float)

        return selected

    def _allocate_parent_slots(self, ranks, ideas_to_breed):
        """
        Allocate parent slots based on hybrid fitness scores with caps to prevent convergence.

        Args:
            ranks: Dict mapping idea indices to fitness score
            ideas_to_breed: Number of children to produce (determines total parent slots needed)

        Returns:
            Dict mapping idea indices to number of parent slots allocated
        """
        if not ranks or ideas_to_breed <= 0:
            return {}

        # Sort ideas by score (higher fitness = better rank)
        sorted_ideas = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        num_ideas = len(sorted_ideas)

        # Total parent slots needed (2 parents per child)
        total_slots = ideas_to_breed * self.breeder.parent_count

        # Define rank-based slot caps that scale with population size
        # These caps ensure diversity while still rewarding good performance
        if num_ideas <= 3:
            # Small population: more equal distribution
            caps = [max(1, total_slots // 2), max(1, total_slots // 3), max(1, total_slots // 4)]
        elif num_ideas <= 5:
            # Medium population: moderate hierarchy
            caps = [max(1, total_slots // 3), max(1, total_slots // 4), max(1, total_slots // 5),
                   max(1, total_slots // 6), max(1, total_slots // 7)]
        else:
            # Large population: steeper hierarchy but still capped
            base_cap = max(1, total_slots // 4)
            caps = []
            for i in range(num_ideas):
                if i == 0:  # Winner
                    caps.append(min(base_cap, max(1, total_slots // 3)))
                elif i < 3:  # Top 3
                    caps.append(min(base_cap // 2, max(1, total_slots // 5)))
                elif i < num_ideas // 2:  # Top half
                    caps.append(max(1, total_slots // 8))
                else:  # Bottom half
                    caps.append(max(1, total_slots // 12))

        # Ensure caps don't exceed available ideas
        caps = caps[:num_ideas]

        # Allocate slots with caps
        allocation = {}
        remaining_slots = total_slots

        # First pass: allocate slots respecting caps
        for i, (idea_idx, _) in enumerate(sorted_ideas):
            if remaining_slots <= 0:
                break

            if i < len(caps):
                slots_to_allocate = min(caps[i], remaining_slots)
            else:
                # For ideas beyond cap list, give minimal allocation
                slots_to_allocate = min(1, remaining_slots)

            allocation[idea_idx] = slots_to_allocate
            remaining_slots -= slots_to_allocate

        # Second pass: distribute remaining slots if any, starting from top
        if remaining_slots > 0:
            for i, (idea_idx, _) in enumerate(sorted_ideas):
                if remaining_slots <= 0:
                    break

                # Add one more slot if under cap
                current_cap = caps[i] if i < len(caps) else 1
                if allocation.get(idea_idx, 0) < current_cap:
                    allocation[idea_idx] = allocation.get(idea_idx, 0) + 1
                    remaining_slots -= 1

        # Log allocation for transparency
        print(f"Parent slot allocation for {ideas_to_breed} children ({total_slots} slots):")
        for i, (idea_idx, score) in enumerate(sorted_ideas):
            slots = allocation.get(idea_idx, 0)
            print(f"  Rank {i+1} (fitness {score:.3f}): {slots} slots")

        return allocation

    def _select_parents_from_slots(self, parent_slots, available_indices):
        """
        Select parents randomly from allocated slots.

        Args:
            parent_slots: Dict mapping idea indices to number of slots
            available_indices: List of available idea indices

        Returns:
            List of parent indices for breeding
        """
        if not parent_slots:
            return []

        # Create a weighted pool based on allocated slots
        parent_pool = []
        for idea_idx, slots in parent_slots.items():
            if idea_idx in available_indices:
                parent_pool.extend([idea_idx] * slots)

        if len(parent_pool) < self.breeder.parent_count:
            # Fallback: if not enough parents in pool, use available indices
            print(f"Warning: Only {len(parent_pool)} parents in pool, need {self.breeder.parent_count}")
            return self.random_choice(
                available_indices,
                size=min(self.breeder.parent_count, len(available_indices)),
                replace=False,
            ).tolist()

        # Simple random selection without replacement
        selected_parents = []
        pool_copy = parent_pool.copy()

        for _ in range(self.breeder.parent_count):
            if not pool_copy:
                break

            # Select random parent from pool
            selected_idx = self.random_choice(len(pool_copy))
            parent_idx = pool_copy.pop(selected_idx)

            # Avoid selecting the same parent twice for this breeding
            if parent_idx not in selected_parents:
                selected_parents.append(parent_idx)
            else:
                # If we selected a duplicate, try to find a different one
                available_alternatives = [p for p in set(pool_copy) if p not in selected_parents]
                if available_alternatives:
                    alternative = self.random_choice(available_alternatives)
                    selected_parents.append(alternative)
                    # Remove the alternative from pool
                    pool_copy = [p for p in pool_copy if p != alternative]
                else:
                    # If no alternatives, allow the duplicate (shouldn't happen often)
                    selected_parents.append(parent_idx)

        return selected_parents
