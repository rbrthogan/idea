from typing import List, Dict, Any, Callable, Awaitable, Optional
import random
import secrets
import threading
import numpy as np
import asyncio
import uuid
import math
import json
from pathlib import Path
from datetime import datetime
from idea.models import Idea
from idea.config import DEFAULT_CREATIVE_TEMP, DEFAULT_TOP_P, PROGRESS_JITTER_MAX_SECONDS
from idea.llm import Ideator, Formatter, Critic, Breeder, Oracle
from idea.prompts.loader import list_available_templates, get_prompts, get_prompts_from_dict
from idea.diversity import DiversityCalculator
from idea import database as db

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
        max_budget: Optional[float] = None,
        mutation_rate: float = 0.2,
        seed_context_pool_size: Optional[int] = None,
        replacement_rate: float = 0.5,
        fitness_alpha: float = 0.7,
        age_decay_rate: float = 0.25,
        age_decay_floor: float = 0.35,
        context_novelty_threshold: float = 0.8,
        context_novelty_max_attempts: int = 2,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,  # Required for Firestore storage
        template_data: Optional[Dict[str, Any]] = None,  # Custom template data from Firestore
    ):
        self.user_id = user_id  # User ID for scoped storage
        self.idea_type = idea_type or get_default_template_id()
        self.template_data = template_data  # Store custom template data for LLM agents
        self._template_prompt_wrapper = None
        self.pop_size = pop_size
        self.generations = generations
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
        self.random_seed = int(random_seed) if random_seed is not None else int(secrets.randbits(64))
        self._py_rng = random.Random(self.random_seed)
        self._np_rng = np.random.default_rng(self.random_seed)
        self._py_rng_lock = threading.Lock()
        self._np_rng_lock = threading.Lock()
        self.population: List[Idea] = []
        # TODO: make this configurable with a dropdown list for each LLM type using the following models:
        # gemini-1.5-flash, gemini-2.0-flash-exp, gemini-2.0-flash-thinking-exp-01-21

        # Initialize LLM components with appropriate temperatures
        print(f"Initializing agents with creative_temp={creative_temp}, top_p={top_p}, thinking_budget={thinking_budget}")

        self.ideator = Ideator(
            provider="google_generative_ai",
            model_name=model_type,
            temperature=creative_temp,
            top_p=top_p,
            thinking_budget=thinking_budget,
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
            api_key=api_key,
            random_seed=self._random_randbits(64),
        )
        self.breeder = Breeder(
            provider="google_generative_ai",
            model_name=model_type,
            temperature=creative_temp,
            top_p=top_p,
            thinking_budget=thinking_budget,
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
        self.specific_prompts = []  # List of specific prompts generated from contexts (translation layer)
        self.breeding_prompts = []  # List of lists: breeding prompts for each generation (empty for gen 0)
        self.tournament_history = []  # List of tournament details per generation

        # Initialize diversity calculator
        self.diversity_calculator = DiversityCalculator(api_key=api_key)
        self.diversity_history = []  # List of diversity metrics for each generation

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

        # Concurrency control
        self.concurrency_limit = 10
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)

        # Store configuration for checkpointing/resuming
        self.model_type = model_type
        self.creative_temp = creative_temp
        self.top_p = top_p
        self.mutation_rate = mutation_rate
        self.replacement_rate = max(0.0, min(1.0, float(replacement_rate)))
        self.fitness_alpha = max(0.0, min(1.0, float(fitness_alpha)))
        self.age_decay_rate = max(0.0, float(age_decay_rate))
        self.age_decay_floor = max(0.0, min(1.0, float(age_decay_floor)))

        # Running normalization state (persisted) to keep fitness scores comparable across generations.
        self.fitness_elo_stats = {"count": 0, "mean": 0.0, "m2": 0.0}
        self.fitness_diversity_stats = {"count": 0, "mean": 0.0, "m2": 0.0}

        # Evolution identity and tracking
        self.evolution_id = None  # Unique ID for this evolution (UUID)
        self.evolution_name = None  # Human-readable name
        self.created_at = None  # ISO timestamp when evolution started
        self.updated_at = None  # ISO timestamp of last update

        # Legacy checkpoint tracking (for backwards compatibility)
        self.checkpoint_id = None  # Set when evolution starts
        self.current_generation = 0  # Tracks which generation we're on
        self.checkpoint_callback = None  # Callback for saving checkpoints
        self._sync_typed_state_from_attrs()

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
        from idea.evolution_types import EvolutionConfig, EvolutionIdentity, EvolutionRuntimeState

        self.config_state = EvolutionConfig(
            idea_type=self.idea_type,
            pop_size=self.pop_size,
            generations=self.generations,
            model_type=self.model_type,
            creative_temp=self.creative_temp,
            top_p=self.top_p,
            tournament_rounds=self.tournament_rounds,
            tournament_count=self.tournament_count,
            full_tournament_rounds=self.full_tournament_rounds,
            thinking_budget=self.thinking_budget,
            max_budget=self.max_budget,
            mutation_rate=self.mutation_rate,
            seed_context_pool_size=self.seed_context_pool_size,
            replacement_rate=self.replacement_rate,
            fitness_alpha=self.fitness_alpha,
            age_decay_rate=self.age_decay_rate,
            age_decay_floor=self.age_decay_floor,
        )
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

    def initialize_evolution(self, name: str = None):
        """Initialize a new evolution with ID, name, and timestamps."""
        self.evolution_id = str(uuid.uuid4())
        self.evolution_name = name or self.generate_default_name()
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
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
    ) -> Optional['EvolutionEngine']:
        from idea.evolution_persistence import EvolutionRepository

        return await EvolutionRepository.load_evolution_for_user(
            cls,
            user_id,
            evolution_id,
            api_key=api_key,
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
    def _load_from_file(cls, file_path: Path, api_key: Optional[str] = None) -> Optional['EvolutionEngine']:
        from idea.evolution_persistence import EvolutionRepository

        return EvolutionRepository.load_from_file(cls, file_path, api_key=api_key)

    @classmethod
    def _restore_from_state(
        cls,
        state: Dict[str, Any],
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> 'EvolutionEngine':
        from idea.evolution_persistence import EvolutionRepository

        return EvolutionRepository.restore_from_state(
            cls,
            state,
            api_key=api_key,
            user_id=user_id,
        )

    @classmethod
    def load_checkpoint(cls, checkpoint_id: str, api_key: Optional[str] = None) -> Optional['EvolutionEngine']:
        from idea.evolution_persistence import EvolutionRepository

        return EvolutionRepository.load_checkpoint(cls, checkpoint_id, api_key=api_key)

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
        # Get input and output tokens from each component
        ideator_input = getattr(self.ideator, 'input_token_count', 0)
        ideator_output = getattr(self.ideator, 'output_token_count', 0)
        formatter_input = getattr(self.formatter, 'input_token_count', 0)
        formatter_output = getattr(self.formatter, 'output_token_count', 0)
        critic_input = getattr(self.critic, 'input_token_count', 0)
        critic_output = getattr(self.critic, 'output_token_count', 0)
        breeder_input = getattr(self.breeder, 'input_token_count', 0)
        breeder_output = getattr(self.breeder, 'output_token_count', 0)

        oracle_input = getattr(self.oracle, 'input_token_count', 0)
        oracle_output = getattr(self.oracle, 'output_token_count', 0)

        # Calculate totals
        total_input = ideator_input + formatter_input + critic_input + breeder_input + oracle_input
        total_output = ideator_output + formatter_output + critic_output + breeder_output + oracle_output
        total = total_input + total_output

        # Get pricing information from config
        from idea.config import model_prices_per_million_tokens

        # Get model names for each agent
        ideator_model = getattr(self.ideator, 'model_name', 'gemini-2.0-flash')
        formatter_model = getattr(self.formatter, 'model_name', 'gemini-2.0-flash')
        critic_model = getattr(self.critic, 'model_name', 'gemini-2.0-flash')
        breeder_model = getattr(self.breeder, 'model_name', 'gemini-2.0-flash')
        oracle_model = getattr(self.oracle, 'model_name', 'gemini-2.0-flash')

        # Default pricing if model not found in config
        default_price = {"input": 0.1, "output": 0.4}

        # Get pricing for each model
        ideator_pricing = model_prices_per_million_tokens.get(ideator_model, default_price)
        formatter_pricing = model_prices_per_million_tokens.get(formatter_model, default_price)
        critic_pricing = model_prices_per_million_tokens.get(critic_model, default_price)
        breeder_pricing = model_prices_per_million_tokens.get(breeder_model, default_price)
        oracle_pricing = model_prices_per_million_tokens.get(oracle_model, default_price)

        # Calculate cost for each component
        ideator_input_cost = (ideator_pricing["input"] * ideator_input) / 1_000_000
        ideator_output_cost = (ideator_pricing["output"] * ideator_output) / 1_000_000
        formatter_input_cost = (formatter_pricing["input"] * formatter_input) / 1_000_000
        formatter_output_cost = (formatter_pricing["output"] * formatter_output) / 1_000_000
        critic_input_cost = (critic_pricing["input"] * critic_input) / 1_000_000
        critic_output_cost = (critic_pricing["output"] * critic_output) / 1_000_000
        breeder_input_cost = (breeder_pricing["input"] * breeder_input) / 1_000_000
        breeder_output_cost = (breeder_pricing["output"] * breeder_output) / 1_000_000
        oracle_input_cost = (oracle_pricing["input"] * oracle_input) / 1_000_000
        oracle_output_cost = (oracle_pricing["output"] * oracle_output) / 1_000_000

        # Calculate total costs
        total_input_cost = ideator_input_cost + formatter_input_cost + critic_input_cost + breeder_input_cost + oracle_input_cost
        total_output_cost = ideator_output_cost + formatter_output_cost + critic_output_cost + breeder_output_cost + oracle_output_cost
        total_cost = total_input_cost + total_output_cost

        token_data = {
            'ideator': {
                'total': self.ideator.total_token_count,
                'input': ideator_input,
                'output': ideator_output,
                'model': ideator_model,
                'cost': ideator_input_cost + ideator_output_cost
            },
            'formatter': {
                'total': self.formatter.total_token_count,
                'input': formatter_input,
                'output': formatter_output,
                'model': formatter_model,
                'cost': formatter_input_cost + formatter_output_cost
            },
            'critic': {
                'total': self.critic.total_token_count,
                'input': critic_input,
                'output': critic_output,
                'model': critic_model,
                'cost': critic_input_cost + critic_output_cost
            },
            'breeder': {
                'total': self.breeder.total_token_count,
                'input': breeder_input,
                'output': breeder_output,
                'model': breeder_model,
                'cost': breeder_input_cost + breeder_output_cost
            },
            'oracle': {
                'total': self.oracle.total_token_count,
                'input': oracle_input,
                'output': oracle_output,
                'model': oracle_model,
                'cost': oracle_input_cost + oracle_output_cost
            },
            'total': total,
            'total_input': total_input,
            'total_output': total_output,
            'cost': {
                'input_cost': total_input_cost,
                'output_cost': total_output_cost,
                'total_cost': total_cost,
                'currency': 'USD'
            },
            'models': {
                'ideator': ideator_model,
                'formatter': formatter_model,
                'critic': critic_model,
                'breeder': breeder_model,
                'oracle': oracle_model
            }
        }

        diagnostics_by_agent = {
            "ideator": getattr(self.ideator, "get_diagnostics", lambda: {})(),
            "formatter": getattr(self.formatter, "get_diagnostics", lambda: {})(),
            "critic": getattr(self.critic, "get_diagnostics", lambda: {})(),
            "breeder": getattr(self.breeder, "get_diagnostics", lambda: {})(),
            "oracle": getattr(self.oracle, "get_diagnostics", lambda: {})(),
        }
        diagnostic_events_by_agent = {
            "ideator": getattr(self.ideator, "get_diagnostic_events", lambda: [])(),
            "formatter": getattr(self.formatter, "get_diagnostic_events", lambda: [])(),
            "critic": getattr(self.critic, "get_diagnostic_events", lambda: [])(),
            "breeder": getattr(self.breeder, "get_diagnostic_events", lambda: [])(),
            "oracle": getattr(self.oracle, "get_diagnostic_events", lambda: [])(),
        }
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
            pricing = model_prices_per_million_tokens.get(model_id, default_price)
            est_cost = (
                pricing['input'] * total_input / 1_000_000
                + pricing['output'] * total_output / 1_000_000
            )
            estimates[model_id] = {'name': model_name, 'cost': est_cost}

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
        if not self.diversity_calculator.is_enabled() or len(population) < 2:
            return default_scores

        embeddings = await self._get_or_compute_embeddings_for_ideas(population)

        valid_embeddings = []
        valid_indices = []
        for idx, embedding in enumerate(embeddings):
            if embedding is not None:
                valid_embeddings.append(embedding)
                valid_indices.append(idx)

        if len(valid_embeddings) < 2:
            return default_scores

        centroid = np.mean(valid_embeddings, axis=0)
        diversity_scores = default_scores.copy()
        for emb_idx, pop_idx in enumerate(valid_indices):
            embedding = valid_embeddings[emb_idx]
            distance = float(np.sqrt(np.sum((embedding - centroid) ** 2)))
            diversity_scores[pop_idx] = distance

        return diversity_scores

    async def _score_population_fitness(
        self, population: List[Any], ranks: Dict[int, float]
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute normalized hybrid fitness for the current population.

        fitness = alpha * elo_norm + (1 - alpha) * diversity_norm
        """
        if not population or not ranks:
            return {}

        valid_indices = [idx for idx in ranks.keys() if 0 <= idx < len(population)]
        if not valid_indices:
            return {}

        diversity_scores = await self._calculate_population_diversity_scores(population)
        elo_values = [float(ranks[idx]) for idx in valid_indices]
        diversity_values = [float(diversity_scores.get(idx, 0.0)) for idx in valid_indices]

        self._update_running_stats(self.fitness_elo_stats, elo_values)
        self._update_running_stats(self.fitness_diversity_stats, diversity_values)

        fitness_map: Dict[int, Dict[str, float]] = {}
        for idx in valid_indices:
            elo = float(ranks[idx])
            diversity = float(diversity_scores.get(idx, 0.0))
            elo_norm = self._normalize_with_running_stats(elo, self.fitness_elo_stats)
            diversity_norm = self._normalize_with_running_stats(
                diversity, self.fitness_diversity_stats
            )
            fitness = (self.fitness_alpha * elo_norm) + (
                (1.0 - self.fitness_alpha) * diversity_norm
            )
            fitness_map[idx] = {
                "elo": elo,
                "diversity": diversity,
                "elo_norm": elo_norm,
                "diversity_norm": diversity_norm,
                "fitness": fitness,
            }

        return fitness_map

    def _score_survival_with_age_decay(
        self, population: List[Any], fitness_map: Dict[int, Dict[str, float]], target_generation: int
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
