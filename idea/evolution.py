from typing import List, Dict, Any, Callable, Awaitable, Optional
import random
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
        tournament_rounds: int = 1,
        tournament_count: Optional[float] = None,
        full_tournament_rounds: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        max_budget: Optional[float] = None,
        mutation_rate: float = 0.2,
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
        self.api_key = api_key
        self.population: List[Idea] = []
        # TODO: make this configurable with a dropdown list for each LLM type using the following models:
        # gemini-1.5-flash, gemini-2.0-flash-exp, gemini-2.0-flash-thinking-exp-01-21

        # Initialize LLM components with appropriate temperatures
        print(f"Initializing agents with creative_temp={creative_temp}, top_p={top_p}, thinking_budget={thinking_budget}")

        self.ideator = Ideator(provider="google_generative_ai", model_name=model_type, temperature=creative_temp, top_p=top_p, thinking_budget=thinking_budget, api_key=api_key)

        # Always use 2.5 Flash for formatting as it has better instruction following for structured output
        # than 2.0 Flash or older models
        self.formatter = Formatter(provider="google_generative_ai", model_name="gemini-2.5-flash", api_key=api_key)

        critic_model_name = "gemini-2.5-flash" if model_type == "gemini-2.5-pro" else model_type
        self.critic = Critic(provider="google_generative_ai", model_name=critic_model_name, temperature=creative_temp, top_p=top_p, thinking_budget=thinking_budget, api_key=api_key)
        self.breeder = Breeder(provider="google_generative_ai", model_name=model_type, temperature=creative_temp, top_p=top_p, thinking_budget=thinking_budget, mutation_rate=mutation_rate, api_key=api_key)

        self.oracle = Oracle(provider="google_generative_ai", model_name=model_type, temperature=creative_temp, top_p=top_p, thinking_budget=thinking_budget, api_key=api_key)

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

        # Evolution identity and tracking
        self.evolution_id = None  # Unique ID for this evolution (UUID)
        self.evolution_name = None  # Human-readable name
        self.created_at = None  # ISO timestamp when evolution started
        self.updated_at = None  # ISO timestamp of last update

        # Legacy checkpoint tracking (for backwards compatibility)
        self.checkpoint_id = None  # Set when evolution starts
        self.current_generation = 0  # Tracks which generation we're on
        self.checkpoint_callback = None  # Callback for saving checkpoints

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
                        await asyncio.sleep(random.uniform(0, PROGRESS_JITTER_MAX_SECONDS))

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

    def reset_stop_state(self):
        """Reset the stop state for a new evolution"""
        self.stop_requested = False
        self.is_stopped = False

    def _get_template_prompts(self):
        """Resolve prompt wrapper for current template, preferring the engine-scoped custom template snapshot."""
        if self.template_data:
            if self._template_prompt_wrapper is None:
                self._template_prompt_wrapper = get_prompts_from_dict(self.template_data)
            return self._template_prompt_wrapper
        return get_prompts(self.idea_type)

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Serialize the current evolution state for checkpointing.
        Returns a dictionary that can be saved to JSON and later restored.
        """
        def serialize_idea(idea):
            """Convert an idea dict to a serializable format"""
            if not isinstance(idea, dict):
                return idea
            result = {}
            for key, value in idea.items():
                if isinstance(value, uuid.UUID):
                    result[key] = str(value)
                elif isinstance(value, list):
                    result[key] = [str(v) if isinstance(v, uuid.UUID) else v for v in value]
                elif hasattr(value, '__dict__'):
                    # Handle Idea objects
                    result[key] = {'title': getattr(value, 'title', None), 'content': getattr(value, 'content', '')}
                else:
                    result[key] = value
            return result

        state = {
            'checkpoint_id': self.checkpoint_id,
            'checkpoint_time': datetime.now().isoformat(),
            'status': 'paused' if self.stop_requested else 'in_progress',

            # Configuration
            'config': {
                'idea_type': self.idea_type,
                'pop_size': self.pop_size,
                'generations': self.generations,
                'model_type': self.model_type,
                'creative_temp': self.creative_temp,
                'top_p': self.top_p,
                'tournament_rounds': self.tournament_rounds,
                'tournament_count': self.tournament_count,
                'full_tournament_rounds': self.full_tournament_rounds,
                'thinking_budget': self.thinking_budget,
                'max_budget': self.max_budget,
                'mutation_rate': self.mutation_rate,
            },

            # Evolution state
            'current_generation': self.current_generation,
            'population': [serialize_idea(idea) for idea in self.population],
            'history': [[serialize_idea(idea) for idea in gen] for gen in self.history],
            'contexts': self.contexts,
            'specific_prompts': self.specific_prompts,
            'breeding_prompts': self.breeding_prompts,
            'tournament_history': self.tournament_history,
            'diversity_history': self.diversity_history,

            # Cost tracking
            'avg_idea_cost': self.avg_idea_cost,
            'avg_tournament_cost': self.avg_tournament_cost,

            # Token counts from agents
            'token_counts': self.get_total_token_count(),
        }
        if self.template_data:
            state['template_data'] = self.template_data
        return state

    def set_name(self, name: str):
        """Set the human-readable name for this evolution."""
        self.evolution_name = name
        self.updated_at = datetime.now().isoformat()

    def generate_default_name(self) -> str:
        """Generate a default name based on template and date."""
        if self.template_data and self.template_data.get("name"):
            template_name = self.template_data.get("name")
        else:
            # Get template display name
            try:
                templates = list_available_templates()
                template_info = templates.get(self.idea_type, {})
                template_name = template_info.get('name', self.idea_type)
            except Exception:
                template_name = self.idea_type or 'Evolution'

        # Get template display name
        date_str = datetime.now().strftime('%b %d, %Y')
        return f"{template_name} - {date_str}"

    def initialize_evolution(self, name: str = None):
        """Initialize a new evolution with ID, name, and timestamps."""
        self.evolution_id = str(uuid.uuid4())
        self.evolution_name = name or self.generate_default_name()
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        # Also set legacy checkpoint_id for compatibility
        self.checkpoint_id = self.evolution_id[:18].replace('-', '')

    async def save_checkpoint(self, status: str = 'in_progress') -> Optional[str]:
        """
        Save the current state to Firestore.
        Returns the evolution_id on success, None on failure.
        """
        try:
            if not self.user_id:
                print("❌ Cannot save evolution: user_id not set")
                return None

            # Initialize if this is a legacy evolution without an ID
            if not self.evolution_id:
                self.initialize_evolution()

            self.updated_at = datetime.now().isoformat()

            state = self.get_checkpoint_state()
            state['evolution_id'] = self.evolution_id
            state['name'] = self.evolution_name
            state['status'] = status
            state['created_at'] = self.created_at
            state['updated_at'] = self.updated_at

            # Save to Firestore
            await db.save_evolution(self.user_id, self.evolution_id, state)

            print(f"💾 Evolution saved to Firestore: {self.evolution_name} ({self.evolution_id})")
            return self.evolution_id

        except Exception as e:
            print(f"❌ Failed to save evolution: {e}")
            import traceback
            traceback.print_exc()
            return None

    @classmethod
    async def list_evolutions_for_user(cls, user_id: str) -> List[Dict[str, Any]]:
        """
        List all evolutions for a user from Firestore.
        Returns a list of evolution metadata.
        """
        try:
            evolutions = await db.list_evolutions(user_id)
            # Transform the data to match expected format
            result = []
            for data in evolutions:
                config = data.get('config', {})
                history = data.get('history', [])
                result.append({
                    'id': data.get('evolution_id') or data.get('id'),
                    'name': data.get('name', 'Unnamed'),
                    'status': data.get('status', 'unknown'),
                    'created_at': data.get('created_at'),
                    'updated_at': data.get('updated_at'),
                    'generation': data.get('current_generation', len(history)),
                    'total_generations': config.get('generations', len(history)),
                    'idea_type': config.get('idea_type', 'unknown'),
                    'model_type': config.get('model_type', 'unknown'),
                    'pop_size': config.get('pop_size', len(history[0]) if history else 0),
                    'total_ideas': sum(len(gen) for gen in history),
                })
            return result
        except Exception as e:
            print(f"Error listing evolutions: {e}")
            return []

    @classmethod
    async def list_checkpoints_for_user(cls, user_id: str) -> List[Dict[str, Any]]:
        """
        List all checkpoints for a user from Firestore.
        Returns a list of checkpoint metadata.
        """
        try:
            checkpoints = await db.list_checkpoints(user_id)
            result = []
            for data in checkpoints:
                config = data.get('config', {})
                result.append({
                    'id': data.get('checkpoint_id') or data.get('id'),
                    'time': data.get('checkpoint_time') or data.get('updated_at'),
                    'status': data.get('status', 'unknown'),
                    'generation': data.get('current_generation', 0),
                    'total_generations': config.get('generations', 0),
                    'idea_type': config.get('idea_type', 'unknown'),
                    'model_type': config.get('model_type', 'unknown'),
                    'pop_size': config.get('pop_size', 0),
                })
            return result
        except Exception as e:
            print(f"Error listing checkpoints: {e}")
            return []

    @classmethod
    async def load_evolution_for_user(cls, user_id: str, evolution_id: str, api_key: Optional[str] = None) -> Optional['EvolutionEngine']:
        """
        Load an evolution engine from Firestore.
        Returns a configured EvolutionEngine instance, or None on failure.
        """
        try:
            state = await db.get_evolution(user_id, evolution_id)
            if not state:
                print(f"❌ Evolution not found: {evolution_id}")
                return None

            config = state.get("config", {})
            idea_type = config.get("idea_type")
            if idea_type and not state.get("template_data"):
                templates = list_available_templates()
                is_valid_system_template = (
                    idea_type in templates and "error" not in templates.get(idea_type, {})
                )
                if not is_valid_system_template:
                    user_template = await db.get_user_template(user_id, idea_type)
                    if user_template:
                        state["template_data"] = user_template

            engine = cls._restore_from_state(state, api_key=api_key, user_id=user_id)
            return engine
        except Exception as e:
            print(f"❌ Failed to load evolution: {e}")
            import traceback
            traceback.print_exc()
            return None

    @classmethod
    async def rename_evolution_for_user(cls, user_id: str, evolution_id: str, new_name: str) -> bool:
        """Rename an evolution in Firestore."""
        try:
            result = await db.rename_evolution(user_id, evolution_id, new_name)
            if result:
                print(f"✅ Evolution renamed to: {new_name}")
            return result
        except Exception as e:
            print(f"❌ Failed to rename evolution: {e}")
            return False

    @classmethod
    async def delete_evolution_for_user(cls, user_id: str, evolution_id: str) -> bool:
        """Delete an evolution from Firestore."""
        try:
            result = await db.delete_evolution(user_id, evolution_id)
            if result:
                print(f"🗑️ Evolution deleted: {evolution_id}")
            return result
        except Exception as e:
            print(f"❌ Failed to delete evolution: {e}")
            return False

    @classmethod
    def _load_from_file(cls, file_path: Path, api_key: Optional[str] = None) -> Optional['EvolutionEngine']:
        """Load evolution engine from a file path."""
        try:
            with open(file_path) as f:
                state = json.load(f)

            return cls._restore_from_state(state, api_key=api_key)
        except Exception as e:
            print(f"❌ Failed to load from {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    @classmethod
    def _restore_from_state(cls, state: Dict[str, Any], api_key: Optional[str] = None, user_id: Optional[str] = None) -> 'EvolutionEngine':
        """Restore an EvolutionEngine from a state dictionary."""
        config = state.get('config', {})
        tournament_rounds = config.get('tournament_rounds', 1)
        full_tournament_rounds = config.get('full_tournament_rounds')
        tournament_count = config.get('tournament_count')

        # Create a new engine with the saved configuration
        engine = cls(
            idea_type=config.get('idea_type'),
            pop_size=config.get('pop_size', 5),
            generations=config.get('generations', 3),
            model_type=config.get('model_type', 'gemini-2.0-flash'),
            creative_temp=config.get('creative_temp', DEFAULT_CREATIVE_TEMP),
            top_p=config.get('top_p', DEFAULT_TOP_P),
            tournament_rounds=tournament_rounds,
            tournament_count=tournament_count,
            full_tournament_rounds=full_tournament_rounds,
            thinking_budget=config.get('thinking_budget'),
            max_budget=config.get('max_budget'),
            mutation_rate=config.get('mutation_rate', 0.2),
            api_key=api_key,
            user_id=user_id or state.get('user_id'),
            template_data=state.get('template_data'),
        )

        # Restore evolution identity
        engine.evolution_id = state.get('evolution_id')
        engine.evolution_name = state.get('name')
        engine.created_at = state.get('created_at')
        engine.updated_at = state.get('updated_at')

        # Restore legacy checkpoint ID for compatibility
        engine.checkpoint_id = state.get('checkpoint_id') or (engine.evolution_id[:18].replace('-', '') if engine.evolution_id else None)
        engine.current_generation = state.get('current_generation', 0)
        engine.contexts = state.get('contexts', [])
        engine.specific_prompts = state.get('specific_prompts', [])
        engine.breeding_prompts = state.get('breeding_prompts', [])
        engine.tournament_history = state.get('tournament_history', [])
        engine.diversity_history = state.get('diversity_history', [])
        engine.avg_idea_cost = state.get('avg_idea_cost', 0.0)
        engine.avg_tournament_cost = state.get('avg_tournament_cost', 0.0)

        # Restore population and history
        def deserialize_idea(idea_data):
            """Convert serialized idea back to proper format"""
            if not isinstance(idea_data, dict):
                return idea_data

            result = dict(idea_data)
            # Ensure ID is a UUID
            if 'id' in result and isinstance(result['id'], str):
                try:
                    result['id'] = uuid.UUID(result['id'])
                except ValueError:
                    result['id'] = uuid.uuid4()

            # Convert parent_ids back to UUIDs
            if 'parent_ids' in result:
                result['parent_ids'] = [
                    uuid.UUID(pid) if isinstance(pid, str) else pid
                    for pid in result['parent_ids']
                ]

            # Restore Idea object if present
            if 'idea' in result and isinstance(result['idea'], dict):
                result['idea'] = Idea(
                    title=result['idea'].get('title'),
                    content=result['idea'].get('content', '')
                )

            return result

        engine.population = [deserialize_idea(idea) for idea in state.get('population', [])]
        engine.history = [[deserialize_idea(idea) for idea in gen] for gen in state.get('history', [])]

        print(f"✅ Evolution loaded: {engine.evolution_name or 'unnamed'} (gen {engine.current_generation}/{engine.generations})")
        return engine

    @classmethod
    def load_checkpoint(cls, checkpoint_id: str, api_key: Optional[str] = None) -> Optional['EvolutionEngine']:
        """
        Load an evolution engine from a checkpoint (legacy or new format).
        Tries unified evolutions first, then legacy checkpoints.
        Returns a configured EvolutionEngine instance, or None on failure.
        """
        # First try unified evolutions directory
        evolution_path = EVOLUTIONS_DIR / f"{checkpoint_id}.json"
        if evolution_path.exists():
            return cls._load_from_file(evolution_path, api_key=api_key)

        # Fall back to legacy checkpoint directory
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{checkpoint_id}.json"
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return None

        # Use shared loading logic
        return cls._load_from_file(checkpoint_path, api_key=api_key)

    @classmethod
    def delete_checkpoint(cls, checkpoint_id: str) -> bool:
        """Delete a checkpoint file."""
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{checkpoint_id}.json"
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"🗑️ Checkpoint deleted: {checkpoint_path}")
                return True
            return False
        except Exception as e:
            print(f"❌ Failed to delete checkpoint: {e}")
            return False

    async def _complete_initial_seeding(
        self,
        progress_callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ):
        """
        Complete the initial seeding phase (Generation 0) if it was interrupted.
        This handles the case where evolution was stopped during initial population creation.
        """
        existing_count = len(self.population)
        needed_count = self.pop_size - existing_count
        print(f"🌱 Completing initial seeding: {existing_count} existing, need {needed_count} more")

        # Generate remaining seed ideas
        async def generate_single_seed():
            context_pool = await asyncio.to_thread(self.ideator.generate_context, self.idea_type)
            idea_text, specific_prompt = await asyncio.to_thread(
                self.ideator.generate_idea_from_context, context_pool, self.idea_type
            )
            return context_pool, {"id": uuid.uuid4(), "idea": idea_text, "parent_ids": []}, specific_prompt

        for i in range(needed_count):
            if self.stop_requested:
                print("Stop requested during seeding completion")
                return

            await progress_callback({
                "current_generation": 0,
                "total_generations": self.generations,
                "is_running": True,
                "status_message": f"Creating seed idea {existing_count + i + 1}/{self.pop_size}...",
                "progress": ((existing_count + i) / self.pop_size) * 50  # First half of gen 0
            })

            context_pool, idea, prompt = await generate_single_seed()
            self.contexts.append(context_pool)
            self.population.append(idea)
            self.specific_prompts.append(prompt)

        # Refine all ideas
        print("Refining initial population...")
        refined_population = []
        for i, idea in enumerate(self.population):
            if self.stop_requested:
                print("Stop requested during refinement")
                return

            await progress_callback({
                "current_generation": 0,
                "total_generations": self.generations,
                "is_running": True,
                "status_message": f"Refining idea {i + 1}/{len(self.population)}...",
                "progress": 50 + ((i + 1) / len(self.population)) * 50  # Second half of gen 0
            })

            refined_idea = await asyncio.to_thread(self.critic.refine, idea, self.idea_type)
            formatted_idea = await asyncio.to_thread(self.formatter.format_idea, refined_idea, self.idea_type)
            refined_population.append(formatted_idea)

        self.population = refined_population
        self.history = [self.population.copy()]

        # Calculate initial diversity
        await self._calculate_and_store_diversity()

        # Save checkpoint for completed generation 0
        self.current_generation = 1
        await self.save_checkpoint(status='in_progress')

        # Send update
        token_counts = self.get_total_token_count()
        await progress_callback({
            "current_generation": 0,
            "total_generations": self.generations,
            "is_running": True,
            "history": self.history,
            "contexts": self.contexts,
            "specific_prompts": self.specific_prompts,
            "progress": 100 / (self.generations + 1),
            "status_message": "Generation 0 complete!",
            "token_counts": token_counts,
            "diversity_history": self.diversity_history.copy() if self.diversity_history else []
        })

        print(f"✅ Initial seeding complete. Population size: {len(self.population)}")

    async def resume_evolution_with_updates(
        self,
        progress_callback: Callable[[Dict[str, Any]], Awaitable[None]],
        additional_generations: int = 0
    ):
        """
        Resume evolution from the current state.
        Can also be used to continue a completed evolution for more generations.

        Args:
            progress_callback: Async function that will be called with progress updates
            additional_generations: If > 0, add this many generations to the existing target
        """
        try:
            # Reset stop state
            self.reset_stop_state()

            # Optionally extend the number of generations
            if additional_generations > 0:
                self.generations += additional_generations
                print(f"📈 Extended evolution by {additional_generations} generations. New total: {self.generations}")

            start_gen = self.current_generation
            print(f"🔄 Resuming evolution from generation {start_gen}/{self.generations}")

            # Handle incomplete Generation 0 (initial seeding not complete)
            if start_gen == 0 and len(self.population) < self.pop_size:
                print(f"⚠️ Generation 0 incomplete ({len(self.population)}/{self.pop_size} ideas). Restarting seeding...")
                await self._complete_initial_seeding(progress_callback)
                # After seeding, we've completed generation 0
                start_gen = 1
                self.current_generation = 1

            # Send initial progress update with restored state
            token_counts = self.get_total_token_count()
            await progress_callback({
                "current_generation": start_gen,
                "total_generations": self.generations,
                "is_running": True,
                "is_resuming": True,
                "history": self.history,
                "contexts": self.contexts,
                "specific_prompts": self.specific_prompts,
                "breeding_prompts": self.breeding_prompts,
                "progress": (start_gen / self.generations) * 100 if self.generations > 0 else 0,
                "status_message": f"Resuming from generation {start_gen}...",
                "token_counts": token_counts,
                "diversity_history": self.diversity_history.copy() if self.diversity_history else []
            })

            # Calculate steps for progress tracking
            est_tournament_rounds = max(1, self.tournament_rounds)
            steps_per_gen = self.pop_size + est_tournament_rounds + 1
            remaining_gens = self.generations - start_gen
            total_steps = remaining_gens * steps_per_gen
            current_step = 0

            # Elite idea tracking for continuity
            elite_idea = None
            elite_breeding_prompt = None

            for gen in range(start_gen, self.generations):
                # Check for stop request
                if self.stop_requested:
                    self.is_stopped = True
                    self.current_generation = gen
                    print(f"Stop requested - evolution halted at generation {gen}")
                    checkpoint_path = await self.save_checkpoint(status='paused')
                    token_counts = self.get_total_token_count()
                    await progress_callback({
                        "current_generation": gen,
                        "total_generations": self.generations,
                        "is_running": False,
                        "is_stopped": True,
                        "is_resumable": True,
                        "checkpoint_id": self.checkpoint_id,
                        "history": self.history,
                        "contexts": self.contexts,
                        "specific_prompts": self.specific_prompts,
                        "breeding_prompts": self.breeding_prompts,
                        "progress": (gen / self.generations) * 100,
                        "stop_message": f"Evolution paused at generation {gen}. You can resume.",
                        "token_counts": token_counts,
                        "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                    })
                    return

                print(f"Starting generation {gen + 1}...")

                new_population = []
                generation_breeding_prompts = []
                random.shuffle(self.population)

                # Handle elite idea from previous generation
                elite_processed = False
                if elite_idea is not None:
                    print(f"🌟 Processing elite idea for generation {gen + 1}...")
                    refined_elite = await asyncio.to_thread(self.critic.refine, elite_idea, self.idea_type)
                    formatted_elite = await asyncio.to_thread(self.formatter.format_idea, refined_elite, self.idea_type)

                    if isinstance(formatted_elite, dict):
                        formatted_elite["elite_selected"] = True
                        formatted_elite["elite_source_id"] = elite_idea.get("id")
                        formatted_elite["elite_source_generation"] = gen
                    else:
                        formatted_elite = {
                            "id": uuid.uuid4(),
                            "idea": formatted_elite,
                            "parent_ids": [],
                            "elite_selected": True,
                            "elite_source_id": elite_idea.get("id"),
                            "elite_source_generation": gen
                        }

                    new_population.append(formatted_elite)
                    generation_breeding_prompts.append(elite_breeding_prompt)
                    elite_processed = True

                current_pop_size = len(self.population)
                ideas_to_breed = current_pop_size - (1 if elite_processed else 0)

                elite_idea = None
                elite_breeding_prompt = None

                # Run global Swiss tournament
                print(f"Running Swiss tournament across {len(self.population)} ideas for {self.tournament_rounds} rounds...")
                tournament_rounds_details: List[Dict[str, Any]] = []
                global_ranks = await asyncio.to_thread(
                    self.critic.get_tournament_ranks,
                    self.population,
                    self.idea_type,
                    self.tournament_rounds,
                    None,
                    tournament_rounds_details,
                    self.full_tournament_rounds,
                )
                self._set_tournament_history(gen + 1, tournament_rounds_details)

                if self.stop_requested:
                    continue  # Will be caught at top of next iteration

                # Allocate parent slots and breed
                global_parent_slots = self._allocate_parent_slots(global_ranks, ideas_to_breed)

                breeding_tasks_data = []
                for _ in range(ideas_to_breed):
                    if global_parent_slots:
                        parent_indices = self._select_parents_from_slots(global_parent_slots, list(global_ranks.keys()))
                        parent_ideas = [self.population[idx] for idx in parent_indices]
                    else:
                        parent_indices = np.random.choice(list(global_ranks.keys()), size=self.breeder.parent_count, replace=False)
                        parent_ideas = [self.population[idx] for idx in parent_indices]
                    breeding_tasks_data.append(parent_ideas)

                for parent_ideas in breeding_tasks_data:
                    if self.stop_requested:
                        break
                    new_idea = await asyncio.to_thread(self.breeder.breed, parent_ideas, self.idea_type)
                    prompt = new_idea.get("specific_prompt") if isinstance(new_idea, dict) else None
                    refined_idea = await asyncio.to_thread(self.critic.refine, new_idea, self.idea_type)
                    formatted_idea = await asyncio.to_thread(self.formatter.format_idea, refined_idea, self.idea_type)
                    new_population.append(formatted_idea)
                    generation_breeding_prompts.append(prompt)

                    # Update progress
                    current_step += 1
                    progress_pct = ((gen - start_gen) * steps_per_gen + current_step) / total_steps * 100
                    await progress_callback({
                        "current_generation": gen + 1,
                        "total_generations": self.generations,
                        "is_running": True,
                        "progress": progress_pct,
                        "status_message": f"Breeding idea {len(new_population)}/{ideas_to_breed + (1 if elite_processed else 0)}..."
                    })

                if self.stop_requested:
                    continue

                # Update population
                self.population = new_population
                self.history.append(self.population.copy())
                self.breeding_prompts.append(generation_breeding_prompts)

                print(f"Generation {gen + 1} complete. Population size: {len(self.population)}")

                # Calculate diversity
                await self._calculate_and_store_diversity()

                # Apply Oracle
                if self.oracle:
                    try:
                        oracle_result = self.oracle.analyze_and_diversify(self.history, self.idea_type)
                        replace_idx = await self._find_least_interesting_idea_idx(self.population)
                        idea_prompt = oracle_result["idea_prompt"]
                        prompts = self._get_template_prompts()
                        extended_prompt = idea_prompt
                        if hasattr(prompts, 'template') and prompts.template.special_requirements:
                            extended_prompt = f"{idea_prompt}\n\nConstraints:\n{prompts.template.special_requirements}"

                        new_idea = self.ideator.generate_text(extended_prompt)
                        oracle_idea = {
                            "id": uuid.uuid4(),
                            "idea": new_idea,
                            "parent_ids": [],
                            "oracle_generated": True,
                            "oracle_analysis": oracle_result["oracle_analysis"]
                        }
                        refined_oracle_idea = self.critic.refine(oracle_idea, self.idea_type)
                        formatted_oracle_idea = self.formatter.format_idea(refined_oracle_idea, self.idea_type)
                        if not formatted_oracle_idea.get("oracle_generated", False):
                            formatted_oracle_idea["oracle_generated"] = True
                            formatted_oracle_idea["oracle_analysis"] = oracle_idea.get("oracle_analysis", "")

                        old_idea = self.population[replace_idx]
                        old_idea_id = str(old_idea.get("id", "")) if isinstance(old_idea, dict) else ""
                        if old_idea_id:
                            await self._remove_embedding(old_idea_id)

                        self.population[replace_idx] = formatted_oracle_idea
                        if self.breeding_prompts:
                            self.breeding_prompts[-1][replace_idx] = idea_prompt
                        self.history[-1] = self.population.copy()
                    except Exception as e:
                        print(f"Oracle failed: {e}")

                # Elite selection
                if gen < self.generations - 1:
                    try:
                        most_diverse_idx = await self._find_most_diverse_idea_idx(self.population)
                        elite_idea = self.population[most_diverse_idx].copy() if isinstance(self.population[most_diverse_idx], dict) else self.population[most_diverse_idx]
                        if isinstance(self.population[most_diverse_idx], dict):
                            self.population[most_diverse_idx]["elite_selected_source"] = True
                            self.population[most_diverse_idx]["elite_target_generation"] = gen + 1
                            self.history[-1] = self.population.copy()
                        if self.breeding_prompts and self.breeding_prompts[-1] and most_diverse_idx < len(self.breeding_prompts[-1]):
                            elite_breeding_prompt = self.breeding_prompts[-1][most_diverse_idx]
                    except Exception as e:
                        print(f"Elite selection failed: {e}")
                        elite_idea = None

                # Update generation tracking and save checkpoint
                self.current_generation = gen + 1
                checkpoint_status = 'in_progress' if gen < self.generations - 1 else 'complete'
                await self.save_checkpoint(status=checkpoint_status)

                # Send progress update
                token_counts = self.get_total_token_count()
                await progress_callback({
                    "current_generation": gen + 1,
                    "total_generations": self.generations,
                    "is_running": True,
                    "history": self.history,
                    "contexts": self.contexts,
                    "specific_prompts": self.specific_prompts,
                    "breeding_prompts": self.breeding_prompts,
                    "progress": ((gen + 1) / self.generations) * 100,
                    "checkpoint_saved": True,
                    "checkpoint_id": self.checkpoint_id,
                    "token_counts": token_counts,
                    "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                })

            # Evolution complete
            print("Evolution complete!")
            await self.save_checkpoint(status='complete')
            token_counts = self.get_total_token_count()
            await progress_callback({
                "current_generation": self.generations,
                "total_generations": self.generations,
                "is_running": False,
                "history": self.history,
                "contexts": self.contexts,
                "specific_prompts": self.specific_prompts,
                "breeding_prompts": self.breeding_prompts,
                "progress": 100,
                "token_counts": token_counts,
                "diversity_history": self.diversity_history.copy() if self.diversity_history else []
            })

        except Exception as e:
            import traceback
            print(f"Error in resume evolution: {e}")
            print(traceback.format_exc())
            await self.save_checkpoint(status='error')
            await progress_callback({
                "is_running": False,
                "error": str(e),
                "is_resumable": True,
                "checkpoint_id": self.checkpoint_id,
                "diversity_history": self.diversity_history.copy() if self.diversity_history else []
            })

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

    async def run_evolution_with_updates(self, progress_callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Runs the evolution process with progress updates

        Args:
            progress_callback: Async function that will be called with progress updates
        """
        try:
            # Reset stop state at the beginning
            self.reset_stop_state()

            # Initialize checkpoint ID for this evolution run
            if not self.checkpoint_id:
                self.checkpoint_id = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
            self.current_generation = 0

            # Seed the initial population
            print("Generating initial population (Generation 0)...")
            self.population = []
            self.specific_prompts = []
            self.contexts = []

            # Calculate total steps: Gen 0 (Seed + Refine) + Gen 1..N (Swiss Rounds + Breed+Refine + Elite)
            est_tournament_rounds = max(1, self.tournament_rounds)
            # Each breeding generation: tournament_rounds + pop_size breeding + 1 for elite processing
            steps_per_gen = self.pop_size + est_tournament_rounds + 1
            total_steps = (2 * self.pop_size) + (self.generations * steps_per_gen)

            # --- Parallel Seeding ---
            async def generate_single_seed():
                context_pool = await asyncio.to_thread(self.ideator.generate_context, self.idea_type)
                idea_text, specific_prompt = await asyncio.to_thread(self.ideator.generate_idea_from_context, context_pool, self.idea_type)
                return context_pool, {"id": uuid.uuid4(), "idea": idea_text, "parent_ids": []}, specific_prompt

            seed_tasks = [generate_single_seed for _ in range(self.pop_size)]

            base_info = {
                "current_generation": 0,
                "total_generations": self.generations,
                "is_running": True,
            }

            seed_results = await self._run_batch_with_progress(
                tasks=seed_tasks,
                progress_callback=progress_callback,
                base_progress_info=base_info,
                start_step=0,
                total_steps=total_steps,
                description_template="Seeding idea {completed}/{total}..."
            )

            if self.stop_requested:
                # Handle stop during seeding
                completed_results = [r for r in seed_results if r is not None]
                for context_pool, idea, prompt in completed_results:
                    self.contexts.append(context_pool)
                    self.population.append(idea)
                    self.specific_prompts.append(prompt)

                await progress_callback({
                    **base_info,
                    "is_running": False,
                    "is_stopped": True,
                    "history": [self.population] if self.population else [],
                    "contexts": self.contexts,
                    "specific_prompts": self.specific_prompts,
                    "stop_message": f"Evolution stopped during initial generation",
                    "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                })
                return

            # Unpack results
            for context_pool, idea, prompt in seed_results:
                self.contexts.append(context_pool)
                self.population.append(idea)
                self.specific_prompts.append(prompt)

            # --- Parallel Refinement ---
            print("Refining initial population...")

            async def refine_single(idea):
                refined_idea = await asyncio.to_thread(self.critic.refine, idea, self.idea_type)
                formatted_idea = await asyncio.to_thread(self.formatter.format_idea, refined_idea, self.idea_type)
                return formatted_idea

            # Create tasks capturing the specific idea for each iteration
            refine_tasks = [lambda i=idea: refine_single(i) for idea in self.population]

            refined_results = await self._run_batch_with_progress(
                tasks=refine_tasks,
                progress_callback=progress_callback,
                base_progress_info=base_info,
                start_step=self.pop_size,
                total_steps=total_steps,
                description_template="Refining idea {completed}/{total}..."
            )

            if self.stop_requested:
                # Handle stop during refinement
                # Update population with whatever finished
                for i, result in enumerate(refined_results):
                    if result is not None:
                        self.population[i] = result

                await progress_callback({
                    **base_info,
                    "is_running": False,
                    "is_stopped": True,
                    "history": [self.population],
                    "contexts": self.contexts,
                    "specific_prompts": self.specific_prompts,
                    "stop_message": f"Evolution stopped during refinement",
                    "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                })
                return

            # Update population with refined ideas
            self.population = refined_results

            # Update costs after refinement batch
            token_counts = self.get_total_token_count()
            current_cost = token_counts['cost']['total_cost']
            self.avg_idea_cost = current_cost / self.pop_size

            # Calculate estimated total cost
            total_ideas_to_generate = self.pop_size * (self.generations + 1)
            remaining_ideas = total_ideas_to_generate - self.pop_size
            remaining_tournaments = self.generations
            estimated_total_cost = current_cost + (remaining_ideas * self.avg_idea_cost) + (remaining_tournaments * self.avg_tournament_cost)
            token_counts['cost']['estimated_total_cost'] = estimated_total_cost

            # Check budget
            if self.check_budget():
                print(f"Budget limit reached: ${current_cost:.4f} >= ${self.max_budget:.4f}")
                self.stop_requested = True
                self.is_stopped = True
                await progress_callback({
                    **base_info,
                    "is_running": False,
                    "is_stopped": True,
                    "history": [self.population],
                    "contexts": self.contexts,
                    "specific_prompts": self.specific_prompts,
                    "stop_message": f"Evolution stopped: Budget limit reached (${current_cost:.2f} / ${self.max_budget:.2f})",
                    "token_counts": token_counts,
                    "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                })
                return

            # Send final update for this phase
            await progress_callback({
                **base_info,
                "history": [self.population],
                "contexts": self.contexts,
                "specific_prompts": self.specific_prompts,
                "progress": (2 * self.pop_size / total_steps) * 100,
                "token_counts": token_counts,
                "diversity_history": self.diversity_history.copy() if self.diversity_history else []
            })

            # Small delay
            await asyncio.sleep(0.1)

            self.history = [self.population.copy()]

            # Calculate initial diversity for generation 0
            initial_diversity = await self._calculate_and_store_diversity()

            # Run evolution for specified number of generations
            elite_idea = None
            elite_breeding_prompt = None

            for gen in range(self.generations):
                # Check for stop request at the beginning of each generation
                if self.stop_requested:
                    self.is_stopped = True
                    self.current_generation = gen
                    print(f"Stop requested - evolution halted after generation {gen}")

                    # Save checkpoint so evolution can be resumed
                    checkpoint_path = await self.save_checkpoint(status='paused')

                    # Calculate token counts for the final update
                    token_counts = self.get_total_token_count()
                    await progress_callback({
                        "current_generation": gen,
                        "total_generations": self.generations,
                        "is_running": False,
                        "is_stopped": True,
                        "is_resumable": True,
                        "checkpoint_id": self.checkpoint_id,
                        "history": self.history,
                        "contexts": self.contexts,
                        "specific_prompts": self.specific_prompts,
                        "breeding_prompts": self.breeding_prompts,
                        "progress": ((self.pop_size + gen * self.pop_size) / (self.pop_size * (self.generations + 1))) * 100,
                        "stop_message": f"Evolution paused after generation {gen}. You can resume this evolution.",
                        "token_counts": token_counts,
                        "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                    })
                    return

                print(f"Starting generation {gen + 1}...")

                new_population = []
                generation_breeding_prompts = []  # Collect breeding prompts for this generation
                random.shuffle(self.population)

                # Handle elite idea from previous generation (if available)
                elite_processed = False
                if elite_idea is not None:
                    print(f"🌟 Processing elite idea for generation {gen + 1}...")

                    # Refine and format the elite idea
                    refined_elite = await asyncio.to_thread(self.critic.refine, elite_idea, self.idea_type)
                    formatted_elite = await asyncio.to_thread(self.formatter.format_idea, refined_elite, self.idea_type)

                    # Mark this idea as elite (most creative/original) and preserve source
                    # Ensure formatted_elite is a dictionary (format_idea should return dict for dict input)
                    print(f"🌟 DEBUG: Elite idea before metadata: {type(formatted_elite)}, keys: {list(formatted_elite.keys()) if isinstance(formatted_elite, dict) else 'N/A'}")

                    if isinstance(formatted_elite, dict):
                        formatted_elite["elite_selected"] = True
                        formatted_elite["elite_source_id"] = elite_idea.get("id")
                        formatted_elite["elite_source_generation"] = gen
                        print(f"🌟 DEBUG: Elite metadata added, keys now: {list(formatted_elite.keys())}")
                    else:
                        # Fallback: convert to dict if needed
                        formatted_elite = {
                            "id": uuid.uuid4(),
                            "idea": formatted_elite,
                            "parent_ids": [],
                            "elite_selected": True,
                            "elite_source_id": elite_idea.get("id"),
                            "elite_source_generation": gen
                        }
                        print(f"🌟 DEBUG: Elite idea converted to dict with keys: {list(formatted_elite.keys())}")

                    print(f"🌟 DEBUG: Final elite idea has elite_selected: {formatted_elite.get('elite_selected')}")

                    # Add to new population
                    new_population.append(formatted_elite)
                    generation_breeding_prompts.append(elite_breeding_prompt)  # Use the original breeding prompt if available

                    # Extract title for logging
                    elite_title = "Unknown"
                    if isinstance(formatted_elite, dict) and "idea" in formatted_elite:
                        idea_obj = formatted_elite["idea"]
                        if hasattr(idea_obj, 'title'):
                            elite_title = idea_obj.title

                    print(f"🌟 Most creative idea '{elite_title}' added to generation {gen + 1}")
                    elite_processed = True

                # Calculate how many ideas we need to breed (total minus elite if processed)
                current_pop_size = len(self.population)
                ideas_to_breed = current_pop_size - (1 if elite_processed else 0)
                print(f"Generating {ideas_to_breed} new ideas via breeding for generation {gen + 1} (plus {1 if elite_processed else 0} creative)")

                # Reset elite for next iteration
                elite_idea = None
                elite_breeding_prompt = None

                # Step 1: Run global Swiss tournament to create global ranking
                print(f"Running Swiss tournament across {len(self.population)} ideas for {self.tournament_rounds} rounds...")

                # Measure tournament cost
                tournament_start_cost = self.get_total_token_count()['cost']['total_cost']

                global_ranks = {}

                # Calculate base steps for this generation (needed for progress calculation)
                # Gen 0 (2*pop) + Previous Gens (gen * steps_per_gen)
                current_gen_start_step = (2 * self.pop_size) + (gen * steps_per_gen)

                # Define base progress info for this generation (for use in callbacks)
                gen_base_progress_info = {
                    "current_generation": gen + 1,
                    "total_generations": self.generations,
                    "is_running": True,
                }

                # Capture the loop from the outer async scope for thread-safe callbacks
                loop = asyncio.get_running_loop()

                pairs_per_round = max(1, len(self.population) // 2)
                total_pairs = pairs_per_round * max(1, self.tournament_rounds)

                def thread_safe_callback(completed, total):
                    round_num = min(self.tournament_rounds, (completed // pairs_per_round) + 1)
                    tournament_fraction = (completed / total_pairs) if total_pairs else 1.0
                    async def send_update():
                        await progress_callback({
                            **gen_base_progress_info,
                            "progress": ((current_gen_start_step + (tournament_fraction * max(1, self.tournament_rounds))) / total_steps) * 100,
                            "status_message": f"Running Swiss round {round_num}/{self.tournament_rounds}...",
                        })

                    asyncio.run_coroutine_threadsafe(send_update(), loop)

                tournament_rounds_details: List[Dict[str, Any]] = []
                global_ranks = await asyncio.to_thread(
                    self.critic.get_tournament_ranks,
                    self.population,
                    self.idea_type,
                    self.tournament_rounds,
                    thread_safe_callback,
                    tournament_rounds_details,
                    self.full_tournament_rounds,
                )
                self._set_tournament_history(gen + 1, tournament_rounds_details)

                if self.stop_requested:
                     # Handle stop during tournaments
                     await progress_callback({
                        "current_generation": gen + 1,
                        "total_generations": self.generations,
                        "is_running": False,
                        "is_stopped": True,
                        "history": self.history,
                        "contexts": self.contexts,
                        "specific_prompts": self.specific_prompts,
                        "breeding_prompts": self.breeding_prompts,
                        "stop_message": f"Evolution stopped during tournaments",
                        "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                    })
                     return

                # Log top ranks for transparency
                print("Swiss tournament rankings:")
                for idea_idx, rank in sorted(global_ranks.items(), key=lambda x: x[1], reverse=True):
                    idea_obj = self.population[idea_idx]["idea"]
                    title = idea_obj.title if hasattr(idea_obj, 'title') else "Untitled"
                    print(f"  {title} - ELO {rank}")

                # Update average tournament cost
                tournament_end_cost = self.get_total_token_count()['cost']['total_cost']
                current_tournament_cost = tournament_end_cost - tournament_start_cost

                # Update moving average (or just set it if it's the first one)
                if self.avg_tournament_cost == 0:
                    self.avg_tournament_cost = current_tournament_cost
                else:
                    # Simple moving average
                    self.avg_tournament_cost = (self.avg_tournament_cost + current_tournament_cost) / 2

                # Step 2: Allocate parent slots globally across entire population
                print("\nAllocating parent slots across entire population...")
                global_parent_slots = self._allocate_parent_slots(global_ranks, ideas_to_breed)

                # Step 3: Generate children using global parent selection
                print(f"Generating {ideas_to_breed} children using global parent pool...")

                # Pre-select parents for all children to be generated
                breeding_tasks_data = []
                for _ in range(ideas_to_breed):
                    if global_parent_slots:
                        parent_indices = self._select_parents_from_slots(global_parent_slots, list(global_ranks.keys()))
                        parent_ideas = [self.population[idx] for idx in parent_indices]
                    else:
                        # Fallback to random selection if allocation fails
                        parent_indices = np.random.choice(list(global_ranks.keys()), size=self.breeder.parent_count, replace=False)
                        parent_ideas = [self.population[idx] for idx in parent_indices]
                    breeding_tasks_data.append(parent_ideas)

                async def breed_single_child(parent_ideas):
                    new_idea = await asyncio.to_thread(self.breeder.breed, parent_ideas, self.idea_type)

                    # Extract breeding prompt
                    prompt = None
                    if isinstance(new_idea, dict) and "specific_prompt" in new_idea:
                        prompt = new_idea["specific_prompt"]

                    # Refine
                    refined_idea = await asyncio.to_thread(self.critic.refine, new_idea, self.idea_type)

                    # Format
                    formatted_idea = await asyncio.to_thread(self.formatter.format_idea, refined_idea, self.idea_type)

                    return formatted_idea, prompt

                breeding_tasks = [lambda p=p: breed_single_child(p) for p in breeding_tasks_data]

                # Base steps calculation
                # Start of breeding is after tournaments
                breeding_start_step = current_gen_start_step + max(1, self.tournament_rounds)

                # Add elite progress if any
                current_gen_base = 1 if elite_processed else 0

                breeding_results = await self._run_batch_with_progress(
                    tasks=breeding_tasks,
                    progress_callback=progress_callback,
                    base_progress_info={
                        "current_generation": gen + 1,
                        "total_generations": self.generations,
                        "is_running": True,
                    },
                    start_step=breeding_start_step + current_gen_base,
                    total_steps=total_steps,
                    description_template="Breeding and refining idea {completed}/{total}..."
                )

                if self.stop_requested:
                    # Handle stop during breeding
                    completed_results = [r for r in breeding_results if r is not None]
                    for idea, prompt in completed_results:
                        new_population.append(idea)
                        generation_breeding_prompts.append(prompt)

                    if new_population:
                        self.history.append(new_population)

                    await progress_callback({
                        "current_generation": gen + 1,
                        "total_generations": self.generations,
                        "is_running": False,
                        "is_stopped": True,
                        "history": self.history,
                        "contexts": self.contexts,
                        "specific_prompts": self.specific_prompts,
                        "breeding_prompts": self.breeding_prompts,
                        "stop_message": f"Evolution stopped during breeding",
                        "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                    })
                    return

                # Process results
                for result in breeding_results:
                    if result is None:
                        generation_breeding_prompts.append(None)
                        continue

                    idea, prompt = result
                    new_population.append(idea)
                    generation_breeding_prompts.append(prompt)

                # Update costs after breeding batch
                token_counts = self.get_total_token_count()
                current_cost = token_counts['cost']['total_cost']

                total_ideas = self.pop_size * (self.generations + 1)
                completed_ideas = self.pop_size + (gen * self.pop_size) + len(new_population)
                remaining_ideas_in_run = total_ideas - completed_ideas
                remaining_tournaments = self.generations - 1 - gen

                estimated_total_cost = current_cost + (remaining_ideas_in_run * self.avg_idea_cost) + (remaining_tournaments * self.avg_tournament_cost)
                token_counts['cost']['estimated_total_cost'] = estimated_total_cost

                # Check budget
                if self.check_budget():
                    print(f"Budget limit reached: ${current_cost:.4f} >= ${self.max_budget:.4f}")
                    self.stop_requested = True
                    self.is_stopped = True
                    if new_population:
                        self.history.append(new_population)
                    await progress_callback({
                        "current_generation": gen + 1,
                        "total_generations": self.generations,
                        "is_running": False,
                        "is_stopped": True,
                        "history": self.history,
                        "contexts": self.contexts,
                        "specific_prompts": self.specific_prompts,
                        "breeding_prompts": self.breeding_prompts,
                        "stop_message": f"Evolution stopped: Budget limit reached (${current_cost:.2f} / ${self.max_budget:.2f})",
                        "token_counts": token_counts,
                        "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                    })
                    return

                # Send final update for this generation
                # Create a copy of the history with the current generation's progress
                history_copy = self.history.copy()
                history_copy.append(new_population.copy())

                # Include the current generation's breeding prompts
                breeding_prompts_with_current = self.breeding_prompts.copy()
                breeding_prompts_with_current.append(generation_breeding_prompts.copy())

                await progress_callback({
                    "current_generation": gen + 1,
                    "total_generations": self.generations,
                    "is_running": True,
                    "history": history_copy,
                    "contexts": self.contexts,
                    "specific_prompts": self.specific_prompts,
                    "breeding_prompts": breeding_prompts_with_current,
                    "progress": ((breeding_start_step + current_gen_base + len(new_population)) / total_steps) * 100,
                    "token_counts": token_counts,
                    "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                })

                # Small delay
                await asyncio.sleep(0.1)

                # Update population with new ideas
                self.population = new_population
                self.history.append(self.population)

                # Store the breeding prompts for this generation
                self.breeding_prompts.append(generation_breeding_prompts)

                print(f"Generation {gen + 1} complete. Population size: {len(self.population)}")
                print(f"Collected {len(generation_breeding_prompts)} breeding prompts for generation {gen + 1}")

                # Calculate diversity for this generation
                generation_diversity = await self._calculate_and_store_diversity()

                # Apply Oracle for diversity enhancement (if enabled)
                if self.oracle:
                    try:
                        print("Oracle analyzing population for diversity enhancement...")
                        print(f"Population size before Oracle: {len(self.population)}")
                        print(f"History generations: {len(self.history)}")

                        oracle_result = self.oracle.analyze_and_diversify(
                            self.history, self.idea_type
                        )

                        print(f"Oracle result: {oracle_result}")

                        # Replace existing idea with more diverse one using embedding-based selection
                        replace_idx = await self._find_least_interesting_idea_idx(self.population)

                        # Generate a new idea using the oracle's prompt, extended with special requirements
                        idea_prompt = oracle_result["idea_prompt"]

                        # Get the special requirements and extend the Oracle prompt with them
                        prompts = self._get_template_prompts()
                        extended_prompt = idea_prompt

                        # If there are special requirements, append them to the Oracle prompt
                        if hasattr(prompts, 'template') and prompts.template.special_requirements:
                            extended_prompt = f"{idea_prompt}\n\nConstraints:\n{prompts.template.special_requirements}"

                        new_idea = self.ideator.generate_text(extended_prompt)

                        # Create the new idea structure
                        oracle_idea = {
                            "id": uuid.uuid4(),
                            "idea": new_idea,
                            "parent_ids": [],
                            "oracle_generated": True,
                            "oracle_analysis": oracle_result["oracle_analysis"]
                        }
                        # refine the idea
                        refined_oracle_idea = self.critic.refine(oracle_idea, self.idea_type)
                        formatted_oracle_idea = self.formatter.format_idea(refined_oracle_idea, self.idea_type)

                        # Ensure Oracle metadata is preserved after formatting
                        if not formatted_oracle_idea.get("oracle_generated", False):
                            print("WARNING: Oracle metadata lost during formatting! Restoring...")
                            formatted_oracle_idea["oracle_generated"] = True
                            formatted_oracle_idea["oracle_analysis"] = oracle_idea.get("oracle_analysis", "Oracle analysis was lost during formatting")

                        old_idea = self.population[replace_idx]
                        old_title = "Unknown"
                        if isinstance(old_idea, dict) and "idea" in old_idea:
                            idea_obj = old_idea["idea"]
                            if hasattr(idea_obj, 'title'):
                                old_title = idea_obj.title

                        # Update embedding storage: remove old idea's embedding
                        old_idea_id = str(old_idea.get("id", "")) if isinstance(old_idea, dict) else ""
                        if old_idea_id:
                            await self._remove_embedding(old_idea_id)
                            print(f"🗑️ Removed embedding for replaced idea: '{old_title}'")

                        self.population[replace_idx] = formatted_oracle_idea

                        # Also update the corresponding prompt so the UI is consistent
                        if self.breeding_prompts:
                            self.breeding_prompts[-1][replace_idx] = idea_prompt

                        print(f"Oracle replaced idea '{old_title}' at index {replace_idx} (least interesting by embedding distance) with more diverse alternative")
                        print(f"Final Oracle idea has metadata: oracle_generated={formatted_oracle_idea.get('oracle_generated')}, has_analysis={'oracle_analysis' in formatted_oracle_idea}")

                        # Store embedding for new Oracle idea
                        # Note: The embedding will be computed and stored when _get_or_compute_embeddings_for_ideas is called next time
                        # This is efficient because it avoids computing the embedding immediately

                        # Update the history with Oracle's changes
                        self.history[-1] = self.population.copy()
                        print(f"Updated history with Oracle changes. Final population size: {len(self.population)}")

                        # Calculate token counts for the update
                        token_counts = self.get_total_token_count()
                        # Immediately update the UI with Oracle changes
                        await progress_callback({
                            "current_generation": gen + 1,
                            "total_generations": self.generations,
                            "is_running": True,
                            "history": self.history,
                            "contexts": self.contexts,
                            "specific_prompts": self.specific_prompts,
                            "breeding_prompts": self.breeding_prompts,
                            "progress": ((gen + 1) / self.generations) * 100,
                            "oracle_update": True,  # Flag to indicate this is an Oracle update
                            "token_counts": token_counts,
                            "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                        })
                    except Exception as e:
                        print(f"Oracle failed with error: {e}. Continuing without Oracle enhancement.")
                        import traceback
                        traceback.print_exc()

                # Elite selection: Pass the most diverse idea directly to the next generation (if not the last generation)
                if gen < self.generations - 1:  # Only do elite selection if there's a next generation
                    try:
                        print("🌟 Performing elite selection for next generation...")
                        most_diverse_idx = await self._find_most_diverse_idea_idx(self.population)
                        elite_idea = self.population[most_diverse_idx].copy() if isinstance(self.population[most_diverse_idx], dict) else self.population[most_diverse_idx]

                        # Mark the SOURCE idea in the current generation as selected for elite
                        # This is what the frontend will see
                        if isinstance(self.population[most_diverse_idx], dict):
                            self.population[most_diverse_idx]["elite_selected_source"] = True
                            self.population[most_diverse_idx]["elite_target_generation"] = gen + 1
                            # Update history to reflect this change
                            self.history[-1] = self.population.copy()
                            print(f"🌟 DEBUG: Marked source idea at index {most_diverse_idx} as elite_selected_source")

                        # Get the corresponding breeding prompt if available
                        if self.breeding_prompts and self.breeding_prompts[-1] and most_diverse_idx < len(self.breeding_prompts[-1]):
                            elite_breeding_prompt = self.breeding_prompts[-1][most_diverse_idx]

                        # Extract title for logging
                        elite_title = "Unknown"
                        if isinstance(elite_idea, dict) and "idea" in elite_idea:
                            idea_obj = elite_idea["idea"]
                            if hasattr(idea_obj, 'title'):
                                elite_title = idea_obj.title

                        print(f"🌟 Most creative idea selected for next generation: '{elite_title}' (will be refined and formatted)")

                        # Calculate token counts for the update
                        token_counts = self.get_total_token_count()
                        # Send an update to notify frontend about elite selection
                        await progress_callback({
                            "current_generation": gen + 1,
                            "total_generations": self.generations,
                            "is_running": True,
                            "history": self.history,
                            "contexts": self.contexts,
                            "specific_prompts": self.specific_prompts,
                            "breeding_prompts": self.breeding_prompts,
                            "progress": ((gen + 1) / self.generations) * 100,
                            "elite_selection_update": True,  # Flag to indicate elite selection update
                            "token_counts": token_counts,
                            "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                        })
                    except Exception as e:
                        print(f"Creative selection failed with error: {e}. Continuing without creative selection.")
                        elite_idea = None

                # Update current generation tracking
                self.current_generation = gen + 1

                # Auto-save checkpoint after each generation completes
                checkpoint_status = 'in_progress' if gen < self.generations - 1 else 'complete'
                checkpoint_path = await self.save_checkpoint(status=checkpoint_status)
                if checkpoint_path:
                    # Notify frontend about checkpoint
                    await progress_callback({
                        "current_generation": gen + 1,
                        "total_generations": self.generations,
                        "is_running": True,
                        "checkpoint_saved": True,
                        "checkpoint_id": self.checkpoint_id,
                    })

            # Mark evolution as complete (only if not stopped)
            if not self.stop_requested:
                # Calculate progress
                progress_percent = ((self.pop_size + (gen + 1) * self.pop_size) / (self.pop_size * (self.generations + 1))) * 100

                # Calculate estimated total cost
                token_counts = self.get_total_token_count()
                current_cost = token_counts['cost']['total_cost']
                estimated_total_cost = 0
                if progress_percent > 0:
                    estimated_total_cost = current_cost / (progress_percent / 100)

                token_counts['cost']['estimated_total_cost'] = estimated_total_cost

                # Check budget
                if self.check_budget():
                    print(f"Budget limit reached: ${current_cost:.4f} >= ${self.max_budget:.4f}")
                    self.stop_requested = True
                    self.is_stopped = True
                    await progress_callback({
                        "current_generation": gen + 1,
                        "total_generations": self.generations,
                        "is_running": False,
                        "is_stopped": True,
                        "history": self.history,
                        "contexts": self.contexts,
                        "specific_prompts": self.specific_prompts,
                        "breeding_prompts": self.breeding_prompts,
                        "progress": progress_percent,
                        "stop_message": f"Evolution stopped: Budget limit reached (${current_cost:.2f} / ${self.max_budget:.2f})",
                        "token_counts": token_counts,
                        "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                    })
                    return

                await progress_callback({
                    "current_generation": gen + 1,
                    "total_generations": self.generations,
                    "is_running": True,
                    "history": self.history,
                    "contexts": self.contexts,
                    "specific_prompts": self.specific_prompts,
                    "breeding_prompts": self.breeding_prompts,
                    "progress": progress_percent,
                    "token_counts": token_counts,
                    "diversity_history": self.diversity_history.copy() if self.diversity_history else []
                })

            print("Evolution complete!")

            # Final update with complete stats
            token_counts = self.get_total_token_count()
            token_counts['cost']['estimated_total_cost'] = token_counts['cost']['total_cost'] # Final cost is actual cost

            await progress_callback({
                "current_generation": self.generations,
                "total_generations": self.generations,
                "is_running": False,
                "history": self.history,
                "contexts": self.contexts,
                "specific_prompts": self.specific_prompts,
                "breeding_prompts": self.breeding_prompts,
                "progress": 100,
                "token_counts": token_counts,
                "diversity_history": self.diversity_history.copy() if self.diversity_history else []
            })
            # Print final diversity summary
            if self.diversity_history:
                print("\n🎯 FINAL DIVERSITY SUMMARY 🎯")
                print("Evolution complete! Here's how diversity evolved:")
                for i, div_data in enumerate(self.diversity_history):
                    if div_data.get("enabled", False) and "error" not in div_data:
                        gen_label = "Initial" if i == 0 else f"Gen {i}"
                        score = div_data.get("diversity_score", 0.0)
                        print(f"  {gen_label}: Diversity = {score:.4f}")
                print("=" * 50)

        except Exception as e:
            import traceback
            print(f"Error in evolution: {e}")
            print(traceback.format_exc())
            await progress_callback({
                "is_running": False,
                "error": str(e),
                "diversity_history": self.diversity_history.copy() if self.diversity_history else []
            })

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

    def _allocate_parent_slots(self, ranks, ideas_to_breed):
        """
        Allocate parent slots based on tournament ranks with caps to prevent convergence.

        Args:
            ranks: Dict mapping idea indices to ELO ratings
            ideas_to_breed: Number of children to produce (determines total parent slots needed)

        Returns:
            Dict mapping idea indices to number of parent slots allocated
        """
        if not ranks or ideas_to_breed <= 0:
            return {}

        # Sort ideas by rank (higher ELO = better rank)
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
        for i, (idea_idx, elo) in enumerate(sorted_ideas):
            slots = allocation.get(idea_idx, 0)
            print(f"  Rank {i+1} (ELO {elo:.0f}): {slots} slots")

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
            return np.random.choice(available_indices, size=min(self.breeder.parent_count, len(available_indices)), replace=False).tolist()

        # Simple random selection without replacement
        selected_parents = []
        pool_copy = parent_pool.copy()

        for _ in range(self.breeder.parent_count):
            if not pool_copy:
                break

            # Select random parent from pool
            selected_idx = np.random.choice(len(pool_copy))
            parent_idx = pool_copy.pop(selected_idx)

            # Avoid selecting the same parent twice for this breeding
            if parent_idx not in selected_parents:
                selected_parents.append(parent_idx)
            else:
                # If we selected a duplicate, try to find a different one
                available_alternatives = [p for p in set(pool_copy) if p not in selected_parents]
                if available_alternatives:
                    alternative = np.random.choice(available_alternatives)
                    selected_parents.append(alternative)
                    # Remove the alternative from pool
                    pool_copy = [p for p in pool_copy if p != alternative]
                else:
                    # If no alternatives, allow the duplicate (shouldn't happen often)
                    selected_parents.append(parent_idx)

        return selected_parents
