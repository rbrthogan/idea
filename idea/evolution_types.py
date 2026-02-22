from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvolutionConfig:
    idea_type: str
    pop_size: int
    generations: int
    model_type: str
    creative_temp: float
    top_p: float
    tournament_rounds: int
    tournament_count: float
    full_tournament_rounds: int
    thinking_budget: Optional[int]
    thinking_level: Optional[str]
    max_budget: Optional[float]
    mutation_rate: float
    seed_context_pool_size: Optional[int]
    replacement_rate: float
    fitness_alpha: float
    age_decay_rate: float
    age_decay_floor: float


@dataclass
class EvolutionIdentity:
    evolution_id: Optional[str] = None
    evolution_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    checkpoint_id: Optional[str] = None


@dataclass
class EvolutionRuntimeState:
    current_generation: int = 0
    population: List[Any] = field(default_factory=list)
    history: List[List[Any]] = field(default_factory=list)
    contexts: List[Any] = field(default_factory=list)
    specific_prompts: List[Any] = field(default_factory=list)
    breeding_prompts: List[List[Any]] = field(default_factory=list)
    tournament_history: List[Dict[str, Any]] = field(default_factory=list)
    diversity_history: List[Dict[str, Any]] = field(default_factory=list)
    avg_idea_cost: float = 0.0
    avg_tournament_cost: float = 0.0
    stop_requested: bool = False
    is_stopped: bool = False


@dataclass
class GenerationWorkState:
    generation_index: int
    start_generation: int
    total_generations: int
    total_steps: int
    steps_per_generation: int
    elite_idea: Optional[Any] = None
    elite_breeding_prompt: Optional[str] = None
