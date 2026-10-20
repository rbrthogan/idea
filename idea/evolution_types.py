from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from idea.config import DEFAULT_MODEL


class EvolutionRunConfig(BaseModel):
    """Canonical, validated settings for one evolution run.

    Authentication, ownership, custom-template snapshots, and injected
    evaluator runners are constructor context rather than reusable run
    settings, so they deliberately remain outside this model.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    DEFAULT_IDEA_TYPE: ClassVar[str] = "airesearch"
    ENGINE_CONTEXT_PARAMETERS: ClassVar[frozenset[str]] = frozenset(
        {
            "self",
            "api_key",
            "user_id",
            "template_data",
            "evaluation_runners",
            "artifact_store",
        }
    )

    idea_type: str = DEFAULT_IDEA_TYPE
    pop_size: int = Field(default=5, ge=1, le=100)
    generations: int = Field(default=3, ge=1, le=500)
    model_type: str = DEFAULT_MODEL
    creative_temp: float = Field(default=1.5, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    random_seed: Optional[int] = Field(default=None, ge=0, le=(2**64) - 1)
    tournament_rounds: int = Field(default=1, ge=1, le=1000)
    tournament_count: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    full_tournament_rounds: Optional[int] = Field(default=None, ge=1, le=1000)
    thinking_budget: Optional[int] = Field(default=None, ge=-1, le=131072)
    thinking_level: Optional[str] = None
    max_budget: Optional[float] = Field(default=None, ge=0.0)
    mutation_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    independent_birth_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    single_parent_birth_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    diversity_representation: Literal["criterion_summaries", "full_text"] = "criterion_summaries"
    novelty_neighbors: int = Field(default=3, ge=1, le=10)
    seed_context_pool_size: Optional[int] = Field(default=None, ge=1, le=8)
    refine_all_ideas: bool = False
    survivor_refinement_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    island_count: int = Field(default=2, ge=1, le=5)
    migration_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    migration_novelty_protection_quantile: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
    )
    replacement_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    fitness_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    age_decay_rate: float = Field(default=0.25, ge=0.0, le=10.0)
    age_decay_floor: float = Field(default=0.35, ge=0.0, le=1.0)
    context_novelty_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    context_novelty_max_attempts: int = Field(default=2, ge=0, le=10)
    evaluation: Optional[Dict[str, Any]] = None

    @staticmethod
    def _bounded_float(value: Any, *, default: float, lower: float, upper: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(numeric):
            return default
        return max(lower, min(upper, numeric))

    @staticmethod
    def _non_negative_float(value: Any, *, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(numeric):
            return default
        return max(0.0, numeric)

    @field_validator("idea_type", mode="before")
    @classmethod
    def normalize_idea_type(cls, value: Any) -> str:
        if value is None:
            return cls.DEFAULT_IDEA_TYPE
        normalized = str(value).strip()
        return normalized or cls.DEFAULT_IDEA_TYPE

    @field_validator("model_type", mode="before")
    @classmethod
    def normalize_model_type(cls, value: Any) -> str:
        if value is None:
            return DEFAULT_MODEL
        normalized = str(value).strip()
        return normalized or DEFAULT_MODEL

    @field_validator("pop_size", "generations", mode="before")
    @classmethod
    def coerce_positive_counts(cls, value: Any, info: ValidationInfo) -> int:
        defaults = {"pop_size": 5, "generations": 3}
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = defaults[info.field_name]
        return max(1, numeric)

    @field_validator("creative_temp", mode="before")
    @classmethod
    def coerce_creative_temp(cls, value: Any) -> float:
        return cls._bounded_float(value, default=1.5, lower=0.0, upper=2.0)

    @field_validator(
        "top_p",
        "mutation_rate",
        "survivor_refinement_rate",
        "migration_rate",
        "migration_novelty_protection_quantile",
        "replacement_rate",
        "fitness_alpha",
        "age_decay_floor",
        "context_novelty_threshold",
        mode="before",
    )
    @classmethod
    def coerce_unit_intervals(cls, value: Any, info: ValidationInfo) -> float:
        defaults = {
            "top_p": 0.9,
            "mutation_rate": 0.3,
            "survivor_refinement_rate": 0.1,
            "migration_rate": 0.05,
            "migration_novelty_protection_quantile": 0.8,
            "replacement_rate": 0.5,
            "fitness_alpha": 0.7,
            "age_decay_floor": 0.35,
            "context_novelty_threshold": 0.8,
        }
        return cls._bounded_float(
            value,
            default=defaults[info.field_name],
            lower=0.0,
            upper=1.0,
        )

    @field_validator("age_decay_rate", mode="before")
    @classmethod
    def coerce_age_decay_rate(cls, value: Any) -> float:
        return cls._non_negative_float(value, default=0.25)

    @field_validator("tournament_rounds", mode="before")
    @classmethod
    def coerce_tournament_rounds(cls, value: Any) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return 1
        return max(1, numeric)

    @field_validator("full_tournament_rounds", mode="before")
    @classmethod
    def coerce_full_tournament_rounds(cls, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        return numeric if numeric > 0 else None

    @field_validator("tournament_count", mode="before")
    @classmethod
    def coerce_tournament_count(cls, value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return max(0.0, numeric)

    @field_validator("thinking_budget", mode="before")
    @classmethod
    def coerce_thinking_budget(cls, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @field_validator("thinking_level", mode="before")
    @classmethod
    def normalize_thinking_level(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None

    @field_validator("max_budget", mode="before")
    @classmethod
    def coerce_max_budget(cls, value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return max(0.0, numeric)

    @field_validator("seed_context_pool_size", mode="before")
    @classmethod
    def coerce_seed_context_pool_size(cls, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return None

    @field_validator("refine_all_ideas", mode="before")
    @classmethod
    def coerce_refine_all_ideas(cls, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
            return False
        return bool(value)

    @field_validator("island_count", mode="before")
    @classmethod
    def coerce_island_count(cls, value: Any) -> int:
        try:
            return max(1, min(5, int(value)))
        except (TypeError, ValueError):
            return 1

    @field_validator("context_novelty_max_attempts", mode="before")
    @classmethod
    def coerce_context_novelty_max_attempts(cls, value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 2

    @field_validator("random_seed", mode="before")
    @classmethod
    def coerce_random_seed(cls, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        return int(value)

    @field_validator("evaluation", mode="before")
    @classmethod
    def normalize_evaluation(cls, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, Mapping):
            return dict(value)
        raise ValueError("evaluation must be a mapping or Pydantic model")

    @model_validator(mode="after")
    def derive_population_dependent_settings(self) -> "EvolutionRunConfig":
        if self.independent_birth_rate + self.single_parent_birth_rate > 1.0 + 1e-12:
            raise ValueError("Independent and single-parent birth rates must sum to at most 1")
        if self.full_tournament_rounds is None:
            self.full_tournament_rounds = max(1, self.pop_size - 1)
        if self.tournament_count is None:
            self.tournament_count = (
                self.tournament_rounds / self.full_tournament_rounds
            )
        self.island_count = max(1, min(max(1, self.pop_size // 2), self.island_count))
        return self

    @classmethod
    def from_stored_mapping(cls, stored: Optional[Mapping[str, Any]]) -> "EvolutionRunConfig":
        """Parse legacy/current snake_case persistence data.

        Older snapshots may contain retired fields such as ``tournament_size``
        and omit newer settings.  Known fields are validated; retired fields
        are ignored and all missing values receive stable model defaults.
        """

        source = dict(stored or {})
        # Raising the new-run default must not alter legacy runs on resume.
        source.setdefault("mutation_rate", 0.2)
        # Legacy vectors and running normalization must keep their original
        # meaning on resume. New runs default to criterion summaries.
        source.setdefault("diversity_representation", "full_text")
        source.setdefault("novelty_neighbors", 5)
        if "evaluation" not in source and "evaluation_config" in source:
            source["evaluation"] = source["evaluation_config"]
        known = {
            name: source[name]
            for name in cls.model_fields
            if name in source
        }
        return cls.model_validate(known)

    @classmethod
    def from_engine_kwargs(cls, values: Mapping[str, Any]) -> "EvolutionRunConfig":
        """Build from an EvolutionEngine constructor-local mapping."""

        source = dict(values)
        if "evaluation" not in source and "evaluation_config" in source:
            source["evaluation"] = source["evaluation_config"]
        known = {
            name: source[name]
            for name in cls.model_fields
            if name in source
        }
        return cls.model_validate(known)

    @classmethod
    def from_engine(cls, engine: Any) -> "EvolutionRunConfig":
        """Snapshot run settings from a live engine without a field whitelist."""

        values = {
            name: getattr(engine, name)
            for name in cls.model_fields
            if name != "evaluation" and hasattr(engine, name)
        }
        evaluation = getattr(engine, "evaluation_config", None)
        if evaluation is not None:
            values["evaluation"] = evaluation
        return cls.model_validate(values)

    def to_persistence_dict(self) -> Dict[str, Any]:
        """Return the canonical snake_case persistence mapping."""

        return self.model_dump(mode="json")

    def engine_kwargs(self) -> Dict[str, Any]:
        """Return settings using EvolutionEngine constructor parameter names."""

        values = self.to_persistence_dict()
        values["evaluation_config"] = values.pop("evaluation")
        return values

    @classmethod
    def engine_parameter_names(cls) -> frozenset[str]:
        """Names covered in EvolutionEngine's constructor signature."""

        names = set(cls.model_fields)
        names.remove("evaluation")
        names.add("evaluation_config")
        return frozenset(names)


# Compatibility name used by EvolutionEngine's existing typed-state snapshot.
EvolutionConfig = EvolutionRunConfig


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
    island_diversity_history: List[Dict[str, Any]] = field(default_factory=list)
    migration_history: List[Dict[str, Any]] = field(default_factory=list)
    comparison_cache: Dict[str, Any] = field(default_factory=dict)
    algorithm_evidence_version: int = 2
    quality_archive: List[Any] = field(default_factory=list)
    gene_catalogue: Dict[str, Any] = field(default_factory=dict)
    generation_draft: Optional[Dict[str, Any]] = None
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
