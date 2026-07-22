from __future__ import annotations

import math
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


SCHEMA_VERSION = 1

Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    ),
]


class ComponentKind(str, Enum):
    LLM_PAIRWISE = "llm_pairwise"
    EMBEDDING_DIVERSITY = "embedding_diversity"
    PYTHON_METRIC = "python_metric"
    LLM_CODE_ARTIFACT = "llm_code_artifact"
    LLM_ARTIFACT_SCORE = "llm_artifact_score"


class EvaluationScope(str, Enum):
    CANDIDATE = "candidate"
    SELECTION_POOL = "selection_pool"
    GLOBAL_POPULATION = "global_population"


class FailurePolicy(str, Enum):
    """How aggregation should handle missing or failed evaluator output."""

    FAIL_RUN = "fail_run"
    PENALIZE = "penalize"
    OMIT = "omit"


class EvaluationStatus(str, Enum):
    OK = "ok"
    CACHED = "cached"
    FAILED = "failed"
    SKIPPED = "skipped"


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class MetricNormalization(str, Enum):
    NONE = "none"
    BOUNDS = "bounds"
    MIN_MAX = "min_max"
    RUNNING_ZSCORE = "running_zscore"


class FitnessContributionStatus(str, Enum):
    USED = "used"
    PENALIZED = "penalized"
    OMITTED = "omitted"


class VersionedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION


class BaseComponentConfig(VersionedModel):
    id: Identifier
    kind: ComponentKind
    enabled: bool = True
    scope: EvaluationScope = EvaluationScope.CANDIDATE
    failure_policy: FailurePolicy = FailurePolicy.FAIL_RUN
    component_version: str = Field(default="1", min_length=1, max_length=64)
    requires: List[Identifier] = Field(default_factory=list)
    provides: List[Identifier] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("requires", "provides")
    @classmethod
    def validate_unique_evidence_keys(cls, values: List[str]) -> List[str]:
        if len(values) != len(set(values)):
            raise ValueError("component evidence keys must be unique")
        return values


class LLMPairwiseComponentConfig(BaseComponentConfig):
    kind: Literal[ComponentKind.LLM_PAIRWISE] = ComponentKind.LLM_PAIRWISE
    scope: EvaluationScope = EvaluationScope.SELECTION_POOL
    metric_name: Identifier = "quality_elo"
    model: Optional[str] = None
    rounds: Optional[int] = Field(default=None, ge=1)

    @field_validator("model")
    @classmethod
    def reject_unsupported_model_override(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            raise ValueError(
                "per-component LLM model overrides are not supported in schema version 1"
            )
        return value


class EmbeddingDiversityComponentConfig(BaseComponentConfig):
    kind: Literal[ComponentKind.EMBEDDING_DIVERSITY] = ComponentKind.EMBEDDING_DIVERSITY
    scope: EvaluationScope = EvaluationScope.GLOBAL_POPULATION
    metric_name: Identifier = "diversity"


class PythonMetricComponentConfig(BaseComponentConfig):
    kind: Literal[ComponentKind.PYTHON_METRIC] = ComponentKind.PYTHON_METRIC
    scope: EvaluationScope = EvaluationScope.CANDIDATE
    script_ref: str = Field(min_length=1, max_length=512)
    execution_profile: Identifier = "python-metric-v1"
    metric_names: List[Identifier] = Field(min_length=1)
    timeout_seconds: float = Field(default=30.0, gt=0, le=3600)

    @field_validator("metric_names")
    @classmethod
    def validate_unique_metric_names(cls, values: List[str]) -> List[str]:
        if len(values) != len(set(values)):
            raise ValueError("metric_names must be unique")
        return values

    @field_validator("script_ref")
    @classmethod
    def validate_script_reference(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        windows_path = PureWindowsPath(value)
        posix_path = PurePosixPath(normalized)
        if (
            "\x00" in value
            or posix_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or any(part in {"", ".", ".."} for part in normalized.split("/"))
            or not normalized.endswith(".py")
        ):
            raise ValueError(
                "script_ref must be a traversal-free relative .py reference"
            )
        return normalized


class LLMCodeArtifactComponentConfig(BaseComponentConfig):
    kind: Literal[ComponentKind.LLM_CODE_ARTIFACT] = ComponentKind.LLM_CODE_ARTIFACT
    scope: EvaluationScope = EvaluationScope.CANDIDATE
    model: Optional[str] = None
    execution_profile: Identifier
    artifact_role: Identifier = "primary"
    source_role: Identifier = "source"
    timeout_seconds: float = Field(default=30.0, gt=0, le=3600)


class LLMArtifactScoreComponentConfig(BaseComponentConfig):
    kind: Literal[ComponentKind.LLM_ARTIFACT_SCORE] = ComponentKind.LLM_ARTIFACT_SCORE
    scope: EvaluationScope = EvaluationScope.CANDIDATE
    model: Optional[str] = None
    artifact_role: Identifier = "primary"
    metric_name: Identifier = "artifact_score"
    rubric: Optional[str] = Field(default=None, max_length=50_000)


EvaluationComponentConfig = Annotated[
    Union[
        LLMPairwiseComponentConfig,
        EmbeddingDiversityComponentConfig,
        PythonMetricComponentConfig,
        LLMCodeArtifactComponentConfig,
        LLMArtifactScoreComponentConfig,
    ],
    Field(discriminator="kind"),
]


class MetricConfig(VersionedModel):
    """Declares how one evaluator metric contributes to selection fitness."""

    component_id: Identifier
    metric_name: Identifier
    weight: float = Field(default=1.0, ge=0)
    direction: MetricDirection = MetricDirection.MAXIMIZE
    normalization: MetricNormalization = MetricNormalization.MIN_MAX
    bounds: Optional[Tuple[float, float]] = None
    clamp: bool = True
    failure_policy: Optional[FailurePolicy] = None
    penalty_score: float = 0.0

    @model_validator(mode="after")
    def validate_normalization(self) -> "MetricConfig":
        if self.bounds is not None:
            lower, upper = self.bounds
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError("metric bounds must be finite")
            if lower >= upper:
                raise ValueError("metric bounds must satisfy lower < upper")
        if self.normalization == MetricNormalization.BOUNDS and self.bounds is None:
            raise ValueError("bounds normalization requires bounds")
        return self

    @property
    def key(self) -> str:
        return f"{self.component_id}.{self.metric_name}"


class FitnessAggregationConfig(VersionedModel):
    enabled: bool = True
    strategy: Literal["weighted_sum"] = "weighted_sum"
    metrics: List[MetricConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_metrics(self) -> "FitnessAggregationConfig":
        keys = [metric.key for metric in self.metrics]
        if len(keys) != len(set(keys)):
            raise ValueError("fitness metrics must be unique by component_id and metric_name")
        if self.enabled and not any(metric.weight > 0 for metric in self.metrics):
            raise ValueError("enabled fitness aggregation requires at least one positive metric weight")
        return self


class EvaluationSuiteConfig(VersionedModel):
    enabled: bool = True
    allow_generated_code_execution: bool = False
    components: List[EvaluationComponentConfig] = Field(default_factory=list)
    fitness: FitnessAggregationConfig = Field(
        default_factory=lambda: FitnessAggregationConfig(enabled=False)
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_component_references(self) -> "EvaluationSuiteConfig":
        component_ids = [component.id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("evaluation component ids must be unique")

        singleton_kinds = {
            ComponentKind.LLM_PAIRWISE,
            ComponentKind.EMBEDDING_DIVERSITY,
        }
        for kind in singleton_kinds:
            enabled_ids = [
                component.id
                for component in self.components
                if component.enabled and component.kind == kind
            ]
            if len(enabled_ids) > 1:
                raise ValueError(
                    f"schema version 1 supports only one enabled {kind.value} "
                    "component; found: "
                    + ", ".join(enabled_ids)
                )

        components_by_id = {component.id: component for component in self.components}
        for metric in self.fitness.metrics:
            component = components_by_id.get(metric.component_id)
            if component is None:
                raise ValueError(
                    f"fitness metric {metric.key!r} references an unknown component"
                )
            if not component.enabled and metric.weight > 0:
                raise ValueError(
                    f"fitness metric {metric.key!r} references a disabled component"
                )
            emitted_metrics: set[str]
            if isinstance(
                component,
                (LLMPairwiseComponentConfig, EmbeddingDiversityComponentConfig),
            ):
                emitted_metrics = {component.metric_name}
            elif isinstance(component, PythonMetricComponentConfig):
                emitted_metrics = set(component.metric_names)
            elif isinstance(component, LLMArtifactScoreComponentConfig):
                emitted_metrics = {component.metric_name}
            else:
                emitted_metrics = set()
            if metric.metric_name not in emitted_metrics:
                emitted = ", ".join(sorted(emitted_metrics)) or "none"
                raise ValueError(
                    f"fitness metric {metric.key!r} is not emitted by component "
                    f"{component.id!r}; emitted metrics: {emitted}"
                )
        if self.enabled and not self.components:
            raise ValueError("an enabled evaluation suite requires at least one component")
        generated_code_components = [
            component.id
            for component in self.components
            if component.enabled and component.kind == ComponentKind.LLM_CODE_ARTIFACT
        ]
        if generated_code_components and not self.allow_generated_code_execution:
            raise ValueError(
                "generated-code components require explicit "
                "allow_generated_code_execution=true: "
                + ", ".join(generated_code_components)
            )
        return self


class ArtifactRef(VersionedModel):
    id: Identifier
    role: Identifier
    media_type: str = Field(min_length=1, max_length=255)
    storage_key: str = Field(min_length=1, max_length=2048)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    size_bytes: int = Field(ge=0)
    producer_id: Optional[Identifier] = None
    content_digest: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(VersionedModel):
    candidate_id: Identifier
    component_id: Identifier
    component_kind: Optional[ComponentKind] = None
    component_version: Optional[str] = Field(default=None, min_length=1, max_length=64)
    scope: Optional[EvaluationScope] = None
    generation: Optional[int] = Field(default=None, ge=0)
    status: EvaluationStatus
    metrics: Dict[str, float] = Field(default_factory=dict)
    normalized_metrics: Dict[str, float] = Field(default_factory=dict)
    artifact_ids: List[Identifier] = Field(default_factory=list)
    rationale: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = Field(default=None, ge=0)
    input_digest: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("metrics", "normalized_metrics")
    @classmethod
    def validate_finite_metrics(cls, metrics: Dict[str, float]) -> Dict[str, float]:
        for name, value in metrics.items():
            if not name or not str(name).strip():
                raise ValueError("metric names must not be empty")
            if not math.isfinite(float(value)):
                raise ValueError(f"metric {name!r} must be finite")
        return metrics

    @field_validator("artifact_ids")
    @classmethod
    def validate_unique_artifact_ids(cls, values: List[str]) -> List[str]:
        if len(values) != len(set(values)):
            raise ValueError("artifact_ids must be unique")
        return values

    @model_validator(mode="after")
    def validate_status_payload(self) -> "EvaluationRecord":
        if self.status == EvaluationStatus.FAILED and not self.error:
            raise ValueError("failed evaluation records require an error")
        if self.status in {EvaluationStatus.OK, EvaluationStatus.CACHED} and self.error:
            raise ValueError("successful evaluation records cannot include an error")
        return self


class EvaluationBatch(VersionedModel):
    generation: Optional[int] = Field(default=None, ge=0)
    records: List[EvaluationRecord] = Field(default_factory=list)
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_unique_entries(self) -> "EvaluationBatch":
        record_keys = [
            (record.candidate_id, record.component_id) for record in self.records
        ]
        if len(record_keys) != len(set(record_keys)):
            raise ValueError(
                "evaluation records must be unique by candidate_id and component_id"
            )

        artifact_ids = [artifact.id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("artifact ids must be unique within an evaluation batch")

        known_artifacts = set(artifact_ids)
        unknown_refs = {
            artifact_id
            for record in self.records
            for artifact_id in record.artifact_ids
            if artifact_id not in known_artifacts
        }
        if unknown_refs:
            raise ValueError(
                "evaluation records reference artifacts absent from the batch: "
                + ", ".join(sorted(unknown_refs))
            )
        return self

    def candidate_ids(self) -> List[str]:
        return list(dict.fromkeys(record.candidate_id for record in self.records))

    def record_for(
        self,
        candidate_id: str,
        component_id: str,
    ) -> Optional[EvaluationRecord]:
        for record in self.records:
            if (
                record.candidate_id == candidate_id
                and record.component_id == component_id
            ):
                return record
        return None


class FitnessContribution(VersionedModel):
    metric_key: str
    status: FitnessContributionStatus
    weight: float = Field(ge=0)
    raw_value: Optional[float] = None
    score_value: Optional[float] = None
    evaluation_status: Optional[EvaluationStatus] = None
    reason: Optional[str] = None


class FitnessScore(VersionedModel):
    candidate_id: Identifier
    score: Optional[float] = None
    used_weight: float = Field(default=0.0, ge=0)
    components: Dict[str, FitnessContribution] = Field(default_factory=dict)
    omitted_metrics: List[str] = Field(default_factory=list)
    penalized_metrics: List[str] = Field(default_factory=list)

    @property
    def is_partial(self) -> bool:
        return bool(self.omitted_metrics or self.penalized_metrics)
