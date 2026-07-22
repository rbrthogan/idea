from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Union

from .models import (
    EmbeddingDiversityComponentConfig,
    EvaluationSuiteConfig,
    FailurePolicy,
    FitnessAggregationConfig,
    LLMPairwiseComponentConfig,
    MetricConfig,
    MetricDirection,
    MetricNormalization,
)


def resolve_legacy_evaluation_suite(fitness_alpha: Any = 0.7) -> EvaluationSuiteConfig:
    """Map legacy Elo/diversity fitness settings to a typed evaluator suite."""

    try:
        alpha = float(fitness_alpha)
    except (TypeError, ValueError):
        alpha = 0.7
    if not math.isfinite(alpha):
        alpha = 0.7
    alpha = max(0.0, min(1.0, alpha))

    return EvaluationSuiteConfig(
        components=[
            LLMPairwiseComponentConfig(
                id="quality_elo",
                metric_name="quality_elo",
                failure_policy=FailurePolicy.PENALIZE,
            ),
            EmbeddingDiversityComponentConfig(
                id="diversity",
                metric_name="diversity",
                failure_policy=FailurePolicy.PENALIZE,
            ),
        ],
        fitness=FitnessAggregationConfig(
            metrics=[
                MetricConfig(
                    component_id="quality_elo",
                    metric_name="quality_elo",
                    weight=alpha,
                    direction=MetricDirection.MAXIMIZE,
                    normalization=MetricNormalization.RUNNING_ZSCORE,
                    penalty_score=0.5,
                ),
                MetricConfig(
                    component_id="diversity",
                    metric_name="diversity",
                    weight=1.0 - alpha,
                    direction=MetricDirection.MAXIMIZE,
                    normalization=MetricNormalization.RUNNING_ZSCORE,
                    penalty_score=0.5,
                ),
            ]
        ),
        metadata={"source": "legacy_fitness_alpha", "fitness_alpha": alpha},
    )


def resolve_evaluation_suite(
    config: Optional[Union[EvaluationSuiteConfig, Mapping[str, Any]]] = None,
    *,
    fitness_alpha: Any = 0.7,
) -> EvaluationSuiteConfig:
    """Return an explicit suite, or synthesize the legacy default when absent."""

    if config is None:
        return resolve_legacy_evaluation_suite(fitness_alpha)
    if isinstance(config, EvaluationSuiteConfig):
        return config
    return EvaluationSuiteConfig.model_validate(dict(config))
