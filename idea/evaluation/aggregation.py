from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

from .models import (
    EvaluationBatch,
    EvaluationRecord,
    EvaluationStatus,
    EvaluationSuiteConfig,
    FailurePolicy,
    FitnessAggregationConfig,
    FitnessContribution,
    FitnessContributionStatus,
    FitnessScore,
    MetricConfig,
    MetricDirection,
    MetricNormalization,
)


class FitnessAggregationError(RuntimeError):
    """Base error for invalid or unusable evaluation evidence."""


class RequiredMetricError(FitnessAggregationError):
    def __init__(
        self,
        *,
        candidate_id: str,
        metric_key: str,
        reason: str,
    ) -> None:
        self.candidate_id = candidate_id
        self.metric_key = metric_key
        self.reason = reason
        super().__init__(
            f"required metric {metric_key!r} is unavailable for "
            f"candidate {candidate_id!r}: {reason}"
        )


class FitnessAggregator:
    """Aggregate typed evaluator output into per-candidate weighted fitness.

    Weights are normalized over the metrics actually used for each candidate.
    This means an evaluator configured with ``FailurePolicy.OMIT`` can be absent
    without implicitly reducing the candidate's total score.
    """

    def __init__(
        self,
        config: Union[EvaluationSuiteConfig, FitnessAggregationConfig],
    ) -> None:
        if isinstance(config, EvaluationSuiteConfig):
            self.suite_config: Optional[EvaluationSuiteConfig] = config
            self.config = config.fitness
            self._component_failure_policies = {
                component.id: component.failure_policy
                for component in config.components
            }
            self._enabled = bool(config.enabled and config.fitness.enabled)
        else:
            self.suite_config = None
            self.config = config
            self._component_failure_policies = {}
            self._enabled = bool(config.enabled)

    def aggregate(
        self,
        batch: EvaluationBatch,
        candidate_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, FitnessScore]:
        resolved_candidate_ids = self._resolve_candidate_ids(batch, candidate_ids)
        if not self._enabled:
            return {
                candidate_id: FitnessScore(candidate_id=candidate_id)
                for candidate_id in resolved_candidate_ids
            }

        records = {
            (record.candidate_id, record.component_id): record
            for record in batch.records
        }
        min_max_ranges = self._build_min_max_ranges(
            records,
            resolved_candidate_ids,
        )

        return {
            candidate_id: self._aggregate_candidate(
                candidate_id,
                records,
                min_max_ranges,
            )
            for candidate_id in resolved_candidate_ids
        }

    @staticmethod
    def _resolve_candidate_ids(
        batch: EvaluationBatch,
        candidate_ids: Optional[Iterable[str]],
    ) -> List[str]:
        if candidate_ids is None:
            return batch.candidate_ids()
        return list(dict.fromkeys(str(candidate_id) for candidate_id in candidate_ids))

    def _build_min_max_ranges(
        self,
        records: Dict[Tuple[str, str], EvaluationRecord],
        candidate_ids: List[str],
    ) -> Dict[str, Tuple[float, float]]:
        ranges: Dict[str, Tuple[float, float]] = {}
        for metric in self.config.metrics:
            if metric.normalization != MetricNormalization.MIN_MAX:
                continue
            values: List[float] = []
            for candidate_id in candidate_ids:
                record = records.get((candidate_id, metric.component_id))
                if record is None or record.status not in {
                    EvaluationStatus.OK,
                    EvaluationStatus.CACHED,
                }:
                    continue
                if metric.metric_name in record.metrics:
                    values.append(float(record.metrics[metric.metric_name]))
            if values:
                ranges[metric.key] = (min(values), max(values))
        return ranges

    def _aggregate_candidate(
        self,
        candidate_id: str,
        records: Dict[Tuple[str, str], EvaluationRecord],
        min_max_ranges: Dict[str, Tuple[float, float]],
    ) -> FitnessScore:
        numerator = 0.0
        used_weight = 0.0
        components: Dict[str, FitnessContribution] = {}
        omitted_metrics: List[str] = []
        penalized_metrics: List[str] = []

        for metric in self.config.metrics:
            if metric.weight == 0:
                components[metric.key] = FitnessContribution(
                    metric_key=metric.key,
                    status=FitnessContributionStatus.OMITTED,
                    weight=0,
                    reason="zero weight",
                )
                omitted_metrics.append(metric.key)
                continue

            record = records.get((candidate_id, metric.component_id))
            if (
                record is not None
                and record.status in {EvaluationStatus.OK, EvaluationStatus.CACHED}
                and metric.metric_name in record.metrics
            ):
                raw_value = float(record.metrics[metric.metric_name])
                if (
                    metric.normalization == MetricNormalization.RUNNING_ZSCORE
                    and metric.metric_name in record.normalized_metrics
                ):
                    normalized_value = float(
                        record.normalized_metrics[metric.metric_name]
                    )
                    score_value = (
                        normalized_value
                        if metric.direction == MetricDirection.MAXIMIZE
                        else 1.0 - normalized_value
                    )
                else:
                    score_value = self._normalize_metric(
                        metric,
                        raw_value,
                        min_max_ranges.get(metric.key),
                    )
                components[metric.key] = FitnessContribution(
                    metric_key=metric.key,
                    status=FitnessContributionStatus.USED,
                    weight=metric.weight,
                    raw_value=raw_value,
                    score_value=score_value,
                    evaluation_status=record.status,
                )
                numerator += metric.weight * score_value
                used_weight += metric.weight
                continue

            reason = self._missing_reason(record, metric)
            policy = self._failure_policy(metric)
            if policy == FailurePolicy.FAIL_RUN:
                raise RequiredMetricError(
                    candidate_id=candidate_id,
                    metric_key=metric.key,
                    reason=reason,
                )
            if policy == FailurePolicy.PENALIZE:
                components[metric.key] = FitnessContribution(
                    metric_key=metric.key,
                    status=FitnessContributionStatus.PENALIZED,
                    weight=metric.weight,
                    score_value=metric.penalty_score,
                    evaluation_status=record.status if record else None,
                    reason=reason,
                )
                numerator += metric.weight * metric.penalty_score
                used_weight += metric.weight
                penalized_metrics.append(metric.key)
                continue

            components[metric.key] = FitnessContribution(
                metric_key=metric.key,
                status=FitnessContributionStatus.OMITTED,
                weight=metric.weight,
                evaluation_status=record.status if record else None,
                reason=reason,
            )
            omitted_metrics.append(metric.key)

        score = numerator / used_weight if used_weight > 0 else None
        return FitnessScore(
            candidate_id=candidate_id,
            score=score,
            used_weight=used_weight,
            components=components,
            omitted_metrics=omitted_metrics,
            penalized_metrics=penalized_metrics,
        )

    def _failure_policy(self, metric: MetricConfig) -> FailurePolicy:
        if metric.failure_policy is not None:
            return metric.failure_policy
        return self._component_failure_policies.get(
            metric.component_id,
            FailurePolicy.FAIL_RUN,
        )

    @staticmethod
    def _missing_reason(
        record: Optional[EvaluationRecord],
        metric: MetricConfig,
    ) -> str:
        if record is None:
            return "evaluation record is missing"
        if record.status == EvaluationStatus.FAILED:
            return record.error or "evaluation failed"
        if record.status == EvaluationStatus.SKIPPED:
            return "evaluation was skipped"
        return f"evaluation did not produce metric {metric.metric_name!r}"

    @staticmethod
    def _normalize_metric(
        metric: MetricConfig,
        raw_value: float,
        observed_range: Optional[Tuple[float, float]],
    ) -> float:
        if metric.normalization == MetricNormalization.NONE:
            return (
                raw_value
                if metric.direction == MetricDirection.MAXIMIZE
                else -raw_value
            )

        if metric.normalization == MetricNormalization.RUNNING_ZSCORE:
            raise FitnessAggregationError(
                f"metric {metric.key!r} requires a normalized value from its "
                "stateful evaluator"
            )

        if metric.normalization == MetricNormalization.BOUNDS:
            assert metric.bounds is not None
            lower, upper = metric.bounds
        else:
            if observed_range is None:
                normalized = 0.5
                return normalized
            lower, upper = observed_range

        if upper == lower:
            normalized = 0.5
        else:
            normalized = (raw_value - lower) / (upper - lower)
            if metric.clamp:
                normalized = max(0.0, min(1.0, normalized))

        if metric.direction == MetricDirection.MINIMIZE:
            normalized = 1.0 - normalized
        return normalized
