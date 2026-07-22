import pytest

from idea.evaluation import (
    EmbeddingDiversityComponentConfig,
    EvaluationBatch,
    EvaluationRecord,
    EvaluationStatus,
    EvaluationSuiteConfig,
    FailurePolicy,
    FitnessAggregationConfig,
    FitnessAggregator,
    FitnessContributionStatus,
    LLMPairwiseComponentConfig,
    MetricConfig,
    MetricDirection,
    MetricNormalization,
    RequiredMetricError,
)


def record(candidate, component, metrics=None, status=EvaluationStatus.OK, error=None):
    return EvaluationRecord(
        candidate_id=candidate,
        component_id=component,
        status=status,
        metrics=metrics or {},
        error=error,
    )


def suite(*metrics, quality_policy=FailurePolicy.FAIL_RUN, diversity_policy=FailurePolicy.FAIL_RUN):
    metric_names_by_component = {}
    for metric in metrics:
        metric_names_by_component.setdefault(metric.component_id, metric.metric_name)
    return EvaluationSuiteConfig(
        components=[
            LLMPairwiseComponentConfig(
                id="quality",
                metric_name=metric_names_by_component.get("quality", "quality_elo"),
                failure_policy=quality_policy,
            ),
            EmbeddingDiversityComponentConfig(
                id="diversity",
                metric_name=metric_names_by_component.get("diversity", "diversity"),
                failure_policy=diversity_policy,
            ),
        ],
        fitness=FitnessAggregationConfig(metrics=list(metrics)),
    )


def test_weighted_bounds_normalization_respects_maximize_and_minimize():
    config = suite(
        MetricConfig(
            component_id="quality",
            metric_name="quality_elo",
            weight=3,
            normalization=MetricNormalization.BOUNDS,
            bounds=(0, 1),
        ),
        MetricConfig(
            component_id="diversity",
            metric_name="latency",
            weight=1,
            direction=MetricDirection.MINIMIZE,
            normalization=MetricNormalization.BOUNDS,
            bounds=(0, 100),
        ),
    )
    batch = EvaluationBatch(
        records=[
            record("a", "quality", {"quality_elo": 0.8}),
            record("a", "diversity", {"latency": 20}),
        ]
    )

    result = FitnessAggregator(config).aggregate(batch)["a"]

    assert result.score == pytest.approx(0.8)
    assert result.used_weight == 4
    assert result.components["diversity.latency"].score_value == pytest.approx(0.8)


def test_population_min_max_normalization_and_constant_values():
    config = suite(
        MetricConfig(
            component_id="quality",
            metric_name="quality_elo",
            normalization=MetricNormalization.MIN_MAX,
        ),
        MetricConfig(
            component_id="diversity",
            metric_name="cost",
            direction=MetricDirection.MINIMIZE,
            normalization=MetricNormalization.MIN_MAX,
        ),
    )
    batch = EvaluationBatch(
        records=[
            record("a", "quality", {"quality_elo": 10}),
            record("a", "diversity", {"cost": 2}),
            record("b", "quality", {"quality_elo": 20}),
            record("b", "diversity", {"cost": 4}),
        ]
    )

    results = FitnessAggregator(config).aggregate(batch)

    assert results["a"].score == pytest.approx(0.5)
    assert results["b"].score == pytest.approx(0.5)

    constant_batch = EvaluationBatch(
        records=[
            record("a", "quality", {"quality_elo": 7}),
            record("a", "diversity", {"cost": 7}),
        ]
    )
    assert FitnessAggregator(config).aggregate(constant_batch)["a"].score == pytest.approx(0.5)


def test_none_normalization_leaves_maximize_raw_and_negates_minimize_raw():
    config = suite(
        MetricConfig(
            component_id="quality",
            metric_name="quality_elo",
            normalization=MetricNormalization.NONE,
        ),
        MetricConfig(
            component_id="diversity",
            metric_name="cost",
            direction=MetricDirection.MINIMIZE,
            normalization=MetricNormalization.NONE,
        ),
    )
    batch = EvaluationBatch(
        records=[
            record("a", "quality", {"quality_elo": 4}),
            record("a", "diversity", {"cost": 3}),
        ]
    )

    result = FitnessAggregator(config).aggregate(batch)["a"]

    assert result.score == pytest.approx(0.5)
    assert result.components["quality.quality_elo"].score_value == 4
    assert result.components["diversity.cost"].score_value == -3


def test_optional_omission_renormalizes_remaining_weight_per_candidate():
    config = suite(
        MetricConfig(
            component_id="quality",
            metric_name="quality_elo",
            weight=0.75,
            normalization=MetricNormalization.BOUNDS,
            bounds=(0, 1),
        ),
        MetricConfig(
            component_id="diversity",
            metric_name="diversity",
            weight=0.25,
            normalization=MetricNormalization.BOUNDS,
            bounds=(0, 1),
        ),
        diversity_policy=FailurePolicy.OMIT,
    )
    batch = EvaluationBatch(
        records=[
            record("a", "quality", {"quality_elo": 0.8}),
            record(
                "a",
                "diversity",
                status=EvaluationStatus.FAILED,
                error="embedding unavailable",
            ),
            record("b", "quality", {"quality_elo": 0.6}),
            record("b", "diversity", {"diversity": 0.2}),
        ]
    )

    results = FitnessAggregator(config).aggregate(batch)

    assert results["a"].score == pytest.approx(0.8)
    assert results["a"].used_weight == pytest.approx(0.75)
    assert results["a"].omitted_metrics == ["diversity.diversity"]
    assert results["a"].components["diversity.diversity"].status == FitnessContributionStatus.OMITTED
    assert results["b"].score == pytest.approx(0.5)


def test_metric_policy_can_override_component_policy_and_penalize():
    config = suite(
        MetricConfig(
            component_id="quality",
            metric_name="quality_elo",
            weight=1,
            normalization=MetricNormalization.BOUNDS,
            bounds=(0, 1),
        ),
        MetricConfig(
            component_id="diversity",
            metric_name="diversity",
            weight=1,
            normalization=MetricNormalization.BOUNDS,
            bounds=(0, 1),
            failure_policy=FailurePolicy.PENALIZE,
            penalty_score=0.1,
        ),
    )
    batch = EvaluationBatch(
        records=[record("a", "quality", {"quality_elo": 0.9})]
    )

    result = FitnessAggregator(config).aggregate(batch, candidate_ids=["a"])["a"]

    assert result.score == pytest.approx(0.5)
    assert result.penalized_metrics == ["diversity.diversity"]
    assert result.components["diversity.diversity"].status == FitnessContributionStatus.PENALIZED


def test_required_failure_raises_with_candidate_and_metric_context():
    config = suite(
        MetricConfig(
            component_id="quality",
            metric_name="quality_elo",
            normalization=MetricNormalization.NONE,
        )
    )
    batch = EvaluationBatch(
        records=[
            record(
                "a",
                "quality",
                status=EvaluationStatus.FAILED,
                error="judge timed out",
            )
        ]
    )

    with pytest.raises(RequiredMetricError, match="judge timed out") as exc_info:
        FitnessAggregator(config).aggregate(batch)

    assert exc_info.value.candidate_id == "a"
    assert exc_info.value.metric_key == "quality.quality_elo"


def test_all_optional_metrics_missing_produces_explicit_unscored_result():
    config = suite(
        MetricConfig(
            component_id="quality",
            metric_name="quality_elo",
            failure_policy=FailurePolicy.OMIT,
        )
    )

    result = FitnessAggregator(config).aggregate(
        EvaluationBatch(),
        candidate_ids=["a"],
    )["a"]

    assert result.score is None
    assert result.used_weight == 0
    assert result.omitted_metrics == ["quality.quality_elo"]


def test_cached_records_are_scored_like_successful_records():
    config = suite(
        MetricConfig(
            component_id="quality",
            metric_name="quality_elo",
            normalization=MetricNormalization.BOUNDS,
            bounds=(0, 1),
        )
    )
    batch = EvaluationBatch(
        records=[
            record(
                "a",
                "quality",
                {"quality_elo": 0.7},
                status=EvaluationStatus.CACHED,
            )
        ]
    )

    result = FitnessAggregator(config).aggregate(batch)["a"]

    assert result.score == pytest.approx(0.7)
    assert result.components["quality.quality_elo"].evaluation_status == EvaluationStatus.CACHED


def test_stateful_normalization_uses_evaluator_supplied_value():
    config = suite(
        MetricConfig(
            component_id="quality",
            metric_name="quality_elo",
            normalization=MetricNormalization.RUNNING_ZSCORE,
        )
    )
    batch = EvaluationBatch(
        records=[
            EvaluationRecord(
                candidate_id="a",
                component_id="quality",
                status=EvaluationStatus.OK,
                metrics={"quality_elo": 1725.0},
                normalized_metrics={"quality_elo": 0.73},
            )
        ]
    )

    result = FitnessAggregator(config).aggregate(batch)["a"]

    assert result.score == pytest.approx(0.73)
    assert result.components["quality.quality_elo"].raw_value == 1725.0


def test_stateful_normalization_requires_evaluator_supplied_value():
    config = suite(
        MetricConfig(
            component_id="quality",
            metric_name="quality_elo",
            normalization=MetricNormalization.RUNNING_ZSCORE,
        )
    )

    with pytest.raises(Exception, match="requires a normalized value"):
        FitnessAggregator(config).aggregate(
            EvaluationBatch(
                records=[record("a", "quality", {"quality_elo": 1725.0})]
            )
        )


def test_evaluator_normalized_value_only_applies_to_stateful_normalization():
    config = suite(
        MetricConfig(
            component_id="quality",
            metric_name="quality_elo",
            normalization=MetricNormalization.BOUNDS,
            bounds=(0, 100),
        )
    )
    batch = EvaluationBatch(
        records=[
            EvaluationRecord(
                candidate_id="a",
                component_id="quality",
                status=EvaluationStatus.OK,
                metrics={"quality_elo": 25.0},
                normalized_metrics={"quality_elo": 0.99},
            )
        ]
    )

    result = FitnessAggregator(config).aggregate(batch)["a"]

    assert result.score == pytest.approx(0.25)
