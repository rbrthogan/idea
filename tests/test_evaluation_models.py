import pytest
from pydantic import ValidationError

from idea.evaluation import (
    ArtifactRef,
    ComponentKind,
    EvaluationBatch,
    EvaluationRecord,
    EvaluationStatus,
    EvaluationSuiteConfig,
    FailurePolicy,
    FitnessAggregationConfig,
    LLMPairwiseComponentConfig,
    LLMCodeArtifactComponentConfig,
    MetricConfig,
    MetricNormalization,
    PythonMetricComponentConfig,
    resolve_evaluation_suite,
    resolve_legacy_evaluation_suite,
)


DIGEST = "a" * 64


def test_suite_config_round_trips_discriminated_components():
    suite = EvaluationSuiteConfig(
        components=[
            LLMPairwiseComponentConfig(id="quality"),
            PythonMetricComponentConfig(
                id="numeric",
                script_ref="scripts/score.py",
                metric_names=["score"],
                failure_policy=FailurePolicy.OMIT,
            ),
        ],
        fitness=FitnessAggregationConfig(
            metrics=[
                MetricConfig(
                    component_id="quality",
                    metric_name="quality_elo",
                    weight=0.7,
                ),
                MetricConfig(
                    component_id="numeric",
                    metric_name="score",
                    weight=0.3,
                    normalization=MetricNormalization.BOUNDS,
                    bounds=(0, 100),
                ),
            ]
        ),
    )

    restored = EvaluationSuiteConfig.model_validate(
        suite.model_dump(mode="json")
    )

    assert restored.schema_version == 1
    assert restored.components[0].kind == ComponentKind.LLM_PAIRWISE
    assert isinstance(restored.components[1], PythonMetricComponentConfig)
    assert restored.fitness.metrics[1].bounds == (0.0, 100.0)


def test_versioned_models_reject_unknown_schema_versions():
    with pytest.raises(ValidationError, match="schema_version"):
        EvaluationSuiteConfig.model_validate(
            {
                "schema_version": 2,
                "enabled": False,
                "components": [],
                "fitness": {"enabled": False, "metrics": []},
            }
        )


def test_suite_rejects_duplicate_or_unknown_component_references():
    duplicate = LLMPairwiseComponentConfig(id="quality")
    with pytest.raises(ValidationError, match="component ids must be unique"):
        EvaluationSuiteConfig(
            components=[duplicate, duplicate.model_copy()],
            fitness=FitnessAggregationConfig(
                metrics=[
                    MetricConfig(
                        component_id="quality",
                        metric_name="quality_elo",
                    )
                ]
            ),
        )

    with pytest.raises(ValidationError, match="unknown component"):
        EvaluationSuiteConfig(
            components=[duplicate],
            fitness=FitnessAggregationConfig(
                metrics=[
                    MetricConfig(
                        component_id="missing",
                        metric_name="score",
                    )
                ]
            ),
        )


@pytest.mark.parametrize(
    ("components", "kind"),
    [
        (
            [
                LLMPairwiseComponentConfig(id="quality-a"),
                LLMPairwiseComponentConfig(id="quality-b"),
            ],
            "llm_pairwise",
        ),
        (
            [
                {
                    "id": "diversity-a",
                    "kind": "embedding_diversity",
                },
                {
                    "id": "diversity-b",
                    "kind": "embedding_diversity",
                },
            ],
            "embedding_diversity",
        ),
    ],
)
def test_schema_v1_rejects_duplicate_enabled_singleton_evaluator_kinds(
    components,
    kind,
):
    with pytest.raises(ValidationError, match=f"only one enabled {kind}"):
        EvaluationSuiteConfig(
            components=components,
            fitness=FitnessAggregationConfig(
                metrics=[
                    MetricConfig(
                        component_id=components[0].id
                        if isinstance(components[0], LLMPairwiseComponentConfig)
                        else components[0]["id"],
                        metric_name="quality_elo"
                        if kind == "llm_pairwise"
                        else "diversity",
                    )
                ]
            ),
        )


def test_suite_rejects_fitness_metrics_the_component_does_not_emit():
    with pytest.raises(ValidationError, match="not emitted"):
        EvaluationSuiteConfig(
            components=[LLMPairwiseComponentConfig(id="quality")],
            fitness=FitnessAggregationConfig(
                metrics=[
                    MetricConfig(
                        component_id="quality",
                        metric_name="misspelled_score",
                    )
                ]
            ),
        )

    with pytest.raises(ValidationError, match="emitted metrics: none"):
        EvaluationSuiteConfig(
            allow_generated_code_execution=True,
            components=[
                LLMCodeArtifactComponentConfig(
                    id="render",
                    execution_profile="matplotlib-v1",
                )
            ],
            fitness=FitnessAggregationConfig(
                metrics=[
                    MetricConfig(
                        component_id="render",
                        metric_name="score",
                    )
                ]
            ),
        )


def test_pairwise_component_rejects_unimplemented_model_override():
    with pytest.raises(ValidationError, match="model overrides are not supported"):
        LLMPairwiseComponentConfig(
            id="quality",
            model="gemini-example",
        )


@pytest.mark.parametrize(
    "script_ref",
    ["../score.py", "/tmp/score.py", r"C:\\score.py", "scripts/../score.py", "score.sh"],
)
def test_python_metric_rejects_unsafe_script_references(script_ref):
    with pytest.raises(ValidationError, match="traversal-free"):
        PythonMetricComponentConfig(
            id="numeric",
            script_ref=script_ref,
            metric_names=["score"],
        )


def test_bounds_normalization_requires_ordered_finite_bounds():
    with pytest.raises(ValidationError, match="requires bounds"):
        MetricConfig(
            component_id="score",
            metric_name="value",
            normalization=MetricNormalization.BOUNDS,
        )

    with pytest.raises(ValidationError, match="lower < upper"):
        MetricConfig(
            component_id="score",
            metric_name="value",
            normalization=MetricNormalization.BOUNDS,
            bounds=(10, 10),
        )


def test_artifacts_records_and_batches_are_typed_and_versioned():
    artifact = ArtifactRef(
        id="plot",
        role="primary",
        media_type="image/png",
        storage_key="users/u/evolutions/e/plot.png",
        sha256=DIGEST,
        size_bytes=123,
        producer_id="render",
        content_digest=DIGEST,
    )
    record = EvaluationRecord(
        candidate_id="candidate-a",
        component_id="visual-judge",
        status=EvaluationStatus.OK,
        metrics={"usefulness": 0.9},
        artifact_ids=[artifact.id],
        input_digest=DIGEST,
    )
    batch = EvaluationBatch(generation=2, records=[record], artifacts=[artifact])

    assert batch.record_for("candidate-a", "visual-judge") == record
    assert batch.candidate_ids() == ["candidate-a"]
    assert batch.model_dump(mode="json")["records"][0]["status"] == "ok"


def test_failed_records_require_errors_and_metrics_must_be_finite():
    with pytest.raises(ValidationError, match="require an error"):
        EvaluationRecord(
            candidate_id="candidate-a",
            component_id="score",
            status=EvaluationStatus.FAILED,
        )

    with pytest.raises(ValidationError, match="finite number"):
        EvaluationRecord(
            candidate_id="candidate-a",
            component_id="score",
            status=EvaluationStatus.OK,
            metrics={"score": float("nan")},
        )


def test_legacy_resolver_maps_fitness_alpha_to_quality_and_diversity():
    suite = resolve_legacy_evaluation_suite(0.8)

    assert [component.id for component in suite.components] == [
        "quality_elo",
        "diversity",
    ]
    assert [metric.weight for metric in suite.fitness.metrics] == pytest.approx([0.8, 0.2])
    assert all(
        metric.normalization == MetricNormalization.RUNNING_ZSCORE
        for metric in suite.fitness.metrics
    )
    assert suite.metadata == {
        "source": "legacy_fitness_alpha",
        "fitness_alpha": 0.8,
    }

    assert resolve_legacy_evaluation_suite(2).fitness.metrics[0].weight == 1.0
    assert resolve_legacy_evaluation_suite(-1).fitness.metrics[0].weight == 0.0
    assert resolve_evaluation_suite(None, fitness_alpha="bad").fitness.metrics[0].weight == 0.7
    assert resolve_legacy_evaluation_suite(float("nan")).fitness.metrics[0].weight == 0.7
    assert resolve_legacy_evaluation_suite(float("inf")).fitness.metrics[0].weight == 0.7


def test_resolve_evaluation_suite_accepts_explicit_serialized_config():
    explicit = resolve_legacy_evaluation_suite(0.25)

    restored = resolve_evaluation_suite(explicit.model_dump(mode="json"), fitness_alpha=0.9)

    assert restored == explicit


def test_components_round_trip_evidence_dependencies():
    component = LLMPairwiseComponentConfig(
        id="quality",
        requires=["candidate.text"],
        provides=["metric.quality_elo"],
    )

    restored = LLMPairwiseComponentConfig.model_validate(
        component.model_dump(mode="json")
    )

    assert restored.requires == ["candidate.text"]
    assert restored.provides == ["metric.quality_elo"]


def test_generated_code_component_requires_explicit_execution_opt_in():
    component = LLMCodeArtifactComponentConfig(
        id="render",
        execution_profile="matplotlib-prime-v1",
        provides=["artifact.primary", "artifact.source"],
    )

    with pytest.raises(ValidationError, match="explicit"):
        EvaluationSuiteConfig(
            components=[component],
            fitness=FitnessAggregationConfig(enabled=False),
        )

    suite = EvaluationSuiteConfig(
        allow_generated_code_execution=True,
        components=[component],
        fitness=FitnessAggregationConfig(enabled=False),
    )

    assert suite.allow_generated_code_execution is True
