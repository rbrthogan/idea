import asyncio

import pytest
from pydantic import ValidationError

from idea.evaluation import (
    EvaluationStatus,
    PythonMetricComponentConfig,
    PythonMetricEvaluator,
    ScriptResultError,
    ScriptResultManifest,
    parse_script_result,
)


def test_script_result_contract_parses_metrics_and_artifact_manifest():
    result = parse_script_result(
        b'''{
          "schema_version": 1,
          "status": "ok",
          "metrics": {"score": 0.82, "plddt": 91.4},
          "artifacts": [{
            "path": "visualization.png",
            "role": "primary",
            "media_type": "image/png"
          }],
          "summary": "Rendered successfully"
        }'''
    )

    assert result.metrics == {"score": 0.82, "plddt": 91.4}
    assert result.artifacts[0].path == "visualization.png"


@pytest.mark.parametrize(
    "path",
    ["../escape.png", "/tmp/escape.png", "nested/../escape.png", "C:\\escape.png"],
)
def test_script_result_contract_rejects_unsafe_artifact_paths(path):
    with pytest.raises(ValidationError, match="relative"):
        parse_script_result(
            {
                "status": "ok",
                "artifacts": [
                    {"path": path, "role": "primary", "media_type": "image/png"}
                ],
            }
        )


def test_script_result_contract_rejects_nonfinite_and_oversized_json():
    with pytest.raises(ScriptResultError, match="non-finite"):
        parse_script_result('{"status":"ok","metrics":{"score":NaN}}')

    with pytest.raises(ScriptResultError, match="size limit"):
        parse_script_result("x" * 20, max_bytes=10)

    with pytest.raises(ScriptResultError, match="size limit"):
        parse_script_result({"status": "ok", "summary": "x" * 20}, max_bytes=10)


def test_failed_script_result_requires_error_and_no_metrics():
    with pytest.raises(ValidationError, match="require an error"):
        ScriptResultManifest(status="failed")

    with pytest.raises(ValidationError, match="cannot publish metrics"):
        ScriptResultManifest(
            status="failed",
            error="boom",
            metrics={"score": 0.0},
        )


class _Runner:
    async def run(self, candidate, **kwargs):
        if candidate["id"] == "bad":
            raise RuntimeError("container timed out")
        return ScriptResultManifest(
            status="ok",
            metrics={"score": 0.75, "aux": 2.0},
            summary="measured",
        )


def test_python_metric_evaluator_uses_runner_boundary_and_records_failures():
    population = [
        {"id": "good", "idea": "A"},
        {"id": "bad", "idea": "B"},
    ]
    evaluator = PythonMetricEvaluator(
        PythonMetricComponentConfig(
            id="numeric",
            script_ref="evaluators/score.py",
            metric_names=["score"],
        )
    )

    result = asyncio.run(
        evaluator.evaluate_population(
            population,
            _Runner(),
            evaluation_generation=4,
        )
    )

    assert result.scores == {0: 0.75}
    assert result.batch.records[0].status == EvaluationStatus.OK
    assert result.batch.records[0].metrics == {"score": 0.75}
    assert result.batch.records[1].status == EvaluationStatus.FAILED
    assert result.batch.records[1].error == "container timed out"
    assert population[1]["evaluations"]["numeric"]["status"] == "failed"


def test_python_metric_evaluator_requires_all_configured_metrics():
    evaluator = PythonMetricEvaluator(
        PythonMetricComponentConfig(
            id="numeric",
            script_ref="evaluators/score.py",
            metric_names=["score", "missing"],
        )
    )

    result = asyncio.run(
        evaluator.evaluate_population(
            [{"id": "good", "idea": "A"}],
            _Runner(),
        )
    )

    assert result.batch.records[0].status == EvaluationStatus.FAILED
    assert "omitted configured metrics" in result.batch.records[0].error
