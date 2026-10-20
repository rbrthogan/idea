import asyncio

import pytest

from idea.artifact_store import ArtifactStore
from idea.evaluation.artifacts import LLMCodeArtifactProducer
from idea.evaluation.models import EvaluationStatus, LLMCodeArtifactComponentConfig
from idea.evaluation.repair import (
    ArtifactAttemptError,
    ArtifactAttemptFailure,
    ArtifactFailureStage,
    ArtifactRepairExhaustedError,
    ArtifactRepairPolicy,
    run_artifact_repair_loop,
)


def _execution_failure(message="generated script failed"):
    return ArtifactAttemptError(
        ArtifactAttemptFailure(
            stage=ArtifactFailureStage.EXECUTION,
            message=message,
            stderr="Traceback: ValueError: bad plot",
            exit_code=1,
            missing_outputs=("visualization.png",),
            output_contract_failures=(
                "result.json did not declare a primary image/png artifact",
            ),
        )
    )


def test_success_never_calls_repairer():
    repair_calls = []

    result = asyncio.run(
        run_artifact_repair_loop(
            "print('valid')",
            lambda source: {"source": source, "png": b"png"},
            repairer=lambda request: repair_calls.append(request),
        )
    )

    assert repair_calls == []
    assert result.repair_attempts == 0
    assert [attempt.outcome for attempt in result.attempts] == ["ok"]


def test_structured_failure_is_sent_then_success_is_audited():
    executed_sources = []
    repair_payloads = []

    def execute(source):
        executed_sources.append(source)
        if len(executed_sources) == 1:
            raise _execution_failure()
        return b"\x89PNG\r\n\x1a\n"

    def repairer(request):
        repair_payloads.append(request.to_payload())
        return "print('fixed')"

    result = asyncio.run(
        run_artifact_repair_loop(
            "raise ValueError('bad plot')",
            execute,
            repairer=repairer,
        )
    )

    assert result.source_code == "print('fixed')\n"
    assert result.repair_attempts == 1
    assert [attempt.outcome for attempt in result.attempts] == ["failed", "ok"]
    assert len({attempt.source_sha256 for attempt in result.attempts}) == 2
    payload = repair_payloads[0]
    assert payload["schema_version"] == 1
    assert payload["failure"]["stage"] == "execution"
    assert payload["failure"]["exit_code"] == 1
    assert payload["failure"]["missing_outputs"] == ["visualization.png"]
    assert payload["attempt_history"] == [payload["failure"]]
    assert payload["repair_attempt"] == 1
    assert payload["max_repair_attempts"] == 1


def test_repair_is_bounded_and_final_cause_is_auditable():
    repair_calls = []

    def execute(_source):
        raise _execution_failure("visualization.png is missing")

    def repairer(request):
        repair_calls.append(request)
        return f"print('repair {request.repair_attempt}')"

    with pytest.raises(ArtifactRepairExhaustedError) as raised:
        asyncio.run(
            run_artifact_repair_loop(
                "print('initial')",
                execute,
                repairer=repairer,
                policy=ArtifactRepairPolicy(max_repair_attempts=2),
            )
        )

    diagnostics = raised.value.diagnostics["repair_loop"]
    assert len(repair_calls) == 2
    assert diagnostics["execution_attempts"] == 3
    assert diagnostics["repair_attempts"] == 2
    assert diagnostics["final_cause"]["message"] == "visualization.png is missing"


def test_repeated_repair_source_is_not_reexecuted_and_feedback_escalates():
    executed_sources = []
    repair_requests = []

    def execute(source):
        executed_sources.append(source)
        if "fixed" not in source:
            raise _execution_failure("container timed out")
        return b"png"

    def repairer(request):
        repair_requests.append(request)
        if request.repair_attempt < 3:
            return "print('initial')"
        return "print('fixed')"

    result = asyncio.run(
        run_artifact_repair_loop(
            "print('initial')",
            execute,
            repairer=repairer,
            policy=ArtifactRepairPolicy(max_repair_attempts=4),
        )
    )

    assert len(repair_requests) == 3
    assert len(executed_sources) == 2
    assert repair_requests[1].repair_feedback
    assert "materially different" in repair_requests[1].repair_feedback[0]
    assert result.repair_attempts == 3


def test_unstructured_infrastructure_failure_does_not_call_repairer():
    repair_calls = []

    with pytest.raises(RuntimeError, match="Docker daemon is unavailable"):
        asyncio.run(
            run_artifact_repair_loop(
                "print('valid')",
                lambda _source: (_ for _ in ()).throw(
                    RuntimeError("Docker daemon is unavailable")
                ),
                repairer=lambda request: repair_calls.append(request),
            )
        )

    assert repair_calls == []


def test_invalid_repair_source_becomes_explicit_final_cause():
    with pytest.raises(ArtifactRepairExhaustedError) as raised:
        asyncio.run(
            run_artifact_repair_loop(
                "print('initial')",
                lambda _source: (_ for _ in ()).throw(_execution_failure()),
                repairer=lambda _request: "",
            )
        )

    assert raised.value.final_cause == {
        "stage": "repair",
        "message": "repair adapter returned no Python source",
    }


def test_producer_persists_attempts_and_final_cause(tmp_path):
    class FailingRunner:
        async def run(self, *_args, **_kwargs):
            await run_artifact_repair_loop(
                "print('initial')",
                lambda _source: (_ for _ in ()).throw(
                    _execution_failure("visualization.png is missing")
                ),
                repairer=lambda _request: "print('fixed')",
            )
            raise AssertionError("repair loop should have raised")

    producer = LLMCodeArtifactProducer(
        LLMCodeArtifactComponentConfig(
            id="render",
            execution_profile="local-container-v1",
            failure_policy="omit",
        ),
        ArtifactStore(tmp_path / "artifacts"),
    )
    candidate = {
        "id": "candidate-1",
        "idea": {"title": "Prime plot", "content": "Render primes."},
    }

    result = asyncio.run(
        producer.evaluate_population([candidate], FailingRunner())
    )

    record = result.batch.records[0]
    assert record.status == EvaluationStatus.FAILED
    repair_loop = record.metadata["execution"]["repair_loop"]
    assert repair_loop["execution_attempts"] == 2
    assert repair_loop["repair_attempts"] == 1
    assert repair_loop["final_cause"]["missing_outputs"] == [
        "visualization.png"
    ]
