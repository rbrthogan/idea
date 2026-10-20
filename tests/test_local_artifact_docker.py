"""Opt-in real-Docker checks for the local artifact production boundary."""

import asyncio
import os
from types import SimpleNamespace

import pytest

from idea.artifact_store import ArtifactStore
from idea.evaluation.artifacts import ArtifactStoreLoader, LLMCodeArtifactProducer
from idea.evaluation.local_artifacts import (
    GeminiContainerArtifactRunner,
    LocalArtifactExecutionError,
)
from idea.evaluation.models import (
    EvaluationStatus,
    LLMCodeArtifactComponentConfig,
)


pytestmark = [pytest.mark.live, pytest.mark.skipif(
    os.getenv("IDEA_RUN_DOCKER_TESTS") != "1",
    reason="set IDEA_RUN_DOCKER_TESTS=1 to exercise the real Docker boundary",
)]

RUNNER_IMAGE = "idea-artifact-runner:py310-v4"
VALID_SOURCE = """
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
ax.plot([0, 1, 2], [0, 1, 0])
ax.set_title("Controlled local artifact")
fig.savefig("visualization.png")
plt.close(fig)

with open("result.json", "w", encoding="utf-8") as output:
    json.dump({
        "schema_version": 1,
        "status": "ok",
        "metrics": {"structure_signal": 0.5},
        "artifacts": [{
            "path": "visualization.png",
            "role": "primary",
            "media_type": "image/png"
        }],
        "summary": "Controlled local artifact completed.",
        "error": None
    }, output)
""".strip()


class _Response:
    text = VALID_SOURCE
    usage_metadata = {
        "prompt_token_count": 10,
        "candidates_token_count": 20,
        "total_token_count": 30,
    }


def _runner():
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_kwargs: _Response())
    )
    return GeminiContainerArtifactRunner(
        enabled=True,
        client=client,
        image=RUNNER_IMAGE,
    )


def _repairing_runner(repair_requests):
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **_kwargs: SimpleNamespace(
                text="raise ValueError('controlled initial failure')",
                usage_metadata={},
            )
        )
    )

    def repairer(request):
        repair_requests.append(request.to_payload())
        return VALID_SOURCE

    return GeminiContainerArtifactRunner(
        enabled=True,
        client=client,
        image=RUNNER_IMAGE,
        repair_hook=repairer,
    )


def _config():
    return LLMCodeArtifactComponentConfig(
        id="render",
        execution_profile="gemini-visualization-v1",
        provides=[
            "artifact.primary",
            "artifact.source",
            "artifact.metadata",
        ],
    )


def test_controlled_local_artifact_produces_persisted_png_source_and_metadata(
    tmp_path,
):
    candidate = {
        "id": "controlled-candidate",
        "idea": {"title": "Controlled", "content": "Plot a small line."},
    }
    store = ArtifactStore(tmp_path / "artifact-store")
    producer = LLMCodeArtifactProducer(_config(), store)

    result = asyncio.run(
        producer.evaluate_population(
            [candidate],
            _runner(),
            evaluation_generation=0,
        )
    )

    assert result.batch.records[0].status == EvaluationStatus.OK
    refs = candidate["artifact_refs"]
    assert {ref["role"] for ref in refs} == {"primary", "source", "metadata"}
    primary = next(ref for ref in refs if ref["role"] == "primary")
    assert ArtifactStoreLoader(store).load(primary).startswith(b"\x89PNG\r\n\x1a\n")
    assert (
        candidate["evaluations"]["render"]["metadata"]["provenance"][
            "execution_attempts"
        ]
        == 1
    )


def test_controlled_local_artifact_rejects_missing_result_contract():
    source = VALID_SOURCE.split('with open("result.json"', 1)[0]

    with pytest.raises(
        LocalArtifactExecutionError,
        match="result.json was not created",
    ) as failure:
        _runner()._execute_source(source, max_png_bytes=1024 * 1024)

    assert failure.value.diagnostics["stage"] == "artifact_contract_validation"
    assert failure.value.diagnostics["container_exit_code"] == 3


def test_controlled_local_artifact_repairs_once_without_external_model_cost(
    tmp_path,
):
    repair_requests = []
    candidate = {
        "id": "controlled-repair-candidate",
        "idea": {"title": "Controlled repair", "content": "Plot a small line."},
    }
    store = ArtifactStore(tmp_path / "artifact-store")
    producer = LLMCodeArtifactProducer(_config(), store)

    result = asyncio.run(
        producer.evaluate_population(
            [candidate],
            _repairing_runner(repair_requests),
            evaluation_generation=0,
        )
    )

    record = result.batch.records[0]
    assert record.status == EvaluationStatus.OK
    assert len(repair_requests) == 1
    assert repair_requests[0]["failure"]["stage"] == "execution"
    provenance = record.metadata["provenance"]
    assert provenance["execution_attempts"] == 2
    assert provenance["repair_attempts"] == 1
    assert provenance["repair_loop"]["outcome"] == "ok"
    assert [attempt["outcome"] for attempt in provenance["repair_loop"]["attempts"]] == [
        "failed",
        "ok",
    ]
    primary = next(
        ref for ref in candidate["artifact_refs"] if ref["role"] == "primary"
    )
    assert ArtifactStoreLoader(store).load(primary).startswith(b"\x89PNG\r\n\x1a\n")
