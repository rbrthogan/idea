import asyncio
import os

import pytest

from idea.evaluation.local_artifacts import (
    GeminiContainerArtifactRunner,
    local_container_runtime_status,
)
from idea.evaluation.ollama_repair import OllamaArtifactRepairer
from idea.evaluation.repair import (
    ArtifactAttemptFailure,
    ArtifactFailureStage,
    ArtifactRepairRequest,
)


pytestmark = [pytest.mark.live, pytest.mark.skipif(
    os.environ.get("IDEA_RUN_OLLAMA_LIVE_TESTS") != "1",
    reason="set IDEA_RUN_OLLAMA_LIVE_TESTS=1 for the local Ollama integration",
)]


BROKEN_VISUALIZATION = """import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 4)
ax.scatter([2, 3, 5, 7, 11], [1, 2, 3, 4, 5])
ax.set_title("Prime positions")
fig.savefig("visualization.png", dpi=100)
plt.close(fig)

result = {
    "schema_version": 1,
    "status": "ok",
    "metrics": {"structure_signal": 0.5},
    "artifacts": [{
        "path": "visualization.png",
        "role": "primary",
        "media_type": "image/png",
    }],
    "summary": "Prime position smoke test.",
    "error": None,
}
with open("result.json", "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"""


def test_default_qwen_repairs_and_container_executes_broken_visualization():
    request = ArtifactRepairRequest(
        source_code=BROKEN_VISUALIZATION,
        failure=ArtifactAttemptFailure(
            stage=ArtifactFailureStage.SOURCE_VALIDATION,
            message="generated Python is invalid: '(' was never closed",
            stderr="'(' was never closed",
        ),
        repair_attempt=1,
        max_repair_attempts=2,
    )
    repairer = OllamaArtifactRepairer(timeout_seconds=300)

    repaired = asyncio.run(repairer(request))

    compile(repaired, "<qwen-repaired-visualization>", "exec")
    assert "visualization.png" in repaired
    assert "result.json" in repaired
    usage = repairer.get_usage_summary()
    assert usage["requests"] == 1
    assert usage["input"] > 0
    assert usage["output"] > 0

    runtime = local_container_runtime_status()
    if not runtime["available"]:
        pytest.skip(f"local artifact container unavailable: {runtime['image']}")
    runner = GeminiContainerArtifactRunner(enabled=True)
    png = runner._execute_source(repaired, max_png_bytes=5 * 1024 * 1024)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
