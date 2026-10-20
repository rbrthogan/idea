import asyncio

import pytest

from idea.evaluation.ollama_repair import OllamaArtifactRepairer
from idea.evaluation.repair import (
    ArtifactAttemptFailure,
    ArtifactFailureStage,
    ArtifactRepairRequest,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, path, *, json):
        self.calls.append((path, json))
        return _Response(self.payload)


def _request():
    return ArtifactRepairRequest(
        source_code="print(undefined_name)",
        failure=ArtifactAttemptFailure(
            stage=ArtifactFailureStage.EXECUTION,
            message="generated source failed",
            stderr="NameError: name 'undefined_name' is not defined",
            exit_code=1,
        ),
        repair_attempt=1,
        max_repair_attempts=2,
    )


def test_ollama_repairer_sends_bounded_failure_and_returns_plain_source():
    client = _Client(
        {
            "message": {"content": "```python\nprint('fixed')\n```"},
            "prompt_eval_count": 120,
            "eval_count": 16,
            "total_duration": 4_000_000,
            "load_duration": 1_000_000,
        }
    )
    repairer = OllamaArtifactRepairer(client=client)

    source = asyncio.run(repairer(_request()))

    assert source == "print('fixed')\n"
    path, payload = client.calls[0]
    assert path == "/api/chat"
    assert payload["model"] == "qwen3.5:9b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["num_ctx"] == 16384
    assert payload["options"]["num_predict"] == 4096
    prompt = payload["messages"][1]["content"]
    assert "NameError" in prompt
    assert "print(undefined_name)" in prompt
    assert "complete replacement Python source only" in prompt
    assert repairer.get_usage_summary() == {
        "requests": 1,
        "input": 120,
        "output": 16,
        "total_duration_ns": 4_000_000,
        "load_duration_ns": 1_000_000,
        "model": "qwen3.5:9b",
        "provider": "ollama",
        "cost": 0.0,
    }


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://127.0.0.1:11434?token=secret",
        "http://user:password@localhost:11434",
    ],
)
def test_ollama_repairer_rejects_non_loopback_or_credentialed_urls(url):
    with pytest.raises(ValueError, match="loopback"):
        OllamaArtifactRepairer(base_url=url, client=_Client({}))


def test_ollama_repairer_rejects_empty_source_response():
    repairer = OllamaArtifactRepairer(
        client=_Client({"message": {"content": ""}})
    )

    with pytest.raises(ValueError, match="no Python source"):
        asyncio.run(repairer(_request()))


def test_repeated_timeout_prompt_requires_bounded_fallback_and_includes_history():
    timeout_failure = ArtifactAttemptFailure(
        stage=ArtifactFailureStage.EXECUTION,
        message="local artifact container timed out",
        timed_out=True,
    )
    request = ArtifactRepairRequest(
        source_code="expensive_spectral_plot()",
        failure=timeout_failure,
        repair_attempt=3,
        max_repair_attempts=4,
        attempt_history=(timeout_failure, timeout_failure, timeout_failure),
        repair_feedback=("Previous source repeated a failed version.",),
    )

    prompt = OllamaArtifactRepairer._prompt(request)

    assert "repair attempt 3 of 4" in prompt
    assert "timeout failure 3" in prompt
    assert "materially different" in prompt
    assert "bounded representative sample" in prompt
    assert '"attempt_history"' in prompt
    assert "Previous source repeated" in prompt
