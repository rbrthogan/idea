import asyncio
import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from idea.evaluation.local_artifacts import (
    GeminiContainerArtifactRunner,
    LocalArtifactExecutionError,
    OpenAIContainerArtifactRunner,
    _container_error_text,
)
from idea.evaluation.models import LLMCodeArtifactComponentConfig
from idea.evaluation.repair import ArtifactRepairRequest


class _Response:
    def __init__(self, text):
        self.text = text


class _OpenAISourceLLM:
    def __init__(self, source):
        self.source = source
        self.calls = []
        self.usage = {
            "total": 0,
            "input": 0,
            "output": 0,
            "thoughts": 0,
            "cached_input": 0,
            "cache_write_input": 0,
            "uncached_input": 0,
            "tool_input": 0,
            "requests": 0,
            "model": "gpt-6-luna",
            "cost": 0.0,
        }

    def generate_text(self, prompt, temperature, top_p, response_schema):
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "response_schema": response_schema,
                "max_tokens": self.MAX_TOKENS,
            }
        )
        self.usage.update(
            {
                "total": 128,
                "input": 48,
                "output": 64,
                "thoughts": 16,
                "uncached_input": 48,
                "requests": 1,
                "cost": 0.00004,
            }
        )
        return self.source

    def get_token_usage_summary(self):
        return dict(self.usage)


def _container_runner_namespace():
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "docker"
        / "artifact-runner"
        / "runner.py"
    )
    return runpy.run_path(str(runner_path))


def test_container_contract_requires_valid_result(tmp_path):
    namespace = _container_runner_namespace()
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ok",
                "metrics": {"structure_signal": 0.25},
                "artifacts": [
                    {
                        "path": "visualization.png",
                        "role": "primary",
                        "media_type": "image/png",
                    }
                ],
                "summary": "Controlled render completed.",
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    namespace["_validate_result"](result_path)


def test_container_contract_rejects_result_without_structure_signal(tmp_path):
    namespace = _container_runner_namespace()
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ok",
                "metrics": {},
                "artifacts": [],
                "summary": "Missing required metric.",
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="structure_signal"):
        namespace["_validate_result"](result_path)


def test_local_runner_accepts_plain_python_source():
    source = GeminiContainerArtifactRunner._response_source(
        _Response("import matplotlib.pyplot as plt\nplt.plot([1, 2])")
    )

    assert source == "import matplotlib.pyplot as plt\nplt.plot([1, 2])\n"


def test_local_runner_unwraps_markdown_fence():
    source = GeminiContainerArtifactRunner._response_source(
        _Response("```python\nprint('hello')\n```")
    )

    assert source == "print('hello')\n"


def test_local_runner_keeps_legacy_json_compatibility():
    source = GeminiContainerArtifactRunner._response_source(
        _Response('{"source_code":"print(\\"hello\\")\\n"}')
    )

    assert source == 'print("hello")\n'


def test_local_runner_reports_bounded_container_error():
    def fake_run(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout=b"",
                stderr=b"ValueError: bad generated plot\n",
            )
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0)
        raise AssertionError(command)

    runner = GeminiContainerArtifactRunner(
        enabled=True,
        api_key="test",
        image="idea-artifact-runner:py310-v1",
    )
    with (
        patch("idea.evaluation.local_artifacts.subprocess.run", side_effect=fake_run),
        patch("idea.evaluation.local_artifacts.run_bounded", side_effect=fake_run),
        pytest.raises(
            LocalArtifactExecutionError,
            match="ValueError: bad generated plot",
        ),
    ):
        runner._execute_source("raise ValueError('bad')", max_png_bytes=1024)


def test_container_error_omits_docker_platform_warning():
    error = _container_error_text(
        b"WARNING: The requested image's platform does not match the host\n"
        b"ValueError: bad generated plot\n"
    )

    assert error == "ValueError: bad generated plot"


def test_local_runner_repairs_generated_source_once():
    requests = []

    def generate_content(**kwargs):
        requests.append(kwargs)
        return _Response("raise ValueError('bad plot')")

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    repair_calls = []

    def repair_hook(request):
        assert isinstance(request, ArtifactRepairRequest)
        repair_calls.append(request.to_payload())
        return "print('fixed plot')"

    runner = GeminiContainerArtifactRunner(
        enabled=True,
        client=client,
        image="idea-artifact-runner:py310-v1",
        repair_hook=repair_hook,
    )
    config = LLMCodeArtifactComponentConfig(
        id="render",
        execution_profile="gemini-visualization-v1",
    )

    with patch.object(
        runner,
        "_execute_source",
        side_effect=[
            LocalArtifactExecutionError("local artifact script failed: ValueError"),
            b"\x89PNG\r\n\x1a\nfixed",
        ],
    ) as execute:
        result = asyncio.run(
            runner.run(
                {"idea": {"title": "Test", "content": "Plot a line."}},
                candidate_id="candidate-1",
                generation=0,
                config=config,
            )
        )

    assert execute.call_count == 2
    assert len(requests) == 1
    assert len(repair_calls) == 1
    assert result.source_code == "print('fixed plot')\n"
    assert result.provenance["source_repaired"] is True
    assert result.provenance["execution_attempts"] == 2
    assert result.provenance["repair_attempts"] == 1
    assert result.provenance["repair_provider"] == "function"


def test_openai_local_runner_uses_luna_for_source_and_local_container_for_execution():
    source_llm = _OpenAISourceLLM("print('rendered')")
    runner = OpenAIContainerArtifactRunner(
        enabled=True,
        source_llm=source_llm,
        image="idea-artifact-runner:py310-v1",
    )
    config = LLMCodeArtifactComponentConfig(
        id="render",
        execution_profile="matplotlib-prime-v1",
        max_output_tokens=4096,
    )

    with patch.object(
        runner,
        "_execute_source",
        return_value=b"\x89PNG\r\n\x1a\nrendered",
    ):
        result = asyncio.run(
            runner.run(
                {"idea": {"title": "Prime map", "content": "Plot prime gaps."}},
                candidate_id="candidate-1",
                generation=2,
                config=config,
            )
        )

    assert len(source_llm.calls) == 1
    assert source_llm.calls[0]["max_tokens"] == 4096
    assert "<candidate-json>" in source_llm.calls[0]["prompt"]
    assert "finish within 30 seconds" in source_llm.calls[0]["prompt"]
    assert "dense eigendecompositions above 250 by 250" in source_llm.calls[0]["prompt"]
    assert result.source_code == "print('rendered')\n"
    assert result.provenance["execution_location"] == "local_container"
    assert result.provenance["source_provider"] == "openai"
    assert result.provenance["source_model"] == "gpt-6-luna"
    assert result.provenance["source_reasoning_effort"] == "low"
    assert result.usage["input"] == 48
    assert runner.get_token_usage_summary()["cost"] == pytest.approx(0.00004)


def test_local_runner_uses_restricted_docker_boundary(tmp_path):
    png = b"\x89PNG\r\n\x1a\nminimal"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(command, 0, stdout=png)
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0)
        raise AssertionError(command)

    runner = GeminiContainerArtifactRunner(
        enabled=True,
        api_key="test",
        image="idea-artifact-runner:py310-v1",
    )
    with patch("idea.evaluation.local_artifacts.subprocess.run", side_effect=fake_run), \
         patch("idea.evaluation.local_artifacts.run_bounded", side_effect=fake_run):
        result = runner._execute_source("print('safe')", max_png_bytes=1024)

    command = next(command for command in commands if command[:2] == ["docker", "run"])
    assert result == png
    assert ["--network", "none"] == command[
        command.index("--network"):command.index("--network") + 2
    ]
    assert "--read-only" in command
    assert ["--cap-drop", "ALL"] == command[
        command.index("--cap-drop"):command.index("--cap-drop") + 2
    ]
    assert "no-new-privileges" in command
    assert any(part.startswith("/output:") and "size=32m" in part for part in command)
    assert not any(str(Path(__file__).resolve().parents[1]) in part for part in command)
