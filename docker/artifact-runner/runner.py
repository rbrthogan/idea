"""Trusted container entrypoint for generated visualization scripts."""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_RESULT_BYTES = 64 * 1024
MAX_IMAGE_DIMENSION = 1600


def _safe_error(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", message)
    return " ".join(message.split())[:1000]


def _regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} was not created as a regular file")


def _validate_png(path: Path) -> None:
    from PIL import Image

    _regular_file(path, "visualization.png")
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError("visualization.png is not a PNG")
        width, height = image.size
        if (
            width <= 0
            or height <= 0
            or width > MAX_IMAGE_DIMENSION
            or height > MAX_IMAGE_DIMENSION
        ):
            raise ValueError(
                "visualization.png dimensions must be between 1 and "
                f"{MAX_IMAGE_DIMENSION} pixels"
            )
        image.verify()


def _validate_result(path: Path) -> None:
    _regular_file(path, "result.json")
    if path.stat().st_size > MAX_RESULT_BYTES:
        raise ValueError(f"result.json exceeds {MAX_RESULT_BYTES} bytes")
    with path.open('rb') as stream:
        raw = stream.read(MAX_RESULT_BYTES + 1)
    if len(raw) > MAX_RESULT_BYTES:
        raise ValueError(f"result.json exceeds {MAX_RESULT_BYTES} bytes")
    try:
        result = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("result.json is not valid UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("result.json must contain one JSON object")
    if result.get("schema_version") != 1:
        raise ValueError("result.json schema_version must be 1")
    if result.get("status") != "ok":
        raise ValueError('result.json status must be "ok"')

    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or "structure_signal" not in metrics:
        raise ValueError("result.json metrics must include structure_signal")
    for name, value in metrics.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("result.json metrics must be finite numeric values")

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("result.json artifacts must be a list")
    primary = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("path") == "visualization.png"
        and artifact.get("role") == "primary"
        and artifact.get("media_type") == "image/png"
    ]
    if len(primary) != 1:
        raise ValueError(
            "result.json must declare visualization.png exactly once as "
            "the primary image/png artifact"
        )
    summary = result.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("result.json summary must be non-empty")
    if result.get("error") is not None:
        raise ValueError("successful result.json error must be null")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(2)

    # The validator lives in a separate interpreter. Generated source cannot
    # monkeypatch its functions or inherit saved host-facing descriptors.
    # Docker bounds the temporary filesystem and per-file writes; the host
    # independently caps pipe output even for malicious /proc descriptor use.
    with tempfile.TemporaryFile() as errors:
        try:
            child = subprocess.run(
                [sys.executable, '-I', sys.argv[1]],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=errors, close_fds=True, check=False,
                timeout=max(1, min(300, float(os.getenv('IDEA_EXECUTION_TIMEOUT', '30')))),
            )
        except subprocess.TimeoutExpired:
            print('Generated source exceeded its execution deadline', file=sys.stderr)
            raise SystemExit(124)
        if child.returncode != 0:
            errors.seek(0)
            detail = errors.read(2048).decode('utf-8', errors='replace')
            print(_safe_error(RuntimeError(detail or f'generated source exited {child.returncode}')), file=sys.stderr)
            raise SystemExit(1)

    output = Path("/output/visualization.png")
    result = Path("/output/result.json")
    try:
        _validate_png(output)
        _validate_result(result)
    except BaseException as exc:
        print(_safe_error(exc), file=sys.stderr)
        raise SystemExit(3)
    sys.stdout.buffer.write(output.read_bytes())
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
