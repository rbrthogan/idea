from __future__ import annotations

import json
import math
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCRIPT_RESULT_SCHEMA_VERSION = 1
MAX_SCRIPT_RESULT_BYTES = 1024 * 1024


class ScriptResultError(ValueError):
    """A script result did not satisfy the evaluator handoff contract."""


class ScriptArtifactOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=512)
    role: str = Field(default="primary", min_length=1, max_length=128)
    media_type: str = Field(min_length=1, max_length=255)

    @field_validator("path")
    @classmethod
    def validate_relative_output_path(cls, path: str) -> str:
        if "\x00" in path:
            raise ValueError("artifact path must not contain NUL bytes")
        normalized = path.replace("\\", "/")
        windows_path = PureWindowsPath(path)
        posix_path = PurePosixPath(normalized)
        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or any(part in {"", ".", ".."} for part in normalized.split("/"))
        ):
            raise ValueError("artifact path must be relative and traversal-free")
        return normalized


class ComputedResultManifest(BaseModel):
    """Strict result returned by a computed evaluator or artifact producer."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[SCRIPT_RESULT_SCHEMA_VERSION] = SCRIPT_RESULT_SCHEMA_VERSION
    status: Literal["ok", "failed"]
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifacts: List[ScriptArtifactOutput] = Field(default_factory=list)
    summary: Optional[str] = Field(default=None, max_length=4000)
    error: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, metrics: Dict[str, float]) -> Dict[str, float]:
        for name, value in metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("metric names must be non-empty strings")
            if not math.isfinite(float(value)):
                raise ValueError(f"metric {name!r} must be finite")
        return metrics

    @model_validator(mode="after")
    def validate_status(self) -> "ComputedResultManifest":
        if self.status == "ok" and self.error:
            raise ValueError("successful script results cannot include an error")
        if self.status == "failed" and not self.error:
            raise ValueError("failed script results require an error")
        if self.status == "failed" and self.metrics:
            raise ValueError("failed script results cannot publish metrics")
        return self


def parse_script_result(
    payload: Union[bytes, bytearray, str, Mapping[str, Any]],
    *,
    max_bytes: int = MAX_SCRIPT_RESULT_BYTES,
) -> ComputedResultManifest:
    """Parse a bounded script result payload with strict JSON semantics."""

    if isinstance(payload, Mapping):
        try:
            encoded = json.dumps(
                dict(payload),
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ScriptResultError(
                "script result mapping must contain finite JSON values"
            ) from exc
    elif isinstance(payload, str):
        encoded = payload.encode("utf-8")
    elif isinstance(payload, (bytes, bytearray)):
        encoded = bytes(payload)
    else:
        raise ScriptResultError("script result must be JSON bytes, text, or a mapping")

    if not encoded:
        raise ScriptResultError("script result is empty")
    if len(encoded) > max_bytes:
        raise ScriptResultError("script result exceeds the size limit")

    try:
        decoded = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ScriptResultError("script result must be UTF-8 JSON") from exc

    def reject_constant(value: str) -> None:
        raise ScriptResultError(f"non-finite JSON number is not allowed: {value}")

    try:
        data = json.loads(decoded, parse_constant=reject_constant)
    except ScriptResultError:
        raise
    except json.JSONDecodeError as exc:
        raise ScriptResultError("script result is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ScriptResultError("script result root must be an object")
    return ComputedResultManifest.model_validate(data)


# Compatibility name for checked-in and isolated Python script integrations.
ScriptResultManifest = ComputedResultManifest


__all__ = [
    "MAX_SCRIPT_RESULT_BYTES",
    "SCRIPT_RESULT_SCHEMA_VERSION",
    "ComputedResultManifest",
    "ScriptArtifactOutput",
    "ScriptResultError",
    "ScriptResultManifest",
    "parse_script_result",
]
