"""Bounded repair orchestration for locally executed visualization programs.

The loop is intentionally independent of both the model and the sandbox. A
container runner supplies an ``execute`` callable and translates its failures
into :class:`ArtifactAttemptError`. An optional local repair adapter receives a
structured payload only after source execution or output-contract validation
fails. Provider/generation errors therefore never trigger a repair request.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Generic, Optional, Protocol, Tuple, TypeVar

from .artifacts import ArtifactProductionError


REPAIR_PAYLOAD_SCHEMA_VERSION = 1
MAX_REPAIR_ATTEMPTS = 4
MAX_FAILURE_TEXT_CHARS = 8_000

ExecutionValue = TypeVar("ExecutionValue")
MaybeAwaitable = ExecutionValue | Awaitable[ExecutionValue]


class ArtifactFailureStage(str, Enum):
    """Repairable failure boundaries after visualization source exists."""

    SOURCE_VALIDATION = "source_validation"
    EXECUTION = "execution"
    OUTPUT_CONTRACT = "output_contract"


def _bounded_text(value: Any, *, fallback: str = "") -> str:
    text = str(value or fallback)
    return text[:MAX_FAILURE_TEXT_CHARS]


@dataclass(frozen=True)
class ArtifactAttemptFailure:
    """Structured evidence from one failed sandbox execution."""

    stage: ArtifactFailureStage
    message: str
    stderr: str = ""
    exit_code: Optional[int] = None
    timed_out: bool = False
    missing_outputs: Tuple[str, ...] = ()
    output_contract_failures: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "message",
            _bounded_text(self.message, fallback="artifact execution failed"),
        )
        object.__setattr__(self, "stderr", _bounded_text(self.stderr))
        object.__setattr__(
            self,
            "missing_outputs",
            tuple(_bounded_text(value) for value in self.missing_outputs),
        )
        object.__setattr__(
            self,
            "output_contract_failures",
            tuple(_bounded_text(value) for value in self.output_contract_failures),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "message": self.message,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "missing_outputs": list(self.missing_outputs),
            "output_contract_failures": list(self.output_contract_failures),
        }


class ArtifactAttemptError(ArtifactProductionError):
    """A generated program failed execution or artifact validation."""

    def __init__(self, failure: ArtifactAttemptFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


@dataclass(frozen=True)
class ArtifactRepairRequest:
    """Versioned request sent to a local repair adapter."""

    source_code: str
    failure: ArtifactAttemptFailure
    repair_attempt: int
    max_repair_attempts: int
    attempt_history: Tuple[ArtifactAttemptFailure, ...] = ()
    repair_feedback: Tuple[str, ...] = ()
    schema_version: int = REPAIR_PAYLOAD_SCHEMA_VERSION

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_code": self.source_code,
            "failure": self.failure.to_payload(),
            "attempt_history": [
                failure.to_payload() for failure in self.attempt_history
            ],
            "repair_feedback": list(self.repair_feedback),
            "repair_attempt": self.repair_attempt,
            "max_repair_attempts": self.max_repair_attempts,
            "response_contract": {
                "type": "python_source",
                "format": "plain_text",
                "full_replacement": True,
            },
        }


class ArtifactRepairer(Protocol):
    """Optional local-model boundary. No implementation is enabled by default."""

    def __call__(
        self,
        request: ArtifactRepairRequest,
    ) -> str | Awaitable[str]: ...


@dataclass(frozen=True)
class ArtifactRepairPolicy:
    """Hard limits for a single generated program's repair loop."""

    max_repair_attempts: int = 1
    repair_timeout_seconds: float = 180.0
    max_source_bytes: int = 512 * 1024

    def __post_init__(self) -> None:
        if not 0 <= self.max_repair_attempts <= MAX_REPAIR_ATTEMPTS:
            raise ValueError(
                f"max_repair_attempts must be between 0 and {MAX_REPAIR_ATTEMPTS}"
            )
        if not 0 < self.repair_timeout_seconds <= 600:
            raise ValueError(
                "repair_timeout_seconds must be greater than 0 and at most 600"
            )
        if self.max_source_bytes <= 0:
            raise ValueError("max_source_bytes must be positive")


@dataclass(frozen=True)
class ArtifactAttemptRecord:
    """Bounded audit record; source contents stay in the repair request only."""

    execution_attempt: int
    source_sha256: str
    source_bytes: int
    outcome: str
    failure: Optional[ArtifactAttemptFailure] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "execution_attempt": self.execution_attempt,
            "source_sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "outcome": self.outcome,
        }
        if self.failure is not None:
            payload["failure"] = self.failure.to_payload()
        return payload


@dataclass(frozen=True)
class ArtifactRepairLoopResult(Generic[ExecutionValue]):
    """Successful execution plus the source and complete attempt history."""

    value: ExecutionValue
    source_code: str
    attempts: Tuple[ArtifactAttemptRecord, ...]
    repair_attempts: int

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "schema_version": REPAIR_PAYLOAD_SCHEMA_VERSION,
            "outcome": "ok",
            "execution_attempts": len(self.attempts),
            "repair_attempts": self.repair_attempts,
            "attempts": [attempt.to_payload() for attempt in self.attempts],
            "final_cause": None,
        }


class ArtifactRepairExhaustedError(ArtifactProductionError):
    """The bounded loop could not produce a valid primary artifact."""

    def __init__(
        self,
        *,
        attempts: Tuple[ArtifactAttemptRecord, ...],
        repair_attempts: int,
        max_repair_attempts: int,
        final_cause: Dict[str, Any],
    ) -> None:
        message = _bounded_text(
            final_cause.get("message"),
            fallback="artifact repair loop exhausted",
        )
        super().__init__(
            f"artifact repair exhausted after {len(attempts)} execution "
            f"attempt(s): {message}"
        )
        self.attempts = attempts
        self.repair_attempts = repair_attempts
        self.final_cause = dict(final_cause)
        self.diagnostics = {
            "repair_loop": {
                "schema_version": REPAIR_PAYLOAD_SCHEMA_VERSION,
                "outcome": "failed",
                "execution_attempts": len(attempts),
                "repair_attempts": repair_attempts,
                "max_repair_attempts": max_repair_attempts,
                "attempts": [attempt.to_payload() for attempt in attempts],
                "final_cause": self.final_cause,
            }
        }


def _source_record(
    source_code: str,
    *,
    execution_attempt: int,
    outcome: str,
    failure: Optional[ArtifactAttemptFailure] = None,
) -> ArtifactAttemptRecord:
    encoded = source_code.encode("utf-8")
    return ArtifactAttemptRecord(
        execution_attempt=execution_attempt,
        source_sha256=hashlib.sha256(encoded).hexdigest(),
        source_bytes=len(encoded),
        outcome=outcome,
        failure=failure,
    )


def _validated_source(source_code: Any, *, max_source_bytes: int) -> str:
    if not isinstance(source_code, str) or not source_code.strip():
        raise ValueError("repair adapter returned no Python source")
    normalized = source_code.strip() + "\n"
    if len(normalized.encode("utf-8")) > max_source_bytes:
        raise ValueError(
            f"repaired Python exceeds the {max_source_bytes}-byte limit"
        )
    return normalized


async def _resolve(value: MaybeAwaitable[ExecutionValue]) -> ExecutionValue:
    if inspect.isawaitable(value):
        return await value
    return value


async def run_artifact_repair_loop(
    source_code: str,
    execute: Callable[[str], MaybeAwaitable[ExecutionValue]],
    *,
    repairer: Optional[ArtifactRepairer] = None,
    policy: Optional[ArtifactRepairPolicy] = None,
) -> ArtifactRepairLoopResult[ExecutionValue]:
    """Execute source and make at most ``policy.max_repair_attempts`` repairs.

    Only :class:`ArtifactAttemptError` is repairable. Model/provider failures,
    sandbox setup errors, cancellations, and arbitrary application exceptions
    pass through without calling the repair adapter.
    """

    resolved_policy = policy or ArtifactRepairPolicy()
    current_source = _validated_source(
        source_code,
        max_source_bytes=resolved_policy.max_source_bytes,
    )
    attempts: list[ArtifactAttemptRecord] = []
    repair_attempts = 0
    seen_source_hashes = {
        hashlib.sha256(current_source.encode("utf-8")).hexdigest()
    }

    while True:
        execution_attempt = len(attempts) + 1
        try:
            value = await _resolve(execute(current_source))
        except ArtifactAttemptError as exc:
            attempts.append(
                _source_record(
                    current_source,
                    execution_attempt=execution_attempt,
                    outcome="failed",
                    failure=exc.failure,
                )
            )
            if (
                repairer is None
                or repair_attempts >= resolved_policy.max_repair_attempts
            ):
                raise ArtifactRepairExhaustedError(
                    attempts=tuple(attempts),
                    repair_attempts=repair_attempts,
                    max_repair_attempts=resolved_policy.max_repair_attempts,
                    final_cause=exc.failure.to_payload(),
                ) from exc

            repair_feedback: list[str] = []
            while repair_attempts < resolved_policy.max_repair_attempts:
                repair_attempts += 1
                request = ArtifactRepairRequest(
                    source_code=current_source,
                    failure=exc.failure,
                    repair_attempt=repair_attempts,
                    max_repair_attempts=resolved_policy.max_repair_attempts,
                    attempt_history=tuple(
                        attempt.failure
                        for attempt in attempts
                        if attempt.failure is not None
                    ),
                    repair_feedback=tuple(repair_feedback),
                )
                try:
                    repaired = await asyncio.wait_for(
                        _resolve(repairer(request)),
                        timeout=resolved_policy.repair_timeout_seconds,
                    )
                    candidate_source = _validated_source(
                        repaired,
                        max_source_bytes=resolved_policy.max_source_bytes,
                    )
                except asyncio.TimeoutError as repair_exc:
                    raise ArtifactRepairExhaustedError(
                        attempts=tuple(attempts),
                        repair_attempts=repair_attempts,
                        max_repair_attempts=resolved_policy.max_repair_attempts,
                        final_cause={
                            "stage": "repair",
                            "message": (
                                "local repair adapter exceeded its "
                                f"{resolved_policy.repair_timeout_seconds:g}-second timeout"
                            ),
                        },
                    ) from repair_exc
                except (TypeError, ValueError) as repair_exc:
                    raise ArtifactRepairExhaustedError(
                        attempts=tuple(attempts),
                        repair_attempts=repair_attempts,
                        max_repair_attempts=resolved_policy.max_repair_attempts,
                        final_cause={
                            "stage": "repair",
                            "message": str(repair_exc),
                        },
                    ) from repair_exc
                except Exception as repair_exc:
                    raise ArtifactRepairExhaustedError(
                        attempts=tuple(attempts),
                        repair_attempts=repair_attempts,
                        max_repair_attempts=resolved_policy.max_repair_attempts,
                        final_cause={
                            "stage": "repair",
                            "message": (
                                f"{repair_exc.__class__.__name__}: "
                                f"{str(repair_exc) or 'local repair adapter failed'}"
                            ),
                        },
                    ) from repair_exc

                candidate_digest = hashlib.sha256(
                    candidate_source.encode("utf-8")
                ).hexdigest()
                if candidate_digest in seen_source_hashes:
                    repair_feedback.append(
                        "The previous repair repeated source that has already failed. "
                        "Produce a materially different implementation."
                    )
                    continue
                seen_source_hashes.add(candidate_digest)
                current_source = candidate_source
                break
            else:
                raise ArtifactRepairExhaustedError(
                    attempts=tuple(attempts),
                    repair_attempts=repair_attempts,
                    max_repair_attempts=resolved_policy.max_repair_attempts,
                    final_cause={
                        "stage": "repair",
                        "message": "local repair adapter repeatedly returned previously failed source",
                    },
                ) from exc
        else:
            attempts.append(
                _source_record(
                    current_source,
                    execution_attempt=execution_attempt,
                    outcome="ok",
                )
            )
            return ArtifactRepairLoopResult(
                value=value,
                source_code=current_source,
                attempts=tuple(attempts),
                repair_attempts=repair_attempts,
            )


__all__ = [
    "ArtifactAttemptError",
    "ArtifactAttemptFailure",
    "ArtifactAttemptRecord",
    "ArtifactFailureStage",
    "ArtifactRepairExhaustedError",
    "ArtifactRepairLoopResult",
    "ArtifactRepairPolicy",
    "ArtifactRepairRequest",
    "ArtifactRepairer",
    "MAX_REPAIR_ATTEMPTS",
    "REPAIR_PAYLOAD_SCHEMA_VERSION",
    "run_artifact_repair_loop",
]
