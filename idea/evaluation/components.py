from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

from .models import (
    ComputedComponentConfig,
    EmbeddingDiversityComponentConfig,
    EvaluationBatch,
    EvaluationRecord,
    EvaluationStatus,
    LLMPairwiseComponentConfig,
)
from .script_contract import ComputedResultManifest


_VALID_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FALLBACK_IDENTIFIER_FIELD = "_evaluation_candidate_id"
_TRANSIENT_CANDIDATE_IDENTITY_KEYS = frozenset(
    {
        _FALLBACK_IDENTIFIER_FIELD,
        "artifact_outputs",
        "artifact_refs",
        "artifact_ids",
        "auto_match_count",
        "diversity_score",
        "elo",
        "evaluations",
        "fitness",
        "generation_metrics",
        "rating_evaluations",
        "ratings",
        "selection_metrics",
        "survival_metrics",
    }
)


def candidate_identifier(candidate: Any, index: int) -> str:
    """Return a stable evaluator-safe identifier for a legacy candidate shape."""

    raw_id = candidate.get("id") if isinstance(candidate, dict) else None
    if raw_id is not None:
        identifier = str(raw_id).strip()
        if _VALID_IDENTIFIER.fullmatch(identifier):
            return identifier

    if isinstance(candidate, dict):
        persisted_fallback = candidate.get(_FALLBACK_IDENTIFIER_FIELD)
        if persisted_fallback is not None:
            identifier = str(persisted_fallback).strip()
            if _VALID_IDENTIFIER.fullmatch(identifier):
                return identifier

    identity_payload = candidate
    if isinstance(candidate, dict):
        identity_payload = {
            key: value
            for key, value in candidate.items()
            if key not in _TRANSIENT_CANDIDATE_IDENTITY_KEYS
        }

    try:
        serialized = json.dumps(identity_payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = repr(identity_payload)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
    identifier = f"candidate-{index}-{digest}"
    if isinstance(candidate, dict):
        candidate[_FALLBACK_IDENTIFIER_FIELD] = identifier
    return identifier


def attach_evaluation_record(candidate: Any, record: EvaluationRecord) -> None:
    """Attach a record without mutating a shallow-copied ancestor mapping."""

    if not isinstance(candidate, dict):
        return
    existing = candidate.get("evaluations")
    evaluations = dict(existing) if isinstance(existing, dict) else {}
    evaluations[record.component_id] = record.model_dump(mode="json")
    candidate["evaluations"] = evaluations


@dataclass(frozen=True)
class PopulationEvaluationResult:
    scores: Dict[int, float]
    batch: EvaluationBatch
    details: List[Dict[str, Any]]


def _record_provenance(
    component: Any,
    evaluation_generation: Optional[int],
) -> Dict[str, Any]:
    return {
        "component_kind": component.kind,
        "component_version": component.component_version,
        "scope": component.scope,
        "generation": evaluation_generation,
    }


def _failed_population_result(
    population: List[Any],
    *,
    component: Any,
    error: Exception,
    evaluation_generation: Optional[int],
    details: Optional[List[Dict[str, Any]]] = None,
) -> PopulationEvaluationResult:
    error_message = str(error) or error.__class__.__name__
    records: List[EvaluationRecord] = []
    for index, candidate in enumerate(population):
        record = EvaluationRecord(
            candidate_id=candidate_identifier(candidate, index),
            component_id=component.id,
            **_record_provenance(component, evaluation_generation),
            status=EvaluationStatus.FAILED,
            error=error_message,
            metadata={
                "kind": component.kind.value,
                "generation": evaluation_generation,
            },
        )
        attach_evaluation_record(candidate, record)
        records.append(record)

    return PopulationEvaluationResult(
        scores={},
        batch=EvaluationBatch(
            generation=evaluation_generation,
            records=records,
            metadata={"component_id": component.id},
        ),
        details=details if details is not None else [],
    )


class LLMPairwiseEvaluator:
    """Adapter that exposes the current Critic/Swiss judge as an evaluator."""

    def __init__(
        self,
        config: LLMPairwiseComponentConfig,
        *,
        artifact_loader: Any = None,
    ):
        self.config = config
        self.artifact_loader = artifact_loader

    def evaluate_population(
        self,
        critic: Any,
        population: List[Any],
        idea_type: str,
        *,
        rounds: int,
        progress_callback: Optional[Callable[[int, int], None]],
        details: List[Dict[str, Any]],
        full_tournament_rounds: int,
        should_stop: Callable[[], bool],
        comparison_cache: Dict[str, Any],
        evaluation_generation: Optional[int],
        early_stop_top_n: Optional[int] = None,
    ) -> PopulationEvaluationResult:
        resolved_rounds = self.config.rounds or rounds
        try:
            required_artifact_roles = [
                requirement.split(".", 1)[1]
                for requirement in self.config.requires
                if requirement.startswith("artifact.")
            ]
            configure_artifacts = getattr(
                critic,
                "configure_artifact_evidence",
                None,
            )
            if callable(configure_artifacts):
                configure_artifacts(
                    loader=self.artifact_loader,
                    required_roles=required_artifact_roles,
                )
            elif required_artifact_roles:
                raise RuntimeError(
                    "pairwise critic does not support required artifact evidence"
                )
            try:
                raw_scores = critic.get_tournament_ranks(
                    population,
                    idea_type,
                    rounds=resolved_rounds,
                    progress_callback=progress_callback,
                    details=details,
                    full_tournament_rounds=full_tournament_rounds,
                    should_stop=should_stop,
                    comparison_cache=comparison_cache,
                    evaluation_generation=evaluation_generation,
                    early_stop_top_n=early_stop_top_n,
                )
            except TypeError as exc:
                # Preserve compatibility with test doubles and older Critic-like
                # integrations while keeping the public evaluator contract stable.
                if "unexpected keyword argument" not in str(exc):
                    raise
                raw_scores = critic.get_tournament_ranks(
                    population,
                    idea_type,
                    resolved_rounds,
                    progress_callback,
                    details,
                    full_tournament_rounds,
                    should_stop=should_stop,
                )
        except Exception as exc:
            return _failed_population_result(
                population,
                component=self.config,
                error=exc,
                evaluation_generation=evaluation_generation,
                details=details,
            )

        scores: Dict[int, float] = {}
        records: List[EvaluationRecord] = []
        for index, candidate in enumerate(population):
            candidate_id = candidate_identifier(candidate, index)
            raw_score = raw_scores.get(index) if isinstance(raw_scores, dict) else None
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = math.nan

            if math.isfinite(score):
                scores[index] = score
                record = EvaluationRecord(
                    candidate_id=candidate_id,
                    component_id=self.config.id,
                    **_record_provenance(self.config, evaluation_generation),
                    status=EvaluationStatus.OK,
                    metrics={self.config.metric_name: score},
                    metadata={
                        "kind": self.config.kind.value,
                        "generation": evaluation_generation,
                    },
                )
            else:
                record = EvaluationRecord(
                    candidate_id=candidate_id,
                    component_id=self.config.id,
                    **_record_provenance(self.config, evaluation_generation),
                    status=EvaluationStatus.FAILED,
                    error="pairwise evaluator did not return a finite rating",
                    metadata={
                        "kind": self.config.kind.value,
                        "generation": evaluation_generation,
                    },
                )
            attach_evaluation_record(candidate, record)
            records.append(record)

        batch = EvaluationBatch(
            generation=evaluation_generation,
            records=records,
            metadata={"component_id": self.config.id},
        )
        return PopulationEvaluationResult(scores=scores, batch=batch, details=details)


class EmbeddingDiversityEvaluator:
    """Adapter around the engine's existing per-candidate diversity scorer."""

    def __init__(self, config: EmbeddingDiversityComponentConfig):
        self.config = config

    async def evaluate_population(
        self,
        population: List[Any],
        score_provider: Callable[[List[Any]], Awaitable[Dict[int, float]]],
        *,
        evaluation_generation: Optional[int] = None,
    ) -> PopulationEvaluationResult:
        try:
            raw_scores = await score_provider(population)
        except Exception as exc:
            return _failed_population_result(
                population,
                component=self.config,
                error=exc,
                evaluation_generation=evaluation_generation,
            )
        scores: Dict[int, float] = {}
        records: List[EvaluationRecord] = []

        for index, candidate in enumerate(population):
            candidate_id = candidate_identifier(candidate, index)
            raw_score = raw_scores.get(index) if isinstance(raw_scores, dict) else None
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = math.nan

            if math.isfinite(score):
                scores[index] = score
                record = EvaluationRecord(
                    candidate_id=candidate_id,
                    component_id=self.config.id,
                    **_record_provenance(self.config, evaluation_generation),
                    status=EvaluationStatus.OK,
                    metrics={self.config.metric_name: score},
                    metadata={
                        "kind": self.config.kind.value,
                        "generation": evaluation_generation,
                    },
                )
            else:
                record = EvaluationRecord(
                    candidate_id=candidate_id,
                    component_id=self.config.id,
                    **_record_provenance(self.config, evaluation_generation),
                    status=EvaluationStatus.FAILED,
                    error="diversity evaluator did not return a finite score",
                    metadata={
                        "kind": self.config.kind.value,
                        "generation": evaluation_generation,
                    },
                )
            attach_evaluation_record(candidate, record)
            records.append(record)

        batch = EvaluationBatch(
            generation=evaluation_generation,
            records=records,
            metadata={"component_id": self.config.id},
        )
        return PopulationEvaluationResult(scores=scores, batch=batch, details=[])


class ComputedEvaluatorRunner(Protocol):
    """Execution boundary supplied by a trusted or isolated backend."""

    async def run(
        self,
        candidate: Any,
        *,
        candidate_id: str,
        generation: Optional[int],
        config: ComputedComponentConfig,
    ) -> ComputedResultManifest: ...


class ComputedEvaluator:
    """Turn a strict runner result into maximize-only candidate measurements.

    This class deliberately does not prescribe how evaluation happens. A caller
    supplies a trusted in-process runner or an isolated backend, which may also
    enrich the candidate with artifacts before pairwise judging. All declared
    numeric outputs must already follow the convention that larger is fitter.
    """

    def __init__(self, config: ComputedComponentConfig):
        self.config = config

    async def evaluate_population(
        self,
        population: List[Any],
        runner: ComputedEvaluatorRunner,
        *,
        evaluation_generation: Optional[int] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> PopulationEvaluationResult:
        records: List[EvaluationRecord] = []
        primary_scores: Dict[int, float] = {}
        primary_metric = self.config.metric_names[0]

        for index, candidate in enumerate(population):
            candidate_id = candidate_identifier(candidate, index)
            if should_stop is not None and should_stop():
                record = EvaluationRecord(
                    candidate_id=candidate_id,
                    component_id=self.config.id,
                    **_record_provenance(self.config, evaluation_generation),
                    status=EvaluationStatus.SKIPPED,
                    metadata={
                        "kind": self.config.kind.value,
                        "generation": evaluation_generation,
                        "runner_ref": self.config.runner_ref,
                        "reason": "run stop or budget limit reached",
                    },
                )
                attach_evaluation_record(candidate, record)
                records.append(record)
                continue
            try:
                try:
                    result = await asyncio.wait_for(
                        runner.run(
                            candidate,
                            candidate_id=candidate_id,
                            generation=evaluation_generation,
                            config=self.config,
                        ),
                        timeout=self.config.timeout_seconds,
                    )
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        "computed evaluator exceeded "
                        f"its {self.config.timeout_seconds:g}-second timeout"
                    ) from exc
                missing = [
                    name for name in self.config.metric_names if name not in result.metrics
                ]
                if result.status != "ok":
                    raise ValueError(result.error or "computed evaluator failed")
                if missing:
                    raise ValueError(
                        "script result omitted configured metrics: " + ", ".join(missing)
                    )
                metrics = {
                    name: float(result.metrics[name])
                    for name in self.config.metric_names
                }
                if result.artifacts and isinstance(candidate, dict):
                    existing_outputs = candidate.get("artifact_outputs")
                    artifact_outputs = (
                        dict(existing_outputs)
                        if isinstance(existing_outputs, dict)
                        else {}
                    )
                    artifact_outputs[self.config.id] = [
                        artifact.model_dump(mode="json")
                        for artifact in result.artifacts
                    ]
                    candidate["artifact_outputs"] = artifact_outputs
                primary_scores[index] = metrics[primary_metric]
                record = EvaluationRecord(
                    candidate_id=candidate_id,
                    component_id=self.config.id,
                    **_record_provenance(self.config, evaluation_generation),
                    status=EvaluationStatus.OK,
                    metrics=metrics,
                    rationale=result.summary,
                    metadata={
                        "kind": self.config.kind.value,
                        "generation": evaluation_generation,
                        "artifact_outputs": len(result.artifacts),
                        "runner_ref": self.config.runner_ref,
                    },
                )
            except Exception as exc:
                record = EvaluationRecord(
                    candidate_id=candidate_id,
                    component_id=self.config.id,
                    **_record_provenance(self.config, evaluation_generation),
                    status=EvaluationStatus.FAILED,
                    error=str(exc) or exc.__class__.__name__,
                    metadata={
                        "kind": self.config.kind.value,
                        "generation": evaluation_generation,
                        "runner_ref": self.config.runner_ref,
                    },
                )

            attach_evaluation_record(candidate, record)
            records.append(record)

        return PopulationEvaluationResult(
            scores=primary_scores,
            batch=EvaluationBatch(
                generation=evaluation_generation,
                records=records,
                metadata={"component_id": self.config.id},
            ),
            details=[],
        )
