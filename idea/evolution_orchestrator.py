from __future__ import annotations

import asyncio
from idea.provider_execution import await_provider_call
import math
import random
import uuid
import copy
from typing import Any, Awaitable, Callable, Dict, List, Optional

from idea.evolution_progress import ProgressEmitter, terminal_status_fields
from idea.evolution_types import GenerationWorkState
from idea.models import merge_candidate_records
from idea.template_runtime import generation_prompt
from idea.genes import empty_gene_catalogue, reconcile_gene_catalogue, valid_gene_observation
from idea.gene_pool import consolidate_gene_pool


class EvolutionOrchestrator:
    """Canonical evolution execution flow used by both new and resumed runs."""

    def __init__(
        self,
        engine: Any,
        progress_callback: Callable[[Dict[str, Any]], Awaitable[None]],
    ):
        self.engine = engine
        version = int(getattr(engine, "algorithm_evidence_version", 1))
        if version > 2:
            raise ValueError("Unsupported future algorithm evidence version")
        engine.algorithm_evidence_version = 2
        if not isinstance(getattr(engine, "quality_archive", None), list):
            engine.quality_archive = []
        if not isinstance(getattr(engine, "gene_catalogue", None), dict):
            engine.gene_catalogue = empty_gene_catalogue()
        def provider_should_stop():
            if bool(getattr(engine, "stop_requested", False)):
                return True
            check_budget = getattr(engine, "check_budget", None)
            if callable(check_budget) and check_budget():
                engine.stop_requested = True
                return True
            return False
        for name in ("ideator", "critic", "breeder", "gene_manager", "formatter", "oracle", "diversity_calculator"):
            agent = getattr(engine, name, None)
            if agent is not None:
                agent.cancel_check = provider_should_stop
        self.progress_callback = progress_callback
        self.progress = ProgressEmitter(engine)
        self._draft_save_lock = asyncio.Lock()

    def _comparison_cache(self) -> Dict[str, Any]:
        cache = getattr(self.engine, "comparison_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self.engine, "comparison_cache", cache)
        return cache

    def _get_tournament_ranks(
        self,
        population: List[Any],
        *,
        rounds: int,
        progress_callback: Optional[Callable[[int, int], None]],
        details: List[Dict[str, Any]],
        full_tournament_rounds: int,
        should_stop: Callable[[], bool],
        evaluation_generation: int,
        early_stop_top_n: Optional[int] = None,
    ) -> Dict[int, float]:
        evaluator = getattr(self.engine, "quality_evaluator", None)
        if evaluator is not None:
            config = getattr(evaluator, "config", None)
            if config is not None:
                self.engine.critic.judge_component_id = config.id
                self.engine.critic.judge_component_version = getattr(config, "component_version", "1")
            result = evaluator.evaluate_population(
                self.engine.critic,
                population,
                self.engine.idea_type,
                rounds=rounds,
                progress_callback=progress_callback,
                details=details,
                full_tournament_rounds=full_tournament_rounds,
                should_stop=should_stop,
                comparison_cache=self._comparison_cache(),
                evaluation_generation=evaluation_generation,
                early_stop_top_n=early_stop_top_n,
            )
            self.engine.last_evaluation_batch = result.batch
            return result.scores

        # Engines with an explicit suite but no pairwise component intentionally
        # skip LLM judging. Uniform placeholder ratings preserve legacy
        # selection aliases while other configured evaluators supply fitness.
        if hasattr(self.engine, "evaluation_config"):
            return {index: 1500.0 for index in range(len(population))}

        try:
            return self.engine.critic.get_tournament_ranks(
                population,
                self.engine.idea_type,
                rounds=rounds,
                progress_callback=progress_callback,
                details=details,
                full_tournament_rounds=full_tournament_rounds,
                should_stop=should_stop,
                comparison_cache=self._comparison_cache(),
                evaluation_generation=evaluation_generation,
                early_stop_top_n=early_stop_top_n,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return self.engine.critic.get_tournament_ranks(
                population,
                self.engine.idea_type,
                rounds,
                progress_callback,
                details,
                full_tournament_rounds,
                should_stop=should_stop,
            )

    @staticmethod
    def _extract_context_terms(context_pool: str) -> set[str]:
        if not context_pool:
            return set()
        return {
            token.strip().lower()
            for token in str(context_pool).split(",")
            if token and token.strip()
        }

    @staticmethod
    def _jaccard_similarity(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    @staticmethod
    def _extract_idea_body_text(idea_payload: Any) -> str:
        if idea_payload is None:
            return ""

        candidate = idea_payload
        if isinstance(candidate, dict) and "idea" in candidate:
            candidate = candidate.get("idea")

        if hasattr(candidate, "content"):
            return str(getattr(candidate, "content", "") or "")

        if isinstance(candidate, dict):
            if "content" in candidate:
                return str(candidate.get("content") or "")
            if "idea" in candidate:
                return str(candidate.get("idea") or "")
            return ""

        return str(candidate or "")

    @classmethod
    def _is_usable_idea(cls, idea_payload: Any) -> bool:
        text = cls._extract_idea_body_text(idea_payload).strip()
        if not text:
            return False
        return text.lower() not in {
            "no response",
            "no response.",
            "model response was empty.",
            "none",
            "null",
        }

    @staticmethod
    def _record_parent_breeding_genotypes(
        parent_ideas: List[Any],
        parent_genotypes: Any,
        source_generation: int,
        child_generation: int,
        child_id: Any,
    ) -> None:
        if not isinstance(parent_genotypes, list):
            return

        parents_by_id = {
            str(parent.get("id")): parent
            for parent in parent_ideas
            if isinstance(parent, dict) and parent.get("id") is not None
        }
        child_id_text = str(child_id) if child_id is not None else None

        for index, genotype_record in enumerate(parent_genotypes):
            if not isinstance(genotype_record, dict):
                continue

            genotype = str(genotype_record.get("genotype") or "").strip()
            if not genotype:
                continue

            parent_id = genotype_record.get("parent_id")
            parent = parents_by_id.get(str(parent_id)) if parent_id is not None else None
            if parent is None and index < len(parent_ideas):
                candidate = parent_ideas[index]
                parent = candidate if isinstance(candidate, dict) else None
            if parent is None:
                continue

            entry = {
                "genotype": genotype,
                "source_generation": source_generation,
                "child_generation": child_generation,
            }
            for key in ("raw_genotype", "source", "gene_pool_revision"):
                if key in genotype_record:
                    entry[key] = genotype_record[key]
            if child_id_text:
                entry["child_id"] = child_id_text

            parent.setdefault("breeding_genotypes", []).append(entry)

    def _record_generation_diagnostic(self, key: str, detail: Optional[str] = None) -> None:
        for agent_name in ("formatter", "critic", "ideator", "breeder", "gene_manager", "oracle"):
            agent = getattr(self.engine, agent_name, None)
            increment = getattr(agent, "_increment_diagnostic", None)
            if callable(increment):
                increment(key, detail=detail)
                return

    @staticmethod
    def _finite_or_none(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric != numeric:
            return None
        return numeric

    def _attach_tree_metrics(
        self,
        population: List[Any],
        fitness_map: Dict[int, Dict[str, Any]],
        survival_scores: Dict[int, Dict[str, float]],
        *,
        evaluation_generation: int,
        target_generation: int,
    ) -> None:
        generation_key = str(evaluation_generation)
        for idx, idea in enumerate(population):
            if not isinstance(idea, dict):
                continue
            fitness_data = fitness_map.get(idx, {})
            survival_data = survival_scores.get(idx, {})
            selection_metrics = {
                "evaluation_generation": evaluation_generation,
            }
            for key in ("elo", "diversity", "elo_norm", "diversity_norm", "fitness"):
                numeric = self._finite_or_none(fitness_data.get(key))
                if numeric is not None:
                    selection_metrics[key] = numeric
                    if key == "elo":
                        idea["elo"] = numeric
                        ratings = idea.get("ratings")
                        if not isinstance(ratings, dict):
                            ratings = {}
                        ratings["auto"] = numeric
                        idea["ratings"] = ratings
                    elif key == "diversity":
                        idea["diversity_score"] = numeric
                    elif key == "fitness":
                        idea["fitness"] = numeric
            components = fitness_data.get("components")
            if isinstance(components, dict):
                selection_metrics["components"] = dict(components)
            idea["selection_metrics"] = selection_metrics

            existing_generation_metrics = idea.get("generation_metrics")
            if isinstance(existing_generation_metrics, dict):
                generation_metrics = dict(existing_generation_metrics)
            else:
                generation_metrics = {}
            if generation_key not in generation_metrics:
                generation_metrics[generation_key] = {
                    key: value
                    for key, value in selection_metrics.items()
                    if key in {"evaluation_generation", "elo", "diversity", "fitness"}
                }
            idea["generation_metrics"] = generation_metrics

            survival_metrics = {
                "survived_to_generation": target_generation,
            }
            for key in ("age", "age_decay", "survival_score"):
                numeric = self._finite_or_none(survival_data.get(key))
                if numeric is not None:
                    survival_metrics[key] = numeric
            idea["survival_metrics"] = survival_metrics
        self._archive_evaluated_candidates(population, fitness_map)

    def _archive_evaluated_candidates(self, population, fitness_map):
        """Retain bounded immutable revisions with successful quality evidence."""
        by_id = {str(idea.get("id")): idea for idea in self.engine.quality_archive
                 if isinstance(idea, dict) and "archive_quality_score" in idea}
        quality = getattr(self.engine, "quality_evaluator", None)
        for idx, candidate in enumerate(population):
            if not isinstance(candidate, dict):
                continue
            evaluations = candidate.get("evaluations") or {}
            if quality is not None:
                config = getattr(quality, "config", None)
                if config is None:
                    continue
                evidence = evaluations.get(config.id) or {}
                if evidence.get("status") not in {"ok", "cached"}:
                    continue
                score = self._finite_or_none(fitness_map.get(idx, {}).get("elo"))
            elif evaluations:
                if any(record.get("status") not in {"ok", "cached"}
                       for record in evaluations.values()):
                    continue
                score = self._finite_or_none(fitness_map.get(idx, {}).get("fitness"))
            else:
                continue
            if score is None or not math.isfinite(score):
                continue
            archived = copy.deepcopy(candidate)
            archived["archive_quality_score"] = score
            by_id[str(candidate.get("id"))] = archived
        self.engine.quality_archive = sorted(
            by_id.values(), key=lambda candidate: candidate["archive_quality_score"],
            reverse=True,
        )[:50]

    async def _prepare_tournament_evidence(
        self,
        population: List[Any],
        *,
        evaluation_generation: int,
    ) -> None:
        """Run artifact/computed producers before a judge consumes their evidence."""

        prepare_for_judging = getattr(
            self.engine,
            "_prepare_population_for_judging",
            None,
        )
        if callable(prepare_for_judging):
            batch = await prepare_for_judging(
                population,
                evaluation_generation=evaluation_generation,
            )
            metadata = getattr(batch, "metadata", {})
            if (
                isinstance(metadata, dict)
                and metadata.get("artifact_evidence_degraded")
            ):
                failed = [
                    record
                    for record in getattr(batch, "records", [])
                    if getattr(
                        getattr(record, "status", None),
                        "value",
                        getattr(record, "status", None),
                    )
                    == "failed"
                ]
                detail = (
                    getattr(failed[0], "error", None)
                    if failed
                    else "artifact evidence was unavailable"
                )
                self._record_generation_diagnostic(
                    "artifact_evidence_degraded",
                    detail=(
                        f"{detail}; tournament continued with text-only judging"
                    ),
                )

    async def _run_tournament_and_attach_tree_metrics(
        self,
        population: List[Any],
        *,
        generation_index: int,
        target_generation: int,
        rounds: int,
        full_rounds: int,
        island_id: Optional[int] = None,
        early_stop_top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not population:
            return []

        await self._prepare_tournament_evidence(
            population,
            evaluation_generation=generation_index,
        )

        tournament_start_cost = self.engine.get_total_token_count()["cost"]["total_cost"]
        tournament_rounds_details: List[Dict[str, Any]] = []

        def run_tournament():
            return self._get_tournament_ranks(
                population,
                rounds=max(1, int(rounds or 1)),
                progress_callback=None,
                details=tournament_rounds_details,
                full_tournament_rounds=max(1, int(full_rounds or 1)),
                should_stop=lambda: self.engine.stop_requested,
                evaluation_generation=generation_index,
                early_stop_top_n=early_stop_top_n,
            )

        global_ranks = await await_provider_call(run_tournament)
        if island_id is not None:
            for detail in tournament_rounds_details:
                if isinstance(detail, dict):
                    detail["island_id"] = island_id

        tournament_end_cost = self.engine.get_total_token_count()["cost"]["total_cost"]
        current_tournament_cost = tournament_end_cost - tournament_start_cost
        if self.engine.avg_tournament_cost == 0:
            self.engine.avg_tournament_cost = current_tournament_cost
        else:
            self.engine.avg_tournament_cost = (
                self.engine.avg_tournament_cost + current_tournament_cost
            ) / 2

        if self.engine.stop_requested:
            return tournament_rounds_details

        fitness_map = await self.engine._score_population_fitness(population, global_ranks)
        if not fitness_map:
            fitness_map = {
                idx: {
                    "elo": float(global_ranks.get(idx, 0.0)),
                    "diversity": 0.0,
                    "elo_norm": 0.5,
                    "diversity_norm": 0.5,
                    "fitness": 0.5,
                }
                for idx in global_ranks.keys()
            }

        survival_scores = self.engine._score_survival_with_age_decay(
            population,
            fitness_map,
            target_generation=target_generation,
        )
        self._attach_tree_metrics(
            population,
            fitness_map,
            survival_scores,
            evaluation_generation=generation_index,
            target_generation=target_generation,
        )
        return tournament_rounds_details

    async def _score_final_generation_for_continuation(self) -> None:
        if not self.engine.population:
            return

        generation_index = self.engine.generations
        await self.progress.emit(
            self.progress_callback,
            {
                "current_generation": generation_index,
                "total_generations": self.engine.generations,
                "is_running": True,
                "progress": 99,
                "status_message": "Running final Swiss tournament...",
            },
        )

        tournament_rounds_details: List[Dict[str, Any]] = []
        if self._islands_enabled():
            for island_id, entries in enumerate(self._population_island_entries(assign_missing=True)):
                if self.engine.stop_requested:
                    break
                island_population = [entry["idea"] for entry in entries]
                if not island_population:
                    continue
                local_rounds = self._local_tournament_rounds(len(island_population))
                tournament_rounds_details.extend(
                    await self._run_tournament_and_attach_tree_metrics(
                        island_population,
                        generation_index=generation_index,
                        target_generation=generation_index,
                        rounds=local_rounds,
                        full_rounds=max(1, len(island_population) - 1),
                        island_id=island_id,
                    )
                )
        else:
            tournament_rounds_details = await self._run_tournament_and_attach_tree_metrics(
                self.engine.population,
                generation_index=generation_index,
                target_generation=generation_index,
                rounds=self._local_tournament_rounds(len(self.engine.population)),
                full_rounds=max(1, len(self.engine.population)-1),
            )

        self.engine._set_tournament_history(generation_index, tournament_rounds_details)
        if self.engine.history:
            self.engine.history[-1] = self.engine.population.copy()

    def _legacy_refine_all_enabled(self) -> bool:
        return bool(getattr(self.engine, "refine_all_ideas", False))

    def _survivor_refinement_rate(self) -> float:
        try:
            raw_rate = getattr(self.engine, "survivor_refinement_rate", 0.1)
            return max(0.0, min(1.0, float(raw_rate)))
        except (TypeError, ValueError):
            return 0.1

    def _should_refine_survivor(self) -> bool:
        rate = self._survivor_refinement_rate()
        if self._legacy_refine_all_enabled() or rate <= 0:
            return False
        if rate >= 1:
            return True
        random_uniform = getattr(self.engine, "random_uniform", None)
        draw = random_uniform(0.0, 1.0) if callable(random_uniform) else random.random()
        return float(draw) < rate

    def _island_count(self) -> int:
        try:
            configured = int(getattr(self.engine, "island_count", 2) or 2)
            population_size = max(1, len(getattr(self.engine, "population", [])) or
                                  int(getattr(self.engine, "pop_size", 1) or 1))
            return max(1, min(configured, max(1, population_size // 2)))
        except (TypeError, ValueError):
            return 1

    def _islands_enabled(self) -> bool:
        return self._island_count() > 1

    def _initial_island_ids(self, population_size: int) -> List[int]:
        engine_ids = getattr(self.engine, "_initial_island_ids", None)
        if callable(engine_ids):
            return engine_ids(population_size)

        count = min(self._island_count(), max(1, population_size))
        base_size = population_size // count
        remainder = population_size % count
        island_ids: List[int] = []
        for island_id in range(count):
            island_ids.extend(
                [island_id] * (base_size + (1 if island_id < remainder else 0))
            )
        return island_ids

    def _get_idea_island_id(self, idea: Any, *, fallback: int = 0) -> int:
        engine_reader = getattr(self.engine, "_get_idea_island_id", None)
        if callable(engine_reader):
            return engine_reader(idea, fallback=fallback)

        count = max(1, self._island_count())
        if isinstance(idea, dict):
            try:
                return max(0, min(count - 1, int(idea.get("island_id", fallback))))
            except (TypeError, ValueError):
                pass
        return max(0, min(count - 1, int(fallback or 0)))

    def _assign_idea_island_id(
        self,
        idea: Any,
        island_id: int,
        *,
        force: bool = False,
        origin: Optional[int] = None,
    ) -> Any:
        if not isinstance(idea, dict):
            return idea
        if not force and not self._islands_enabled() and "island_id" not in idea:
            return idea

        normalized = self._get_idea_island_id({"island_id": island_id}, fallback=0)
        idea["island_id"] = normalized
        if origin is not None:
            idea["origin_island_id"] = self._get_idea_island_id(
                {"island_id": origin},
                fallback=normalized,
            )
        else:
            idea.setdefault("origin_island_id", normalized)
        return idea

    def _population_island_entries(
        self,
        population: Optional[List[Any]] = None,
        *,
        assign_missing: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        current_population = self.engine.population if population is None else population
        if hasattr(self.engine, "_population_island_entries"):
            return self.engine._population_island_entries(
                current_population,
                assign_missing=assign_missing,
            )

        island_count = min(self._island_count(), max(1, len(current_population)))
        fallback_ids = self._initial_island_ids(len(current_population))
        groups: List[List[Dict[str, Any]]] = [[] for _ in range(island_count)]
        for global_idx, idea in enumerate(current_population):
            fallback = fallback_ids[global_idx] if global_idx < len(fallback_ids) else 0
            island_id = self._get_idea_island_id(idea, fallback=fallback)
            if assign_missing:
                self._assign_idea_island_id(idea, island_id, force=True)
            groups[island_id].append({"global_index": global_idx, "idea": idea})
        return groups

    def _next_refill_island_id(self) -> int:
        target_sizes_method = getattr(self.engine, "_compute_island_sizes", None)
        if callable(target_sizes_method):
            target_sizes = target_sizes_method(self.engine.pop_size)
        else:
            target_sizes = [
                self._initial_island_ids(self.engine.pop_size).count(island_id)
                for island_id in range(self._island_count())
            ]
        counts = [0] * len(target_sizes)
        fallback_ids = self._initial_island_ids(len(self.engine.population))
        for idx, idea in enumerate(self.engine.population):
            fallback = fallback_ids[idx] if idx < len(fallback_ids) else 0
            island_id = self._get_idea_island_id(idea, fallback=fallback)
            if 0 <= island_id < len(counts):
                counts[island_id] += 1

        deficits = [
            (target_sizes[island_id] - counts[island_id], island_id)
            for island_id in range(len(target_sizes))
        ]
        deficits.sort(key=lambda row: (-row[0], row[1]))
        return deficits[0][1] if deficits and deficits[0][0] > 0 else 0

    async def _prepare_idea_for_population(self, idea: Any, *, refine: bool) -> Any:
        candidate = idea
        if refine:
            refined = await await_provider_call(
                self.engine.critic.refine, candidate, self.engine.idea_type
            )
            candidate = merge_candidate_records(candidate, refined)
        formatted = await await_provider_call(
            self.engine.formatter.format_idea, candidate, self.engine.idea_type
        )
        result = merge_candidate_records(candidate, formatted)
        if refine and self._content_identity(idea) != self._content_identity(result):
            result = self._invalidate_changed_candidate(result)
            if isinstance(idea, dict) and isinstance(result, dict) and idea.get("evaluations"):
                result["id"] = uuid.uuid4()
                result["parent_ids"] = [str(idea["id"])] if idea.get("id") else []
        return result

    async def _observe_candidate_genes(self, candidate: Any) -> None:
        """Observe completed prose without changing eligibility or selection."""
        observer = getattr(self.engine, "gene_manager", None) or getattr(self.engine, "breeder", None)
        observe = getattr(observer, "observe_genes", None)
        if not isinstance(candidate, dict) or not callable(observe):
            return
        if self.engine.stop_requested:
            return
        if self.engine.check_budget():
            self.engine.stop_requested = True
            return
        try:
            await await_provider_call(observe, candidate, self.engine.idea_type)
        except Exception:
            # Observational failures must never discard an otherwise usable idea.
            candidate["gene_observation"] = {
                "version": 1, "status": "failed", "genes": [],
                "error": "Gene extraction could not be completed.",
            }
            self._record_generation_diagnostic("gene_observation_failures")
        self._sync_gene_observation_to_history(candidate)

    def _sync_gene_observation_to_history(self, candidate: Any) -> None:
        """Keep repaired evidence on separately deserialized copies of a candidate."""
        if not isinstance(candidate, dict) or not candidate.get("id"):
            return
        observation = candidate.get("gene_observation")
        fingerprint = observation.get("fingerprint") if isinstance(observation, dict) else None
        if not isinstance(fingerprint, str) or not valid_gene_observation(observation, fingerprint):
            return
        identity = str(candidate["id"])
        phenotype = self._content_identity(candidate)
        for population in self.engine.history:
            for historical in population:
                if (not isinstance(historical, dict) or historical is candidate
                        or str(historical.get("id")) != identity
                        or self._content_identity(historical) != phenotype):
                    continue
                previous = historical.get("gene_observation")
                if isinstance(previous, dict):
                    # Different extraction settings are separate evidence, even
                    # when the prose is unchanged. Never replace them by accident.
                    if previous.get("fingerprint") not in (None, fingerprint):
                        continue
                    if (previous.get("mapping_status") == "ok"
                            and observation.get("mapping_status") != "ok"):
                        continue
                # Ratings and other generation-specific metadata must stay local.
                historical["gene_observation"] = copy.deepcopy(observation)

    async def _observe_population_genes(self, generation: int) -> None:
        """Reconcile free extractions once per population, outside breeding."""
        observer = getattr(self.engine, "gene_manager", None) or getattr(self.engine, "breeder", None)
        if self.engine.stop_requested or not callable(getattr(observer, "observe_genes", None)):
            return
        await self.progress.emit(self.progress_callback, {
            "current_generation": generation, "is_running": True,
            "status_message": "Recording candidate genes...",
        })
        for candidate in self.engine.population:
            await self._observe_candidate_genes(candidate)
            if self.engine.stop_requested:
                return
        await self.progress.emit(self.progress_callback, {
            "current_generation": generation, "is_running": True,
            "status_message": "Consolidating the shared gene pool...",
        })
        if self.engine.check_budget():
            self.engine.stop_requested = True
            return
        try:
            manager = getattr(self.engine, "gene_manager", None)
            if manager is not None:
                self.engine.gene_catalogue = await await_provider_call(
                    consolidate_gene_pool, self.engine.population, self.engine.history,
                    self.engine.gene_catalogue, manager, self.engine.idea_type, generation,
                )
            else:
                # Legacy/injected engines and standalone observation tools.
                self.engine.gene_catalogue = await await_provider_call(
                    reconcile_gene_catalogue, self.engine.population,
                    self.engine.gene_catalogue, self.engine.breeder,
                    self.engine.idea_type, generation,
                )
        except Exception:
            self._record_generation_diagnostic("gene_catalogue_failures")
        for candidate in self.engine.population:
            self._sync_gene_observation_to_history(candidate)
        mapping_failed = self.engine.gene_catalogue.get("mapping_status") == "failed"
        if mapping_failed:
            self._record_generation_diagnostic("gene_catalogue_failures")
        await self.progress.emit(self.progress_callback, {
            "current_generation": generation, "is_running": True,
            "history": self.engine.history,
            "gene_catalogue": self.engine.gene_catalogue,
            "status_message": ("Genes recorded; consolidation unavailable. Saved factors or raw genes will be used."
                               if mapping_failed else "Candidate genes recorded."),
        })

    @staticmethod
    def _content_identity(candidate: Any) -> tuple[str, str]:
        value = candidate.get("idea", candidate) if isinstance(candidate, dict) else candidate
        if isinstance(value, dict):
            return str(value.get("title") or ""), str(value.get("content") or "")
        if hasattr(value, "content"):
            return str(getattr(value, "title", "") or ""), str(value.content or "")
        return "", str(value or "")

    @staticmethod
    def _invalidate_changed_candidate(candidate: Any) -> Any:
        if not isinstance(candidate, dict):
            return candidate
        result = copy.deepcopy(candidate)
        for key in ("elo", "fitness", "diversity_score", "selection_metrics",
                    "survival_metrics", "generation_metrics", "evaluations",
                    "rating_evaluations", "auto_match_count", "artifact_refs",
                    "artifact_ids", "artifact_outputs", "artifacts", "_evaluation_candidate_id",
                    "gene_observation", "gene_projection"):
            result.pop(key, None)
        result["ratings"] = {}
        return result

    async def _maybe_refine_survivor(
        self,
        survivor: Any,
        *,
        target_generation: int,
    ) -> Any:
        if not self._should_refine_survivor():
            return survivor

        source_id = (
            str(survivor.get("id"))
            if isinstance(survivor, dict) and survivor.get("id")
            else None
        )
        try:
            refined = await self._prepare_idea_for_population(survivor, refine=True)
        except Exception as exc:
            self._record_generation_diagnostic("survivor_refinement_failures", detail=str(exc))
            return survivor

        if not self._is_usable_idea(refined):
            self._record_generation_diagnostic("invalid_survivor_refinements")
            return survivor
        if self._content_identity(refined) == self._content_identity(survivor):
            return survivor

        if isinstance(refined, dict):
            refined_survivor = dict(refined)
        else:
            refined_survivor = {
                "idea": refined,
                "parent_ids": [],
            }

        refined_survivor["id"] = uuid.uuid4()
        refined_survivor["parent_ids"] = [source_id] if source_id else []
        refined_survivor["birth_generation"] = target_generation
        refined_survivor["survived_to_generation"] = target_generation
        refined_survivor["survivor_refined"] = True
        refined_survivor["birth_type"] = "survivor_refinement"
        if isinstance(survivor, dict) and "island_id" in survivor:
            self._assign_idea_island_id(
                refined_survivor,
                self._get_idea_island_id(survivor),
                force=True,
                origin=survivor.get("origin_island_id"),
            )
        if source_id:
            refined_survivor["survivor_refined_from_id"] = source_id
        refined_survivor["survivor_refinement_rate"] = self._survivor_refinement_rate()
        return refined_survivor

    async def _generate_single_seed(self) -> tuple[str, Dict[str, Any], str]:
        context_pool = await await_provider_call(
            self.engine.ideator.generate_context, self.engine.idea_type
        )
        idea_text, specific_prompt = await await_provider_call(
            self.engine.ideator.generate_idea_from_context,
            context_pool,
            self.engine.idea_type,
        )
        return (
            context_pool,
            {
                "id": uuid.uuid4(),
                "idea": idea_text,
                "parent_ids": [],
                "birth_generation": 0,
                "birth_type": "independent",
            },
            specific_prompt,
        )

    async def _ensure_seed_novelty(
        self,
        seed_result: tuple[str, Dict[str, Any], str],
        existing_context_sets: List[set[str]],
    ) -> tuple[str, Dict[str, Any], str]:
        threshold = float(getattr(self.engine, "context_novelty_threshold", 0.8))
        max_attempts = int(getattr(self.engine, "context_novelty_max_attempts", 2))
        max_attempts = max(0, max_attempts)

        context_pool, idea, prompt = seed_result
        candidate_terms = self._extract_context_terms(context_pool)

        if not existing_context_sets or not candidate_terms:
            existing_context_sets.append(candidate_terms)
            return context_pool, idea, prompt

        max_similarity = max(
            self._jaccard_similarity(candidate_terms, existing_terms)
            for existing_terms in existing_context_sets
        )

        attempts = 0
        while max_similarity >= threshold and attempts < max_attempts:
            attempts += 1
            increment_diagnostic = getattr(self.engine.ideator, "_increment_diagnostic", None)
            if callable(increment_diagnostic):
                increment_diagnostic("seed_context_novelty_regenerations")
            context_pool, idea, prompt = await self._generate_single_seed()
            candidate_terms = self._extract_context_terms(context_pool)
            if not candidate_terms:
                break
            max_similarity = max(
                self._jaccard_similarity(candidate_terms, existing_terms)
                for existing_terms in existing_context_sets
            )

        if max_similarity >= threshold:
            increment_diagnostic = getattr(self.engine.ideator, "_increment_diagnostic", None)
            if callable(increment_diagnostic):
                increment_diagnostic("seed_context_novelty_guardrail_exhausted")

        existing_context_sets.append(candidate_terms)
        return context_pool, idea, prompt

    async def run(
        self,
        *,
        start_generation: int,
        mode: str,
        additional_generations: int = 0,
    ) -> None:
        try:
            # A stop received after dispatch must survive startup. Explicit
            # resume of an already-paused engine is the sole reset exception.
            if self.engine.stop_requested and (mode == "new" or not self.engine.is_stopped):
                await self._stop_before_provider_work()
                await self._save_paused_run()
                return
            self.engine.reset_stop_state()
            if await self._stop_before_provider_work():
                await self._save_paused_run()
                return

            if additional_generations > 0:
                if self.engine.generations + additional_generations > 500:
                    raise ValueError("Total generations cannot exceed 500")
                self.engine.generations += additional_generations
                print(
                    f"📈 Extended evolution by {additional_generations} generations. "
                    f"New total: {self.engine.generations}"
                )

            if mode == "new":
                await self._run_new(start_generation=start_generation)
            elif mode in {"resume", "continue"}:
                await self._run_resume(start_generation=start_generation)
            else:
                raise ValueError(f"Unsupported orchestrator mode: {mode}")

            # Seeding, tournaments, breeding and budget checks all have safe
            # early exits. Persist every one before acknowledging the stop.
            if self.engine.stop_requested:
                await self._save_paused_run()

        except Exception as exc:
            import traceback

            print(f"Error in orchestrator run: {exc}")
            print(traceback.format_exc())
            save_error = None
            try:
                await self.engine.save_checkpoint(status="error")
            except Exception as checkpoint_error:
                save_error = str(checkpoint_error)
                self.engine.last_save_error = save_error
            await self.progress.emit(
                self.progress_callback,
                {
                    **terminal_status_fields("error"),
                    "status_message": (
                        "Evolution failed. The latest progress could not be saved."
                        if save_error else "Evolution failed. Saved progress is available to resume."
                    ),
                    "is_running": False,
                    "is_stopped": bool(self.engine.is_stopped),
                    "error": str(exc),
                    "is_resumable": save_error is None and bool(self.engine.checkpoint_id),
                    "save_error": save_error,
                    "durably_saved": save_error is None,
                    "checkpoint_id": self.engine.checkpoint_id,
                    "diversity_history": self.engine.diversity_history.copy()
                    if self.engine.diversity_history
                    else [],
                },
            )

    async def _save_paused_run(self) -> None:
        self.engine.is_stopped = True
        if self.engine.current_generation == 0 and self.engine.population:
            self.engine.history = [self.engine.population.copy()]
        await self.engine.save_checkpoint(status="paused")
        await self.progress.emit(self.progress_callback, {
            **terminal_status_fields("paused", budget_reached=self.engine.check_budget()),
            "current_generation": self.engine.current_generation,
            "is_running": False,
            "is_stopped": True,
            "is_resumable": True,
            "durably_saved": True,
            "checkpoint_id": self.engine.checkpoint_id,
            "token_counts": self.engine.get_total_token_count(),
            "history": self.engine.history,
            "gene_catalogue": self.engine.gene_catalogue,
            "generation_draft": getattr(self.engine, "generation_draft", None),
            **self._draft_progress_message(),
        })

    def _draft_progress_message(self):
        draft = getattr(self.engine, "generation_draft", None)
        if not draft:
            return {}
        entries = draft["islands"].values()
        complete = sum(result is not None for entry in entries for result in entry["results"])
        message = (f"Generation {draft['generation']} is incomplete; {complete} completed offspring saved. "
                   "Resume will reuse them and finish the remaining work.")
        return {"status_message": message, "stop_message": message}

    async def _stop_before_provider_work(self) -> bool:
        budget_reached = self.engine.check_budget()
        if not self.engine.stop_requested and not budget_reached:
            return False
        self.engine.stop_requested = True
        self.engine.is_stopped = True
        await self.progress.emit(self.progress_callback, {
            "current_generation": self.engine.current_generation,
            "is_running": False,
            "is_stopped": True,
            "stop_message": "Budget limit reached before scheduling work" if budget_reached
                            else "Evolution stopped before scheduling work",
            "token_counts": self.engine.get_total_token_count(),
        })
        return True

    async def _run_new(self, *, start_generation: int) -> None:
        if not self.engine.checkpoint_id:
            from datetime import datetime

            self.engine.checkpoint_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        self.engine.current_generation = 0
        self.engine.population = []
        self.engine.specific_prompts = []
        self.engine.contexts = []
        self.engine.history = []
        self.engine.breeding_prompts = []
        self.engine.tournament_history = []
        self.engine.comparison_cache = {}
        self.engine.quality_archive = []
        self.engine.gene_catalogue = empty_gene_catalogue()
        self.engine.generation_draft = None
        self.engine.diversity_history = []
        self.engine.island_diversity_history = []
        self.engine.migration_history = []

        print("Generating initial population (Generation 0)...")

        est_tournament_rounds = self._planned_generation_tournament_rounds()
        estimated_children_per_gen = self.engine._compute_replacement_count(self.engine.pop_size)
        steps_per_gen = estimated_children_per_gen + est_tournament_rounds + 1
        total_steps = (
            (2 * self.engine.pop_size)
            + (self.engine.generations * steps_per_gen)
            + est_tournament_rounds
        )

        await self._seed_initial_population(total_steps=total_steps)
        if self.engine.stop_requested:
            return

        self.engine.history = [self.engine.population.copy()]
        await self.engine._calculate_and_store_diversity()
        await self._observe_population_genes(0)
        if self.engine.stop_requested:
            return

        token_counts = self.engine.get_total_token_count()
        current_cost = token_counts["cost"]["total_cost"]
        self.engine.avg_idea_cost = current_cost / max(1, self.engine.pop_size)

        children_per_gen = self.engine._compute_replacement_count(self.engine.pop_size)
        remaining_ideas = self.engine.generations * children_per_gen
        remaining_tournaments = self.engine.generations + 1
        estimated_total_cost = current_cost + (
            remaining_ideas * self.engine.avg_idea_cost
        ) + (remaining_tournaments * self.engine.avg_tournament_cost)
        token_counts["cost"]["estimated_total_cost"] = estimated_total_cost

        base_payload = self.progress.base(
            current_generation=0, is_running=True, include_core_state=True
        )
        base_payload.update(
            {
                "progress": (2 * self.engine.pop_size / max(1, total_steps)) * 100,
                "token_counts": token_counts,
            }
        )
        await self.progress.emit(self.progress_callback, base_payload)

        if self.engine.check_budget():
            self.engine.stop_requested = True
            self.engine.is_stopped = True
            stop_payload = self.progress.base(
                current_generation=0,
                is_running=False,
                include_core_state=True,
            )
            stop_payload.update(
                {
                    "is_stopped": True,
                    "stop_message": (
                        "Evolution stopped: Budget limit reached "
                        f"(${current_cost:.2f} / ${self.engine.max_budget:.2f})"
                    ),
                    "token_counts": token_counts,
                }
            )
            await self.progress.emit(self.progress_callback, stop_payload)
            return

        await asyncio.sleep(0.1)

        work_state = GenerationWorkState(
            generation_index=0,
            start_generation=start_generation,
            total_generations=self.engine.generations,
            total_steps=total_steps,
            steps_per_generation=steps_per_gen,
        )

        await self._evolve_generations(
            start_generation=0,
            total_steps=total_steps,
            step_offset=(2 * self.engine.pop_size),
            work_state=work_state,
            resuming=False,
        )

    async def _run_resume(self, *, start_generation: int) -> None:
        if start_generation == 0 and len(self.engine.population) < self.engine.pop_size:
            print(
                f"⚠️ Generation 0 incomplete ({len(self.engine.population)}/{self.engine.pop_size} ideas). "
                "Restarting seeding..."
            )
            await self._complete_initial_seeding()
            if self.engine.stop_requested:
                return
            start_generation = self.engine.current_generation

        print(
            f"🔄 Resuming evolution from generation {start_generation}/{self.engine.generations}"
        )

        await self._observe_population_genes(start_generation)
        if self.engine.stop_requested:
            return

        token_counts = self.engine.get_total_token_count()
        initial_payload = self.progress.base(
            current_generation=start_generation,
            is_running=True,
            include_core_state=True,
        )
        initial_payload.update(
            {
                "is_resuming": True,
                "progress": (start_generation / self.engine.generations) * 100
                if self.engine.generations > 0
                else 0,
                "status_message": f"Resuming from generation {start_generation}...",
                "token_counts": token_counts,
            }
        )
        await self.progress.emit(self.progress_callback, initial_payload)

        est_tournament_rounds = self._planned_generation_tournament_rounds()
        estimated_children_per_gen = self.engine._compute_replacement_count(self.engine.pop_size)
        steps_per_gen = estimated_children_per_gen + est_tournament_rounds + 1
        remaining_gens = self.engine.generations - start_generation
        total_steps = max(1, (remaining_gens * steps_per_gen) + est_tournament_rounds)

        work_state = GenerationWorkState(
            generation_index=start_generation,
            start_generation=start_generation,
            total_generations=self.engine.generations,
            total_steps=total_steps,
            steps_per_generation=steps_per_gen,
        )

        await self._evolve_generations(
            start_generation=start_generation,
            total_steps=total_steps,
            step_offset=0,
            work_state=work_state,
            resuming=True,
        )

    async def _seed_initial_population(self, *, total_steps: int) -> None:
        if await self._stop_before_provider_work():
            return
        seed_island_ids = self._initial_island_ids(self.engine.pop_size)

        async def generate_single_seed_for_island(island_id: int):
            context_pool, idea, prompt = await self._generate_single_seed()
            self._assign_idea_island_id(idea, island_id, force=self._islands_enabled())
            return context_pool, idea, prompt, island_id

        seed_tasks = [
            (lambda island_id=island_id: generate_single_seed_for_island(island_id))
            for island_id in seed_island_ids
        ]

        base_info = {
            "current_generation": 0,
            "total_generations": self.engine.generations,
            "is_running": True,
        }

        seed_results = await self.engine._run_batch_with_progress(
            tasks=seed_tasks,
            progress_callback=self.progress_callback,
            base_progress_info=base_info,
            start_step=0,
            total_steps=total_steps,
            description_template="Seeding idea {completed}/{total}...",
        )

        if self.engine.stop_requested:
            completed_results = [r for r in seed_results if r is not None]
            for context_pool, idea, prompt, _island_id in completed_results:
                self.engine.contexts.append(context_pool)
                self.engine.population.append(idea)
                self.engine.specific_prompts.append(prompt)

            payload = self.progress.base(
                current_generation=0,
                is_running=False,
                include_core_state=False,
            )
            payload.update(
                {
                    "is_stopped": True,
                    "history": [self.engine.population] if self.engine.population else [],
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "stop_message": "Evolution stopped during initial generation",
                }
            )
            await self.progress.emit(self.progress_callback, payload)
            return

        existing_context_sets_by_island: Dict[int, List[set[str]]] = {
            island_id: [] for island_id in range(self._island_count())
        }
        for idx, seed_result in enumerate(seed_results):
            island_id = seed_island_ids[idx] if idx < len(seed_island_ids) else 0
            if seed_result is None:
                increment_diagnostic = getattr(self.engine.ideator, "_increment_diagnostic", None)
                if callable(increment_diagnostic):
                    increment_diagnostic("seed_generation_failures")
                try:
                    context_pool, idea, prompt = await self._generate_single_seed()
                    self._assign_idea_island_id(idea, island_id, force=self._islands_enabled())
                    seed_result = (context_pool, idea, prompt, island_id)
                except Exception as exc:
                    if callable(increment_diagnostic):
                        increment_diagnostic("seed_generation_failures", detail=str(exc))
                    continue

            context_pool, idea, prompt, island_id = seed_result
            context_pool, idea, prompt = await self._ensure_seed_novelty(
                (context_pool, idea, prompt),
                existing_context_sets_by_island.setdefault(island_id, []),
            )
            self._assign_idea_island_id(idea, island_id, force=self._islands_enabled())
            self.engine.contexts.append(context_pool)
            self.engine.population.append(idea)
            self.engine.specific_prompts.append(prompt)

        async def refine_single(idea, island_id: Optional[int] = None):
            formatted = await self._prepare_idea_for_population(
                idea,
                refine=self._legacy_refine_all_enabled(),
            )
            if island_id is not None:
                self._assign_idea_island_id(
                    formatted,
                    island_id,
                    force=self._islands_enabled(),
                    origin=(
                        idea.get("origin_island_id")
                        if isinstance(idea, dict)
                        else island_id
                    ),
                )
            return formatted

        refine_tasks = [
            (
                lambda i=idea, island_id=self._get_idea_island_id(idea, fallback=0): refine_single(
                    i,
                    island_id,
                )
            )
            for idea in self.engine.population
        ]

        refined_results = await self.engine._run_batch_with_progress(
            tasks=refine_tasks,
            progress_callback=self.progress_callback,
            base_progress_info=base_info,
            start_step=self.engine.pop_size,
            total_steps=total_steps,
            description_template=(
                "Refining idea {completed}/{total}..."
                if self._legacy_refine_all_enabled()
                else "Formatting idea {completed}/{total}..."
            ),
        )

        if self.engine.stop_requested:
            for i, result in enumerate(refined_results):
                if result is not None and self._is_usable_idea(result):
                    self.engine.population[i] = result

            payload = self.progress.base(
                current_generation=0,
                is_running=False,
                include_core_state=False,
            )
            payload.update(
                {
                    "is_stopped": True,
                    "history": [self.engine.population],
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "stop_message": "Evolution stopped during refinement",
                }
            )
            await self.progress.emit(self.progress_callback, payload)
            return

        filtered_population: List[Any] = []
        filtered_contexts: List[str] = []
        filtered_prompts: List[str] = []
        for idx, result in enumerate(refined_results):
            if result is None or not self._is_usable_idea(result):
                self._record_generation_diagnostic("invalid_formatted_seed_ideas")
                continue
            filtered_population.append(result)
            filtered_contexts.append(self.engine.contexts[idx])
            filtered_prompts.append(self.engine.specific_prompts[idx])

        self.engine.population = filtered_population
        self.engine.contexts = filtered_contexts
        self.engine.specific_prompts = filtered_prompts

        refill_attempts = 0
        max_refill_attempts = max(3, self.engine.pop_size * 4)
        while (
            not self.engine.stop_requested
            and len(self.engine.population) < self.engine.pop_size
            and refill_attempts < max_refill_attempts
        ):
            refill_attempts += 1
            try:
                island_id = self._next_refill_island_id()
                seed_result = await self._generate_single_seed()
                context_pool, idea, prompt = seed_result
                self._assign_idea_island_id(idea, island_id, force=self._islands_enabled())
                context_pool, idea, prompt = await self._ensure_seed_novelty(
                    (context_pool, idea, prompt),
                    existing_context_sets_by_island.setdefault(island_id, []),
                )
                formatted_idea = await refine_single(idea, island_id)
            except Exception as exc:
                self._record_generation_diagnostic("seed_generation_failures", detail=str(exc))
                continue

            if not self._is_usable_idea(formatted_idea):
                self._record_generation_diagnostic("invalid_formatted_seed_ideas")
                continue

            self.engine.contexts.append(context_pool)
            self.engine.population.append(formatted_idea)
            self.engine.specific_prompts.append(prompt)

        if len(self.engine.population) < self.engine.pop_size:
            raise RuntimeError(
                "Failed to produce a full non-empty initial population "
                f"({len(self.engine.population)}/{self.engine.pop_size})."
            )

    async def _complete_initial_seeding(self) -> None:
        existing_count = len(self.engine.population)
        needed_count = self.engine.pop_size - existing_count
        print(
            f"🌱 Completing initial seeding: {existing_count} existing, need {needed_count} more"
        )
        fallback_ids = self._initial_island_ids(max(self.engine.pop_size, existing_count))
        for idx, idea in enumerate(self.engine.population):
            self._assign_idea_island_id(
                idea,
                fallback_ids[idx] if idx < len(fallback_ids) else 0,
                force=self._islands_enabled(),
            )

        existing_context_sets_by_island: Dict[int, List[set[str]]] = {
            island_id: [] for island_id in range(self._island_count())
        }
        for idx, ctx in enumerate(self.engine.contexts):
            fallback = fallback_ids[idx] if idx < len(fallback_ids) else 0
            idea = self.engine.population[idx] if idx < len(self.engine.population) else None
            island_id = self._get_idea_island_id(idea, fallback=fallback)
            existing_context_sets_by_island.setdefault(island_id, []).append(
                self._extract_context_terms(ctx)
            )

        for i in range(needed_count):
            if self.engine.stop_requested:
                print("Stop requested during seeding completion")
                return

            await self.progress.emit(
                self.progress_callback,
                {
                    "current_generation": 0,
                    "total_generations": self.engine.generations,
                    "is_running": True,
                    "status_message": (
                        f"Creating seed idea {existing_count + i + 1}/{self.engine.pop_size}..."
                    ),
                    "progress": ((existing_count + i) / self.engine.pop_size) * 50,
                },
            )

            seed_result = None
            for _ in range(3):
                try:
                    seed_result = await self._generate_single_seed()
                    break
                except Exception as exc:
                    increment_diagnostic = getattr(self.engine.ideator, "_increment_diagnostic", None)
                    if callable(increment_diagnostic):
                        increment_diagnostic("seed_generation_failures", detail=str(exc))
            if seed_result is None:
                continue

            context_pool, idea, prompt = seed_result
            island_id = self._next_refill_island_id()
            self._assign_idea_island_id(idea, island_id, force=self._islands_enabled())
            context_pool, idea, prompt = await self._ensure_seed_novelty(
                (context_pool, idea, prompt),
                existing_context_sets_by_island.setdefault(island_id, []),
            )
            self._assign_idea_island_id(idea, island_id, force=self._islands_enabled())
            self.engine.contexts.append(context_pool)
            self.engine.population.append(idea)
            self.engine.specific_prompts.append(prompt)

        print("Refining initial population...")
        refined_population = []
        for i, idea in enumerate(self.engine.population):
            if self.engine.stop_requested:
                print("Stop requested during refinement")
                return

            await self.progress.emit(
                self.progress_callback,
                {
                    "current_generation": 0,
                    "total_generations": self.engine.generations,
                    "is_running": True,
                    "status_message": f"Refining idea {i + 1}/{len(self.engine.population)}...",
                    "progress": 50 + ((i + 1) / len(self.engine.population)) * 50,
                },
            )

            try:
                island_id = self._get_idea_island_id(idea, fallback=0)
                formatted_idea = await self._prepare_idea_for_population(
                    idea,
                    refine=self._legacy_refine_all_enabled(),
                )
                self._assign_idea_island_id(
                    formatted_idea,
                    island_id,
                    force=self._islands_enabled(),
                    origin=(
                        idea.get("origin_island_id")
                        if isinstance(idea, dict)
                        else island_id
                    ),
                )
            except Exception as exc:
                self._record_generation_diagnostic("seed_generation_failures", detail=str(exc))
                continue

            if self._is_usable_idea(formatted_idea):
                refined_population.append(formatted_idea)
            elif self._is_usable_idea(idea):
                self._record_generation_diagnostic("invalid_formatted_seed_ideas")
                refined_population.append(idea)
            else:
                self._record_generation_diagnostic("invalid_formatted_seed_ideas")

        if not refined_population:
            raise RuntimeError("Unable to recover any usable ideas during initial seeding resume.")

        self.engine.population = refined_population
        self.engine.history = [self.engine.population.copy()]

        await self.engine._calculate_and_store_diversity()

        self.engine.current_generation = 1
        await self.engine.save_checkpoint(status="in_progress")

        token_counts = self.engine.get_total_token_count()
        payload = self.progress.base(
            current_generation=0,
            is_running=True,
            include_core_state=True,
        )
        payload.update(
            {
                "progress": 100 / (self.engine.generations + 1),
                "status_message": "Generation 0 complete!",
                "token_counts": token_counts,
            }
        )
        await self.progress.emit(self.progress_callback, payload)

        print(f"✅ Initial seeding complete. Population size: {len(self.engine.population)}")

    def _local_tournament_rounds(self, population_size):
        full_rounds = max(1, population_size-1)
        count = getattr(self.engine, "tournament_count", None)
        if count is None:
            return max(1, int(getattr(self.engine, "tournament_rounds", 1) or 1))
        return max(1, int(math.floor(float(count)*full_rounds + 0.5)))

    def _planned_generation_tournament_rounds(self):
        size = self.engine.pop_size
        islands = self._island_count() if self._islands_enabled() else 1
        return sum(self._local_tournament_rounds(size//islands + (idx < size % islands))
                   for idx in range(islands))

    def _birth_plans(self, capacities, generation):
        from idea.births import distribute_births
        return distribute_births(
            capacities, getattr(self.engine, "independent_birth_rate", 0.0),
            getattr(self.engine, "single_parent_birth_rate", 0.0), generation)

    def _draft_entry(self, gen, island_id=None):
        draft = getattr(self.engine, "generation_draft", None)
        if not isinstance(draft, dict) or draft.get("generation") != gen + 1:
            return None
        return draft.get("islands", {}).get(str(island_id if island_id is not None else 0))

    def _build_birth_tasks(self, population, fitness, operators, gen, island_id=None,
                           *, survivors=None, fallback=None, tournament_rounds=None):
        entry = self._draft_entry(gen, island_id)
        if entry is not None:
            plans = [(plan["operator"], plan["parents"]) for plan in entry["plans"]]
            return self._birth_tasks_from_plan(plans, gen, island_id, entry)
        indices = sorted(fitness) or list(range(len(population)))
        crossovers = operators.count("crossover")
        slots = self.engine._allocate_parent_slots(fitness, crossovers) if crossovers else {}
        single_available = list(indices)
        plans = []
        for operator in operators:
            parent_indices = []
            if operator == "crossover":
                parent_indices = self.engine._select_parents_from_slots(slots, indices)
                if len(parent_indices) < 2:
                    parent_indices = self.engine.random_choice(
                        indices, size=min(2, len(indices)), replace=False).tolist()
            elif operator == "single_parent":
                if not single_available:
                    single_available = list(indices)
                weights = [max(0.0, float(fitness.get(idx, 0.0))) for idx in single_available]
                total = sum(weights)
                idx = int(self.engine.random_choice(
                    single_available, p=[weight/total for weight in weights] if total else None))
                single_available.remove(idx)
                parent_indices = [idx]
            # A singleton cannot cross over; use a genuine single-parent variant.
            if operator == "crossover" and len(parent_indices) == 1:
                operator = "single_parent"
            plans.append((operator, [population[idx] for idx in parent_indices]))

        entry = self._draft_entry(gen, island_id)
        if entry is None and survivors is not None:
            draft = getattr(self.engine, "generation_draft", None)
            if not isinstance(draft, dict) or draft.get("generation") != gen + 1:
                draft = {"version": 1, "generation": gen + 1, "islands": {}}
                self.engine.generation_draft = draft
            entry = {"island_id": island_id, "capacity": len(population),
                     "survivors": copy.deepcopy(survivors), "fallback": copy.deepcopy(fallback or []),
                     "plans": [{"operator": operator, "parents": copy.deepcopy(parents)} for operator, parents in plans],
                     "results": [None] * len(plans), "tournament_rounds": copy.deepcopy(tournament_rounds or [])}
            draft["islands"][str(island_id if island_id is not None else 0)] = entry
        return self._birth_tasks_from_plan(plans, gen, island_id, entry)

    def _birth_tasks_from_plan(self, plans, gen, island_id, entry=None):
        async def create_birth(index, operator, parents):
            if entry is not None and entry["results"][index] is not None:
                result = entry["results"][index]
                return result["candidate"], result["prompt"]
            if operator == "independent":
                context, new_idea, prompt = await self._generate_single_seed()
                new_idea["birth_context"] = context
            else:
                new_idea = await await_provider_call(
                    self.engine.breeder.breed, parents, self.engine.idea_type)
                prompt = new_idea.get("specific_prompt") if isinstance(new_idea, dict) else None
                if isinstance(new_idea, dict):
                    self._record_parent_breeding_genotypes(
                        parents, new_idea.get("parent_genotypes"), source_generation=gen,
                        child_generation=gen+1, child_id=new_idea.get("id"))
            formatted = await self._prepare_idea_for_population(
                new_idea, refine=self._legacy_refine_all_enabled())
            if not isinstance(formatted, dict):
                formatted = {"id": uuid.uuid4(), "idea": formatted,
                             "parent_ids": [str(parent["id"]) for parent in parents]}
            formatted["birth_generation"] = gen+1
            formatted["birth_type"] = operator
            if island_id is not None:
                self._assign_idea_island_id(formatted, island_id, force=True)
            if not self._is_usable_idea(formatted):
                raise ValueError("New candidate was empty after formatting")
            if entry is not None:
                # Persist each complete child before exposing batch progress. Keep
                # drafts separate from evaluated population/history until commit.
                async with self._draft_save_lock:
                    entry["results"][index] = {"candidate": formatted, "prompt": prompt}
                    await self.engine.save_checkpoint(status="paused" if self.engine.stop_requested else "in_progress")
            return formatted, prompt

        return [(lambda index=index, kind=kind, parents=parents: create_birth(index, kind, parents))
                for index, (kind, parents) in enumerate(plans)]

    async def _resume_draft_island(self, gen, entry, base_progress_info, total_steps):
        plans = [(plan["operator"], plan["parents"]) for plan in entry["plans"]]
        tasks = self._birth_tasks_from_plan(plans, gen, entry["island_id"], entry)
        # Completed slots never enter the provider queue again.
        missing = [task for index, task in enumerate(tasks) if entry["results"][index] is None]
        if missing:
            await self.engine._run_batch_with_progress(
                tasks=missing, progress_callback=self.progress_callback,
                base_progress_info=base_progress_info, start_step=0, total_steps=total_steps,
                description_template="Finishing saved generation: child {completed}/{total}...")
        population = copy.deepcopy(entry["survivors"])
        prompts = [None] * len(population)
        for result in entry["results"]:
            if result is not None:
                population.append(result["candidate"])
                prompts.append(result["prompt"])
        if not self.engine.stop_requested:
            existing = {str(c.get("id")) for c in population}
            for candidate in entry["fallback"]:
                if len(population) >= entry["capacity"]:
                    break
                if str(candidate.get("id")) not in existing:
                    population.append(copy.deepcopy(candidate))
                    prompts.append(None)
                    existing.add(str(candidate.get("id")))
        return {"population": population, "prompts": prompts,
                "tournament_rounds": entry["tournament_rounds"], "stopped": self.engine.stop_requested}

    async def _commit_generation_population(self, gen, population, prompts):
        self.engine.population = population
        self.engine.history.append(population)
        self.engine.breeding_prompts.append(prompts)
        self.engine.current_generation = gen + 1
        self.engine.generation_draft = None
        # Save before any optional provider work, including observations/diversity.
        await self.engine.save_checkpoint(status="paused" if self.engine.stop_requested else "in_progress")

    async def _evolve_generations(
        self,
        *,
        start_generation: int,
        total_steps: int,
        step_offset: int,
        work_state: GenerationWorkState,
        resuming: bool,
    ) -> None:
        if self._islands_enabled():
            await self._evolve_island_generations(
                start_generation=start_generation,
                total_steps=total_steps,
                step_offset=step_offset,
                work_state=work_state,
                resuming=resuming,
            )
            return

        for gen in range(start_generation, self.engine.generations):
            if self.engine.stop_requested:
                self.engine.is_stopped = True
                self.engine.current_generation = gen
                print(f"Stop requested - evolution halted after generation {gen}")
                await self.engine.save_checkpoint(status="paused")

                payload = self.progress.base(
                    current_generation=gen,
                    is_running=False,
                    include_core_state=True,
                )
                payload.update(
                    {
                        "is_stopped": True,
                        "is_resumable": True,
                        "checkpoint_id": self.engine.checkpoint_id,
                        "progress": self._generation_progress(
                            gen=gen,
                            start_generation=start_generation,
                            total_steps=total_steps,
                            step_offset=step_offset,
                            step_in_generation=0,
                            steps_per_generation=work_state.steps_per_generation,
                        ),
                        "stop_message": (
                            f"Evolution paused after generation {gen}. You can resume this evolution."
                        ),
                        "token_counts": self.engine.get_total_token_count(),
                    }
                )
                await self.progress.emit(self.progress_callback, payload)
                return

            print(f"Starting generation {gen + 1}...")
            entry = self._draft_entry(gen)
            if entry is not None:
                result = await self._resume_draft_island(gen, entry,
                    {"current_generation": gen + 1, "total_generations": self.engine.generations, "is_running": True},
                    max(1, total_steps))
                if result["stopped"]:
                    return
                new_population, generation_breeding_prompts = result["population"], result["prompts"]
                survivor_count = len(entry["survivors"])
                current_gen_start_step = step_offset + (gen - start_generation) * work_state.steps_per_generation
                breeding_start_step = current_gen_start_step
                breeding_tasks = entry["plans"]
                self.engine._set_tournament_history(gen, result["tournament_rounds"])
            else:
                self.engine.population = self.engine.population.copy()
                self.engine.random_shuffle(self.engine.population)

                current_pop_size = len(self.engine.population)
                ideas_to_breed = self.engine._compute_replacement_count(current_pop_size)
                survivor_count = max(0, current_pop_size - ideas_to_breed)
                planned_rounds = self._local_tournament_rounds(current_pop_size)
                print(
                    f"Generation {gen + 1}: carrying {survivor_count}/{current_pop_size} "
                    f"survivors and breeding {ideas_to_breed} children"
                )

                tournament_start_cost = self.engine.get_total_token_count()["cost"]["total_cost"]

                current_gen_start_step = step_offset + (
                    (gen - start_generation) * work_state.steps_per_generation
                )

                gen_base_progress_info = {
                    "current_generation": gen + 1,
                    "total_generations": self.engine.generations,
                    "is_running": True,
                }

                loop = asyncio.get_running_loop()

                pairs_per_round = max(1, len(self.engine.population) // 2)
                total_pairs = pairs_per_round * max(1, planned_rounds)

                def thread_safe_callback(completed, _total):
                    round_num = min(
                        planned_rounds,
                        (completed // pairs_per_round) + 1,
                    )
                    tournament_fraction = (completed / total_pairs) if total_pairs else 1.0

                    async def send_update():
                        await self.progress_callback(
                            {
                                **gen_base_progress_info,
                                "progress": (
                                    (
                                        current_gen_start_step
                                        + (
                                            tournament_fraction
                                            * max(1, planned_rounds)
                                        )
                                    )
                                    / max(1, total_steps)
                                )
                                * 100,
                                "status_message": (
                                    f"Running Swiss round {round_num}/{planned_rounds}..."
                                ),
                            }
                        )

                    asyncio.run_coroutine_threadsafe(send_update(), loop)

                tournament_rounds_details: List[Dict[str, Any]] = []

                await self._prepare_tournament_evidence(
                    self.engine.population,
                    evaluation_generation=gen,
                )

                def run_tournament():
                    return self._get_tournament_ranks(
                        self.engine.population,
                        rounds=planned_rounds,
                        progress_callback=thread_safe_callback,
                        details=tournament_rounds_details,
                        full_tournament_rounds=self.engine.full_tournament_rounds,
                        should_stop=lambda: self.engine.stop_requested,
                        evaluation_generation=gen,
                        early_stop_top_n=survivor_count,
                    )

                global_ranks = await await_provider_call(run_tournament)
                self.engine._set_tournament_history(gen, tournament_rounds_details)

                if self.engine.stop_requested:
                    payload = self.progress.base(
                        current_generation=gen + 1,
                        is_running=False,
                        include_core_state=True,
                    )
                    payload.update(
                        {
                            "is_stopped": True,
                            "stop_message": "Evolution stopped during tournaments",
                        }
                    )
                    await self.progress.emit(self.progress_callback, payload)
                    return

                tournament_end_cost = self.engine.get_total_token_count()["cost"]["total_cost"]
                current_tournament_cost = tournament_end_cost - tournament_start_cost
                if self.engine.avg_tournament_cost == 0:
                    self.engine.avg_tournament_cost = current_tournament_cost
                else:
                    self.engine.avg_tournament_cost = (
                        self.engine.avg_tournament_cost + current_tournament_cost
                    ) / 2

                fitness_map = await self.engine._score_population_fitness(
                    self.engine.population, global_ranks
                )
                if not fitness_map:
                    fitness_map = {
                        idx: {
                            "elo": float(global_ranks.get(idx, 0.0)),
                            "diversity": 0.0,
                            "elo_norm": 0.5,
                            "diversity_norm": 0.5,
                            "fitness": 0.5,
                        }
                        for idx in global_ranks.keys()
                    }

                next_generation_index = gen + 1
                survival_scores = self.engine._score_survival_with_age_decay(
                    self.engine.population, fitness_map, target_generation=next_generation_index
                )
                self._attach_tree_metrics(
                    self.engine.population,
                    fitness_map,
                    survival_scores,
                    evaluation_generation=gen,
                    target_generation=next_generation_index,
                )

                survivor_indices = self.engine._select_survivor_indices(
                    survival_scores, survivor_count
                )
                if len(survivor_indices) < survivor_count:
                    remaining = [
                        idx for idx in range(current_pop_size) if idx not in set(survivor_indices)
                    ]
                    self.engine.random_shuffle(remaining)
                    survivor_indices.extend(remaining[: (survivor_count - len(survivor_indices))])

                def clone_survivor(idea: Any) -> Any:
                    if not isinstance(idea, dict):
                        return idea
                    cloned = dict(idea)
                    cloned["birth_generation"] = self.engine._get_birth_generation(idea)
                    cloned["survived_to_generation"] = next_generation_index
                    return cloned

                new_population: List[Any] = [
                    clone_survivor(self.engine.population[idx]) for idx in survivor_indices
                ]
                if not self._legacy_refine_all_enabled():
                    new_population = [
                        await self._maybe_refine_survivor(
                            survivor,
                            target_generation=next_generation_index,
                        )
                        for survivor in new_population
                    ]
                generation_breeding_prompts: List[Optional[str]] = [None] * len(new_population)

                parent_fitness = {
                    idx: float(data.get("fitness", 0.0)) for idx, data in fitness_map.items()
                }
                available_parent_indices = sorted(parent_fitness.keys())
                if not available_parent_indices:
                    available_parent_indices = list(range(current_pop_size))
                    parent_fitness = {idx: 1.0 for idx in available_parent_indices}

                operators = self._birth_plans([ideas_to_breed], gen)[0]
                breeding_tasks = self._build_birth_tasks(
                    self.engine.population, parent_fitness, operators, gen,
                    survivors=new_population,
                    fallback=[clone_survivor(self.engine.population[idx]) for idx in sorted(
                        range(current_pop_size), key=lambda i: float(survival_scores.get(i, {}).get("survival_score", 0.0)), reverse=True)],
                    tournament_rounds=tournament_rounds_details)

                breeding_start_step = current_gen_start_step + max(1, planned_rounds)

                breeding_results = await self.engine._run_batch_with_progress(
                    tasks=breeding_tasks,
                    progress_callback=self.progress_callback,
                    base_progress_info=gen_base_progress_info,
                    start_step=breeding_start_step,
                    total_steps=max(1, total_steps),
                    description_template="Breeding and refining idea {completed}/{total}...",
                )

                if self.engine.stop_requested:
                    completed_results = [r for r in breeding_results if r is not None]
                    for idea, prompt in completed_results:
                        if not self._is_usable_idea(idea):
                            self._record_generation_diagnostic("invalid_formatted_bred_ideas")
                            continue
                        new_population.append(idea)
                        generation_breeding_prompts.append(prompt)

                    payload = self.progress.base(
                        current_generation=gen + 1,
                        is_running=False,
                        include_core_state=True,
                    )
                    payload.update(
                        {
                            "is_stopped": True,
                            "stop_message": "Evolution stopped during breeding",
                        }
                    )
                    await self.progress.emit(self.progress_callback, payload)
                    return

                for result in breeding_results:
                    if result is None:
                        continue

                    idea, prompt = result
                    if not self._is_usable_idea(idea):
                        self._record_generation_diagnostic("invalid_formatted_bred_ideas")
                        continue
                    new_population.append(idea)
                    generation_breeding_prompts.append(prompt)

                # Keep generation size stable if some child tasks failed.
                if len(new_population) < current_pop_size:
                    missing = current_pop_size - len(new_population)
                    fallback_order = sorted(
                        range(current_pop_size),
                        key=lambda idx: float(
                            survival_scores.get(idx, {}).get("survival_score", 0.0)
                        ),
                        reverse=True,
                    )
                    picked = set(survivor_indices)
                    for idx in fallback_order:
                        if idx in picked:
                            continue
                        new_population.append(clone_survivor(self.engine.population[idx]))
                        generation_breeding_prompts.append(None)
                        picked.add(idx)
                        missing -= 1
                        if missing <= 0:
                            break

                if len(new_population) > current_pop_size:
                    new_population = new_population[:current_pop_size]
                    generation_breeding_prompts = generation_breeding_prompts[:current_pop_size]

            token_counts = self.engine.get_total_token_count()
            current_cost = token_counts["cost"]["total_cost"]

            expected_children_per_gen = self.engine._compute_replacement_count(
                self.engine.pop_size
            )
            total_children = self.engine.generations * expected_children_per_gen
            produced_children_this_gen = max(0, len(generation_breeding_prompts) - survivor_count)
            produced_children = (gen * expected_children_per_gen) + produced_children_this_gen
            remaining_ideas_in_run = max(0, total_children - produced_children)
            remaining_tournaments = self.engine.generations - gen

            estimated_total_cost = current_cost + (
                remaining_ideas_in_run * self.engine.avg_idea_cost
            ) + (remaining_tournaments * self.engine.avg_tournament_cost)
            token_counts["cost"]["estimated_total_cost"] = estimated_total_cost

            await self._commit_generation_population(gen, new_population, generation_breeding_prompts)

            if self.engine.check_budget():
                self.engine.stop_requested = True
                self.engine.is_stopped = True
                payload = self.progress.base(
                    current_generation=gen + 1,
                    is_running=False,
                    include_core_state=True,
                )
                payload.update(
                    {
                        "is_stopped": True,
                        "stop_message": (
                            "Evolution stopped: Budget limit reached "
                            f"(${current_cost:.2f} / ${self.engine.max_budget:.2f})"
                        ),
                        "token_counts": token_counts,
                    }
                )
                await self.progress.emit(self.progress_callback, payload)
                return

            history_copy = self.engine.history.copy()

            breeding_prompts_with_current = self.engine.breeding_prompts.copy()

            await self.progress.emit(
                self.progress_callback,
                {
                    "current_generation": gen + 1,
                    "total_generations": self.engine.generations,
                    "is_running": True,
                    "history": history_copy,
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "breeding_prompts": breeding_prompts_with_current,
                    "progress": (
                        (
                            breeding_start_step
                + len(breeding_tasks)
                        )
                        / max(1, total_steps)
                    )
                    * 100,
                    "token_counts": token_counts,
                },
            )

            await asyncio.sleep(0.1)


            await self._apply_oracle_if_enabled(gen)
            await self.engine._calculate_and_store_diversity()
            # Commit the generation before optional observation so a pause
            # preserves its candidates and resumes at the correct boundary.
            self.engine.current_generation = gen + 1
            await self._observe_population_genes(gen + 1)
            if self.engine.stop_requested:
                return

            # Final judging still has to finish before this run is complete.
            checkpoint_path = await self.engine.save_checkpoint(status="in_progress")
            if checkpoint_path:
                await self.progress.emit(
                    self.progress_callback,
                    {
                        "current_generation": gen + 1,
                        "total_generations": self.engine.generations,
                        "is_running": True,
                        "checkpoint_saved": True,
                        "checkpoint_id": self.engine.checkpoint_id,
                    },
                )

        if not self.engine.stop_requested:
            await self._score_final_generation_for_continuation()
            if self.engine.stop_requested:
                return
            token_counts = self.engine.get_total_token_count()
            token_counts["cost"]["estimated_total_cost"] = token_counts["cost"]["total_cost"]
            payload = self.progress.base(
                current_generation=self.engine.generations,
                is_running=False,
                include_core_state=True,
            )
            payload.update(
                {
                    **terminal_status_fields("complete"),
                    "is_stopped": False,
                    "is_resumable": False,
                    "durably_saved": True,
                    "checkpoint_id": self.engine.checkpoint_id,
                    "progress": 100,
                    "token_counts": token_counts,
                    "tournament_history": self.engine.tournament_history,
                }
            )
            await self.engine.save_checkpoint(status="complete")
            await self.progress.emit(self.progress_callback, payload)

    async def _evolve_single_island_generation(
        self,
        *,
        gen: int,
        island_id: int,
        island_population: List[Any],
        current_gen_start_step: int,
        total_steps: int,
        gen_base_progress_info: Dict[str, Any],
        birth_operators: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        entry = self._draft_entry(gen, island_id)
        if entry is not None:
            return await self._resume_draft_island(gen, entry, gen_base_progress_info, max(1, total_steps))
        self.engine.random_shuffle(island_population)
        for idea in island_population:
            self._assign_idea_island_id(idea, island_id, force=True)

        current_pop_size = len(island_population)
        ideas_to_breed = self.engine._compute_replacement_count(current_pop_size)
        survivor_count = max(0, current_pop_size - ideas_to_breed)
        next_generation_index = gen + 1
        print(
            f"Generation {next_generation_index}, island {island_id + 1}: "
            f"carrying {survivor_count}/{current_pop_size} survivors and breeding "
            f"{ideas_to_breed} children"
        )

        local_rounds = self._local_tournament_rounds(current_pop_size)
        local_full_rounds = max(1, current_pop_size - 1)
        tournament_start_cost = self.engine.get_total_token_count()["cost"]["total_cost"]
        loop = asyncio.get_running_loop()
        pairs_per_round = max(1, current_pop_size // 2)
        total_pairs = pairs_per_round * local_rounds

        def thread_safe_callback(completed, _total):
            round_num = min(local_rounds, (completed // pairs_per_round) + 1)
            tournament_fraction = (completed / total_pairs) if total_pairs else 1.0

            async def send_update():
                await self.progress_callback(
                    {
                        **gen_base_progress_info,
                        "progress": (
                            (
                                current_gen_start_step
                                + (tournament_fraction * local_rounds)
                            )
                            / max(1, total_steps)
                        )
                        * 100,
                        "status_message": (
                            f"Island {island_id + 1}: Swiss round {round_num}/{local_rounds}..."
                        ),
                    }
                )

            asyncio.run_coroutine_threadsafe(send_update(), loop)

        tournament_rounds_details: List[Dict[str, Any]] = []

        await self._prepare_tournament_evidence(
            island_population,
            evaluation_generation=gen,
        )

        def run_tournament():
            return self._get_tournament_ranks(
                island_population,
                rounds=local_rounds,
                progress_callback=thread_safe_callback,
                details=tournament_rounds_details,
                full_tournament_rounds=local_full_rounds,
                should_stop=lambda: self.engine.stop_requested,
                evaluation_generation=gen,
                early_stop_top_n=survivor_count,
            )

        global_ranks = await await_provider_call(run_tournament)
        for detail in tournament_rounds_details:
            if isinstance(detail, dict):
                detail["island_id"] = island_id

        tournament_end_cost = self.engine.get_total_token_count()["cost"]["total_cost"]
        current_tournament_cost = tournament_end_cost - tournament_start_cost
        if self.engine.avg_tournament_cost == 0:
            self.engine.avg_tournament_cost = current_tournament_cost
        else:
            self.engine.avg_tournament_cost = (
                self.engine.avg_tournament_cost + current_tournament_cost
            ) / 2

        if self.engine.stop_requested:
            return {
                "population": island_population,
                "prompts": [None] * len(island_population),
                "tournament_rounds": tournament_rounds_details,
                "stopped": True,
            }

        fitness_map = await self.engine._score_population_fitness(island_population, global_ranks)
        if not fitness_map:
            fitness_map = {
                idx: {
                    "elo": float(global_ranks.get(idx, 0.0)),
                    "diversity": 0.0,
                    "elo_norm": 0.5,
                    "diversity_norm": 0.5,
                    "fitness": 0.5,
                }
                for idx in global_ranks.keys()
            }

        survival_scores = self.engine._score_survival_with_age_decay(
            island_population,
            fitness_map,
            target_generation=next_generation_index,
        )
        self._attach_tree_metrics(
            island_population,
            fitness_map,
            survival_scores,
            evaluation_generation=gen,
            target_generation=next_generation_index,
        )
        survivor_indices = self.engine._select_survivor_indices(
            survival_scores,
            survivor_count,
        )
        if len(survivor_indices) < survivor_count:
            remaining = [
                idx for idx in range(current_pop_size) if idx not in set(survivor_indices)
            ]
            self.engine.random_shuffle(remaining)
            survivor_indices.extend(remaining[: (survivor_count - len(survivor_indices))])

        def clone_survivor(idea: Any) -> Any:
            if not isinstance(idea, dict):
                return idea
            cloned = dict(idea)
            cloned["birth_generation"] = self.engine._get_birth_generation(idea)
            cloned["survived_to_generation"] = next_generation_index
            self._assign_idea_island_id(
                cloned,
                island_id,
                force=True,
                origin=idea.get("origin_island_id", island_id),
            )
            return cloned

        new_population: List[Any] = [
            clone_survivor(island_population[idx]) for idx in survivor_indices
        ]
        if not self._legacy_refine_all_enabled():
            new_population = [
                self._assign_idea_island_id(
                    await self._maybe_refine_survivor(
                        survivor,
                        target_generation=next_generation_index,
                    ),
                    island_id,
                    force=True,
                    origin=(
                        survivor.get("origin_island_id")
                        if isinstance(survivor, dict)
                        else island_id
                    ),
                )
                for survivor in new_population
            ]
        generation_breeding_prompts: List[Optional[str]] = [None] * len(new_population)

        parent_fitness = {
            idx: float(data.get("fitness", 0.0)) for idx, data in fitness_map.items()
        }
        available_parent_indices = sorted(parent_fitness.keys())
        if not available_parent_indices:
            available_parent_indices = list(range(current_pop_size))
            parent_fitness = {idx: 1.0 for idx in available_parent_indices}

        operators = (birth_operators if birth_operators is not None else
                     self._birth_plans([ideas_to_breed], gen)[0])
        breeding_tasks = self._build_birth_tasks(
            island_population, parent_fitness, operators, gen, island_id,
            survivors=new_population,
            fallback=[clone_survivor(island_population[idx]) for idx in sorted(
                range(current_pop_size), key=lambda i: float(survival_scores.get(i, {}).get("survival_score", 0.0)), reverse=True)],
            tournament_rounds=tournament_rounds_details)

        breeding_start_step = current_gen_start_step + local_rounds
        breeding_results = await self.engine._run_batch_with_progress(
            tasks=breeding_tasks,
            progress_callback=self.progress_callback,
            base_progress_info=gen_base_progress_info,
            start_step=breeding_start_step,
            total_steps=max(1, total_steps),
            description_template=(
                f"Island {island_id + 1}: breeding idea "
                "{completed}/{total}..."
            ),
        )

        if self.engine.stop_requested:
            completed_results = [r for r in breeding_results if r is not None]
            for idea, prompt in completed_results:
                if not self._is_usable_idea(idea):
                    self._record_generation_diagnostic("invalid_formatted_bred_ideas")
                    continue
                self._assign_idea_island_id(idea, island_id, force=True)
                new_population.append(idea)
                generation_breeding_prompts.append(prompt)

            return {
                "population": new_population,
                "prompts": generation_breeding_prompts,
                "tournament_rounds": tournament_rounds_details,
                "stopped": True,
            }

        for result in breeding_results:
            if result is None:
                continue

            idea, prompt = result
            if not self._is_usable_idea(idea):
                self._record_generation_diagnostic("invalid_formatted_bred_ideas")
                continue
            self._assign_idea_island_id(idea, island_id, force=True)
            new_population.append(idea)
            generation_breeding_prompts.append(prompt)

        if len(new_population) < current_pop_size:
            missing = current_pop_size - len(new_population)
            fallback_order = sorted(
                range(current_pop_size),
                key=lambda idx: float(
                    survival_scores.get(idx, {}).get("survival_score", 0.0)
                ),
                reverse=True,
            )
            picked = set(survivor_indices)
            for idx in fallback_order:
                if idx in picked:
                    continue
                new_population.append(clone_survivor(island_population[idx]))
                generation_breeding_prompts.append(None)
                picked.add(idx)
                missing -= 1
                if missing <= 0:
                    break

        if len(new_population) > current_pop_size:
            new_population = new_population[:current_pop_size]
            generation_breeding_prompts = generation_breeding_prompts[:current_pop_size]

        return {
            "population": new_population,
            "prompts": generation_breeding_prompts,
            "tournament_rounds": tournament_rounds_details,
            "stopped": False,
        }

    async def _evolve_island_generations(
        self,
        *,
        start_generation: int,
        total_steps: int,
        step_offset: int,
        work_state: GenerationWorkState,
        resuming: bool,
    ) -> None:
        for gen in range(start_generation, self.engine.generations):
            if self.engine.stop_requested:
                self.engine.is_stopped = True
                self.engine.current_generation = gen
                print(f"Stop requested - island evolution halted after generation {gen}")
                await self.engine.save_checkpoint(status="paused")
                payload = self.progress.base(
                    current_generation=gen,
                    is_running=False,
                    include_core_state=True,
                )
                payload.update(
                    {
                        "is_stopped": True,
                        "is_resumable": True,
                        "checkpoint_id": self.engine.checkpoint_id,
                        "progress": self._generation_progress(
                            gen=gen,
                            start_generation=start_generation,
                            total_steps=total_steps,
                            step_offset=step_offset,
                            step_in_generation=0,
                            steps_per_generation=work_state.steps_per_generation,
                        ),
                        "stop_message": (
                            f"Evolution paused after generation {gen}. You can resume this evolution."
                        ),
                        "token_counts": self.engine.get_total_token_count(),
                    }
                )
                await self.progress.emit(self.progress_callback, payload)
                return

            print(f"Starting generation {gen + 1} across {self._island_count()} islands...")
            island_entries = self._population_island_entries(assign_missing=True)
            current_gen_start_step = step_offset + (
                (gen - start_generation) * work_state.steps_per_generation
            )
            gen_base_progress_info = {
                "current_generation": gen + 1,
                "total_generations": self.engine.generations,
                "is_running": True,
            }

            new_population: List[Any] = []
            generation_breeding_prompts: List[Optional[str]] = []
            tournament_rounds_details: List[Dict[str, Any]] = []

            birth_plans = self._birth_plans(
                [self.engine._compute_replacement_count(len(entries)) for entries in island_entries], gen)
            for island_id, entries in enumerate(island_entries):
                island_population = [entry["idea"] for entry in entries]
                if not island_population:
                    continue

                result = await self._evolve_single_island_generation(
                    gen=gen,
                    island_id=island_id,
                    island_population=island_population,
                    current_gen_start_step=current_gen_start_step,
                    total_steps=total_steps,
                    gen_base_progress_info=gen_base_progress_info,
                    birth_operators=birth_plans[island_id],
                )
                new_population.extend(result["population"])
                generation_breeding_prompts.extend(result["prompts"])
                tournament_rounds_details.extend(result["tournament_rounds"])

                if result["stopped"]:
                    self.engine._set_tournament_history(gen, tournament_rounds_details)
                    payload = self.progress.base(
                        current_generation=gen + 1,
                        is_running=False,
                        include_core_state=True,
                    )
                    payload.update(
                        {
                            "is_stopped": True,
                            "stop_message": "Evolution stopped during island generation",
                        }
                    )
                    await self.progress.emit(self.progress_callback, payload)
                    return

            self.engine._set_tournament_history(gen, tournament_rounds_details)

            token_counts = self.engine.get_total_token_count()
            current_cost = token_counts["cost"]["total_cost"]

            expected_children_per_gen = self.engine._compute_replacement_count(
                self.engine.pop_size
            )
            total_children = self.engine.generations * expected_children_per_gen
            produced_children_this_gen = max(
                0,
                len(generation_breeding_prompts)
                - max(0, len(new_population) - expected_children_per_gen),
            )
            produced_children = (gen * expected_children_per_gen) + produced_children_this_gen
            remaining_ideas_in_run = max(0, total_children - produced_children)
            remaining_tournaments = self.engine.generations - gen
            token_counts["cost"]["estimated_total_cost"] = current_cost + (
                remaining_ideas_in_run * self.engine.avg_idea_cost
            ) + (remaining_tournaments * self.engine.avg_tournament_cost)

            await self._commit_generation_population(gen, new_population, generation_breeding_prompts)

            if self.engine.check_budget():
                self.engine.stop_requested = True
                self.engine.is_stopped = True
                payload = self.progress.base(
                    current_generation=gen + 1,
                    is_running=False,
                    include_core_state=True,
                )
                payload.update(
                    {
                        "is_stopped": True,
                        "stop_message": (
                            "Evolution stopped: Budget limit reached "
                            f"(${current_cost:.2f} / ${self.engine.max_budget:.2f})"
                        ),
                        "token_counts": token_counts,
                    }
                )
                await self.progress.emit(self.progress_callback, payload)
                return

            history_copy = self.engine.history.copy()
            breeding_prompts_with_current = self.engine.breeding_prompts.copy()

            await self.progress.emit(
                self.progress_callback,
                {
                    "current_generation": gen + 1,
                    "total_generations": self.engine.generations,
                    "is_running": True,
                    "history": history_copy,
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "breeding_prompts": breeding_prompts_with_current,
                    "progress": (
                        (current_gen_start_step + work_state.steps_per_generation)
                        / max(1, total_steps)
                    )
                    * 100,
                    "token_counts": token_counts,
                },
            )

            await asyncio.sleep(0.1)


            await self._apply_island_migration(gen)
            await self._apply_oracle_if_enabled(gen)
            await self.engine._calculate_and_store_diversity()
            self.engine.current_generation = gen + 1
            await self._observe_population_genes(gen + 1)
            if self.engine.stop_requested:
                return
            # Final judging still has to finish before this run is complete.
            checkpoint_path = await self.engine.save_checkpoint(status="in_progress")
            if checkpoint_path:
                await self.progress.emit(
                    self.progress_callback,
                    {
                        "current_generation": gen + 1,
                        "total_generations": self.engine.generations,
                        "is_running": True,
                        "checkpoint_saved": True,
                        "checkpoint_id": self.engine.checkpoint_id,
                    },
                )

        if not self.engine.stop_requested:
            await self._score_final_generation_for_continuation()
            if self.engine.stop_requested:
                return
            token_counts = self.engine.get_total_token_count()
            token_counts["cost"]["estimated_total_cost"] = token_counts["cost"]["total_cost"]
            payload = self.progress.base(
                current_generation=self.engine.generations,
                is_running=False,
                include_core_state=True,
            )
            payload.update(
                {
                    **terminal_status_fields("complete"),
                    "is_stopped": False,
                    "is_resumable": False,
                    "durably_saved": True,
                    "checkpoint_id": self.engine.checkpoint_id,
                    "progress": 100,
                    "token_counts": token_counts,
                    "tournament_history": self.engine.tournament_history,
                }
            )
            await self.engine.save_checkpoint(status="complete")
            await self.progress.emit(self.progress_callback, payload)

    def _protected_novelty_indices(
        self,
        scores: Dict[int, float],
        population_size: int,
    ) -> set[int]:
        if population_size <= 1 or not scores:
            return set()
        try:
            quantile = float(
                getattr(self.engine, "migration_novelty_protection_quantile", 0.8)
            )
        except (TypeError, ValueError):
            quantile = 0.8
        quantile = max(0.0, min(1.0, quantile))
        protect_count = int(math.ceil(population_size * (1.0 - quantile)))
        protect_count = max(1, min(population_size - 1, protect_count))
        ranked = sorted(
            range(population_size),
            key=lambda idx: float(scores.get(idx, 0.0)),
            reverse=True,
        )
        return set(ranked[:protect_count])

    def _clone_migrant(
        self,
        idea: Any,
        *,
        source_island_id: int,
        destination_island_id: int,
        generation: int,
    ) -> Any:
        if isinstance(idea, dict):
            clone = dict(idea)
            source_id = str(idea.get("id")) if idea.get("id") else None
            origin_island_id = idea.get("origin_island_id", source_island_id)
        else:
            clone = {
                "idea": idea,
                "parent_ids": [],
            }
            source_id = None
            origin_island_id = source_island_id

        clone["id"] = uuid.uuid4()
        clone["parent_ids"] = [source_id] if source_id else []
        clone["birth_generation"] = idea.get("birth_generation", 0) if isinstance(idea, dict) else 0
        clone["migration_event"] = True
        clone["migration_generation"] = generation + 1
        clone["migrated_from_id"] = source_id
        clone["source_island_id"] = source_island_id
        self._assign_idea_island_id(
            clone,
            destination_island_id,
            force=True,
            origin=origin_island_id,
        )
        return clone

    async def _apply_island_migration(self, gen: int) -> None:
        if not self._islands_enabled():
            return

        try:
            rate = float(getattr(self.engine, "migration_rate", 0.05))
        except (TypeError, ValueError):
            rate = 0.05
        rate = max(0.0, min(1.0, rate))
        if rate <= 0:
            return

        island_entries = self._population_island_entries(assign_missing=True)
        if len(island_entries) <= 1:
            return

        migration_events: List[Dict[str, Any]] = []
        for source_island_id in range(len(island_entries)):
            if source_island_id >= len(island_entries):
                continue
            source_entries = island_entries[source_island_id]
            if len(source_entries) <= 1:
                continue

            random_uniform = getattr(self.engine, "random_uniform", None)
            draw = random_uniform(0.0, 1.0) if callable(random_uniform) else random.random()
            if rate < 1.0 and float(draw) >= rate:
                continue

            destination_ids = [
                island_id
                for island_id, entries in enumerate(island_entries)
                if island_id != source_island_id and entries
            ]
            if not destination_ids:
                continue

            # Ring topology and a single snapshot prevent index-order bias and
            # prohibit retransmitting arrivals in the same migration step.
            destination_island_id = next((source_island_id + offset) % len(island_entries)
                for offset in range(1, len(island_entries))
                if (source_island_id + offset) % len(island_entries) in destination_ids)
            destination_entries = island_entries[destination_island_id]

            source_population = [entry["idea"] for entry in source_entries]
            destination_population = [entry["idea"] for entry in destination_entries]
            source_scores = await self.engine._calculate_population_diversity_scores(
                source_population
            )
            destination_scores = await self.engine._calculate_population_diversity_scores(
                destination_population
            )
            source_protected = self._protected_novelty_indices(
                source_scores,
                len(source_population),
            )
            destination_protected = self._protected_novelty_indices(
                destination_scores,
                len(destination_population),
            )

            candidate_indices = [
                idx for idx in range(len(source_population)) if idx not in source_protected
            ]
            replacement_indices = [
                idx
                for idx in range(len(destination_population))
                if idx not in destination_protected
            ]
            if not candidate_indices or not replacement_indices:
                continue

            if len(candidate_indices) == 1:
                source_local_idx = candidate_indices[0]
            else:
                source_local_idx = int(self.engine.random_choice(candidate_indices))
            replacement_local_idx = min(
                replacement_indices,
                key=lambda idx: float(destination_scores.get(idx, 0.0)),
            )

            source_idea = source_population[source_local_idx]
            destination_entry = destination_entries[replacement_local_idx]
            destination_global_idx = destination_entry["global_index"]
            replaced_idea = destination_entry["idea"]
            migrated_idea = self._clone_migrant(
                source_idea,
                source_island_id=source_island_id,
                destination_island_id=destination_island_id,
                generation=gen,
            )

            replaced_id = (
                str(replaced_idea.get("id"))
                if isinstance(replaced_idea, dict) and replaced_idea.get("id")
                else None
            )
            if replaced_id:
                await self.engine._remove_embedding(replaced_id)

            self.engine.population[destination_global_idx] = migrated_idea
            if self.engine.history:
                self.engine.history[-1] = self.engine.population.copy()
            if (
                self.engine.breeding_prompts
                and destination_global_idx < len(self.engine.breeding_prompts[-1])
            ):
                self.engine.breeding_prompts[-1][destination_global_idx] = None

            event = {
                "generation": gen + 1,
                "from_island_id": source_island_id,
                "to_island_id": destination_island_id,
                "source_idea_id": (
                    str(source_idea.get("id"))
                    if isinstance(source_idea, dict) and source_idea.get("id")
                    else None
                ),
                "migrated_idea_id": (
                    str(migrated_idea.get("id"))
                    if isinstance(migrated_idea, dict) and migrated_idea.get("id")
                    else None
                ),
                "replaced_idea_id": replaced_id,
            }
            migration_events.append(event)

        if migration_events:
            history = getattr(self.engine, "migration_history", None)
            if isinstance(history, list):
                history.extend(migration_events)
            token_counts = self.engine.get_total_token_count()
            await self.progress.emit(
                self.progress_callback,
                {
                    "current_generation": gen + 1,
                    "total_generations": self.engine.generations,
                    "is_running": True,
                    "history": self.engine.history,
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "breeding_prompts": self.engine.breeding_prompts,
                    "migration_update": True,
                    "migration_history": getattr(self.engine, "migration_history", []),
                    "token_counts": token_counts,
                },
            )

    async def _apply_oracle_if_enabled(self, gen: int) -> None:
        if not self.engine.oracle:
            return

        try:
            if self.engine.stop_requested or self.engine.check_budget():
                return
            if getattr(self.engine, "diversity_representation", "full_text") == "criterion_summaries":
                # Give the Oracle compact evidence for newly born candidates
                # too. Aggregate generation metrics are computed after admission.
                await self.engine._get_or_compute_embeddings_for_ideas(self.engine.population)
                if self.engine.stop_requested or self.engine.check_budget():
                    return
            oracle_result = await await_provider_call(
                self.engine.oracle.analyze_and_diversify,
                self.engine.history, self.engine.idea_type,
            )
            replace_idx = await self.engine._find_least_interesting_idea_idx(self.engine.population)
            old_idea = self.engine.population[replace_idx]
            replace_island_id = self._get_idea_island_id(old_idea, fallback=0)
            idea_prompt = oracle_result["idea_prompt"]
            prompts = self.engine._get_template_prompts()
            extended_prompt = generation_prompt(prompts, idea_prompt)

            if self.engine.stop_requested or self.engine.check_budget():
                return
            new_idea = await await_provider_call(self.engine.ideator.generate_text, extended_prompt)
            oracle_idea = {
                "id": uuid.uuid4(),
                "idea": new_idea,
                "parent_ids": [],
                "birth_generation": gen + 1,
                "oracle_generated": True,
                "oracle_analysis": oracle_result["oracle_analysis"],
            }
            self._assign_idea_island_id(
                oracle_idea,
                replace_island_id,
                force=self._islands_enabled() or (
                    isinstance(old_idea, dict) and "island_id" in old_idea
                ),
                origin=(
                    old_idea.get("origin_island_id")
                    if isinstance(old_idea, dict)
                    else replace_island_id
                ),
            )

            formatted_oracle_idea = await self._prepare_idea_for_population(
                oracle_idea,
                refine=self._legacy_refine_all_enabled(),
            )
            self._assign_idea_island_id(
                formatted_oracle_idea,
                replace_island_id,
                force=self._islands_enabled() or (
                    isinstance(old_idea, dict) and "island_id" in old_idea
                ),
                origin=(
                    old_idea.get("origin_island_id")
                    if isinstance(old_idea, dict)
                    else replace_island_id
                ),
            )

            if not formatted_oracle_idea.get("oracle_generated", False):
                formatted_oracle_idea["oracle_generated"] = True
                formatted_oracle_idea["oracle_analysis"] = oracle_idea.get(
                    "oracle_analysis", ""
                )
            formatted_oracle_idea.setdefault("birth_generation", gen + 1)

            if not self._is_usable_idea(formatted_oracle_idea):
                self._record_generation_diagnostic("invalid_oracle_replacements")
                print("Oracle replacement discarded because formatted idea content was empty.")
                return

            # Oracle immigration is deliberate exploration, not a quality duel.
            # Admit valid content directly; it joins normal judging next cycle.
            # Replacement keeps the configured population and cost bounded.
            if self.engine.stop_requested or self.engine.check_budget():
                return
            formatted_oracle_idea["birth_type"] = "oracle"
            formatted_oracle_idea["birth_generation"] = gen+1
            formatted_oracle_idea["oracle_admission"] = "direct_replacement"

            old_idea_id = str(old_idea.get("id", "")) if isinstance(old_idea, dict) else ""
            if old_idea_id:
                await self.engine._remove_embedding(old_idea_id)

            self.engine.population[replace_idx] = formatted_oracle_idea
            if self.engine.breeding_prompts:
                self.engine.breeding_prompts[-1][replace_idx] = idea_prompt

            self.engine.history[-1] = self.engine.population.copy()

            token_counts = self.engine.get_total_token_count()
            await self.progress.emit(
                self.progress_callback,
                {
                    "current_generation": gen + 1,
                    "total_generations": self.engine.generations,
                    "is_running": True,
                    "history": self.engine.history,
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "breeding_prompts": self.engine.breeding_prompts,
                    "progress": ((gen + 1) / max(1, self.engine.generations)) * 100,
                    "oracle_update": True,
                    "token_counts": token_counts,
                },
            )
        except Exception as exc:
            print(f"Oracle failed with error: {exc}. Continuing without Oracle enhancement.")

    async def _select_elite_for_next_generation(
        self, gen: int
    ) -> tuple[Optional[Any], Optional[str]]:
        elite_idea = None
        elite_breeding_prompt = None
        try:
            most_diverse_idx = await self.engine._find_most_diverse_idea_idx(self.engine.population)
            elite_idea = (
                self.engine.population[most_diverse_idx].copy()
                if isinstance(self.engine.population[most_diverse_idx], dict)
                else self.engine.population[most_diverse_idx]
            )

            if isinstance(self.engine.population[most_diverse_idx], dict):
                self.engine.population[most_diverse_idx]["elite_selected_source"] = True
                self.engine.population[most_diverse_idx]["elite_target_generation"] = gen + 1
                self.engine.history[-1] = self.engine.population.copy()

            if (
                self.engine.breeding_prompts
                and self.engine.breeding_prompts[-1]
                and most_diverse_idx < len(self.engine.breeding_prompts[-1])
            ):
                elite_breeding_prompt = self.engine.breeding_prompts[-1][most_diverse_idx]

            token_counts = self.engine.get_total_token_count()
            await self.progress.emit(
                self.progress_callback,
                {
                    "current_generation": gen + 1,
                    "total_generations": self.engine.generations,
                    "is_running": True,
                    "history": self.engine.history,
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "breeding_prompts": self.engine.breeding_prompts,
                    "progress": ((gen + 1) / max(1, self.engine.generations)) * 100,
                    "elite_selection_update": True,
                    "token_counts": token_counts,
                },
            )
        except Exception as exc:
            print(f"Creative selection failed with error: {exc}. Continuing without creative selection.")

        return elite_idea, elite_breeding_prompt

    @staticmethod
    def _generation_progress(
        *,
        gen: int,
        start_generation: int,
        total_steps: int,
        step_offset: int,
        step_in_generation: int,
        steps_per_generation: int,
    ) -> float:
        current = step_offset + ((gen - start_generation) * steps_per_generation) + step_in_generation
        return (current / max(1, total_steps)) * 100
