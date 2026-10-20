from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from idea import database as db
from idea.evolution_progress import terminal_status_fields
from idea.evolution_types import EvolutionRunConfig
from idea.models import candidate_from_wire, candidate_to_wire, encode_json_safe
from idea.prompts.loader import list_available_templates
from idea.genes import empty_gene_catalogue


class EvolutionPersistenceError(RuntimeError):
    """A checkpoint was not durably saved; callers must surface the failure."""

# Legacy filesystem storage directories (kept for backwards compatibility with local dev)
EVOLUTIONS_DIR = Path("data/evolutions")
CHECKPOINT_DIR = Path("data/checkpoints")


def _safe_local_checkpoint_id(checkpoint_id: Any) -> str:
    if not isinstance(checkpoint_id, str):
        raise ValueError("checkpoint_id must be a string")
    checkpoint_id = checkpoint_id.strip()
    if (
        not checkpoint_id
        or len(checkpoint_id) > 128
        or "\x00" in checkpoint_id
        or "/" in checkpoint_id
        or "\\" in checkpoint_id
        or Path(checkpoint_id).name != checkpoint_id
    ):
        raise ValueError("checkpoint_id must be a plain path-free identifier")
    return checkpoint_id


class EvolutionSerializer:
    @staticmethod
    def serialize_idea(idea: Any) -> Any:
        return candidate_to_wire(idea)

    @classmethod
    def to_checkpoint_state(cls, engine: Any) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "checkpoint_id": engine.checkpoint_id,
            "checkpoint_time": datetime.now().isoformat(),
            "status": "paused" if engine.stop_requested else "in_progress",
            "config": EvolutionRunConfig.from_engine(engine).to_persistence_dict(),
            "current_generation": engine.current_generation,
            "population": [cls.serialize_idea(idea) for idea in engine.population],
            "history": [[cls.serialize_idea(idea) for idea in gen] for gen in engine.history],
            "contexts": engine.contexts,
            "specific_prompts": engine.specific_prompts,
            "breeding_prompts": engine.breeding_prompts,
            "tournament_history": engine.tournament_history,
            "diversity_history": engine.diversity_history,
            "island_diversity_history": getattr(engine, "island_diversity_history", []),
            "migration_history": getattr(engine, "migration_history", []),
            "comparison_cache": getattr(engine, "comparison_cache", {}),
            "algorithm_evidence_version": getattr(engine, "algorithm_evidence_version", 2),
            "quality_archive": [cls.serialize_idea(idea) for idea in getattr(engine, "quality_archive", [])],
            "gene_catalogue": getattr(engine, "gene_catalogue", empty_gene_catalogue()),
            "generation_draft": encode_json_safe(getattr(engine, "generation_draft", None)),
            "avg_idea_cost": engine.avg_idea_cost,
            "avg_tournament_cost": engine.avg_tournament_cost,
            "fitness_elo_stats": engine.fitness_elo_stats,
            "fitness_diversity_stats": engine.fitness_diversity_stats,
            "fitness_metric_stats": engine.fitness_metric_stats,
            "token_counts": engine.get_total_token_count(),
        }
        if engine.template_data:
            state["template_data"] = engine.template_data
        if getattr(engine, "source_checkpoint_id", None):
            state["source_checkpoint_id"] = engine.source_checkpoint_id
        persistence_revision = getattr(engine, "persistence_revision", None)
        if persistence_revision is not None:
            state["_save_revision"] = persistence_revision
        return state

    @staticmethod
    def deserialize_idea(idea_data: Any) -> Any:
        return candidate_from_wire(idea_data)

    @classmethod
    def restore_runtime_state(cls, engine: Any, state: Dict[str, Any]) -> None:
        def _coerce_stats(raw: Any) -> Dict[str, float]:
            if not isinstance(raw, dict):
                return {"count": 0, "mean": 0.0, "m2": 0.0}
            try:
                return {
                    "count": int(raw.get("count", 0)),
                    "mean": float(raw.get("mean", 0.0)),
                    "m2": float(raw.get("m2", 0.0)),
                }
            except (TypeError, ValueError):
                return {"count": 0, "mean": 0.0, "m2": 0.0}

        engine.evolution_id = state.get("evolution_id")
        engine.evolution_name = state.get("name")
        engine.source_checkpoint_id = state.get("source_checkpoint_id")
        engine.created_at = state.get("created_at")
        engine.updated_at = state.get("updated_at")
        persistence_revision = state.get("_save_revision")
        engine.persistence_revision = (
            persistence_revision
            if isinstance(persistence_revision, str) and persistence_revision
            else None
        )

        engine.checkpoint_id = state.get("checkpoint_id") or (
            engine.evolution_id[:18].replace("-", "") if engine.evolution_id else None
        )
        engine.current_generation = state.get("current_generation", 0)
        engine.contexts = state.get("contexts", [])
        engine.specific_prompts = state.get("specific_prompts", [])
        engine.breeding_prompts = state.get("breeding_prompts", [])
        engine.tournament_history = state.get("tournament_history", [])
        engine.diversity_history = state.get("diversity_history", [])
        engine.island_diversity_history = state.get("island_diversity_history", [])
        engine.migration_history = state.get("migration_history", [])
        comparison_cache = state.get("comparison_cache", {})
        engine.comparison_cache = comparison_cache if isinstance(comparison_cache, dict) else {}
        version = state.get("algorithm_evidence_version", 1)
        if not isinstance(version, int) or isinstance(version, bool) or version not in {1, 2}:
            raise ValueError("Unsupported algorithm evidence version")
        engine.algorithm_evidence_version = version
        archive = state.get("quality_archive", [])
        engine.quality_archive = [cls.deserialize_idea(idea) for idea in archive] if isinstance(archive, list) else []
        catalogue = state.get("gene_catalogue")
        engine.gene_catalogue = catalogue if isinstance(catalogue, dict) else empty_gene_catalogue()
        draft = state.get("generation_draft")
        if draft is not None and (not isinstance(draft, dict) or draft.get("version") != 1
                                  or not isinstance(draft.get("islands"), dict)
                                  or draft.get("generation") != engine.current_generation + 1):
            raise ValueError("Invalid pending generation draft")
        draft = copy.deepcopy(draft)
        if draft is not None:
            for entry in draft["islands"].values():
                if (not isinstance(entry, dict) or not isinstance(entry.get("plans"), list)
                        or not isinstance(entry.get("results"), list)
                        or len(entry["plans"]) != len(entry["results"])
                        or not isinstance(entry.get("capacity"), int)
                        or not isinstance(entry.get("survivors"), list)
                        or not isinstance(entry.get("fallback"), list)):
                    raise ValueError("Invalid pending generation island")
                for key in ("survivors", "fallback"):
                    entry[key] = [cls.deserialize_idea(c) for c in entry[key]]
                for plan in entry["plans"]:
                    if (not isinstance(plan, dict) or plan.get("operator") not in
                            {"independent", "single_parent", "crossover"}
                            or not isinstance(plan.get("parents"), list)):
                        raise ValueError("Invalid pending birth plan")
                    plan["parents"] = [cls.deserialize_idea(c) for c in plan["parents"]]
                for result in entry["results"]:
                    if result is not None:
                        if not isinstance(result, dict) or "candidate" not in result or "prompt" not in result:
                            raise ValueError("Invalid saved offspring")
                        result["candidate"] = cls.deserialize_idea(result["candidate"])
        engine.generation_draft = draft
        engine.avg_idea_cost = state.get("avg_idea_cost", 0.0)
        engine.avg_tournament_cost = state.get("avg_tournament_cost", 0.0)
        saved_token_counts = state.get("token_counts")
        engine._persisted_token_counts = (
            saved_token_counts if isinstance(saved_token_counts, dict) else None
        )
        raw_metric_stats = state.get("fitness_metric_stats", {})
        if not isinstance(raw_metric_stats, dict):
            raw_metric_stats = {}
        engine.fitness_elo_stats = _coerce_stats(
            state.get(
                "fitness_elo_stats",
                raw_metric_stats.get("quality_elo.quality_elo"),
            )
        )
        engine.fitness_diversity_stats = _coerce_stats(
            state.get(
                "fitness_diversity_stats",
                raw_metric_stats.get("diversity.diversity"),
            )
        )
        engine.fitness_metric_stats = {
            str(metric_key): _coerce_stats(metric_stats)
            for metric_key, metric_stats in raw_metric_stats.items()
        }
        # Keep legacy public attributes and generic state synchronized.
        engine.fitness_metric_stats[
            "quality_elo.quality_elo"
        ] = engine.fitness_elo_stats
        engine.fitness_metric_stats[
            "diversity.diversity"
        ] = engine.fitness_diversity_stats

        engine.population = [cls.deserialize_idea(idea) for idea in state.get("population", [])]
        engine.history = [[cls.deserialize_idea(idea) for idea in gen] for gen in state.get("history", [])]


class EvolutionRepository:
    @staticmethod
    async def save_checkpoint(engine: Any, status: str = "in_progress") -> Optional[str]:
        try:
            if not engine.user_id:
                raise EvolutionPersistenceError("Cannot save evolution without a user id")

            if not engine.evolution_id:
                engine.initialize_evolution()

            engine.updated_at = datetime.now().isoformat()
            state = EvolutionSerializer.to_checkpoint_state(engine)
            state["evolution_id"] = engine.evolution_id
            state["name"] = engine.evolution_name
            state["status"] = status
            check_budget = getattr(engine, "check_budget", None)
            state.update(terminal_status_fields(
                status,
                budget_reached=status == "paused" and callable(check_budget) and check_budget(),
            ))
            state["created_at"] = engine.created_at
            state["updated_at"] = engine.updated_at

            new_revision = await db.save_evolution(
                engine.user_id,
                engine.evolution_id,
                state,
                expected_revision=getattr(engine, "persistence_revision", None),
                **({"run_token": engine._run_token} if getattr(engine, "_run_token", None) else {}),
            )
            engine.persistence_revision = new_revision
            engine.last_save_error = None
            print(f"💾 Evolution saved to Firestore: {engine.evolution_name} ({engine.evolution_id})")
            return engine.evolution_id
        except (db.EvolutionSaveConflict, db.RunOwnershipLost):
            # A stale engine must never report a successful checkpoint or
            # silently overwrite a newer process's state.
            raise
        except Exception as exc:
            engine.last_save_error = "Checkpoint could not be durably saved"
            raise EvolutionPersistenceError("Checkpoint could not be durably saved") from exc

    @staticmethod
    async def list_evolutions_for_user(user_id: str) -> List[Dict[str, Any]]:
        try:
            evolutions = await db.list_evolutions(user_id)
            result: List[Dict[str, Any]] = []
            for data in evolutions:
                config = data.get("config", {})
                history = data.get("history", [])
                history_summary = data.get("_history_summary") or {}
                population_summary = data.get("_population_summary") or {}
                if isinstance(history, list) and history:
                    generation_count = len(history)
                    total_ideas = sum(len(gen) for gen in history if isinstance(gen, list))
                    inferred_pop_size = len(history[0]) if isinstance(history[0], list) else 0
                else:
                    generation_count = int(history_summary.get("generation_count") or 0)
                    total_ideas = int(history_summary.get("total_ideas") or 0)
                    inferred_pop_size = int(
                        population_summary.get("population_size")
                        or history_summary.get("last_generation_size")
                        or 0
                    )
                result.append(
                    {
                        "id": data.get("evolution_id") or data.get("id"),
                        "name": data.get("name", "Unnamed"),
                        "status": data.get("status", "unknown"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "generation": data.get("current_generation", generation_count),
                        "total_generations": config.get("generations", generation_count),
                        "idea_type": config.get("idea_type", "unknown"),
                        "model_type": config.get("model_type", "unknown"),
                        "pop_size": config.get("pop_size", inferred_pop_size),
                        "total_ideas": total_ideas,
                        "config": config,
                    }
                )
            return result
        except db.EvolutionStorageIntegrityError:
            # Corruption is operationally distinct from a missing evolution;
            # callers need the typed failure to alert or recover explicitly.
            raise
        except Exception as exc:
            print(f"Error listing evolutions: {exc}")
            return []

    @staticmethod
    async def list_checkpoints_for_user(user_id: str) -> List[Dict[str, Any]]:
        try:
            checkpoints = await db.list_checkpoints(user_id)
            result: List[Dict[str, Any]] = []
            for data in checkpoints:
                config = data.get("config", {})
                result.append(
                    {
                        "id": data.get("id") or data.get("checkpoint_id"),
                        "time": data.get("checkpoint_time") or data.get("updated_at"),
                        "status": data.get("status", "unknown"),
                        "generation": data.get("current_generation", 0),
                        "total_generations": config.get("generations", 0),
                        "idea_type": config.get("idea_type", "unknown"),
                        "model_type": config.get("model_type", "unknown"),
                        "pop_size": config.get("pop_size", 0),
                        "config": config,
                    }
                )
            return result
        except Exception as exc:
            print(f"Error listing checkpoints: {exc}")
            return []

    @staticmethod
    async def load_evolution_for_user(
        engine_cls: Type[Any],
        user_id: str,
        evolution_id: str,
        api_key: Optional[str] = None,
        evaluation_runners: Optional[Dict[str, Any]] = None,
        artifact_store: Optional[Any] = None,
    ) -> Optional[Any]:
        try:
            state = await db.get_evolution_for_restore(user_id, evolution_id)
            if not state:
                print(f"❌ Evolution not found: {evolution_id}")
                return None

            legacy_checkpoint_id = state.pop("_legacy_checkpoint_id", None)
            if legacy_checkpoint_id:
                # A legacy snapshot may name a newer evolution or carry an old
                # revision. Restore it as a distinct experiment; never replace
                # either the source checkpoint or the newer evolution.
                new_id = str(uuid.uuid4())
                state.update({
                    "evolution_id": new_id,
                    "checkpoint_id": new_id[:18].replace("-", ""),
                    "source_checkpoint_id": legacy_checkpoint_id,
                    "name": state.get("name") or f"Checkpoint {legacy_checkpoint_id}",
                    "created_at": datetime.now().isoformat(),
                })
                state.pop("_save_revision", None)

            config = state.get("config", {})
            idea_type = config.get("idea_type")
            if idea_type and not state.get("template_data"):
                templates = list_available_templates()
                is_valid_system_template = (
                    idea_type in templates and "error" not in templates.get(idea_type, {})
                )
                if not is_valid_system_template:
                    user_template = await db.get_user_template(user_id, idea_type)
                    if user_template:
                        state["template_data"] = user_template

            return EvolutionRepository.restore_from_state(
                engine_cls,
                state,
                api_key=api_key,
                user_id=user_id,
                evaluation_runners=evaluation_runners,
                artifact_store=artifact_store,
            )
        except db.EvolutionStorageIntegrityError:
            # Corruption is operationally distinct from a missing evolution;
            # callers need the typed failure to alert or recover explicitly.
            raise
        except Exception as exc:
            print(f"❌ Failed to load evolution: {exc}")
            import traceback

            traceback.print_exc()
            return None

    @staticmethod
    async def rename_evolution_for_user(user_id: str, evolution_id: str, new_name: str) -> bool:
        try:
            result = await db.rename_evolution(user_id, evolution_id, new_name)
            if result:
                print(f"✅ Evolution renamed to: {new_name}")
            return result
        except Exception as exc:
            print(f"❌ Failed to rename evolution: {exc}")
            return False

    @staticmethod
    async def delete_evolution_for_user(user_id: str, evolution_id: str) -> bool:
        try:
            result = await db.delete_evolution(user_id, evolution_id)
            if result:
                print(f"🗑️ Evolution deleted: {evolution_id}")
            return result
        except Exception as exc:
            print(f"❌ Failed to delete evolution: {exc}")
            return False

    @staticmethod
    def load_from_file(
        engine_cls: Type[Any],
        file_path: Path,
        api_key: Optional[str] = None,
        evaluation_runners: Optional[Dict[str, Any]] = None,
        artifact_store: Optional[Any] = None,
    ) -> Optional[Any]:
        try:
            with open(file_path) as f:
                state = json.load(f)
            return EvolutionRepository.restore_from_state(
                engine_cls,
                state,
                api_key=api_key,
                evaluation_runners=evaluation_runners,
                artifact_store=artifact_store,
            )
        except Exception as exc:
            print(f"❌ Failed to load from {file_path}: {exc}")
            import traceback

            traceback.print_exc()
            return None

    @staticmethod
    def restore_from_state(
        engine_cls: Type[Any],
        state: Dict[str, Any],
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        evaluation_runners: Optional[Dict[str, Any]] = None,
        artifact_store: Optional[Any] = None,
    ) -> Any:
        run_config = EvolutionRunConfig.from_stored_mapping(state.get("config"))

        engine = engine_cls(
            **run_config.engine_kwargs(),
            evaluation_runners=evaluation_runners,
            artifact_store=artifact_store,
            api_key=api_key,
            user_id=user_id or state.get("user_id"),
            template_data=state.get("template_data"),
        )

        EvolutionSerializer.restore_runtime_state(engine, state)
        engine._sync_typed_state_from_attrs()
        print(
            f"✅ Evolution loaded: {engine.evolution_name or 'unnamed'} "
            f"(gen {engine.current_generation}/{engine.generations})"
        )
        return engine

    @staticmethod
    def load_checkpoint(
        engine_cls: Type[Any],
        checkpoint_id: str,
        api_key: Optional[str] = None,
        evaluation_runners: Optional[Dict[str, Any]] = None,
        artifact_store: Optional[Any] = None,
    ) -> Optional[Any]:
        checkpoint_id = _safe_local_checkpoint_id(checkpoint_id)
        evolution_path = EVOLUTIONS_DIR / f"{checkpoint_id}.json"
        if evolution_path.exists():
            return EvolutionRepository.load_from_file(
                engine_cls,
                evolution_path,
                api_key=api_key,
                evaluation_runners=evaluation_runners,
                artifact_store=artifact_store,
            )

        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{checkpoint_id}.json"
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return None

        return EvolutionRepository.load_from_file(
            engine_cls,
            checkpoint_path,
            api_key=api_key,
            evaluation_runners=evaluation_runners,
            artifact_store=artifact_store,
        )

    @staticmethod
    def delete_checkpoint(checkpoint_id: str) -> bool:
        checkpoint_id = _safe_local_checkpoint_id(checkpoint_id)
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{checkpoint_id}.json"
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"🗑️ Checkpoint deleted: {checkpoint_path}")
                return True
            return False
        except Exception as exc:
            print(f"❌ Failed to delete checkpoint: {exc}")
            return False
