from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from idea import database as db
from idea.config import DEFAULT_CREATIVE_TEMP, DEFAULT_TOP_P
from idea.models import Idea
from idea.prompts.loader import list_available_templates

# Legacy filesystem storage directories (kept for backwards compatibility with local dev)
EVOLUTIONS_DIR = Path("data/evolutions")
CHECKPOINT_DIR = Path("data/checkpoints")


class EvolutionSerializer:
    @staticmethod
    def serialize_idea(idea: Any) -> Any:
        if not isinstance(idea, dict):
            return idea

        result: Dict[str, Any] = {}
        for key, value in idea.items():
            if isinstance(value, uuid.UUID):
                result[key] = str(value)
            elif isinstance(value, list):
                result[key] = [str(v) if isinstance(v, uuid.UUID) else v for v in value]
            elif hasattr(value, "__dict__"):
                result[key] = {
                    "title": getattr(value, "title", None),
                    "content": getattr(value, "content", ""),
                }
            else:
                result[key] = value
        return result

    @classmethod
    def to_checkpoint_state(cls, engine: Any) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "checkpoint_id": engine.checkpoint_id,
            "checkpoint_time": datetime.now().isoformat(),
            "status": "paused" if engine.stop_requested else "in_progress",
            "config": {
                "idea_type": engine.idea_type,
                "pop_size": engine.pop_size,
                "generations": engine.generations,
                "model_type": engine.model_type,
                "creative_temp": engine.creative_temp,
                "top_p": engine.top_p,
                "tournament_rounds": engine.tournament_rounds,
                "tournament_count": engine.tournament_count,
                "full_tournament_rounds": engine.full_tournament_rounds,
                "thinking_budget": engine.thinking_budget,
                "max_budget": engine.max_budget,
                "mutation_rate": engine.mutation_rate,
                "replacement_rate": engine.replacement_rate,
                "fitness_alpha": engine.fitness_alpha,
                "age_decay_rate": engine.age_decay_rate,
                "age_decay_floor": engine.age_decay_floor,
            },
            "current_generation": engine.current_generation,
            "population": [cls.serialize_idea(idea) for idea in engine.population],
            "history": [[cls.serialize_idea(idea) for idea in gen] for gen in engine.history],
            "contexts": engine.contexts,
            "specific_prompts": engine.specific_prompts,
            "breeding_prompts": engine.breeding_prompts,
            "tournament_history": engine.tournament_history,
            "diversity_history": engine.diversity_history,
            "avg_idea_cost": engine.avg_idea_cost,
            "avg_tournament_cost": engine.avg_tournament_cost,
            "fitness_elo_stats": engine.fitness_elo_stats,
            "fitness_diversity_stats": engine.fitness_diversity_stats,
            "token_counts": engine.get_total_token_count(),
        }
        if engine.template_data:
            state["template_data"] = engine.template_data
        return state

    @staticmethod
    def deserialize_idea(idea_data: Any) -> Any:
        if not isinstance(idea_data, dict):
            return idea_data

        result = dict(idea_data)
        if "id" in result and isinstance(result["id"], str):
            try:
                result["id"] = uuid.UUID(result["id"])
            except ValueError:
                result["id"] = uuid.uuid4()

        if "parent_ids" in result:
            result["parent_ids"] = [
                uuid.UUID(pid) if isinstance(pid, str) else pid for pid in result["parent_ids"]
            ]

        if "idea" in result and isinstance(result["idea"], dict):
            result["idea"] = Idea(
                title=result["idea"].get("title"),
                content=result["idea"].get("content", ""),
            )

        return result

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
        engine.created_at = state.get("created_at")
        engine.updated_at = state.get("updated_at")

        engine.checkpoint_id = state.get("checkpoint_id") or (
            engine.evolution_id[:18].replace("-", "") if engine.evolution_id else None
        )
        engine.current_generation = state.get("current_generation", 0)
        engine.contexts = state.get("contexts", [])
        engine.specific_prompts = state.get("specific_prompts", [])
        engine.breeding_prompts = state.get("breeding_prompts", [])
        engine.tournament_history = state.get("tournament_history", [])
        engine.diversity_history = state.get("diversity_history", [])
        engine.avg_idea_cost = state.get("avg_idea_cost", 0.0)
        engine.avg_tournament_cost = state.get("avg_tournament_cost", 0.0)
        engine.fitness_elo_stats = _coerce_stats(state.get("fitness_elo_stats"))
        engine.fitness_diversity_stats = _coerce_stats(
            state.get("fitness_diversity_stats")
        )

        engine.population = [cls.deserialize_idea(idea) for idea in state.get("population", [])]
        engine.history = [[cls.deserialize_idea(idea) for idea in gen] for gen in state.get("history", [])]


class EvolutionRepository:
    @staticmethod
    async def save_checkpoint(engine: Any, status: str = "in_progress") -> Optional[str]:
        try:
            if not engine.user_id:
                print("❌ Cannot save evolution: user_id not set")
                return None

            if not engine.evolution_id:
                engine.initialize_evolution()

            engine.updated_at = datetime.now().isoformat()
            state = EvolutionSerializer.to_checkpoint_state(engine)
            state["evolution_id"] = engine.evolution_id
            state["name"] = engine.evolution_name
            state["status"] = status
            state["created_at"] = engine.created_at
            state["updated_at"] = engine.updated_at

            await db.save_evolution(engine.user_id, engine.evolution_id, state)
            print(f"💾 Evolution saved to Firestore: {engine.evolution_name} ({engine.evolution_id})")
            return engine.evolution_id
        except Exception as exc:
            print(f"❌ Failed to save evolution: {exc}")
            import traceback

            traceback.print_exc()
            return None

    @staticmethod
    async def list_evolutions_for_user(user_id: str) -> List[Dict[str, Any]]:
        try:
            evolutions = await db.list_evolutions(user_id)
            result: List[Dict[str, Any]] = []
            for data in evolutions:
                config = data.get("config", {})
                history = data.get("history", [])
                result.append(
                    {
                        "id": data.get("evolution_id") or data.get("id"),
                        "name": data.get("name", "Unnamed"),
                        "status": data.get("status", "unknown"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "generation": data.get("current_generation", len(history)),
                        "total_generations": config.get("generations", len(history)),
                        "idea_type": config.get("idea_type", "unknown"),
                        "model_type": config.get("model_type", "unknown"),
                        "pop_size": config.get("pop_size", len(history[0]) if history else 0),
                        "total_ideas": sum(len(gen) for gen in history),
                    }
                )
            return result
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
                        "id": data.get("checkpoint_id") or data.get("id"),
                        "time": data.get("checkpoint_time") or data.get("updated_at"),
                        "status": data.get("status", "unknown"),
                        "generation": data.get("current_generation", 0),
                        "total_generations": config.get("generations", 0),
                        "idea_type": config.get("idea_type", "unknown"),
                        "model_type": config.get("model_type", "unknown"),
                        "pop_size": config.get("pop_size", 0),
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
    ) -> Optional[Any]:
        try:
            state = await db.get_evolution(user_id, evolution_id)
            if not state:
                print(f"❌ Evolution not found: {evolution_id}")
                return None

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
                engine_cls, state, api_key=api_key, user_id=user_id
            )
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
        engine_cls: Type[Any], file_path: Path, api_key: Optional[str] = None
    ) -> Optional[Any]:
        try:
            with open(file_path) as f:
                state = json.load(f)
            return EvolutionRepository.restore_from_state(engine_cls, state, api_key=api_key)
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
    ) -> Any:
        config = state.get("config", {})
        tournament_rounds = config.get("tournament_rounds", 1)
        full_tournament_rounds = config.get("full_tournament_rounds")
        tournament_count = config.get("tournament_count")

        engine = engine_cls(
            idea_type=config.get("idea_type"),
            pop_size=config.get("pop_size", 5),
            generations=config.get("generations", 3),
            model_type=config.get("model_type", "gemini-2.0-flash"),
            creative_temp=config.get("creative_temp", DEFAULT_CREATIVE_TEMP),
            top_p=config.get("top_p", DEFAULT_TOP_P),
            tournament_rounds=tournament_rounds,
            tournament_count=tournament_count,
            full_tournament_rounds=full_tournament_rounds,
            thinking_budget=config.get("thinking_budget"),
            max_budget=config.get("max_budget"),
            mutation_rate=config.get("mutation_rate", 0.2),
            replacement_rate=config.get("replacement_rate", 0.5),
            fitness_alpha=config.get("fitness_alpha", 0.7),
            age_decay_rate=config.get("age_decay_rate", 0.25),
            age_decay_floor=config.get("age_decay_floor", 0.35),
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
        engine_cls: Type[Any], checkpoint_id: str, api_key: Optional[str] = None
    ) -> Optional[Any]:
        evolution_path = EVOLUTIONS_DIR / f"{checkpoint_id}.json"
        if evolution_path.exists():
            return EvolutionRepository.load_from_file(engine_cls, evolution_path, api_key=api_key)

        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{checkpoint_id}.json"
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return None

        return EvolutionRepository.load_from_file(engine_cls, checkpoint_path, api_key=api_key)

    @staticmethod
    def delete_checkpoint(checkpoint_id: str) -> bool:
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
