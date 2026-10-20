from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict


def terminal_status_fields(status: str, *, budget_reached: bool = False) -> Dict[str, Any]:
    """Use the same terminal explanation in saved checkpoints and live progress."""
    if status == "paused":
        message = (
            "Paused because the cumulative budget limit was reached. Progress saved; choose Use config to start a new experiment with a larger budget."
            if budget_reached else "Evolution paused. Progress saved; you can resume."
        )
        return {
            "status_message": message,
            "stop_message": message,
            "stop_reason": "budget" if budget_reached else "requested",
        }
    if status == "complete":
        return {"status_message": "Evolution complete. All results saved.",
                "stop_message": None, "stop_reason": None}
    if status == "error":
        return {"status_message": "Evolution failed. Saved progress is available to resume.",
                "stop_message": None, "stop_reason": "error"}
    return {}


class ProgressEmitter:
    """Centralizes construction of evolution progress payloads."""

    def __init__(self, engine: Any):
        self.engine = engine

    def base(
        self,
        *,
        current_generation: int,
        is_running: bool,
        include_core_state: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "current_generation": current_generation,
            "total_generations": self.engine.generations,
            "is_running": is_running,
            "diversity_history": self.engine.diversity_history.copy() if self.engine.diversity_history else [],
            "island_diversity_history": (
                self.engine.island_diversity_history.copy()
                if getattr(self.engine, "island_diversity_history", None)
                else []
            ),
            "migration_history": (
                self.engine.migration_history.copy()
                if getattr(self.engine, "migration_history", None)
                else []
            ),
        }

        if include_core_state:
            payload.update(
                {
                    "history": self.engine.history,
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "breeding_prompts": self.engine.breeding_prompts,
                    "gene_catalogue": getattr(self.engine, "gene_catalogue", None),
                    "generation_draft": getattr(self.engine, "generation_draft", None),
                }
            )

        return payload

    def with_tokens(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload["token_counts"] = self.engine.get_total_token_count()
        return payload

    def _backfill_tokens(self, payload: Dict[str, Any]) -> None:
        if "token_counts" in payload:
            return
        get_total_token_count = getattr(self.engine, "get_total_token_count", None)
        if callable(get_total_token_count):
            payload["token_counts"] = get_total_token_count()

    async def emit(
        self,
        progress_callback: Callable[[Dict[str, Any]], Awaitable[None]],
        payload: Dict[str, Any],
    ) -> None:
        self._backfill_tokens(payload)
        if "diversity_history" not in payload:
            payload["diversity_history"] = (
                self.engine.diversity_history.copy() if self.engine.diversity_history else []
            )
        if "island_diversity_history" not in payload:
            payload["island_diversity_history"] = (
                self.engine.island_diversity_history.copy()
                if getattr(self.engine, "island_diversity_history", None)
                else []
            )
        if "migration_history" not in payload:
            payload["migration_history"] = (
                self.engine.migration_history.copy()
                if getattr(self.engine, "migration_history", None)
                else []
            )
        await progress_callback(payload)
