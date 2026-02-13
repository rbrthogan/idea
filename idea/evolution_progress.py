from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict


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
        }

        if include_core_state:
            payload.update(
                {
                    "history": self.engine.history,
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "breeding_prompts": self.engine.breeding_prompts,
                }
            )

        return payload

    def with_tokens(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload["token_counts"] = self.engine.get_total_token_count()
        return payload

    async def emit(
        self,
        progress_callback: Callable[[Dict[str, Any]], Awaitable[None]],
        payload: Dict[str, Any],
    ) -> None:
        if "diversity_history" not in payload:
            payload["diversity_history"] = (
                self.engine.diversity_history.copy() if self.engine.diversity_history else []
            )
        await progress_callback(payload)
