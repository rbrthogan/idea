# User State Management
# Provides per-user evolution state to allow concurrent multi-user operations

from dataclasses import dataclass, field
from asyncio import Queue
from typing import Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime, timedelta
import asyncio

if TYPE_CHECKING:
    from idea.evolution import EvolutionEngine


def _default_autorating_status() -> Dict[str, Any]:
    """Default autorating progress payload used by /api/auto-rate/progress."""
    return {
        "is_running": False,
        "status": "idle",
        "status_message": "",
        "progress": 0,
        "requested_rounds": 0,
        "completed_rounds": 0,
        "total_matches": 0,
        "completed_matches": 0,
        "completed_comparisons": 0,
        "new_comparisons": 0,
        "win_counts": {"A": 0, "B": 0, "tie": 0},
        "ideas": [],
        "token_counts": None,
        "error": None,
        "version": 0,
        "updated_at": 0,
    }


@dataclass
class UserEvolutionState:
    """Encapsulates all evolution state for a single user."""
    engine: Optional['EvolutionEngine'] = None
    queue: Queue = field(default_factory=Queue)
    status: Dict[str, Any] = field(default_factory=lambda: {
        "current_generation": 0,
        "total_generations": 0,
        "is_running": False,
        "history": []
    })
    latest_data: list = field(default_factory=list)
    history_version: int = 0
    last_sent_history_version: int = -1
    last_activity: datetime = field(default_factory=datetime.utcnow)
    run_owner_id: Optional[str] = None
    run_token: Optional[str] = None
    run_task: Optional[asyncio.Task] = None
    run_started: bool = False
    cancel_requested: bool = False
    run_last_write: float = 0.0
    run_last_heartbeat: float = 0.0
    heartbeat_task: Optional[asyncio.Task] = None
    autorating_status: Dict[str, Any] = field(default_factory=_default_autorating_status)
    autorating_task: Optional[asyncio.Task] = None

    def reset_queue(self):
        """Clear the queue to avoid stale updates."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break

    def reset_status(self):
        """Reset status to initial state for a new evolution."""
        self.status = {
            "current_generation": 0,
            "total_generations": 0,
            "is_running": False,
            "history": []
        }
        self.history_version = 0
        self.last_sent_history_version = -1
        self.reset_queue()

    def reset_autorating_status(self):
        """Reset autorating progress payload."""
        self.autorating_status = _default_autorating_status()

    def reset_run_tracking(self):
        """Reset run coordination metadata."""
        self.run_owner_id = None
        self.run_token = None
        self.run_started = False
        self.cancel_requested = False
        self.run_last_write = 0.0
        self.run_last_heartbeat = 0.0
        self.stop_heartbeat()

    def start_heartbeat(self, task: asyncio.Task):
        """Track a heartbeat task for this user's run."""
        self.stop_heartbeat()
        self.heartbeat_task = task

    def stop_heartbeat(self):
        """Stop any active heartbeat task."""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            self.heartbeat_task = None


class UserStateManager:
    """Manages evolution states for all users with automatic cleanup."""

    def __init__(self, idle_timeout_hours: int = 24):
        self._states: Dict[str, UserEvolutionState] = {}
        self._lock = asyncio.Lock()
        self._idle_timeout_hours = idle_timeout_hours

    async def get(self, user_id: str) -> UserEvolutionState:
        """Get or create state for a user."""
        async with self._lock:
            if user_id not in self._states:
                self._states[user_id] = UserEvolutionState()
            state = self._states[user_id]
            state.last_activity = datetime.utcnow()
            return state

    async def cleanup_idle(self) -> int:
        """
        Remove states that have been idle too long.
        Returns count of states removed.
        Only removes states where no evolution is running.
        """
        async with self._lock:
            cutoff = datetime.utcnow() - timedelta(hours=self._idle_timeout_hours)
            to_remove = []
            for user_id, state in self._states.items():
                if (state.last_activity < cutoff and not state.status.get("is_running")
                        and not state.autorating_status.get("is_running")
                        and (state.run_task is None or state.run_task.done())):
                    to_remove.append(user_id)

            for user_id in to_remove:
                close_engine_clients(self._states[user_id].engine)
                del self._states[user_id]

            return len(to_remove)

    async def get_active_count(self) -> int:
        """Get count of users with currently running evolutions."""
        async with self._lock:
            return sum(1 for s in self._states.values() if s.status.get("is_running"))

    async def get_total_count(self) -> int:
        """Get total count of user states in memory."""
        async with self._lock:
            return len(self._states)

    async def close(self) -> None:
        tasks = [state.autorating_task for state in self._states.values()
                 if state.autorating_task is not None and not state.autorating_task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            for state in self._states.values():
                state.stop_heartbeat()
                close_engine_clients(state.engine)
            self._states.clear()


def close_engine_clients(engine) -> None:
    """Close distinct provider transports held by a completed/discarded engine."""
    if engine is None:
        return
    components = [getattr(engine, name, None) for name in (
        "ideator", "formatter", "critic", "breeder", "gene_manager",
        "genotype_encoder", "oracle", "diversity_calculator"
    )]
    components.extend(runner for _, runner in getattr(engine, "pre_judge_evaluators", []))
    seen = set()
    while components:
        component = components.pop()
        if component is None or id(component) in seen:
            continue
        seen.add(id(component))
        components.extend(getattr(component, name, None) for name in (
            "_source_llm", "repair_hook", "gene_encoder"
        ))
        for name in ("client", "_client"):
            client = getattr(component, name, None)
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


# Global singleton instance
user_states = UserStateManager()
