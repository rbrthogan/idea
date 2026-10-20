"""Supervise local evolution jobs at an explicit worker boundary.

This service owns in-process tasks. It is intentionally not a durable queue;
external workers can consume the persisted claim using the same run token.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from idea.evolution_progress import terminal_status_fields

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunJob:
    user_id: str
    evolution_id: str
    run_token: str
    mode: str = "new"
    additional_generations: int = 0


class RunService:
    def __init__(self):
        self.tasks: set[asyncio.Task] = set()
        self._states = {}

    def start(self, state, operation: Callable[[], Awaitable[None]], finalize):
        """Keep a strong reference and acknowledge completion only after cleanup."""
        if state.run_task is not None and not state.run_task.done():
            raise RuntimeError("An evolution worker is still active for this user")
        task = asyncio.create_task(self._supervise(state, operation, finalize), name=f"evolution-{state.run_token}")
        state.run_task = task
        self.tasks.add(task)
        self._states[task] = state
        def completed(done):
            self.tasks.discard(done)
            self._states.pop(done, None)
        task.add_done_callback(completed)
        return task

    async def _supervise(self, state, operation, finalize):
        engine = state.engine
        token = state.run_token
        state.run_started = True
        try:
            if state.cancel_requested:
                raise asyncio.CancelledError
            await operation()
        except asyncio.CancelledError:
            engine.stop_requested = True
            engine.is_stopped = True
            state.status.update({"is_running": False, "is_stopped": True,
                                 "is_resumable": False, "status_message": "Evolution stopped",
                                 "stop_reason": "requested"})
            try:
                if not state.status.get("ownership_lost"):
                    await engine.save_checkpoint(status="paused")
                    state.status["checkpoint_id"] = engine.checkpoint_id
                    check_budget = getattr(engine, "check_budget", None)
                    state.status.update(terminal_status_fields(
                        "paused", budget_reached=callable(check_budget) and check_budget()))
                    state.status.update({"is_resumable": bool(engine.checkpoint_id),
                                         "durably_saved": True})
            except Exception as exc:
                state.status.update({"error": str(exc), "save_failed": True,
                                     "durably_saved": False,
                                     "status_message": "Stopped, but checkpoint could not be saved"})
                logger.exception("Cancellation checkpoint failed")
        except Exception as exc:
            state.status.update({"is_running": False, "error": str(exc),
                                 "is_resumable": False, "status_message": "Evolution failed",
                                 "stop_reason": "error", "stop_message": None})
            logger.exception("Evolution worker failed")
            save = getattr(engine, "save_checkpoint", None)
            if callable(save) and not state.status.get("ownership_lost"):
                try:
                    await save(status="error")
                    checkpoint_id = getattr(engine, "checkpoint_id", None)
                    state.status.update(terminal_status_fields("error"))
                    state.status.update({"checkpoint_id": checkpoint_id,
                                         "is_resumable": bool(checkpoint_id), "durably_saved": True})
                except Exception as save_error:
                    state.status.update({"save_error": str(save_error), "save_failed": True,
                                         "durably_saved": False})
                    logger.exception("Failed worker checkpoint could not be saved")
        finally:
            # A callback from an obsolete job cannot finalize its replacement.
            if state.engine is engine and state.run_token == token:
                state.status["is_running"] = False
                state.reset_queue()
                await state.queue.put(state.status.copy())
                try:
                    await finalize(engine.user_id, state, state.status.copy())
                except Exception:
                    logger.exception("Evolution finalization failed")
                state.reset_run_tracking()
                state.run_task = None

    def request_cancel(self, state) -> bool:
        if state.engine is not None:
            stop = getattr(state.engine, "stop_evolution", None)
            if callable(stop):
                stop()
            else:
                state.engine.stop_requested = True
        if state.run_task is not None and not state.run_task.done():
            if state.run_task is asyncio.current_task():
                return False
            if not state.cancel_requested:
                state.cancel_requested = True
                if state.run_started:
                    state.run_task.cancel()
            return True
        return False

    async def close(self):
        tasks = tuple(self.tasks)
        for task in tasks:
            state = self._states.get(task)
            if state is not None:
                # Signal provider retry guards before cancelling their waiter.
                # Pending supervisors observe this flag before starting work.
                self.request_cancel(state)
            else:
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


run_service = RunService()
