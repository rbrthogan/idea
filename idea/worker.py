"""Cloud Run Job entrypoint; claims are adopted once and credentials stay in storage."""

import argparse
import asyncio
import logging
import uuid
import os
import signal

from idea import database as db
from idea.config import RUN_LEASE_SECONDS, RUN_HEARTBEAT_SECONDS, GLOBAL_MAX_ACTIVE_RUNS, get_model_provider
from idea.evolution import EvolutionEngine
from idea.job_dispatch import parse_job_payload
from idea.run_service import RunJob, RunService
from idea.user_state import UserEvolutionState, close_engine_clients
from idea.blocking_io import run_blocking


logger = logging.getLogger(__name__)


async def execute_job(job: RunJob, *, on_state=None) -> bool:
    """Return false for a duplicate, cancelled, stale, or replaced delivery."""
    from idea import viewer

    owner = f"worker-{uuid.uuid4().hex}"
    if not await db.adopt_active_run(job.user_id, job.evolution_id, job.run_token, owner,
                                     RUN_LEASE_SECONDS, transfer_global_lock=GLOBAL_MAX_ACTIVE_RUNS == 1):
        logger.info("Ignoring job without an adoptable current claim")
        return False

    async def bootstrap_heartbeat():
        while True:
            await asyncio.sleep(RUN_HEARTBEAT_SECONDS)
            if not await db.update_active_run(job.user_id, {}, RUN_LEASE_SECONDS, owner,
                                               run_token=job.run_token, require_live_lease=True):
                raise db.RunOwnershipLost("Worker claim lost during initialization")
            if GLOBAL_MAX_ACTIVE_RUNS == 1:
                if not await db.refresh_global_run_lock(owner, RUN_LEASE_SECONDS, run_token=job.run_token):
                    raise db.RunOwnershipLost("Global worker claim lost during initialization")

    bootstrap = asyncio.create_task(bootstrap_heartbeat())
    state = UserEvolutionState(run_owner_id=owner, run_token=job.run_token)
    service = RunService()
    if on_state is not None:
        on_state(state, service)
    initialization = None
    try:
        async def initialize():
            snapshot = await db.get_evolution(job.user_id, job.evolution_id)
            if not snapshot:
                raise RuntimeError("Queued evolution snapshot is unavailable")
            provider = get_model_provider((snapshot.get("config") or {}).get("model_type"))
            api_key = await db.get_user_api_key(job.user_id, provider)
            if not api_key:
                raise RuntimeError("Provider API key is unavailable for queued evolution")
            _, has_artifacts = await viewer._stored_evolution_artifact_context(job.user_id, job.evolution_id)
            runners, store = (await run_blocking(viewer._web_artifact_runtime,
                             user_id=job.user_id, evolution_id=job.evolution_id, api_key=api_key,
                             execution_mode="remote", model_provider=provider)
                             if has_artifacts else ({}, None))
            engine = await EvolutionEngine.load_evolution_for_user(job.user_id, job.evolution_id,
                                                                   api_key=api_key, evaluation_runners=runners,
                                                                   artifact_store=store)
            state.engine = engine
            if engine is None or engine.unsupported_evaluation_components:
                raise RuntimeError("Queued evolution could not load its evaluators")
            if getattr(engine, "generations", 0) + job.additional_generations > 500:
                raise ValueError("Continuation would exceed 500 total generations")
            return engine

        initialization = asyncio.create_task(initialize())
        await asyncio.wait({initialization, bootstrap}, return_when=asyncio.FIRST_COMPLETED)
        if bootstrap.done():
            bootstrap.result()
        engine = initialization.result()
        engine._run_token = job.run_token
        bootstrap.cancel()
        await asyncio.gather(bootstrap, return_exceptions=True)
        state.status.update({"is_running": True, "evolution_id": job.evolution_id})
        state.run_last_heartbeat = viewer.time.monotonic()
        state.start_heartbeat(asyncio.create_task(viewer._heartbeat_loop(job.user_id, state)))
        if job.mode == "new":
            operation = lambda: viewer.run_evolution_task(state)
        elif job.mode == "resume":
            operation = lambda: viewer.run_resume_evolution_task(state)
        else:
            operation = lambda: viewer.run_continue_evolution_task(state, job.additional_generations)
        current = await db.get_active_run(job.user_id)
        if current and current.get("stop_requested"):
            state.cancel_requested = True
        task = service.start(state, operation, viewer._finalize_run_state)
        await task
        if state.status.get("error"):
            raise RuntimeError(state.status["error"])
        return True
    except asyncio.CancelledError:
        await service.close()
        if state.run_token:
            updated = await db.update_active_run(
                job.user_id, {"status": "paused", "is_running": False, "active": False,
                              "is_stopped": True, "is_resumable": True, "cancel_acknowledged": True},
                0, owner, run_token=job.run_token, require_live_lease=True,
            )
            if updated and GLOBAL_MAX_ACTIVE_RUNS == 1:
                await db.release_global_run_lock(owner, run_token=job.run_token)
        raise
    except Exception as exc:
        await service.close()
        updated = await db.update_active_run(
            job.user_id, {"status": "error", "last_error": str(exc), "is_running": False, "active": False},
            0, owner, run_token=job.run_token, require_live_lease=True,
        )
        if updated and GLOBAL_MAX_ACTIVE_RUNS == 1:
            await db.release_global_run_lock(owner, run_token=job.run_token)
        raise
    finally:
        if initialization is not None and not initialization.done():
            initialization.cancel()
            await asyncio.gather(initialization, return_exceptions=True)
        await service.close()
        bootstrap.cancel()
        await asyncio.gather(bootstrap, return_exceptions=True)
        state.stop_heartbeat()
        close_engine_clients(state.engine)


async def _stop_on_termination(task, *, grace_seconds=8.0, hard_exit=None, request_stop=None):
    if request_stop is not None:
        request_stop()
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=grace_seconds)
    except asyncio.CancelledError:
        pass
    except asyncio.TimeoutError:
        logger.error("Worker termination grace expired; recovery uses the last durable checkpoint")
        (hard_exit or os._exit)(143)
    except Exception:
        logger.exception("Worker failed while terminating")


async def _run_with_signals(job):
    loop = asyncio.get_running_loop()
    request_stop = None
    def register_state(state, service):
        nonlocal request_stop
        request_stop = lambda: service.request_cancel(state)
    task = asyncio.create_task(execute_job(job, on_state=register_state))
    termination = None
    def stop():
        nonlocal termination
        if termination is None:
            termination = asyncio.create_task(_stop_on_termination(task, request_stop=request_stop))
    loop.add_signal_handler(signal.SIGTERM, stop)
    try:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if termination is None:
                if request_stop is not None:
                    request_stop()
                task.cancel()
                raise
            await termination
            return False
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, help="Credential-free JSON RunJob payload")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run_with_signals(parse_job_payload(args.payload)))


if __name__ == "__main__":
    main()
