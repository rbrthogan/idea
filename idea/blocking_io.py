"""Bounded offloading for synchronous SDKs without blocking the web event loop."""

import asyncio
import contextvars
import functools
import threading
import time
from concurrent.futures import ThreadPoolExecutor


_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="idea-io")
_slots = threading.BoundedSemaphore(8)
_in_worker = contextvars.ContextVar("idea_in_io_worker", default=False)
DEFAULT_IO_TIMEOUT = 60.0


async def run_blocking(function, *args, timeout=DEFAULT_IO_TIMEOUT, **kwargs):
    """Bound both admission and waiting; a timed-out worker still occupies its slot.

    Python cannot kill an SDK thread. Keeping its slot until actual completion
    prevents repeated timeouts from spawning an unbounded backlog of workers.
    SDK calls must also use transport deadlines.
    """
    deadline = time.monotonic() + timeout
    while not _slots.acquire(blocking=False):
        if time.monotonic() >= deadline:
            raise TimeoutError("I/O capacity unavailable before deadline")
        await asyncio.sleep(0.01)
    try:
        context = contextvars.copy_context()
        future = _executor.submit(context.run, functools.partial(function, *args, **kwargs))
    except BaseException:
        _slots.release()
        raise
    future.add_done_callback(lambda _future: _slots.release())
    remaining = max(0.001, deadline - time.monotonic())
    wrapped = asyncio.wrap_future(future)
    def observe_completion(done):
        if not done.cancelled():
            done.exception()
    wrapped.add_done_callback(observe_completion)
    # Shield preserves the worker's completion/slot accounting on timeout or cancellation.
    return await asyncio.wait_for(asyncio.shield(wrapped), remaining)


def offload_sync_sdk(function):
    """Retain an async repository API while executing its synchronous SDK in a worker.

    Nested repository calls share the same worker instead of recursively
    submitting to the pool and deadlocking when it is full.
    """
    @functools.wraps(function)
    async def wrapped(*args, **kwargs):
        if _in_worker.get():
            return await function(*args, **kwargs)

        def invoke():
            token = _in_worker.set(True)
            try:
                return asyncio.run(function(*args, **kwargs))
            finally:
                _in_worker.reset(token)

        return await run_blocking(invoke)

    return wrapped
