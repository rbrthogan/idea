"""Keep provider threads observable until their bounded request completes."""

import asyncio
from typing import Any, Callable


async def await_provider_call(function: Callable[..., Any], *args, **kwargs) -> Any:
    """Run blocking I/O without releasing a cancelled waiter before the thread.

    Engine semaphores bound submission. Provider transport timeouts bound drain
    time; cancellation guards prevent scheduling another provider retry. This
    uses the loop's default executor, independently of Firestore's worker pool.
    """
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Multiple cancel requests must not abandon accounting or the worker.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()  # Retrieve a provider exception during cancellation.
        raise
