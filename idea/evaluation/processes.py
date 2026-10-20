"""Bounded subprocess output for generated-code execution boundaries."""

import os
import selectors
import subprocess
import time


class ProcessOutputLimitError(RuntimeError):
    pass


def run_bounded(command, *, timeout, stdout_limit, stderr_limit=64 * 1024):
    """Drain both pipes without ever buffering more than their byte caps.

    Children are always reaped, including timeout, oversized output, and read
    failures. The caller owns cleanup of any external container/process tree.
    """
    buffers = {'stdout': bytearray(), 'stderr': bytearray()}
    limits = {'stdout': stdout_limit, 'stderr': stderr_limit}
    deadline = time.monotonic() + timeout
    with subprocess.Popen(command, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          close_fds=True) as process:
        try:
            with selectors.DefaultSelector() as selector:
                for name in buffers:
                    stream = getattr(process, name)
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, name)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    for key, _ in selector.select(min(remaining, 0.1)):
                        name = key.data
                        # One additional byte establishes overflow without
                        # allocating an arbitrarily large producer chunk.
                        size = min(65536, limits[name] - len(buffers[name]) + 1)
                        chunk = os.read(key.fileobj.fileno(), max(1, size))
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        if len(buffers[name]) + len(chunk) > limits[name]:
                            raise ProcessOutputLimitError(f'{name} exceeded {limits[name]} bytes')
                        buffers[name].extend(chunk)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            process.wait(timeout=remaining)
            return subprocess.CompletedProcess(command, process.returncode,
                                               bytes(buffers['stdout']), bytes(buffers['stderr']))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
