"""Bounded subprocess capture shared by the conformance evidence gates."""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import threading
from collections.abc import Callable
from typing import BinaryIO

MAX_CAPTURE_BYTES = 64 * 1024


def run_bounded(
    argv: list[str],
    timeout: float,
    *,
    output_path: pathlib.Path | None = None,
    max_file_bytes: int = 513 * 1024 * 1024,
    preexec_fn: Callable[[], None] | None = None,
) -> tuple[int, str, str, set[str], bool]:
    """Run a process while bounding both diagnostics and an optional output."""
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=preexec_fn,  # noqa: PLW1509 - single-threaded harness; this installs rlimits
        # Candidate code may fork while retaining these pipes or the output
        # path. A private session gives every descendant a group we can tear
        # down without signalling the conformance host itself.
        start_new_session=os.name == "posix",
    )
    buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    overflow: set[str] = set()
    lock = threading.Lock()

    def terminate_group() -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - CI/release conformance hosts are POSIX
                process.kill()
        except ProcessLookupError:
            pass

    def drain(name: str, stream: BinaryIO) -> None:
        while chunk := stream.read(8192):
            with lock:
                remaining = MAX_CAPTURE_BYTES - len(buffers[name])
                buffers[name].extend(chunk[: max(remaining, 0)])
                if len(chunk) > remaining:
                    overflow.add(name)
                    terminate_group()
                    return

    assert process.stdout is not None and process.stderr is not None
    threads = [
        threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()

    output_watcher: threading.Thread | None = None
    if output_path is not None:

        def watch_output() -> None:
            while process.poll() is None:
                try:
                    if output_path.stat().st_size > max_file_bytes:
                        with lock:
                            overflow.add("output")
                        terminate_group()
                        return
                except FileNotFoundError:
                    pass
                threading.Event().wait(0.01)

        output_watcher = threading.Thread(target=watch_output, daemon=True)
        output_watcher.start()

    timed_out = False
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_group()
        returncode = process.wait()
    finally:
        # Always close the candidate's lifetime before inspecting its output.
        # This also closes inherited pipes after a direct parent exits zero.
        terminate_group()
    for thread in threads:
        thread.join(timeout=5)
    if output_watcher is not None:
        output_watcher.join(timeout=1)
    if (
        output_path is not None
        and output_path.is_file()
        and output_path.stat().st_size >= max_file_bytes
        and returncode != 0
    ):
        overflow.add("output")
    return (
        returncode,
        buffers["stdout"].decode("utf-8", errors="replace"),
        buffers["stderr"].decode("utf-8", errors="replace"),
        overflow,
        timed_out,
    )
