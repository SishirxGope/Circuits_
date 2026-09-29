# [AI-GEN] agent=Claude date=2026-09-29 task=Review fix - parallel cells must not race on one cache file
# reviewed-by: PENDING

"""An exclusive lock around computing one cache file, shared by every process on the machine.

The queue runs cells in parallel (``05_run_queue.sh stageB 12``), and every cell of one
(model, method) wants the same cached E[x^2] or GPTQ/AWQ result. Without a lock they all
compute it, and before this module they all wrote the same ``.partial`` file at once.
With it, one process computes and the others wait, then load what it wrote.

Usage: check for the file, and only if it is missing take the lock and check *again*.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def cache_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``<path>.lock`` (blocking) for the duration."""
    lock_path = Path(path).with_name(Path(path).name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as handle:
        if os.name == "nt":
            import msvcrt

            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:  # LK_LOCK gives up after ~10 s; keep waiting
                    continue
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def partial_path(path: Path, suffix: str) -> Path:
    """A temporary name next to ``path`` that no other process will choose."""
    path = Path(path)
    return path.with_name(f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.partial{suffix}")


__all__ = ["cache_lock", "partial_path"]
