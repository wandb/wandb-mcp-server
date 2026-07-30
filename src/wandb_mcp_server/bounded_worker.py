"""Small, non-blocking background worker queues for optional telemetry sinks."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")
_MAX_DROP_COUNT = 2_147_483_647


class BoundedWorkerQueue(Generic[T]):
    """Run best-effort work without blocking or growing memory without bound.

    Submissions are deliberately non-blocking. The caller remains responsible
    for the canonical local log; this queue is only for optional forwarding.
    """

    def __init__(
        self,
        handler: Callable[[T], object],
        *,
        capacity: int,
        worker_count: int,
        on_drop: Callable[[int], object] | None = None,
        thread_name_prefix: str = "mcp-background",
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        self._handler = handler
        self._queue: queue.Queue[T] = queue.Queue(maxsize=capacity)
        self._on_drop = on_drop
        self._state_lock = threading.Lock()
        self._stopping = False
        self._dropped_count = 0
        self._outstanding_count = 0
        self._threads = [
            threading.Thread(
                target=self._run,
                name=f"{thread_name_prefix}-{index}",
                daemon=True,
            )
            for index in range(worker_count)
        ]
        for thread in self._threads:
            thread.start()

    @property
    def dropped_count(self) -> int:
        with self._state_lock:
            return self._dropped_count

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def outstanding_count(self) -> int:
        with self._state_lock:
            return self._outstanding_count

    @property
    def capacity(self) -> int:
        return self._queue.maxsize

    def submit(self, item: T) -> bool:
        """Queue an item immediately, or report a bounded drop."""
        with self._state_lock:
            if self._stopping:
                return False
            if self._outstanding_count >= self.capacity:
                self._dropped_count = min(_MAX_DROP_COUNT, self._dropped_count + 1)
                dropped_count = self._dropped_count
            else:
                self._outstanding_count += 1
                self._queue.put_nowait(item)
                return True
        # Log at exponentially increasing intervals to avoid an outage
        # turning the drop signal itself into a logging flood.
        if self._on_drop is not None and (dropped_count == 1 or dropped_count & (dropped_count - 1) == 0):
            self._on_drop(dropped_count)
        return False

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Stop accepting work and optionally drain queued work.

        ``cancel_futures`` mirrors ``ThreadPoolExecutor`` enough for existing
        controlled-shutdown callers. Pending events are discarded only when
        explicitly requested.
        """
        with self._state_lock:
            self._stopping = True
        if cancel_futures:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    with self._state_lock:
                        self._outstanding_count -= 1
                    self._queue.task_done()
        if wait:
            self._queue.join()
            for thread in self._threads:
                thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            with self._state_lock:
                should_stop = self._stopping and self._queue.empty()
            if should_stop:
                return
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._handler(item)
            except Exception:
                # Forwarder handlers log their own failures. Keep the worker
                # alive so one unexpected sink error cannot strand the queue.
                pass
            finally:
                with self._state_lock:
                    self._outstanding_count -= 1
                self._queue.task_done()


__all__ = ["BoundedWorkerQueue"]
