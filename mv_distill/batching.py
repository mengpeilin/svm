"""Background batch prefetching (the first pass over the memmaps is disk-bound)."""

from __future__ import annotations

import queue
import threading


class Prefetcher:
    """Runs `fn()` on a background thread, keeping `depth` batches ready."""

    def __init__(self, fn, depth: int = 4):
        self.fn = fn
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                item = self.fn()
            except Exception as exc:  # surface worker failures on the main thread
                self.q.put(exc)
                return
            while not self.stop.is_set():
                try:
                    self.q.put(item, timeout=1.0)
                    break
                except queue.Full:
                    continue

    def next(self):
        item = self.q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.stop.set()
