"""Background batch prefetching (memmap reads overlap the GPU)."""

from __future__ import annotations

import queue
import threading

import numpy as np

_END = object()


class Prefetcher:
    """Runs `fn(rng)` on `threads` background threads (each with its own
    generator, seeded by (seed, thread)), keeping `depth` batches ready."""

    def __init__(self, fn, depth: int = 4, threads: int = 1, seed: int = 0):
        self.fn = fn
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.threads = [threading.Thread(target=self._run, args=(np.random.default_rng([seed, i]),), daemon=True)
                        for i in range(threads)]
        for t in self.threads:
            t.start()

    def _run(self, rng):
        while not self.stop.is_set():
            try:
                item = self.fn(rng)
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


def prefetched(items, depth: int = 4):
    """Iterate the iterable `items` on a background thread, `depth` items ahead."""
    q: queue.Queue = queue.Queue(maxsize=depth)

    def run():
        try:
            for item in items:
                q.put(item)
        except Exception as exc:
            q.put(exc)
            return
        q.put(_END)

    threading.Thread(target=run, daemon=True).start()
    while (item := q.get()) is not _END:
        if isinstance(item, Exception):
            raise item
        yield item
