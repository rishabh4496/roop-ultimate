"""Dynamic batching for the swap inference (Phase 2 cross-frame batching).

The video pipeline runs N worker threads, each doing detect→swap→enhance→paste
on its own frame. On a batch-1 swap model that means N tiny inference calls that
underutilize the GPU. This batcher coalesces the concurrent single-crop swap
calls from those worker threads into one batched inference (RunBatchMulti),
without touching the per-frame logic: a worker just submits its crop and waits
for the result.

A batch fires when it reaches `max_batch`, or `max_wait_ms` after the first
request arrives (so a lone straggler never stalls). One dedicated batcher thread
runs the inference, so the swap context is only ever touched by one thread.

Opt-in via ROOP_BATCH_SWAP_XFRAME=1 (requires a batch-dynamic swap session,
i.e. ROOP_BATCH_SWAP=1). Off by default — zero effect on normal runs.

Robustness: timeout always flushes partial batches; stop() drains and unblocks
every waiter; an exception in the batched run is delivered to that batch's
waiters (they fall back to nothing/raise) instead of hanging forever.
"""

import os
import time
import threading


def xframe_enabled() -> bool:
    return os.environ.get('ROOP_BATCH_SWAP_XFRAME', '0') == '1'


class _Request:
    __slots__ = ('src', 'tgt', 'blob', 'out', 'err', 'ev')

    def __init__(self, src, tgt, blob):
        self.src = src
        self.tgt = tgt
        self.blob = blob
        self.out = None
        self.err = None
        self.ev = threading.Event()


class SwapBatcher:
    def __init__(self, run_fn, guard_fn, max_batch, max_wait_ms=6.0):
        """run_fn(list[(src,tgt,blob)]) -> list[out]; guard_fn() -> context manager
        wrapping the GPU call (pool lease / global lock)."""
        self._run_fn = run_fn
        self._guard_fn = guard_fn
        self._max_batch = max(1, int(max_batch))
        self._max_wait = max(0.0, float(max_wait_ms) / 1000.0)
        self._cond = threading.Condition(threading.Lock())
        self._queue = []
        self._stopped = False
        # ── per-run stats (batch sizes + inference time) ────────────────────
        self._stats_lock = threading.Lock()
        self._n_batches = 0
        self._n_items = 0
        self._run_time = 0.0
        self._hist = {}
        self._thread = threading.Thread(target=self._loop, name='swap_batcher', daemon=True)
        self._thread.start()

    def _record(self, size, dt):
        with self._stats_lock:
            self._n_batches += 1
            self._n_items += size
            self._run_time += dt
            self._hist[size] = self._hist.get(size, 0) + 1

    def report(self):
        """Print a one-shot summary of how well swaps coalesced. avg-batch near 1
        means threads rarely overlap at the swap stage (batching isn't buying
        much); a high avg-batch with low ms/item means it's saturating well."""
        with self._stats_lock:
            nb, ni, rt, hist = self._n_batches, self._n_items, self._run_time, dict(self._hist)
        if nb == 0:
            return
        print("\n==== SWAP BATCHER (ROOP_BATCH_SWAP_XFRAME) ====", flush=True)
        print(f"  batches: {nb}   items: {ni}   avg batch: {ni / nb:.2f}   max: {max(hist)}", flush=True)
        print(f"  batch-size histogram: {dict(sorted(hist.items()))}", flush=True)
        print(f"  swap inference: {rt:.2f}s  "
              f"({1000 * rt / nb:.2f} ms/batch, {1000 * rt / max(ni, 1):.2f} ms/item)", flush=True)
        print("===============================================\n", flush=True)

    # ── producer side (worker threads) ──────────────────────────────────────
    def submit(self, src, tgt, blob):
        """Enqueue a crop; returns a handle. Non-blocking. If the batcher is
        already stopped, runs inline so callers never block forever."""
        req = _Request(src, tgt, blob)
        with self._cond:
            if self._stopped:
                self._run_inline(req)
                return req
            self._queue.append(req)
            # Always wake the batcher: it may be idle-waiting on an empty queue,
            # and a lone request (below max_batch) must still get processed.
            self._cond.notify()
        return req

    def wait(self, req):
        req.ev.wait()
        if req.err is not None:
            raise req.err
        return req.out

    # ── batcher thread ──────────────────────────────────────────────────────
    def _loop(self):
        while True:
            with self._cond:
                while not self._queue and not self._stopped:
                    self._cond.wait()
                if self._stopped and not self._queue:
                    return
                # Have ≥1 request; briefly wait for more to coalesce a batch.
                if len(self._queue) < self._max_batch and not self._stopped and self._max_wait > 0:
                    self._cond.wait(self._max_wait)
                batch = self._queue[:self._max_batch]
                del self._queue[:self._max_batch]
            self._run_batch(batch)

    def _run_batch(self, batch):
        if not batch:
            return
        try:
            with self._guard_fn():
                t0 = time.perf_counter()
                outs = self._run_fn([(r.src, r.tgt, r.blob) for r in batch])
                # A short result is a failed batch contract.  Treat it like
                # any other batch failure so every waiter is released; using
                # zip() directly would strand the unmatched workers forever.
                outs = list(outs)
                if len(outs) != len(batch):
                    raise RuntimeError(
                        'swap batch returned %d outputs for %d requests' %
                        (len(outs), len(batch)))
                self._record(len(batch), time.perf_counter() - t0)
            for r, o in zip(batch, outs):
                r.out = o
                r.ev.set()
        except Exception as e:  # deliver the error to every waiter in this batch
            for r in batch:
                r.err = e
                r.ev.set()

    def _run_inline(self, req):
        try:
            with self._guard_fn():
                t0 = time.perf_counter()
                req.out = self._run_fn([(req.src, req.tgt, req.blob)])[0]
                self._record(1, time.perf_counter() - t0)
        except Exception as e:
            req.err = e
        req.ev.set()

    # ── teardown ────────────────────────────────────────────────────────────
    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout=10.0)
        # Drain anything the thread didn't get to, so no waiter hangs.
        with self._cond:
            leftover, self._queue = self._queue, []
        for r in leftover:
            self._run_inline(r)
