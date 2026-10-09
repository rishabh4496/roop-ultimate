"""Per-key futures for the state-INDEPENDENT half of a face, shared between neighbouring stabilization blocks.

WHY. A parallel-stabilization block primes its filters by running the whole pipeline on the WU frames before its
first output frame and throwing the picture away. Those WU frames are the LAST WU frames of the previous block, which
processes them for real. Every frame in that overlap is therefore swapped, restored and masked twice (19.4% of all
frames at the shipped geometry, 144 of 744 on a 600-frame clip). docs/perf/stab_warmup.md maps which parts of that
work depend on filter state: none of the swap, the restorer or the mask net does (the kps/landmark smoothing is done
once, sequentially, by the tracking pre-pass when ``temporal_detection`` is on); only the mask / enhancer / HF filters
carry state, and they sit AFTER the seam this cache stores.

WHAT. Whichever block reaches an overlap frame first computes the raw stage and publishes ``(fake_frame,
enhanced_frame, scale_factor, swap_model_mask, img_mask)``; the other block takes it instead of running the networks,
and replays only its own filters on top. A block that arrives while the first is still computing WAITS for it
(a per-key future) rather than duplicating the work. Entries are removed on the second visit, so nothing is held
longer than one block's duration, and only overlap frames are cached at all: every other frame is visited once.

CORRECTNESS. The key includes the alignment matrix bytes, so a hit is only ever served for the same crop the consumer
would have computed. If a live per-block kps filter were active (``temporal_detection`` off) M would differ between
blocks and no key would match; ProcessMgr additionally refuses to build the cache in that mode. A producer that fails
publishes nothing and its waiters compute for themselves. The cache never changes what is computed, only whether it is
computed twice, so output must be bit-identical to a run without it (tests/ab_stab_dedup.py).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

import roop.globals

_WAIT_SLICE_S = 0.25
_WAIT_CAP_S = 600.0
# Hard ceiling on what the cache may hold. Measured peak is ~90 MB on 720p (10 workers, 20 blocks a chunk) and scales
# with frame area, so 384 MB covers 1080p with room to spare. Over the cap a producer simply does not publish and the
# second visitor computes for itself - exactly today's behaviour - so the cap can never change a pixel, only the saving.
_DEFAULT_CAP_MB = 384.0


def _cap_bytes() -> int:
    try:
        return int(float(os.environ.get('ROOP_STAB_DEDUP_MB', '') or _DEFAULT_CAP_MB) * 1048576)
    except ValueError:
        return int(_DEFAULT_CAP_MB * 1048576)


class _Slot:
    __slots__ = ('event', 'entry', 'ok', 'nbytes', 't0')

    def __init__(self) -> None:
        self.event = threading.Event()
        self.entry: Optional[Dict[str, Any]] = None
        self.ok = False
        self.nbytes = 0
        self.t0 = time.perf_counter()


def _entry_bytes(entry: Dict[str, Any]) -> int:
    n = 0
    for v in entry.values():
        if isinstance(v, np.ndarray):
            n += int(v.nbytes)
    return n


def _own(a: Any) -> Any:
    """A copy the cache owns, so nothing a producer mutates later can reach a consumer."""
    return a.copy() if isinstance(a, np.ndarray) else a


class StabRawCache:
    """Thread-safe table of raw per-face results for the frames two blocks both process."""

    def __init__(self, block: int, warmup: int, n_frames: int, max_bytes: Optional[int] = None) -> None:
        self.max_bytes = _cap_bytes() if max_bytes is None else int(max_bytes)
        self.block = int(block)
        self.warmup = int(warmup)
        self.n_frames = int(n_frames)
        self._lock = threading.Lock()
        self._table: Dict[Tuple, _Slot] = {}
        self._owned: Dict[int, _Slot] = {}      # thread ident -> its in-flight, unpublished slot
        self._live_bytes = 0
        self.peak_bytes = 0
        self.stats = {'produced': 0, 'hit': 0, 'hit_waited': 0, 'wait_s': 0.0,
                      'bypass': 0, 'failed': 0, 'dropped': 0, 'capped': 0}

    # ── which frames are visited twice ───────────────────────────────────────────────────────────────
    def in_zone(self, gi: int) -> bool:
        """True for the last ``warmup`` frames of a block that a LATER block warms up from.

        Blocks tile the window on a fixed grid of ``block`` frames from frame 0 (the chunk is a whole number of
        blocks), and a block's warm-up is the ``warmup`` frames before its start, so those frames are exactly
        ``gi % block >= block - warmup``. Needs ``warmup <= block`` (otherwise a frame can be visited three times).
        """
        if self.warmup <= 0 or self.warmup > self.block or gi is None or gi < 0 or gi >= self.n_frames:
            return False
        r = int(gi) % self.block
        if r < self.block - self.warmup:
            return False
        return (int(gi) - r + self.block) < self.n_frames      # a following block exists to be the second visitor

    # ── protocol ─────────────────────────────────────────────────────────────────────────────────────
    def acquire(self, key: Tuple) -> Tuple[str, Any]:
        """('hit', entry) | ('produce', slot) | ('bypass', None).

        'produce': the caller must compute the raw stage, then ``publish(slot, entry)``. If it cannot, it just
        returns; ``abandon()`` (called after every frame) releases the slot and wakes anyone waiting.
        """
        ident = threading.get_ident()
        with self._lock:
            slot = self._table.get(key)
            if slot is None:
                slot = _Slot()
                self._table[key] = slot
                self._owned[ident] = slot
                return 'produce', slot
        # Someone else owns it: take it if ready, otherwise wait for the producer.
        waited = 0.0
        if not slot.event.is_set():
            t0 = time.perf_counter()
            while not slot.event.wait(_WAIT_SLICE_S):
                waited = time.perf_counter() - t0
                if not roop.globals.processing or waited > _WAIT_CAP_S:
                    self._count('bypass')
                    return 'bypass', None
            waited = time.perf_counter() - t0
        with self._lock:
            if not slot.ok or slot.entry is None or self._table.get(key) is not slot:
                self.stats['bypass'] += 1
                return 'bypass', None
            entry = slot.entry
            slot.entry = None
            del self._table[key]
            self._live_bytes -= slot.nbytes
            self.stats['hit'] += 1
            if waited > 0.0:
                self.stats['hit_waited'] += 1
                self.stats['wait_s'] += waited
        return 'hit', entry

    def publish(self, slot: _Slot, entry: Dict[str, Any]) -> None:
        nbytes = _entry_bytes(entry)
        with self._lock:
            over = self._live_bytes + nbytes > self.max_bytes
            self._owned.pop(threading.get_ident(), None)
            if over:
                # Over the ceiling: do not hold it. The slot is released as a failure, so the second visitor (waiting
                # or arriving later) computes for itself, exactly as it would without the cache.
                self.stats['capped'] += 1
                for k, v in list(self._table.items()):
                    if v is slot:
                        del self._table[k]
                        break
        if over:
            slot.event.set()
            return
        owned = {k: _own(v) for k, v in entry.items()}
        with self._lock:
            slot.entry = owned
            slot.nbytes = nbytes
            slot.ok = True
            self._live_bytes += nbytes
            self.peak_bytes = max(self.peak_bytes, self._live_bytes)
            self.stats['produced'] += 1
        slot.event.set()

    def abandon(self) -> None:
        """Release this thread's slot if its frame ended without publishing (exception, refusal, early out)."""
        with self._lock:
            slot = self._owned.pop(threading.get_ident(), None)
            if slot is None or slot.event.is_set():
                return
            for k, v in list(self._table.items()):
                if v is slot:
                    del self._table[k]
                    break
            self.stats['failed'] += 1
        slot.event.set()                    # ok stays False: waiters fall back to computing for themselves

    # ── housekeeping ─────────────────────────────────────────────────────────────────────────────────
    def drop_before(self, gi: int) -> None:
        """Free entries whose second visit can no longer happen (frames older than ``gi``)."""
        with self._lock:
            for k in [k for k in self._table if k[0] < gi]:
                slot = self._table.pop(k)
                if slot.ok:
                    self._live_bytes -= slot.nbytes
                    self.stats['dropped'] += 1
                slot.entry = None
                slot.event.set()

    def clear(self) -> None:
        with self._lock:
            for slot in self._table.values():
                if slot.ok:
                    self.stats['dropped'] += 1
                slot.entry = None
                slot.event.set()
            self._table.clear()
            self._owned.clear()
            self._live_bytes = 0

    def _count(self, name: str) -> None:
        with self._lock:
            self.stats[name] += 1

    def summary_line(self) -> str:
        s = self.stats
        return ('[StabDedup] produced %d, served %d (%d after waiting %.1fs total), bypassed %d, producer-failed %d, '
                'dropped unconsumed %d, over the %.0f MB cap %d; peak held %.0f MB (block %d, warm-up %d)'
                % (s['produced'], s['hit'], s['hit_waited'], s['wait_s'], s['bypass'], s['failed'], s['dropped'],
                   self.max_bytes / 1048576.0, s['capped'], self.peak_bytes / 1048576.0, self.block, self.warmup))
