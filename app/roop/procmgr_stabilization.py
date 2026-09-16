"""Stabilization scheduling policy for ProcessMgr.

The methods here answer one cohesive question: how much ordered filter state
and host memory can a parallel stabilization run safely consume? Keeping this
policy separate from the reader, writer, and face operations makes the 4070
and 3060 resource rules auditable without changing their runtime contract.
"""

from __future__ import annotations

import os

import psutil

from roop.degrade import swallowed as _swallowed
from roop.one_euro import _MAX_WARMUP as _MAX_STAB_WARMUP


class StabilizationSchedulingMixin:
    """Own warm-up, RAM budget, and block geometry decisions."""

    def _stab_warmup_frames(self, eps=0.01):
        """How many frames each parallel block must discard before its output is
        trustworthy — asked of the filters themselves rather than hardcoded.

        Every active stabilizer is primed from the wrong seed at a block start,
        so the answer is the SLOWEST of them to forget: whichever needs the most
        frames sets the boundary. `eps` is the residual weight of that seed we
        accept at the boundary (1% by default), which is what "seam-free" means
        here in a way that can actually be checked.

        ROOP_STAB_WARMUP overrides, for A/B-ing a seam against the derivation.
        """
        raw = os.environ.get('ROOP_STAB_WARMUP')
        if raw:
            try:
                return max(0, int(raw))
            except ValueError:
                pass
        need = 0
        # The opt-in temporal engines are stabilizers in every sense that matters
        # here: each keeps a per-track recurrence over frames, so a block that
        # starts mid-clip is primed from the wrong seed exactly as the kps/mask/
        # enhancer filters are. They therefore answer the same question and take
        # part in the same worst-case. The expression engine is deliberately
        # absent -- its `plan()` is read-only and its state is written by the
        # sequential tracking pre-pass, so worker order cannot reach it.
        # The landmark smoother and the HF carry are here for the same reason
        # even though they do NOT force ordered execution: a block that starts
        # mid-clip primes them from the wrong seed exactly as the kps/mask/
        # enhancer filters are primed, so they take part in the worst-case. At
        # their defaults they need 11 and 3 frames, i.e. neither widens the
        # block beyond what `stabilize_face` already asks for.
        _temporal = [engine for engine in (getattr(self, '_temporal_identity', None),
                                           getattr(self, '_temporal_occlusion', None),
                                           getattr(self, '_target_appearance', None),
                                           getattr(self, '_temporal_compositing', None),
                                           getattr(self, '_temporal_quality', None),
                                           getattr(self, '_landmark_smoother', None),
                                           getattr(self, '_hf_stabilizer', None))
                     if engine is not None and getattr(engine, 'enabled', False)]
        for stab in (self.kps_stabilizer, self.enh_stabilizer,
                     self.mask_stabilizer, *_temporal):
            fn = getattr(stab, 'warmup_frames', None)
            if fn is not None:
                try:
                    need = max(need, int(fn(eps)))
                except Exception as _degrade_error:
                    _swallowed("roop/procmgr_stabilization.py:76", _degrade_error, "fallback continued")
                    need = max(need, _MAX_STAB_WARMUP)
        return need

    # A chunk is NOT the only copy of itself. Live at the same moment in
    # _run_stab_parallel, every one of them chunk-sized:
    #     the reader filling the next chunk              1
    #     prefetch_q                                     its capacity
    #     the chunk currently being processed            1
    #     `results`, that chunk's processed output       1
    #     _write_q                                       its capacity
    # So decoded frames alone reach several times the per-chunk budget. The old
    # flat 1536 MB default reserved ~9 GB, which is unremarkable on 32 GB and
    # fatal on 16 GB — measured on a 16 GB / RTX 3060 box, a 40934-frame render
    # died at 12% with ffmpeg's own threads failing malloc (AVERROR(ENOMEM))
    # and numpy unable to allocate a 1.5 MB array.
    #
    # The two chunk queues now use one dedicated slot each. Keep the formula
    # derived from that capacity so the RAM budget and actual queue geometry
    # cannot drift apart again. The scheduler fallback remains for helper
    # calls made before a render has selected the stabilization path.
    _STAB_LIVE_CHUNKS = 6          # fallback only; see _stab_live_chunks()

    def _stab_live_chunks(self):
        """How many chunk-sized frame buffers _run_stab_parallel holds at once."""
        capacity = getattr(self, '_stab_chunk_queue_capacity', None)
        if capacity is None:
            # Keep the scheduler fallback for callers/tests that invoke the
            # geometry helpers outside a fully profiled render.
            capacity = getattr(self._runtime_scheduler, 'queue_capacity', None)
        try:
            capacity = max(1, int(capacity))
        except (TypeError, ValueError):
            return self._STAB_LIVE_CHUNKS
        # reader + processing + results, plus both bounded queues.
        return 3 + 2 * capacity

    def _default_stab_chunk_mb(self, hard_cap=None):
        """Per-chunk budget that keeps every live copy inside a share of the
        RAM this machine actually has free.

        "Every" is counted by `_stab_live_chunks()` from the queue capacities
        in force, not from a constant -- see the note on `_STAB_LIVE_CHUNKS`.
        """
        try:
            memory = psutil.virtual_memory()
            avail_mb = memory.available / (1024.0 ** 2)
            # `available` is the safety-critical measurement.  Keep the original
            # 1536 MB cap when a lightweight psutil implementation (or a test
            # double) cannot report total RAM, rather than returning the cap
            # before applying the available-memory guard.
            total_mb = float(getattr(memory, 'total', 0) or 0) / (1024.0 ** 2)
        except Exception as _degrade_error:
            # If memory telemetry is unavailable, keep the fallback cap as a
            # TOTAL live-buffer budget rather than handing every live chunk a
            # full 1536 MB.  The latter silently multiplies to ~9 GB on the
            # historical six-buffer path and can recreate the 16 GB laptop
            # allocation failure this guard is meant to prevent.
            _swallowed("roop/procmgr_stabilization.py:128", _degrade_error, "fallback continued")
            try:
                return 1536.0 / max(1, int(self._stab_live_chunks()))
            except Exception as _degrade_error:
                _swallowed("roop/procmgr_stabilization.py:136", _degrade_error, "fallback continued")
                return 1536.0 / self._STAB_LIVE_CHUNKS

        if hard_cap is None:
            # Scale budget cap dynamically with system RAM:
            # 64GB+ RAM: 8192 MB cap
            # 32GB RAM:  4096 MB cap (e.g. desktop workstation)
            # <=16GB RAM: 1536 MB cap (e.g. laptop / low-memory)
            if total_mb >= 55000:
                hard_cap = 8192.0
            elif total_mb >= 28000:
                hard_cap = 4096.0
            else:
                hard_cap = 1536.0

        # The desktop profile needs two complete worker rounds before the block
        # queue can redistribute an expensive face-heavy block.  The old 58%
        # share was enough for the historical 12-frame geometry, but the live
        # 720p render uses 24-frame blocks and started with only one round:
        # `10 blocks x 24f` and 22.89 FPS were observed in the active terminal.
        # With the real queue depth of one, five live chunk copies are held.
        # 75% gives that desktop path enough room for 20 blocks at 720p while
        # retaining roughly 2.0 GB of free host RAM in the observed case.
        #
        # This is deliberately keyed to system RAM, not VRAM: the TensorRT pool
        # policy already owns VRAM and stays at 2/2/2 on the 12 GB card.  Keep
        # the 16 GB laptop at 40%, where its live chunks must stay under the
        # tighter host-memory budget.  If a caller has a deeper queue than the
        # production stabilized path, retain the conservative 58% desktop
        # share because the extra copies consume the headroom this optimization
        # relies on. An explicit environment setting remains authoritative on
        # either machine.
        live = self._stab_live_chunks()
        if total_mb >= 28000.0:
            default_share = 0.75 if live <= 5 else 0.58
        else:
            default_share = 0.40
        try:
            share = float(os.environ.get('ROOP_STAB_RAM_SHARE', '') or default_share)
        except ValueError:
            share = default_share
        share = min(0.90, max(0.05, share))
        # The floor is useful only while the requested RAM share can afford it.
        # Applying an unconditional 96 MB floor makes queue depth part of the
        # memory limit in name only: under pressure, 7 live chunks reserve
        # 672 MB and 11 reserve 1056 MB even when the share is below either
        # amount.  Keep the per-chunk budget proportional to the live-buffer
        # count; geometry already falls back to a sequential path when the
        # resulting budget cannot fund a parallel block.  The one-MB floor is
        # only a finite-value guard and is not a RAM reservation.
        budget = max(1.0, min(hard_cap, (avail_mb * share) / live))
        if not getattr(self, '_stab_budget_notified', False):
            self._stab_budget_notified = True
            print(f"[Stabilize] {avail_mb / 1024.0:.1f} GB RAM free of {total_mb / 1024.0:.1f} GB: chunk budget "
                  f"{budget:.0f} MB (cap {hard_cap:.0f} MB), holding {live} live copies "
                  f"(~{budget * live / 1024.0:.1f} GB of frames). "
                  f"ROOP_STAB_CHUNK_MB overrides this exactly.")
        return budget

    def _stab_parallel_geometry(self, threads):
        """(warm_up, block_frames, width) for parallel stabilization.

        Every block pays `warm_up` frames of full-pipeline work it then discards,
        so a block must be several times the warm-up or the priming eats the
        parallelism it is buying. Blocks are therefore 4x the warm-up, and the
        chunk holds as many as the decoded-frame memory budget allows — at 1080p
        a frame is ~6MB, and a slow filter (strength 1.0 -> 39 warm-up frames ->
        156-frame blocks) wants far more of them than fits.

        `width` is how many such blocks fit, i.e. the REAL concurrency of a
        stabilized run. It can come out at 1, which is the caller's cue to use
        the sequential path instead: 1-wide through the chunking machinery is
        single-threaded with extra bookkeeping, and measured SLOWER than simply
        running sequentially (2.65 vs 2.92 fps at strength 1.0).
        """
        wu = self._stab_warmup or self._stab_warmup_frames()
        # The adaptive shrinks below trade warm-up overhead for width, stepping
        # 4*wu -> 2*wu -> wu. For a 4-6 frame filter warm-up that is a good deal:
        # the alternative on a memory-tight machine is width 1, i.e. no
        # parallelism at all. For the opt-in temporal engines, whose warm-ups are
        # 15 and 44 frames, the last step means a block that discards as many
        # frames as it produces -- 100% redundant full-pipeline work. Measured on
        # the locked fixture, that made the occlusion engine SLOWER in parallel
        # (3.75 fps) than pinned to one worker (5.18), while identity still won
        # (5.89 vs 4.34). So those runs set a floor instead, and fall back to
        # sequential when the budget cannot fund it.
        #
        # A multiple of 1 is a no-op: every expression below is already >= wu.
        _floor = max(1, int(getattr(self, '_stab_min_block_multiple', 1) or 1)) * wu
        block = max(_floor,
                    max(2 * wu, 16) if self._runtime_stab_small else max(4 * wu, 24))
        budget_mb = self._default_stab_chunk_mb()
        _env_budget = (os.environ.get('ROOP_STAB_CHUNK_MB', '') or '').strip()
        if _env_budget:
            # Explicit means explicit, the same rule the pool knobs settled on:
            # a control that silently runs a different number than the one it was
            # given is a control that lies. The RAM-derived default applies only
            # when nothing was asked for.
            try:
                budget_mb = float(_env_budget)
            except ValueError:
                pass
        frame_mb = max(0.1, (self._stab_frame_bytes or (1920 * 1080 * 3)) / (1024.0 ** 2))
        fits = max(1, int((budget_mb / frame_mb) // block))
        if fits < threads and threads >= 2 and wu > 0:
            adaptive_block = max(_floor, max(2 * wu, 16))
            adaptive_fits = max(1, int((budget_mb / frame_mb) // adaptive_block))
            if adaptive_fits > fits:
                block = adaptive_block
                fits = adaptive_fits
            if fits < threads and wu > 0:
                adaptive_block2 = max(_floor, max(wu, 12))
                adaptive_fits2 = max(1, int((budget_mb / frame_mb) // adaptive_block2))
                if adaptive_fits2 > fits:
                    block = adaptive_block2
                    fits = adaptive_fits2
        width = max(1, min(threads, fits))

        # BLOCKS PER CHUNK is a separate question from WORKERS, and conflating
        # them cost ~19% of a real render.
        #
        # The chunk used to be sized `width * block` — exactly one block per
        # worker. The dispatch below hands blocks out through a shared queue so
        # an idle worker can take the next one, but with one block each there is
        # never a next one, and the chunk's wall time is gated by its unluckiest
        # block. Per-frame cost is not uniform (face count and size, masking,
        # close-up rescue), so that gate is expensive: measured over 96 chunks of
        # a live 50,646-frame render, the fastest worker sat idle for a median
        # 18.2% of every chunk and up to 54%, totalling 8.0 of 42.5 minutes.
        #
        # The fix is to let a chunk hold every block the SAME memory budget
        # already allows, so there is slack for the queue to redistribute. At
        # 720p that is 14 blocks against 10 workers where it used to be 10.
        #
        # This does NOT make blocks smaller, which is the other way to create
        # slack and the wrong one: a block is 4x the warm-up it discards, so
        # halving it takes redundant priming from 25% to 50% — more than the
        # imbalance it recovers. Block size, warm-up length and therefore the
        # block grid over the video are all unchanged; only how many of them are
        # decoded and dispatched together changes.
        # IT MUST BE A WHOLE MULTIPLE OF `width`, and this is the entire subtlety.
        #
        # A shared queue schedules greedily, so k blocks over n workers take
        # ceil(k/n) rounds and the last round is as long as a full one however
        # few blocks are in it. A PARTIAL extra round is therefore the worst
        # case, not an improvement: simulated with the per-block spread measured
        # in the live log (fast/slow ~0.70 over 10 blocks), against 10 workers —
        #
        #     blocks   10     11     12     14     16     18     20     30    40
        #     eff.   84.7%  60.9%  62.6%  69.0%  75.9%  82.8%  88.6%  90.4% 91.6%
        #
        # 14 blocks — "as many as the budget allows" — is 19% SLOWER than 10,
        # because four workers run two blocks while six sit idle after one. Only
        # multiples of the worker count beat one-per-worker, and the gain is
        # +5% at 2 per worker rising to +8% at 4, not the 19% of idle time the
        # imbalance stat reports: most of that idle is the cost of the final
        # round, which more blocks cannot remove, only amortise.
        #
        # So: whole rounds only. The default takes two rounds only when the
        # current memory budget already fits them; otherwise it retains the
        # one-round path.  The desktop's larger RAM share above is specifically
        # sized to make two rounds available at its observed 720p workload,
        # while the 16 GB laptop keeps its one-round, low-RSS behaviour.
        #
        # A/B on an 8748-frame 720p clip, all arms in one process, then repeated
        # with the order reversed to counterbalance position:
        #
        #     config          forward   reversed   mean
        #     1 round         15.32     15.01      15.17 fps
        #     2 rounds        15.17     -          15.17
        #     4 rounds        16.89     14.99      15.94
        #
        # The SAME configuration measured 14.99 and 16.89 depending only on
        # whether it ran first or last, and inside the reversed pass two
        # adjacent arms gave 14.99 (4 rounds) and 15.01 (1 round). Position
        # moves the number more than the knob does; the apparent +10% in the
        # first, un-counterbalanced pass was ordering.
        #
        # The reason is visible in the same logs: idle was 1.9-3.6% on every arm
        # of that clip. There was no imbalance to recover, so redistributing it
        # could not help. The automatic second round is therefore conditional on
        # already having the host-memory headroom; the knob remains available
        # for explicitly asking for more than two rounds.
        #
        # ROOP_STAB_BLOCKS_PER_WORKER or ROOP_STAB_BLOCKS_PER_THREAD=2 (or more)
        # opts in, and is worth trying on footage whose [STAB CHUNK] lines report
        # a large `imbalance`. Whatever it is set to, the count stays a WHOLE
        # multiple of `width`: a partial round is 19% slower than none (see the
        # table in tests/test_stab_block_dispatch.py).
        try:
            env_want = os.environ.get('ROOP_STAB_BLOCKS_PER_WORKER') or os.environ.get('ROOP_STAB_BLOCKS_PER_THREAD')
            want = int(float(env_want)) if env_want else 0
        except ValueError:
            want = 0
        if want > 0:
            rounds = max(1, min(max(1, fits // width), want))
        else:
            # Auto: on machines with enough RAM budget (fits // width >= 2), enable 2 rounds
            # so work-stealing eliminates worker idle stalls and GPU utilization valleys.
            rounds = 2 if (fits // width) >= 2 else 1
        blocks_per_chunk = width * rounds
        return wu, block, width, blocks_per_chunk
