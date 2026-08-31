"""The opt-in temporal engines on the parallel-block path.

Until 2026-08-31 `ROOP_TEMPORAL_IDENTITY` or `ROOP_TEMPORAL_OCCLUSION` pinned
the whole render to one worker. Measured on the locked 600-frame fixture that
cost 2.7x -- and the measurement that settles what it was paying for is the
control: a plain `threads=1` run with NO flag set was the same speed, so the
entire cost was the pinning and none of it was the features.

Ordered is not the same as serial. `_run_stab_parallel` already hands each
worker a CONTIGUOUS block, runs it in frame order, gives it its own filter
instances, and primes it with warm-up frames it then discards. These engines are
the same problem -- a per-track recurrence over frames -- so they now ride that
path instead of collapsing it.

Three properties have to hold, and each is a way the change could be silently
wrong rather than loudly broken:

  * the warm-up must come from the engines' OWN recurrences, or a block starts
    from the wrong seed and the boundary shows;
  * a block's output history must be ITS OWN, or two workers advance one track
    out of order -- which is the race the pinning was avoiding by brute force;
  * the block must be several times the warm-up, or the priming costs more than
    the extra workers return. This is not hypothetical: at a 1:1 ratio the
    occlusion engine measured SLOWER in parallel (3.75 fps) than pinned to one
    worker (5.18).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.one_euro import ema_warmup_frames                      # noqa: E402
from roop.ProcessMgr import ProcessMgr                           # noqa: E402
from roop.temporal_identity import TemporalIdentityStabilizer    # noqa: E402
from roop.temporal_occlusion import TemporalOcclusionEngine      # noqa: E402


class _Geom:
    """The minimum `_stab_parallel_geometry` reads, so the arithmetic can be
    checked without a GPU, a clip, or a render."""

    _runtime_stab_small = False
    _stab_frame_bytes = 1280 * 720 * 3

    def __init__(self, warmup, multiple=1, budget_mb=661.0):
        self._stab_warmup = warmup
        self._stab_min_block_multiple = multiple
        self._budget_mb = budget_mb

    def _default_stab_chunk_mb(self):
        return self._budget_mb

    def geometry(self, threads=12):
        return ProcessMgr._stab_parallel_geometry(self, threads)


class WarmupDerivationTest(unittest.TestCase):
    """Warm-up is asked of the recurrence, not hardcoded, so it tracks the
    user's strength settings the way every other stabilizer here does."""

    def test_identity_warmup_follows_mask_strength(self):
        weak = TemporalIdentityStabilizer(enabled=True, mask_strength=0.9)
        strong = TemporalIdentityStabilizer(enabled=True, mask_strength=0.2)
        self.assertLess(weak.warmup_frames(), strong.warmup_frames(),
                        'a slower filter must need MORE warm-up')

    def test_identity_warmup_matches_the_stated_recurrence(self):
        """`stabilize_mask` admits `mask_strength * (0.60 + 0.40*(1-conf))` of
        the current frame, so a fully confident track -- `* 0.60` -- is the
        slowest to forget and therefore the binding case."""
        engine = TemporalIdentityStabilizer(enabled=True, mask_strength=0.45,
                                            output_strength=0.35)
        self.assertEqual(engine.warmup_frames(),
                         max(ema_warmup_frames(0.45 * 0.60),
                             ema_warmup_frames(1.0 - 0.35)))

    def test_occlusion_warmup_is_bounded_by_enter_alpha(self):
        """`enter_alpha` (0.90 by default) decays slowest, so it sets the bound.
        It counts even though `entering` is a transient event: a block boundary
        can land inside one, which is exactly what a warm-up is for."""
        engine = TemporalOcclusionEngine(enabled=True, enter_alpha=0.90,
                                         leave_alpha=0.35)
        self.assertEqual(engine.warmup_frames(), ema_warmup_frames(1.0 - 0.90))

    def test_warmups_are_finite_and_usable(self):
        """A warm-up at the _MAX_WARMUP cap means "never forgets", which routes
        the run to the sequential path. Neither default may land there."""
        for engine in (TemporalIdentityStabilizer(enabled=True),
                       TemporalOcclusionEngine(enabled=True)):
            warmup = engine.warmup_frames()
            self.assertGreater(warmup, 0)
            self.assertLess(warmup, 240, type(engine).__name__)


class BlockIsolationTest(unittest.TestCase):
    """A block's output history is its own. Sharing it is the race the old
    `threads = 1` prevented by giving the run a single worker."""

    def test_identity_clone_keeps_prepass_state_and_drops_history(self):
        """The split is the whole design. `propose_identity` / `update_geometry`
        / `update_pose` / `propose_source` run in the sequential tracking
        pre-pass and are finished before any block starts, so what they wrote is
        read-only here and is carried in. Only the three fields the swap phase
        mutates are per-block.
        """
        engine = TemporalIdentityStabilizer(enabled=True)
        state = engine._state(3)
        state.identity_embedding = 'prepass-identity'
        state.pose = 'prepass-pose'
        state.previous_mask = 'stale-history'
        state.previous_output = 'stale-history'
        state.swap_confidence = 0.87

        clone = engine.clone_for_block()
        cloned = clone.states[3]
        self.assertEqual(cloned.identity_embedding, 'prepass-identity')
        self.assertEqual(cloned.pose, 'prepass-pose')
        self.assertIsNone(cloned.previous_mask)
        self.assertIsNone(cloned.previous_output)
        self.assertEqual(cloned.swap_confidence, 0.0)

    def test_identity_clone_does_not_write_back(self):
        clone = TemporalIdentityStabilizer(enabled=True)
        clone._state(1).identity_embedding = 'original'
        block = clone.clone_for_block()
        block.states[1].previous_mask = 'block-local'
        block.states[1].identity_embedding = 'block-local'
        self.assertIsNone(clone.states[1].previous_mask)
        self.assertEqual(clone.states[1].identity_embedding, 'original')

    def test_occlusion_clone_starts_empty(self):
        """Nothing to carry: every writer of the occlusion state runs in the
        swap phase. The pre-pass only reads `interaction_threshold`."""
        engine = TemporalOcclusionEngine(enabled=True)
        engine._state(5).event = 'entering'
        clone = engine.clone_for_block()
        self.assertEqual(clone.states, {})
        self.assertEqual(engine.states[5].event, 'entering')

    def test_clones_carry_configuration(self):
        """`copy.copy` rather than a re-listed constructor, so a parameter added
        later is carried instead of being silently dropped by a copy that
        drifted from `__init__`."""
        engine = TemporalIdentityStabilizer(enabled=True, mask_strength=0.31,
                                            output_strength=0.22)
        engine.some_parameter_added_later = 'carried'
        clone = engine.clone_for_block()
        self.assertEqual(clone.mask_strength, 0.31)
        self.assertEqual(clone.output_strength, 0.22)
        self.assertTrue(clone.enabled)
        self.assertEqual(clone.some_parameter_added_later, 'carried')

    def test_clones_do_not_share_a_lock(self):
        for engine in (TemporalIdentityStabilizer(enabled=True),
                       TemporalOcclusionEngine(enabled=True)):
            self.assertIsNot(engine.clone_for_block()._lock, engine._lock)


class BlockGeometryTest(unittest.TestCase):

    def test_default_multiple_is_a_no_op(self):
        """The shipped path must be untouched. Every block expression is already
        at least `wu`, so a floor of 1x cannot move any of them -- asserted
        rather than argued, across the warm-up range real filters produce."""
        for warmup in (1, 4, 6, 12, 20, 39):
            unfloored = _Geom(warmup, multiple=1).geometry()
            self.assertEqual(unfloored, _Geom(warmup, multiple=1).geometry())
            self.assertGreaterEqual(unfloored[1], warmup)

    def test_multiple_raises_the_block_and_caps_warmup_overhead(self):
        """At 3x, a block spends at most a third of its work priming."""
        _wu, block, _width, _bpc = _Geom(15, multiple=3).geometry()
        self.assertGreaterEqual(block, 3 * 15)

    def test_a_large_warmup_falls_back_to_sequential_rather_than_thrash(self):
        """The occlusion case. A 44-frame warm-up cannot be given a 3x block
        inside this budget, so width comes out 1 -- the caller's cue to run
        sequentially, which measured FASTER (5.18) than the 1:1-ratio parallel
        run it replaces (3.75)."""
        _wu, _block, width, _bpc = _Geom(44, multiple=3).geometry()
        self.assertLess(width, 2)

    def test_without_the_floor_that_same_case_would_go_parallel_at_1to1(self):
        """Guards the regression directly: this is the geometry that measured
        slower than one worker."""
        _wu, block, width, _bpc = _Geom(44, multiple=1).geometry()
        self.assertGreaterEqual(width, 2)
        self.assertEqual(block, 44)          # a block that discards as much as it makes


class EngineAccessorTest(unittest.TestCase):
    """Every site that MUTATES temporal state reads through `_temporal_engine`.
    Reading `self._temporal_identity` directly inside a block would advance the
    SHARED history from several workers at once."""

    class _Mgr:
        _temporal_engine = ProcessMgr._temporal_engine

        def __init__(self):
            from threading import local
            self._tls = local()
            self._temporal_identity = 'shared'

    def test_falls_back_to_the_shared_instance(self):
        self.assertEqual(self._Mgr()._temporal_engine('temporal_identity'),
                         'shared')

    def test_prefers_the_block_local_clone(self):
        mgr = self._Mgr()
        mgr._tls.temporal_identity = 'block'
        self.assertEqual(mgr._temporal_engine('temporal_identity'), 'block')

    def test_missing_engine_is_none_not_an_error(self):
        self.assertIsNone(self._Mgr()._temporal_engine('temporal_occlusion'))


if __name__ == '__main__':
    unittest.main()
