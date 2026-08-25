"""Parallel stabilization must derive its warm-up, not guess it.

Splitting a clip into contiguous blocks and giving each its own smoothing filter
is only output-equivalent to running the clip in order if each block first
discards enough frames to forget the seed it was primed with. Every stabilizer
here is an EMA (`out = a*x + (1-a)*prev`), so the seed's weight after W frames is
exactly (1-a)^W — the warm-up is a solved quantity, not a taste parameter.

The old code used a fixed 4 for every configuration. `a` is derived from the
smoothing strength the user picked, so that was only ever right at the weak end:

    strength   a        (1-a)^4     needed for <=1%
      0.00     0.725      0.57%           4
      0.50     0.580      3.10%           6
      0.75     0.430     10.57%           9
      1.00     0.112     62.28%          39

At strength 1 — the setting that asks for the MOST smoothing — 62% of the seed
survived to the first kept frame of every block, i.e. a visible step at every
boundary. These tests pin the derivation and the two edges that decide whether
parallel stabilization may run at all.

one_euro imports only math + numpy, so the filter half runs functionally. The
scheduler half is checked against source, like test_detector_pools, because
importing ProcessMgr pulls torch (~4s).
"""

import os
import re
import sys
import unittest
from pathlib import Path

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

from roop.one_euro import (                                    # noqa: E402
    _MAX_WARMUP, _alpha, ema_warmup_frames,
    EmaKpsStabilizer, EnhancerStabilizer, KpsStabilizer,
)

PM = Path(APP, 'roop', 'ProcessMgr.py').read_text(encoding='utf-8')


def _code(text):
    text = re.sub(r'""".*?"""', '', text, flags=re.S)
    return '\n'.join(re.sub(r'#.*$', '', ln) for ln in text.splitlines())


PM_CODE = _code(PM)


class WarmupIsSolvedNotGuessed(unittest.TestCase):

    def test_warmup_actually_bounds_the_residual(self):
        """The whole contract: after the returned W frames, the seed's weight is
        at or below eps — and W is not wastefully larger than it needs to be."""
        for alpha in (0.05, 0.1116, 0.2, 0.43, 0.58, 0.725, 0.9):
            for eps in (0.05, 0.01, 0.001):
                W = ema_warmup_frames(alpha, eps)
                with self.subTest(alpha=alpha, eps=eps):
                    self.assertLessEqual((1 - alpha) ** W, eps + 1e-12,
                                         f'W={W} leaves too much seed')
                    self.assertGreater((1 - alpha) ** (W - 1), eps,
                                       f'W={W} is larger than necessary')

    def test_degenerate_alphas(self):
        self.assertEqual(ema_warmup_frames(1.0), 0)    # no memory at all
        self.assertEqual(ema_warmup_frames(1.5), 0)
        self.assertEqual(ema_warmup_frames(0.0), _MAX_WARMUP)   # never forgets
        self.assertEqual(ema_warmup_frames(-1.0), _MAX_WARMUP)

    def test_enhancer_warmup_tracks_strength(self):
        """Stronger smoothing = slower filter = strictly more warm-up. A fixed
        number cannot satisfy both ends."""
        ws = [EnhancerStabilizer(strength=s).warmup_frames()
              for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
        self.assertEqual(ws, sorted(ws), f'not monotonic in strength: {ws}')
        self.assertLess(ws[0], ws[-1])
        self.assertEqual(ws, [4, 5, 6, 9, 39])   # the table in the docstring

    def test_the_old_fixed_four_was_not_enough(self):
        """Guards the reason this exists. If someone reinstates a constant, this
        is the case that proves it wrong."""
        strong = EnhancerStabilizer(strength=1.0)
        a = _alpha(1.0, strong.base_cutoff)
        self.assertGreater((1 - a) ** 4, 0.5,
                           'strength 1.0 with 4 warm-up frames should leave most '
                           'of the seed behind')
        self.assertLessEqual((1 - a) ** strong.warmup_frames(), 0.01)

    def test_kps_stabilizers_expose_the_same_contract(self):
        """Whichever filter is slowest sets the block boundary, so all of them
        have to be askable."""
        for stab in (KpsStabilizer(min_cutoff=0.1), EmaKpsStabilizer(alpha=0.3),
                     EnhancerStabilizer(strength=0.5)):
            with self.subTest(stab=type(stab).__name__):
                W = stab.warmup_frames()
                self.assertIsInstance(W, int)
                self.assertGreaterEqual(W, 1)
                self.assertLessEqual(W, _MAX_WARMUP)

    def test_beta_only_speeds_convergence(self):
        """Both derivations use the still case. That is only safe if motion can
        never make the filter slower — beta and motion_beta only raise cutoff."""
        for cutoff, extra in ((0.02, 0.5), (0.22, 0.1), (0.1, 2.0)):
            self.assertGreater(_alpha(1.0, cutoff + extra), _alpha(1.0, cutoff))


class SchedulerPrefersParallel(unittest.TestCase):

    def test_source_was_found(self):
        self.assertGreater(len(PM_CODE), 20000)

    def test_parallel_is_the_default(self):
        """The old default was '0', which meant every stabilized render silently
        collapsed to ONE worker thread — measured at 2.5-3x the wall time."""
        m = re.search(r"ROOP_STAB_PARALLEL['\"]\s*,\s*['\"]([01])['\"]", PM_CODE)
        self.assertIsNotNone(m, 'ROOP_STAB_PARALLEL default not found')
        self.assertEqual(m.group(1), '1',
                         'parallel stabilization must default ON; off means '
                         'single-threaded rendering')

    def test_warmup_comes_from_the_filters(self):
        self.assertIn('_stab_warmup_frames', PM_CODE)
        self.assertIn('warmup_frames', PM_CODE)
        self.assertNotRegex(
            PM_CODE, r"ROOP_STAB_WARMUP['\"]\s*,\s*['\"]\d",
            'warm-up must be derived from the filter, not defaulted to a constant')

    def test_uncapped_filters_fall_back_to_sequential(self):
        """A filter that cannot forget its seed within the cap has no seam-free
        block size, so the scheduler must give up parallelism rather than ship
        a seam."""
        self.assertIn('_MAX_STAB_WARMUP', PM_CODE)
        self.assertRegex(PM_CODE, r'_stab_warmup\s*>=\s*_MAX_STAB_WARMUP')

    def test_one_wide_falls_back_to_the_sequential_path(self):
        """A slow filter wants blocks so long that only one fits the memory
        budget (strength 1.0 at 1080p: 39 warm-up frames -> 156-frame blocks).
        One block is single-threaded ANYWAY, so paying the chunk buffering and
        block bookkeeping on top of it is pure loss — measured 2.65 fps against
        the sequential path's 2.92. The scheduler must take the faster of two
        equally correct paths."""
        self.assertIn('_stab_parallel_geometry', PM_CODE)
        self.assertRegex(PM_CODE, r'_width\s*<\s*2')
        m = re.search(r'if _width < 2:(.*?)\n\s{8}\S', PM_CODE, re.S)
        self.assertIsNotNone(m, 'the 1-wide downgrade block was not found')
        self.assertIn('use_parallel_stab = False', m.group(1))
        self.assertIn('threads = 1', m.group(1))

    def test_the_one_wide_downgrade_re_asks_the_2pass_question(self):
        """`use_2pass` is computed as `(not use_parallel_stab) and ...` BEFORE
        the frame size is known, so while parallel is still on it always comes
        out False. The 1-wide downgrade then turns parallel off -- and if it
        does not recompute `use_2pass`, a kps-only run drops to ONE worker
        thread for the whole swap even though the 2-pass path exists precisely
        to keep them (smooth sequentially in pass 1, swap multi-threaded in
        pass 2).

        Seen on a 16GB/RTX 3060 box: the render process holds ~9GB, the
        RAM-derived chunk budget falls to 138MB, one 40-frame 720p block fits,
        and the whole pipeline ran at execution_threads=1.
        """
        m = re.search(r'if _width < 2:(.*?)\n\s{8}\S', PM_CODE, re.S)
        self.assertIsNotNone(m)
        block = m.group(1)
        self.assertRegex(
            block, r'use_2pass\s*=',
            'the 1-wide downgrade must recompute use_2pass -- it was decided '
            'from a use_parallel_stab that this block has just changed')
        # ...and the single-thread collapse must be conditional on that answer,
        # not unconditional as it was.
        self.assertRegex(
            block, r'if not use_2pass:',
            'threads = 1 must only happen when 2-pass cannot rescue the run')
        self.assertLess(
            block.index('use_2pass ='), block.index('threads = 1'),
            'use_2pass must be recomputed before the thread count is dropped')

    def test_the_downgrade_says_what_it_costs(self):
        """The user-visible symptom is `execution_threads=1` in the progress
        bar and a render at a fraction of its fps. `_warn_single_worker_on_gpu`
        cannot explain it -- that runs in batch_process, before this drop -- so
        this block is the only place that can, and it has to name the thread
        count rather than only its own chunk geometry."""
        m = re.search(r'if _width < 2:(.*?)\n\s{8}\S', PM_CODE, re.S)
        self.assertIsNotNone(m)
        block = m.group(1)
        self.assertIn('ONE worker thread', block,
                      'the downgrade must state the thread cost in words')
        for setting in ('stabilize_enhancer', 'stabilize_mask'):
            self.assertIn(setting, block,
                          'the message must name the settings that force it')

    def test_geometry_is_shared_not_duplicated(self):
        """The scheduler's downgrade check and the run's actual chunk sizing must
        agree; two copies of the arithmetic would drift and the run would then
        parallelise at a width the scheduler already rejected."""
        self.assertEqual(len(re.findall(r'def _stab_parallel_geometry', PM_CODE)), 1)
        self.assertGreaterEqual(len(re.findall(r'_stab_parallel_geometry\(', PM_CODE)), 3)

    def test_dispatch_uses_the_memory_derived_width(self):
        """The chunk is sized to hold exactly stab_width warm-up-amortising
        blocks; splitting it `threads` ways instead would produce blocks shorter
        than their own warm-up."""
        self.assertIn('stab_width', PM_CODE)
        self.assertRegex(PM_CODE, r'n\s*=\s*max\(1,\s*min\(stab_width')


if __name__ == '__main__':
    unittest.main()
