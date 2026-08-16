"""The "time left" the web UI shows must be the one the terminal is showing.

The UI used to derive it as elapsed * (1 - fraction) / fraction, which silently
assumes every second so far was spent at the rate the finished frames ran at. A
render spends minutes on model loads, TensorRT engine builds and the temporal
pre-pass first, all billed against a frame counter still at zero — and then
extrapolated across the remaining frames. Reported gap on a real run: 66 minutes
in the UI against 28 on the bar.

So the bar publishes what it is displaying and the UI shows that. What is worth
pinning is that the published number really is the terminal's arithmetic (for
both display paths), and that it goes quiet rather than stale when nothing is
counting frames.
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import procmgr_runtime as rt                    # noqa: E402


class FakeBar:
    """Enough of a tqdm to answer _bar_eta_seconds.

    `rate` mirrors tqdm's format_dict: the smoothed EMA rate, or None when the
    EMA has nothing in it yet (which is when tqdm falls back to the average).
    """

    def __init__(self, n, total, elapsed, rate=None, chunked=False, last_rate=None):
        self._d = {"n": n, "total": total, "elapsed": elapsed, "rate": rate}
        self._chunked = chunked
        self._last_rate = last_rate

    @property
    def format_dict(self):
        return dict(self._d)


class BarEtaTest(unittest.TestCase):

    def test_bar_mode_divides_remaining_by_the_smoothed_rate(self):
        """tqdm.format_meter: remaining = (total - n) / rate."""
        bar = FakeBar(n=7233, total=44755, elapsed=767.0, rate=22.5)
        self.assertAlmostEqual(rt._bar_eta_seconds(bar), (44755 - 7233) / 22.5, places=6)

    def test_bar_mode_falls_back_to_the_average_like_tqdm_does(self):
        """rate is None until the EMA has samples; tqdm then uses n / elapsed."""
        bar = FakeBar(n=1000, total=5000, elapsed=100.0, rate=None)
        self.assertAlmostEqual(rt._bar_eta_seconds(bar), 4000 / 10.0, places=6)

    def test_chunked_mode_uses_the_rate_the_chunk_printed(self):
        """A chunked bar never redraws, so tqdm's EMA rate is meaningless here —
        the printed line divides by the chunk's own rate and so must we."""
        bar = FakeBar(n=2000, total=10000, elapsed=500.0,
                      rate=99.0,           # would be wrong to use
                      chunked=True, last_rate=8.0)
        self.assertAlmostEqual(rt._bar_eta_seconds(bar), 8000 / 8.0, places=6)

    def test_chunked_before_the_first_chunk_falls_back_to_the_average(self):
        bar = FakeBar(n=500, total=10000, elapsed=100.0, chunked=True, last_rate=None)
        self.assertAlmostEqual(rt._bar_eta_seconds(bar), 9500 / 5.0, places=6)

    def test_no_estimate_when_there_is_nothing_to_divide(self):
        for bar in (FakeBar(n=0, total=0, elapsed=10.0),          # unknown total
                    FakeBar(n=100, total=100, elapsed=10.0),      # finished
                    FakeBar(n=200, total=100, elapsed=10.0),      # overshot
                    FakeBar(n=0, total=100, elapsed=0.0)):        # nothing timed yet
            self.assertIsNone(rt._bar_eta_seconds(bar))

    def test_a_broken_bar_never_raises_into_the_render(self):
        for junk in (None, object(), "not a bar"):
            self.assertIsNone(rt._bar_eta_seconds(junk))

    def test_matches_the_terminal_and_not_the_old_extrapolation(self):
        """The measured run behind this change, kept as the regression case.

        12m47s in, 7,233 of 44,755 frames, 22.5 fps. The bar said 28 minutes
        left; the UI said 66, because ~7 of those 12 minutes were one-off setup.
        """
        elapsed, n, total, rate = 767.0, 7233, 44755, 22.5
        terminal = rt._bar_eta_seconds(FakeBar(n=n, total=total, elapsed=elapsed, rate=rate))
        old_ui = elapsed * (1 - n / total) / (n / total)
        self.assertAlmostEqual(terminal / 60.0, 27.8, delta=0.3)
        self.assertAlmostEqual(old_ui / 60.0, 66.2, delta=0.3)
        self.assertGreater(old_ui, terminal * 2, 'the bug this pins is a >2x overestimate')


class PublishedEtaTest(unittest.TestCase):

    def setUp(self):
        rt.reset_eta()

    def tearDown(self):
        rt.reset_eta()

    def test_starts_with_nothing_published(self):
        self.assertIsNone(rt.eta_seconds())

    def test_publish_then_read(self):
        rt.publish_eta(FakeBar(n=100, total=1100, elapsed=10.0, rate=10.0))
        self.assertAlmostEqual(rt.eta_seconds(), 100.0, places=6)

    def test_an_unusable_bar_leaves_the_last_good_value_alone(self):
        """Between stages a bar can briefly have nothing to say; that must not
        wipe a figure the UI is currently showing."""
        rt.publish_eta(FakeBar(n=100, total=1100, elapsed=10.0, rate=10.0))
        rt.publish_eta(FakeBar(n=0, total=0, elapsed=0.0))
        self.assertAlmostEqual(rt.eta_seconds(), 100.0, places=6)

    def test_stale_estimates_are_dropped_rather_than_shown(self):
        """Nothing counts frames during encode/mux. An ETA frozen at the last
        stage's figure is worse than none — the UI falls back to its own."""
        rt.publish_eta(FakeBar(n=100, total=1100, elapsed=10.0, rate=10.0))
        rt._eta_state["t"] = time.time() - 3600
        self.assertIsNone(rt.eta_seconds())
        self.assertAlmostEqual(rt.eta_seconds(max_age=7200), 100.0, places=6)

    def test_reset_clears_it_for_the_next_run(self):
        rt.publish_eta(FakeBar(n=100, total=1100, elapsed=10.0, rate=10.0))
        rt.reset_eta()
        self.assertIsNone(rt.eta_seconds())


class ChunkedBarRecordsItsRateTest(unittest.TestCase):
    """_last_rate is what makes the chunked path agree with its own output, so
    the attribute has to exist from construction and be written when a chunk is
    emitted -- checked against the source, since constructing a real bar here
    would print to the test runner's console."""

    def _source(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'roop', 'procmgr_runtime.py')
        with open(path, encoding='utf-8') as fh:
            return fh.read()

    def test_initialised_and_written_on_emit(self):
        src = self._source()
        self.assertIn('self._last_rate = None', src)
        self.assertIn('self._last_rate = rate', src)

    def test_publish_eta_is_called_on_both_progress_paths(self):
        """The swap loop and the temporal pre-pass each have their own bar; if
        only one published, the other stage would show a stale figure."""
        here = os.path.dirname(__file__)
        for rel, needle in ((('..', 'roop', 'ProcessMgr.py'), '_publish_eta(progress)'),
                            (('..', 'roop', 'procmgr_tracking.py'), 'publish_eta(pbar)')):
            with open(os.path.join(here, *rel), encoding='utf-8') as fh:
                self.assertIn(needle, fh.read(), f'{rel[-1]} does not publish an ETA')


if __name__ == '__main__':
    unittest.main()


class SignatureAgreesBetweenRecordAndEstimate(unittest.TestCase):
    """The estimate and the record must key the SAME bucket.

    Two independent payloads reach `signature_from_payload`: the full swap body
    on the record side, and the frontend's small `sigPayload` on the estimate
    side. Nothing connected them, so a field added to _SIG_FIELDS but not to
    sigPayload would make the estimate look up a signature the record never
    writes — the estimate silently falls back to the global average forever, and
    every measurement still lands correctly, so nothing looks broken.
    """

    def setUp(self):
        import re
        from pathlib import Path
        repo = Path(__file__).resolve().parents[2]
        self.calib = (repo / 'app' / 'roop' / 'runtime_calib.py').read_text(encoding='utf-8')
        hook = (repo / 'react-ui' / 'src' / 'components' / 'faceswap'
                / 'useRuntimeEstimate.js').read_text(encoding='utf-8')
        body = hook[hook.index('const sigPayload'):hook.index('const heuristicMsPerFrame')]
        self.sent = set(re.findall(r'^\s+([a-z_0-9]+):', body, re.M))
        self.re = re

    def test_every_signature_field_is_sent_by_the_estimate(self):
        fields = self.re.search(r'_SIG_FIELDS = \[(.*?)\n\]', self.calib, self.re.S).group(1)
        # ("canonical", ["payload", "keys"]) — any ONE of the keys satisfies it.
        rows = self.re.findall(r'\(\s*"[a-z_]+",\s*\[([^\]]*)\]', fields)
        self.assertGreaterEqual(len(rows), 10, '_SIG_FIELDS not parsed')
        for row in rows:
            keys = self.re.findall(r'"([a-z_0-9]+)"', row)
            with self.subTest(field=keys):
                self.assertTrue(self.sent & set(keys),
                                f"signature reads one of {keys}, but sigPayload sends none "
                                f"of them — estimate and record would key different buckets")

    def test_the_merger_count_is_sent_too(self):
        """Appended outside _SIG_FIELDS, so the loop above cannot see it."""
        keys = self.re.findall(r'"(merger_[a-z_]+)"',
                               self.re.search(r'_MERGER_COST_KEYS = \((.*?)\)',
                                              self.calib, self.re.S).group(1))
        self.assertEqual(len(keys), 5)
        for key in keys:
            with self.subTest(key=key):
                self.assertIn(key, self.sent)

    def test_face_size_never_moves_the_signature(self):
        """It costs nothing, so splitting buckets on it would only starve them.

        Asserted on BEHAVIOUR rather than on the name being absent from the
        source — it is named there, in the comment explaining this very rule.
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from roop import runtime_calib
        base = {'swap_model': 'inswapper', 'selected_enhancer': 'GPEN'}
        ref = runtime_calib.signature_from_payload(base)
        for value in (-0.2, -0.05, 0.0, 0.05, 0.2):
            with self.subTest(output_face_scale=value):
                self.assertEqual(
                    runtime_calib.signature_from_payload({**base, 'output_face_scale': value}),
                    ref)

    def test_a_neutral_payload_signs_exactly_as_before(self):
        """The merger part is appended only when active, which is what keeps
        every calibration entry already on disk valid without a _VERSION bump."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from roop import runtime_calib
        base = {'swap_model': 'inswapper', 'selected_enhancer': 'GPEN'}
        off = runtime_calib.signature_from_payload(base)
        zeros = runtime_calib.signature_from_payload(
            {**base, 'merger_grain_match': 0, 'merger_sharpen': 0.0,
             'output_face_scale': 0.15})
        self.assertEqual(off, zeros, 'a neutral merger changed the signature')
        on = runtime_calib.signature_from_payload({**base, 'merger_grain_match': 0.5})
        self.assertNotEqual(off, on, 'an active merger did NOT change the signature')
        self.assertIn('merger=1', on)
        two = runtime_calib.signature_from_payload(
            {**base, 'merger_grain_match': 0.5, 'merger_degrade': 0.2})
        self.assertIn('merger=2', two)
