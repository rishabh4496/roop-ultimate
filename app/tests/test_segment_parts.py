"""The output-parts registry behind the console's part tabs.

A resumable render writes numbered part files into the output folder and merges
them at the end. Those parts are the run's chapters in the UI, so what matters
here is that their FRAME NUMBERING is right — a part labelled "frames
6001-7000" is how a user finds the stretch of video that went wrong, and a
resumed run must continue the numbering rather than restart at 1.

The registry is observational only: nothing it records feeds encoding or resume
(those stay driven by the manifest), so these tests drive it directly rather
than spawning ffmpeg.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import segment_writer as sw                      # noqa: E402


def _writer(tmp, **kw):
    return sw.SegmentedVideoWriter(os.path.join(tmp, "out.mp4"), (64, 64), 30.0,
                                   source_video=os.path.join(tmp, "src.mp4"), **kw)


class PartsRegistryTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        sw.reset_parts()

    def tearDown(self):
        self._tmp.cleanup()
        sw.reset_parts()

    def test_starts_empty(self):
        self.assertEqual(sw.parts_snapshot(), [])
        self.assertEqual(sw.current_part_index(), 0)

    def test_finalized_parts_number_frames_continuously(self):
        w = _writer(self.tmp)
        w._register(1, "a.mp4", 1000, done=True)
        w._register(2, "b.mp4", 1000, done=True)
        w._register(3, "c.mp4", 640, done=True)
        got = [(p["index"], p["first"], p["last"]) for p in sw.parts_snapshot()]
        self.assertEqual(got, [(1, 1, 1000), (2, 1001, 2000), (3, 2001, 2640)])
        self.assertTrue(all(p["done"] for p in sw.parts_snapshot()))
        self.assertEqual(sw.current_part_index(), 3)

    def test_open_part_reports_frames_written_so_far(self):
        w = _writer(self.tmp)
        w._register(1, "a.mp4", 1000, done=True)
        sw._current = {"index": 2, "file": "b.mp4", "first": 1001, "_written": 412,
                       "bytes": 0, "done": False, "inherited": False}
        try:
            live = sw.parts_snapshot()[-1]
            self.assertEqual((live["index"], live["frames"], live["first"], live["last"]),
                             (2, 412, 1001, 1412))
            self.assertFalse(live["done"])
            self.assertEqual(sw.current_part_index(), 2)   # log lines tag part 2
        finally:
            sw._current = None

    def test_resumed_run_continues_the_numbering(self):
        """A resume inherits finished parts; the next one must not restart at 1."""
        w = _writer(self.tmp)
        with open(sw.manifest_path(w.target_video), encoding="utf-8") as fh:
            manifest = json.load(fh)
        for name, n in (("s0.mp4", 1000), ("s1.mp4", 1000)):
            open(os.path.join(self.tmp, name), "wb").close()
            manifest["segments"].append({"file": name, "frames": n})
        with open(sw.manifest_path(w.target_video), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)

        w2 = _writer(self.tmp)                        # same identity → resumes
        self.assertEqual(w2.resume_frames, 2000)
        inherited = sw.parts_snapshot()
        self.assertEqual([(p["index"], p["first"], p["last"]) for p in inherited],
                         [(1, 1, 1000), (2, 1001, 2000)])
        self.assertTrue(all(p["inherited"] for p in inherited))

        w2._register(3, "s2.mp4", 500, done=True)
        self.assertEqual(sw.parts_snapshot()[-1]["first"], 2001)

    def test_a_new_run_clears_the_previous_run_s_parts(self):
        w = _writer(self.tmp)
        w._register(1, "a.mp4", 1000, done=True)
        sw.reset_parts()
        self.assertEqual(sw.parts_snapshot(), [])
        self.assertEqual(sw.current_part_index(), 0)


def _api_log_rule():
    """The real is_counter_line/counter_shape, lifted out of api.py.

    Importing api.py would load the whole model stack, but copying its regexes
    into the test would leave the test passing while the shipped rule drifts —
    so the two definitions and the regexes they use are exec'd straight from the
    source. What is exercised below is the code that actually runs.
    """
    import ast
    import re as _re_mod
    api_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'api.py')
    with open(api_py, encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    wanted_fns = {'is_counter_line', 'counter_shape'}
    wanted_names = {'_COUNTER_RE', '_COUNTER_SHAPE_RE', '_COUNTER_PAREN_RE'}
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in wanted_fns)
            or (isinstance(n, ast.Assign) and any(getattr(t, 'id', None) in wanted_names
                                                  for t in n.targets))]
    found = {getattr(n, 'name', None) for n in body} | {
        t.id for n in body if isinstance(n, ast.Assign) for t in n.targets}
    missing = (wanted_fns | wanted_names) - found
    assert not missing, f'api.py no longer defines: {sorted(missing)}'
    ns = {'_re': _re_mod}
    exec(compile(ast.Module(body=body, type_ignores=[]), 'api.py', 'exec'), ns)
    return ns['is_counter_line'], ns['counter_shape']


is_counter_line, counter_shape = _api_log_rule()


class CounterLineTest(unittest.TestCase):
    """The rule that keeps per-frame counters OUT of the console history:
    a line that changes every poll is pinned, anything that happens once is
    kept — and a stage must still leave exactly one mark behind."""

    def assert_pinned(self, msg):
        self.assertTrue(is_counter_line(msg), f"expected pinned: {msg!r}")

    def assert_logged(self, msg):
        self.assertFalse(is_counter_line(msg), f"expected logged: {msg!r}")

    def _history(self, lines):
        """What _push_log keeps: non-counters, plus the first line of each
        counter shape per phase (a real event ends the phase)."""
        kept, shapes, last = [], set(), ''
        for m in lines:
            if m == last:
                continue
            if is_counter_line(m):
                shape = counter_shape(m)
                if shape in shapes:
                    continue
                shapes.add(shape)
            else:
                shapes = set()
            kept.append(m)
            last = m
        return kept

    def test_a_rate_suffix_that_comes_and_goes_is_not_a_new_stage(self):
        """tqdm only reports "(20.3 FPS)" once it has a rate, and drops it on a
        stall — that must not read as the stage restarting."""
        kept = self._history(['Processing frame 1 / 300',
                              'Processing frame 2 / 300 (12.0 FPS)',
                              'Processing frame 3 / 300',
                              'Processing frame 4 / 300 (11.4 FPS)'])
        self.assertEqual(kept, ['Processing frame 1 / 300'])

    def test_every_stage_leaves_a_mark_in_the_history(self):
        """Dropping counter lines outright also dropped the only evidence the
        SWAP ran — "Processing frame N / M" is all that stage ever says."""
        run = ['▶ Starting job…', 'Analyzing faces',
               'Processing frame 1 / 48501', 'Processing frame 2 / 48501 (11.1 FPS)',
               'Processing frame 4211 / 48501 (20.3 FPS)', 'Processing frame 48501 / 48501',
               'Upscaling frame 1 / 48501', 'Upscaling frame 900 / 48501',
               'Combining (encode + audio)…', '✓ Done']
        kept = self._history(run)
        self.assertEqual(kept, ['▶ Starting job…', 'Analyzing faces',
                                'Processing frame 1 / 48501',
                                'Upscaling frame 1 / 48501',
                                'Combining (encode + audio)…', '✓ Done'])

    def test_a_stage_that_repeats_per_file_is_not_collapsed_across_stages(self):
        """Returning to a stage after another one logs it again."""
        kept = self._history(['Processing frame 1 / 300', 'Processing frame 2 / 300',
                              'Combining (encode + audio)…',
                              'Processing frame 1 / 500', 'Processing frame 2 / 500'])
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept[-1], 'Processing frame 1 / 500')

    def test_frame_counters_are_pinned(self):
        for m in ("Processing frame 6412 / 109376 (20.3 FPS)",
                  "Upscaling frame 12 / 300",
                  "Interpolating frame 5 / 60",
                  "Extracting frames 45%"):
            self.assert_pinned(m)

    def test_one_off_events_are_logged(self):
        for m in ("▶ Starting job…", "✓ Done", "⚠ ffmpeg failed",
                  "Combining (encode + audio)…", "Analyzing faces", "Paused",
                  "✓ part 3 written · frames 2001–3000 · 41 MB"):
            self.assert_logged(m)

    def test_the_rule_under_test_is_the_shipped_one(self):
        """_api_log_rule() asserts the definitions still exist; this pins that
        they were really lifted from api.py rather than defaulting to a stub."""
        self.assertEqual(is_counter_line.__code__.co_filename, 'api.py')
        self.assertEqual(counter_shape('Processing frame 12 / 300 (20.3 FPS)'),
                         'Processing frame # / #')


if __name__ == '__main__':
    unittest.main()
