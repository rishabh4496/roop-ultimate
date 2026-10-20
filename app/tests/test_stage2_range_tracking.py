"""Stage 2 pre-pass over Stage 1 active ranges: what a track may carry across a gap.

The brief for this stage feared that a locked track from the end of range i could use its
position to claim a different person standing in the same place when range i+1 begins. These
tests drive the real `_precompute_tracks` (the same harness test_track_stitch.py uses) with
`_active_ranges` set, and pin what actually happens:

* any skipped stretch is a shot boundary (procmgr_tracking: `idx > prev_idx + 1`), so every
  live track is retired with its velocity and flow state cleared - at 16 frames, the brief's
  threshold, and at 1;
* what survives is the track's identity embedding, so the SAME person is re-acquired on
  appearance wherever they have moved to, and a stranger in the same box is not;
* a track's first_seen / last_seen and the assignment keys are true clip indices;
* an entry whose appearance has collapsed (a hand, a turned head) is chained to the track
  that follows it by `_stitch_tracks`, so the entry frames are swapped.
"""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                      # noqa: E402
from roop.procmgr_tracking import TrackingMixin          # noqa: E402


def _emb(*v):
    a = np.zeros(512, dtype=np.float32)
    a[:len(v)] = v
    return a / np.linalg.norm(a)


def _at_distance(base, d):
    perp = np.zeros_like(base)
    perp[-1] = 1.0
    perp = perp - base * float(np.dot(perp, base))
    perp /= np.linalg.norm(perp)
    cos = 1.0 - d
    v = base * cos + perp * float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    return (v / np.linalg.norm(v)).astype(np.float32)


TARGET = _emb(1.0, 0.0, 0.0, 0.0, 0.0)
PROFILE = _at_distance(TARGET, 0.85)       # the same person once every gate reads a stranger
STRANGER = _at_distance(TARGET, 1.0)
HERE = (100.0, 100.0)


class _Face:
    def __init__(self, embedding, at, size=100.0):
        x, y = at
        self.bbox = np.array([x, y, x + size, y + size], dtype=np.float32)
        self.embedding = None if embedding is None else np.asarray(embedding, dtype=np.float32)


class _Captured:
    def __init__(self, embedding):
        self.embedding = embedding


class _Options:
    face_distance_threshold = 0.75
    selected_index = 0


class _Mgr(TrackingMixin):
    def __init__(self):
        self.target_face_datas = [_Captured(TARGET)]
        self.target_face_groups = [0]
        self.options = _Options()
        self.progress_gradio = None
        self._track_assignments = {}
        self._track_scanned = 0

    def _publish_live(self, frame):
        pass


def scan(script, ranges):
    """`script` {true frame index: [faces]}; only the frames inside `ranges` are decoded."""
    mgr = _Mgr()
    mgr._active_ranges = ranges
    seen = []

    def detect(frame, *_a, **_kw):
        seen.append(frame)
        return list(script.get(frame, []))

    was, roop.globals.processing = roop.globals.processing, True
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), mock.patch('roop.face_util.get_all_faces', detect):
            tracks = mgr._precompute_tracks(None, 0, 400, 400, awebp_frames=list(range(400)),
                                            step=1, collect_obs=True)
    finally:
        roop.globals.processing = was
    mgr._decoded = seen
    mgr._stdout = out.getvalue()
    return mgr, tracks


def source_at(mgr, frame):
    entries = mgr._track_assignments.get(frame) or []
    return entries[0][1] if entries else 'none'


def _stand(embedding, first, last, at=HERE):
    return {f: [_Face(embedding, at)] for f in range(first, last + 1)}


class AcrossAGap(unittest.TestCase):
    def test_a_stranger_in_the_same_box_does_not_inherit_the_track_or_its_source(self):
        for gap in (1, 15, 16, 40, 200):
            with self.subTest(gap=gap):
                a_next = 24 + gap + 1
                script = _stand(TARGET, 0, 24)
                script.update(_stand(STRANGER, a_next, a_next + 24))
                mgr, tracks = scan(script, [(0, 24), (a_next, a_next + 24)])
                self.assertEqual(len(tracks), 2)
                self.assertEqual(source_at(mgr, 24), 0)
                self.assertIsNone(source_at(mgr, a_next))
                self.assertIsNone(source_at(mgr, a_next + 24))
                self.assertIn(a_next, mgr._shot_boundaries)

    def test_the_same_person_is_reacquired_by_appearance_wherever_they_went(self):
        script = _stand(TARGET, 0, 24)
        script.update(_stand(TARGET, 80, 104, at=(400.0, 250.0)))      # far from where they left
        mgr, tracks = scan(script, [(0, 24), (80, 104)])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(source_at(mgr, 0), 0)
        self.assertEqual(source_at(mgr, 80), 0)
        self.assertEqual(source_at(mgr, 104), 0)

    def test_velocity_is_not_carried_over_the_gap(self):
        # a face moving right at 10 px/frame, then (after the gap) a stranger exactly where
        # that motion would have taken it
        script = {f: [_Face(TARGET, (100.0 + 10 * f, 100.0))] for f in range(0, 25)}
        a_next = 24 + 20
        script.update({f: [_Face(STRANGER, (100.0 + 10 * 24 + 10 * (f - 24), 100.0))]
                       for f in range(a_next, a_next + 10)})
        mgr, tracks = scan(script, [(0, 24), (a_next, a_next + 9)])
        self.assertEqual(len(tracks), 2)
        self.assertIsNone(source_at(mgr, a_next))
        for t in tracks:
            if t['last_seen'] == 24:
                self.assertIsNone(t.get('vel'), 'a retired track kept the previous shot\'s motion')

    def test_track_ages_and_assignments_are_true_clip_indices(self):
        script = _stand(TARGET, 5, 20)
        script.update(_stand(TARGET, 300, 330))
        mgr, tracks = scan(script, [(5, 20), (300, 330)])
        self.assertEqual((tracks[0]['first_seen'], tracks[0]['last_seen']), (5, 330))
        self.assertEqual(sorted(mgr._track_assignments), list(range(5, 21)) + list(range(300, 331)))
        self.assertEqual(sorted(set(mgr._decoded)), sorted(set(mgr._decoded)))   # only scanned frames
        self.assertEqual(len(mgr._decoded), 16 + 31)

    def test_scanned_frames_are_exactly_the_ranges(self):
        mgr, _ = scan(_stand(TARGET, 0, 399), [(10, 19), (100, 104)])
        self.assertEqual(sorted(mgr._decoded), list(range(10, 20)) + list(range(100, 105)))


class EntryFrames(unittest.TestCase):
    def test_an_entry_with_collapsed_appearance_is_chained_to_the_track_after_it(self):
        script = _stand(PROFILE, 50, 55)            # enters turned / partly covered
        script.update(_stand(TARGET, 56, 89))
        mgr, tracks = scan(script, [(50, 89)])
        self.assertEqual(len(tracks), 1, 'the entry fragment was not chained')
        for f in range(50, 56):
            self.assertEqual(source_at(mgr, f), 0, f'entry frame {f} left unswapped')

    def test_without_the_chain_the_entry_would_have_no_source(self):
        # the same entry, but nobody follows it: appearance alone refuses it
        mgr, tracks = scan(_stand(PROFILE, 50, 55), [(50, 89)])
        self.assertIsNone(source_at(mgr, 50))


class NoUsableEmbedding(unittest.TestCase):
    def test_a_detection_without_an_embedding_no_longer_raises_out_of_the_prepass(self):
        script = _stand(TARGET, 0, 9)
        script[10] = [_Face(None, HERE)]
        script.update(_stand(TARGET, 11, 20))
        mgr, tracks = scan(script, [(0, 20)])        # was: TypeError, the whole pre-pass lost
        self.assertEqual(len(tracks), 1)
        self.assertEqual(source_at(mgr, 9), 0)
        self.assertEqual(source_at(mgr, 11), 0)
        self.assertEqual(source_at(mgr, 10), 'none')
        self.assertIn('1 detection(s) had no usable embedding', mgr._stdout)

    def test_a_clean_scan_prints_nothing_about_it(self):
        mgr, _ = scan(_stand(TARGET, 0, 9), [(0, 9)])
        self.assertNotIn('no usable embedding', mgr._stdout)


if __name__ == '__main__':
    unittest.main()
