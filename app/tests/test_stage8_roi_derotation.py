"""Stage 8: a track last seen sideways has its ROI stood upright BEFORE the detector sees it.

Measured 2026-10-19 on real Love / Weeds / Monica Bellucci / b1 frames rolled by
+90 / -90 / 180 degrees, with the real `TemporalFaceTracker` driving 150 consecutive
frames per case (the hint comes from the tracker's own history, not an oracle):

    roll    ROI recall, upright-first -> derotated-first
    -90     89.5% -> 96.1%
    +90     91.4% -> 95.9%
    180     82.7% -> 92.0%      (and 10-27% less wall time: fewer rescue ladders)

The upright detector, shown a sideways crop, often "succeeds badly" -- returns a
confident upright face that is not the one there -- and the rotated rescue only runs
on an EMPTY result, so it never got its turn. A track that never rolls past ~45 degrees carries no hint and is untouched (replayed on
four real upright-ish clips: recall identical with and without, hints only where a head
really was rolled). A hint is checked: the face found must roll back to the same quarter
turn. That catches a face with real sideways geometry; it cannot catch a hallucinated
upright-layout one (see `test_hallucinated_upright_geometry_is_NOT_caught...`).
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insightface.app.common import Face                                   # noqa: E402
import roop.globals                                                       # noqa: E402
from roop import face_util, orientation, temporal_tracker                 # noqa: E402
from roop.procmgr_tracking import TrackingMixin                           # noqa: E402
from roop.temporal_tracker import TemporalFaceTracker                     # noqa: E402

CW, ACW, HALF = "rotate_clockwise", "rotate_anticlockwise", "rotate_180"

_UPRIGHT = np.array([[-30, -20], [30, -20], [0, 0], [-25, 35], [25, 35]], np.float64)


def _kps(roll_deg, cx=300.0, cy=240.0):
    """Five keypoints of a face whose chin points at `roll_deg` (0 = down, +90 = image +x)."""
    t = np.radians(roll_deg)
    r = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    return (_UPRIGHT @ r.T + [cx, cy]).astype(np.float32)


def _face(roll_deg, cx=300.0, cy=240.0, emb=None):
    k = _kps(roll_deg, cx, cy)
    return Face(bbox=np.array([cx - 70, cy - 70, cx + 70, cy + 70], np.float32), kps=k,
                det_score=0.95, embedding=emb if emb is not None else np.ones(512, np.float32) / 22.6)


class TrackerRollHint(unittest.TestCase):

    def _tracker_after(self, rolls, frames=1):
        tr = TemporalFaceTracker.from_env()
        for i in range(frames):
            tr.update([_face(r) for r in rolls], i, (480, 640, 3), detection_mode="full")
        return tr

    def test_a_sideways_track_plans_a_derotated_roi(self):
        tr = self._tracker_after([90.0], frames=3)
        plan = tr.plan(3, (480, 640, 3))
        self.assertEqual(plan.mode, "roi")
        self.assertEqual(plan.rotation_action, CW)

    def test_each_quarter_turn(self):
        for roll, want in ((90.0, CW), (-90.0, ACW), (180.0, HALF), (-178.0, HALF)):
            tr = self._tracker_after([roll], frames=3)
            self.assertEqual(tr.plan(3, (480, 640, 3)).rotation_action, want, roll)

    def test_an_upright_track_has_no_hint(self):
        for roll in (0.0, 20.0, -35.0):
            tr = self._tracker_after([roll], frames=3)
            self.assertIsNone(tr.plan(3, (480, 640, 3)).rotation_action, roll)

    def test_tracks_that_disagree_get_no_hint(self):
        tr = TemporalFaceTracker.from_env()
        for i in range(3):
            tr.update([_face(90.0, cx=150.0), _face(0.0, cx=500.0)], i, (480, 640, 3),
                      detection_mode="full")
        plan = tr.plan(3, (480, 640, 3))
        self.assertEqual(plan.mode, "roi")
        self.assertIsNone(plan.rotation_action)

    def test_hysteresis_holds_the_turn_near_the_boundary(self):
        tr = TemporalFaceTracker.from_env()
        for i, roll in enumerate((90.0, 90.0, 50.0, 48.0)):     # 50 > 36: still tilted right
            tr.update([_face(roll)], i, (480, 640, 3), detection_mode="full")
        self.assertEqual(tr.plan(4, (480, 640, 3)).rotation_action, CW)
        tr.update([_face(30.0)], 4, (480, 640, 3), detection_mode="full")  # < 36: back upright
        self.assertIsNone(tr.plan(5, (480, 640, 3)).rotation_action)

    def test_the_switch_restores_the_upright_first_roi(self):
        tr = self._tracker_after([90.0], frames=3)
        with mock.patch.object(temporal_tracker, "ROI_DEROTATE", False):
            self.assertIsNone(tr.plan(3, (480, 640, 3)).rotation_action)


class DerotatedRoi(unittest.TestCase):
    """A faithful upright-biased detector.

    The WORLD face has a true roll. The detector is shown whatever crop it is handed; the
    face's keypoints in that crop are the world keypoints put through the SAME transform
    the turn applies (np.rot90 equivalents), and it reports a face only while that face is
    within `reach` degrees of upright in the crop it sees. `reach` 45 models a normal
    detector (a sideways face is simply missed); a large reach models one that still
    returns a face with its true, sideways geometry.
    """

    def setUp(self):
        self._enrich = mock.patch("roop.face_util._enrich_detected_faces",
                                  side_effect=lambda frame, faces: faces)
        self._enrich.start()
        self.frame = np.zeros((480, 640, 3), np.uint8)
        self.bbox = (260, 180, 380, 300)
        crop, self.cx1, self.cy1 = face_util._roi_window(self.frame, self.bbox, 1.0, 160)
        self.ch, self.cw = crop.shape[:2]
        self.turn = None

    def tearDown(self):
        self._enrich.stop()

    def _stubs(self, world_roll, reach=45.0):
        H, W = self.ch, self.cw
        wk = _kps(world_roll, W / 2.0, H / 2.0).astype(np.float64)
        seen = []

        def mk(name, real):
            def wrapped(a):
                self.turn = name
                return real(a)
            return wrapped

        def to_turned(k):
            x, y = k[:, 0], k[:, 1]
            if self.turn == "cw":
                return np.stack([H - 1 - y, x], 1)
            if self.turn == "acw":
                return np.stack([y, W - 1 - x], 1)
            if self.turn == "180":
                return np.stack([W - 1 - x, H - 1 - y], 1)
            return k

        def raw(img, **kw):
            seen.append(self.turn)
            k = to_turned(wk).astype(np.float32)
            self.turn = None        # the next call is a fresh look unless a turn sets it again
            roll_here = orientation.roll_from_kps(k)
            if roll_here is None or abs(orientation.wrap180(roll_here)) > reach:
                return []
            h, w = img.shape[:2]
            c = k.mean(axis=0)
            return [Face(bbox=np.array([c[0] - 70, c[1] - 70, c[0] + 70, c[1] + 70], np.float32),
                         kps=k, det_score=0.97, embedding=np.ones(512, np.float32) / 22.6)]

        patches = [mock.patch("roop.face_util._detect_faces_raw", side_effect=raw),
                   mock.patch("roop.face_util.rotate_clockwise", mk("cw", face_util.rotate_clockwise)),
                   mock.patch("roop.face_util.rotate_anticlockwise", mk("acw", face_util.rotate_anticlockwise)),
                   mock.patch("roop.face_util.rotate_image_180", mk("180", face_util.rotate_image_180))]
        return patches, seen

    def _run(self, fn, world_roll, reach=45.0):
        patches, seen = self._stubs(world_roll, reach)
        for p in patches:
            p.start()
        try:
            self.turn = None
            return fn(), seen
        finally:
            for p in patches:
                p.stop()

    def test_a_correct_hint_finds_the_face_where_the_picture_has_it(self):
        for roll, action in ((90.0, CW), (-90.0, ACW), (180.0, HALF)):
            faces, seen = self._run(lambda: face_util.get_all_faces_in_roi(
                self.frame, self.bbox, rescue=False, rotation_action=action), roll)
            self.assertEqual(len(faces), 1, roll)
            self.assertEqual(len(seen), 1, "one hinted pass, no fallback needed")
            self.assertLess(abs(orientation.angdiff(orientation.roll_from_face(faces[0]), roll)), 1.5,
                            "un-rotated back to the roll the frame shows")
            want = _kps(roll, self.cw / 2.0, self.ch / 2.0).mean(axis=0) + [self.cx1, self.cy1]
            np.testing.assert_allclose(faces[0].kps.mean(axis=0), want, atol=1.5)   # frame, not crop, coordinates

    def test_a_wrong_hint_on_a_normal_detector_falls_back_to_the_upright_roi(self):
        # the head is upright, the track said CW: the turned crop shows it sideways -> missed
        faces, seen = self._run(lambda: face_util.get_all_faces_in_roi(
            self.frame, self.bbox, rescue=False, rotation_action=CW), 0.0)
        self.assertEqual(len(faces), 1, "the plain pass still finds it")
        self.assertEqual(seen, ["cw", None], "turned look first (missed), then the plain crop")

    def test_a_sideways_geometry_detection_that_disagrees_with_the_hint_is_rejected(self):
        # a tolerant detector returns the TRUE (upright) face from the CW-turned crop, rolled -90
        # there; once un-rotated it is roll 0 != the CW hint, so it must not be accepted.
        crop = self.frame[self.cy1:self.cy1 + self.ch, self.cx1:self.cx1 + self.cw]
        out, _ = self._run(lambda: face_util._roi_faces_derotated(crop, CW), 0.0, reach=100.0)
        self.assertEqual(out, [])

    def test_hallucinated_upright_geometry_is_NOT_caught_by_the_roll_check(self):
        """The limit of this guard, stated rather than hidden. A detector that returns an
        upright-LAYOUT face on the turned crop (the 'succeeds badly' failure) un-rotates to a
        roll equal to the hint, so it agrees with it. The check catches a face with real
        sideways geometry; it cannot catch an invented upright one."""
        patches = [mock.patch("roop.face_util._detect_faces_raw",
                              return_value=[_face(0.0, cx=self.cw / 2.0, cy=self.ch / 2.0)])]
        for p in patches:
            p.start()
        try:
            crop = self.frame[self.cy1:self.cy1 + self.ch, self.cx1:self.cx1 + self.cw]
            self.assertEqual(len(face_util._roi_faces_derotated(crop, CW)), 1)
        finally:
            for p in patches:
                p.stop()

    def test_no_hint_is_exactly_one_plain_call(self):
        faces, seen = self._run(lambda: face_util.get_all_faces_in_roi(
            self.frame, self.bbox, rescue=False), 0.0)
        self.assertEqual(seen, [None])
        self.assertEqual(len(faces), 1)

    def test_an_empty_hinted_pass_falls_through_to_the_old_path(self):
        faces, seen = self._run(lambda: face_util.get_all_faces_in_roi(
            self.frame, self.bbox, rescue=False, rotation_action=HALF), 90.0)
        # world roll 90: the 180-turned crop shows -90 (missed); the plain crop shows +90 (missed too)
        self.assertEqual(faces, [])
        self.assertEqual(seen, ["180", None])


class PrepassPassesTheHint(unittest.TestCase):
    """Drive the real `_precompute_tracks`: after a sideways face is tracked, the ROI call
    must carry `rotation_action`, and an upright clip must never carry one."""

    class _Opts:
        face_distance_threshold = 0.75
        selected_index = 0

    class _Mgr(TrackingMixin):
        def __init__(self):
            self.target_face_datas = []
            self.target_face_groups = []
            self.options = PrepassPassesTheHint._Opts()
            self.progress_gradio = None
            self._track_assignments = {}
            self._track_scanned = 0

        def _publish_live(self, frame):
            pass

    def _scan(self, roll, n=14):
        frames = []
        for i in range(n):
            f = np.zeros((480, 640, 3), np.uint8)
            f[0, 0, 0] = i
            frames.append(f)
        emb = np.ones(512, np.float32) / 22.6
        roi_kwargs = []

        def full(frame, *a, **kw):
            return [_face(roll, emb=emb)]

        def roi(frame, bbox, *a, **kw):
            roi_kwargs.append(dict(kw))
            return [_face(roll, emb=emb)]

        was = roop.globals.processing
        roop.globals.processing = True
        try:
            with mock.patch("roop.face_util.get_all_faces", full), \
                    mock.patch("roop.face_util.get_all_faces_in_roi", roi):
                self._Mgr()._precompute_tracks(None, 0, n, n, awebp_frames=frames,
                                               step=1, collect_obs=True)
        finally:
            roop.globals.processing = was
        return roi_kwargs

    def test_a_sideways_clip_hands_the_roi_detector_its_turn(self):
        calls = self._scan(90.0)
        self.assertTrue(calls, "premise: the scan reached the ROI path")
        self.assertEqual(calls[-1].get("rotation_action"), CW, calls)

    def test_an_upright_clip_never_does(self):
        calls = self._scan(0.0)
        self.assertTrue(calls)
        self.assertTrue(all("rotation_action" not in c for c in calls), calls)


if __name__ == "__main__":
    unittest.main()
