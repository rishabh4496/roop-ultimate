"""Head pose: one type, one adapter, one convention across the source bank.

Four pose representations are live (see roop/head_pose.py). Two defects came from
mixing them, both on the multi-face (V1 .fsz) source-bank path:

  * `FaceSet.canonical_face_index` ranked faces by min(yaw^2 + pitch^2) over
    `face_poses`, which on a V1 set are EPnP angles where a frontal head reads
    ~+-170 degrees. It returned the PROFILE photograph. Measured on the banks in
    roop-keep: riya kansal frontal 175.6 / profile 99.8; akansha frontal -170.4 /
    profile -89.3; the old test fixture's "frontal 2.6" was never what production
    computed, so it passed while the real thing was wrong.
  * The tracking pre-pass asked the bank for a TARGET pose through
    `getattr(target_p, 'yaw', 0.0)`; with temporal identity on, `target_p` is a
    dict, so every frame asked for a frontal face.

Poses below come from projecting the project's own reference head
(`face_util._project_reference`), not from typed-in numbers.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insightface.app.common import Face                              # noqa: E402
from roop.FaceSet import FaceSet                                     # noqa: E402
from roop.face_util import _project_reference, solve_pose_5pt        # noqa: E402
from roop.head_pose import (CANONICAL_ORDER, INSIGHTFACE_ORDER, HeadPose,  # noqa: E402
                            head_pose_from_kps, normalize_pose_tuple)
from roop.procmgr_tracking import TrackingMixin                      # noqa: E402
from roop.pose_source_selector import PoseEstimate, estimate_target_pose  # noqa: E402
from roop.temporal_identity import TemporalIdentityStabilizer        # noqa: E402


def _kps(yaw, pitch=0.0, cx=400.0, cy=300.0, scale=120.0):
    """5 keypoints of the reference head at (yaw, pitch), placed in an image."""
    return (_project_reference(yaw, pitch) * scale + np.array([cx, cy])).astype(np.float32)


def _emb(seed):
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def _face(kps, emb=None):
    k = np.asarray(kps, np.float32)
    lo, hi = k.min(axis=0) - 40, k.max(axis=0) + 40
    return Face(bbox=np.array([lo[0], lo[1], hi[0], hi[1]], np.float32), kps=k,
                det_score=0.9, embedding=_emb(1) if emb is None else emb)


class NormalizePoseTuple(unittest.TestCase):

    def test_canonical_tuple_keeps_its_order(self):
        hp = normalize_pose_tuple((30.0, -5.0, 2.0))
        self.assertEqual(tuple(hp), (30.0, -5.0, 2.0))

    def test_insightface_order_is_permuted_into_yaw_pitch_roll(self):
        # face.pose is (pitch, yaw, roll)
        hp = normalize_pose_tuple(np.array([-5.0, 30.0, 2.0], np.float32), INSIGHTFACE_ORDER)
        self.assertEqual((hp.yaw, hp.pitch, hp.roll), (30.0, -5.0, 2.0))

    def test_the_order_is_never_guessed_from_magnitudes(self):
        a = normalize_pose_tuple((10.0, 40.0, 0.0), CANONICAL_ORDER)
        b = normalize_pose_tuple((10.0, 40.0, 0.0), INSIGHTFACE_ORDER)
        self.assertEqual((a.yaw, a.pitch), (10.0, 40.0))
        self.assertEqual((b.yaw, b.pitch), (40.0, 10.0))

    def test_two_tuples_in_either_order(self):
        self.assertEqual(tuple(normalize_pose_tuple((12.0, 3.0), "ypr")), (12.0, 3.0, 0.0))
        self.assertEqual(tuple(normalize_pose_tuple((3.0, 12.0), "pyr")), (12.0, 3.0, 0.0))

    def test_jaw_solver_four_tuple_keeps_its_first_three(self):
        hp = normalize_pose_tuple((20.0, 4.0, 1.0, 0.3))
        self.assertEqual(tuple(hp), (20.0, 4.0, 1.0))

    def test_dict_and_objects_name_their_fields(self):
        d = normalize_pose_tuple({"yaw": 33.0, "pitch": -2.0, "roll": 1.0, "confidence": 0.5})
        self.assertEqual((d.yaw, d.pitch, d.roll, d.confidence), (33.0, -2.0, 1.0, 0.5))
        est = PoseEstimate(yaw=-41.0, pitch=7.0, roll=3.0, face_scale=80.0, relative_scale=0.1,
                           proportions={}, expression={}, confidence=0.9, off_axis=0.0,
                           perspective_risk=0.0, inverted=False, available=True)
        o = normalize_pose_tuple(est)
        self.assertEqual((o.yaw, o.pitch, o.roll), (-41.0, 7.0, 3.0))
        # the dict form the temporal-identity stabilizer stores
        self.assertEqual(normalize_pose_tuple(est.as_dict()).yaw, -41.0)

    def test_head_pose_passes_through(self):
        hp = HeadPose(1.0, 2.0, 3.0)
        self.assertIs(normalize_pose_tuple(hp), hp)

    def test_unusable_input_is_none_not_an_exception(self):
        for bad in (None, (), (1.0,), "abc", (float("nan"), 0.0, 0.0), {"roll": 3.0},
                    object(), [[1.0, 2.0], [3.0, 4.0], [5.0]]):
            self.assertIsNone(normalize_pose_tuple(bad), bad)
        self.assertIsNone(normalize_pose_tuple((1.0, 2.0, 3.0), order="xyz"))
        self.assertIsNone(normalize_pose_tuple((1.0, 2.0, 3.0), order="yp"))


class HeadPoseType(unittest.TestCase):

    def test_unpack_index_slice_len_and_keys(self):
        hp = HeadPose(10.0, -4.0, 1.5, confidence=0.8)
        yaw, pitch, roll = hp
        self.assertEqual((yaw, pitch, roll), (10.0, -4.0, 1.5))
        self.assertEqual((hp[0], hp[1], hp[2], hp[-1]), (10.0, -4.0, 1.5, 1.5))
        self.assertEqual(hp[:2], (10.0, -4.0))
        self.assertEqual(len(hp), 3)
        self.assertEqual(hp["pitch"], -4.0)
        self.assertEqual(hp.get("nope", "d"), "d")
        self.assertEqual(hp.pitch_up, 4.0)
        np.testing.assert_allclose(np.asarray(hp, dtype=float), [10.0, -4.0, 1.5])

    def test_a_consumer_written_for_the_solvers_tuple_accepts_it(self):
        # face_quality / pose_quality style: len(pose) >= 3 and pose[0], pose[1]
        hp = head_pose_from_kps(_kps(35.0, -10.0))
        self.assertTrue(len(hp) >= 3)
        self.assertAlmostEqual(hp[0], 35.0, delta=0.5)
        self.assertAlmostEqual(hp[1], -10.0, delta=0.5)

    def test_from_kps_matches_the_solver_and_keeps_the_points(self):
        kps = _kps(-52.0, 12.0)
        hp = head_pose_from_kps(kps, jaw_aware=False)
        ref = solve_pose_5pt(kps)
        self.assertEqual((hp.yaw, hp.pitch, hp.roll), tuple(float(v) for v in ref))
        self.assertEqual(hp.landmarks_5pt.shape, (5, 2))
        self.assertIsNone(head_pose_from_kps(None))
        self.assertIsNone(head_pose_from_kps(np.zeros((5, 2), np.float32)))


# Poses of the real "riya kansal" bank, as the production code computes them:
# ensure_face_poses' EPnP angles (frontal = 175.6) beside solve_pose_5pt's yaw
# for the same five photographs (frontal = 1.0). Measured 2026-10-19.
RIYA_EPNP = [(99.8, -11.1), (140.2, 38.2), (175.6, -7.4), (138.9, 37.6), (100.8, -10.2)]
RIYA_SOLVER_YAW_PITCH = [(84.6, -5.7), (39.0, -18.0), (1.0, -17.1), (40.1, -17.4), (83.5, -6.4)]


def _riya_bank(with_epnp=True):
    fs = FaceSet()
    fs.faces = [_face(_kps(y, p), emb=_emb(i)) for i, (y, p) in enumerate(RIYA_SOLVER_YAW_PITCH)]
    if with_epnp:
        fs.face_poses = list(RIYA_EPNP)
    return fs


class V1BankUsesOneConvention(unittest.TestCase):

    def test_epnp_poses_rank_a_profile_as_the_most_frontal(self):
        """Why `face_poses` cannot answer 'which is frontal': this is the old rule."""
        naive = min(range(5), key=lambda i: RIYA_EPNP[i][0] ** 2 + RIYA_EPNP[i][1] ** 2)
        self.assertNotEqual(naive, 2)
        self.assertGreater(abs(RIYA_SOLVER_YAW_PITCH[naive][0]), 80.0)

    def test_canonical_face_index_is_the_frontal_photograph(self):
        fs = _riya_bank()
        self.assertEqual(fs.canonical_face_index(), 2)

    def test_an_averaged_v1_set_keeps_the_mean_identity_in_face_zero(self):
        """`AverageEmbeddings` puts the mean in faces[0]; that face is the swap source."""
        fs = _riya_bank()
        mean_before = np.mean([f.embedding for f in fs.faces], axis=0)
        fs.AverageEmbeddings()
        np.testing.assert_allclose(fs.faces[0].embedding, mean_before, rtol=1e-5)
        self.assertEqual(fs.canonical_face_index(), 0)
        # ...while the pose-aware bank lookup still sees the frontal photograph
        self.assertEqual(fs.select_reference_index_canonical((0.0, -17.0)), 2)

    def test_canonical_poses_recover_the_projected_angles(self):
        fs = _riya_bank()
        for (ey, ep), (gy, gp) in zip(fs.canonical_poses(), RIYA_SOLVER_YAW_PITCH):
            self.assertAlmostEqual(ey, gy, delta=0.6)
            self.assertAlmostEqual(ep, gp, delta=0.6)

    def test_epnp_face_poses_are_left_alone_for_the_swap_time_lookup(self):
        fs = _riya_bank()
        fs.canonical_face_index()
        fs.select_reference_index_canonical((0.0, 0.0))
        self.assertEqual(fs.face_poses, RIYA_EPNP)

    def test_canonical_selection_follows_the_target_pose(self):
        fs = _riya_bank()
        self.assertEqual(fs.select_reference_index_canonical((0.0, -17.0)), 2)
        self.assertEqual(fs.select_reference_index_canonical((80.0, -6.0)), 4)  # 83.5 vs 84.6 -> nearest
        self.assertIn(fs.select_reference_index_canonical((39.0, -18.0)), (1, 3))

    def test_canonical_selection_keeps_the_hysteresis_margin(self):
        fs = _riya_bank()
        # 20 deg is closer to the 39 deg photo than to the frontal one, but not by 15 deg
        self.assertEqual(fs.select_reference_index_canonical((20.0, -17.0), previous_index=2), 2)
        self.assertEqual(fs.select_reference_index_canonical((45.0, -17.0), previous_index=2), 3)

    def test_the_old_selector_given_a_canonical_target_picks_the_profile(self):
        """The defect, kept as a documented contrast: EPnP bank vs canonical target."""
        fs = _riya_bank()
        self.assertNotEqual(fs.select_reference_index(pose=(0.0, -17.0)), 2)

    def test_a_hand_built_set_without_keypoints_keeps_its_legacy_behaviour(self):
        fs = FaceSet()
        fs.faces = [0, 1, 2, 3, 4]
        fs.face_poses = [(82.3, 31.9), (34.2, 3.8), (2.6, 1.1), (35.3, 4.0), (81.4, 25.9)]
        self.assertEqual(fs.canonical_face_index(), 2)

    def test_v2_sets_use_their_metadata_poses_unchanged(self):
        fs = FaceSet()
        fs.faces = [object(), object(), object()]
        fs.format_version = 2
        fs.faceset_metadata = {"sources": []}
        fs.face_poses = [(60.0, 0.0), (1.0, 0.0), (-45.0, 0.0)]
        self.assertEqual(fs.canonical_poses(), fs.face_poses)
        self.assertEqual(fs.canonical_face_index(), 1)


class _Options:
    stabilize_face = False
    stabilize_method = 'one_euro'
    stabilize_min_cutoff = 0.05
    stabilize_beta = 0.02
    use_source_bank = True


class _Mgr(TrackingMixin):
    def __init__(self, bank, identity=False):
        self.options = _Options()
        self.input_face_datas = [bank]
        self._track_source_map = {0: (0, _emb(1))}
        self._track_pose_source_map = {}
        self._temporal_frame_shape = (720, 1280, 3)
        if identity:
            self._temporal_identity = TemporalIdentityStabilizer(enabled=True)


def _pose_track(yaw, pitch, frames=12):
    emb = _emb(1)
    obs = {i: _face(_kps(yaw, pitch), emb) for i in range(frames)}
    return {'id': 0, 'obs': obs, 'emb_mean': emb, 'emb_n': len(obs)}


class PrepassPicksTheRightBankPhoto(unittest.TestCase):
    """`_build_temporal_faces` is the only caller of the V1 branch; drive it."""

    def _selected(self, yaw, pitch, identity):
        out = _Mgr(_riya_bank(), identity=identity)._build_temporal_faces(
            [_pose_track(yaw, pitch)], gap_max=4)
        picks = [f.get('_temporal_source_index') for faces in out.values() for f in faces]
        self.assertEqual(len(picks), 12, picks)
        self.assertNotIn(None, picks, "the V1 branch did not run -- test proves nothing")
        return set(picks)

    def test_frontal_target_gets_the_frontal_photo(self):
        self.assertEqual(self._selected(0.0, -17.0, identity=False), {2})

    def test_turned_target_gets_a_turned_photo(self):
        self.assertTrue(self._selected(38.0, -17.0, identity=False) <= {1, 3})

    def test_with_temporal_identity_on_the_dict_pose_is_read(self):
        # The old getattr(dict, 'yaw', 0.0) asked for yaw 0 here: a frontal photo,
        # for a head turned 38 degrees.
        mgr = _Mgr(_riya_bank(), identity=True)
        out = mgr._build_temporal_faces([_pose_track(38.0, -17.0)], gap_max=4)
        faces = [f for fs in out.values() for f in fs]
        self.assertTrue(all(isinstance(f.get('_pose_v5'), dict) for f in faces),
                        "temporal identity did not publish a dict pose -- test proves nothing")
        self.assertTrue({f.get('_temporal_source_index') for f in faces} <= {1, 3})

    def test_the_estimate_the_branch_reads_is_in_the_banks_convention(self):
        est = estimate_target_pose(_face(_kps(38.0, -17.0)))
        self.assertAlmostEqual(est.yaw, 38.0, delta=1.0)


class _SmoothOptions(_Options):
    stabilize_face = True
    stabilize_landmarks = True
    use_source_bank = False


class _SmoothMgr(TrackingMixin):
    def __init__(self, cuts=()):
        self.options = _SmoothOptions()
        self.input_face_datas = []
        self._track_source_map = {}
        self._track_pose_source_map = {}
        self._temporal_frame_shape = (720, 1280, 3)
        self._shot_boundaries = set(cuts)


def _smoothed(kps_by_frame, cuts=()):
    emb = _emb(3)
    obs = {i: _face(k, emb) for i, k in kps_by_frame.items()}
    track = {'id': 0, 'obs': obs, 'emb_mean': emb, 'emb_n': len(obs)}
    out = _SmoothMgr(cuts)._build_temporal_faces([track], gap_max=4)
    return {i: np.asarray(faces[0]['kps'], np.float64) for i, faces in out.items()}


class SmoothingBehaviour(unittest.TestCase):
    """Stage 6 asks: still faces stay jitter-free, fast turns do not lag, a cut resets."""

    def test_a_stationary_face_loses_most_of_its_detector_jitter(self):
        rng = np.random.default_rng(7)
        base = _kps(0.0)
        raw = {i: base + rng.normal(scale=1.0, size=(5, 2)).astype(np.float32) for i in range(80)}
        out = _smoothed(raw)
        tail = range(20, 80)
        sd_in = np.std([raw[i] - base for i in tail])
        sd_out = np.std([out[i] - base for i in tail])
        self.assertLess(sd_out, 0.6 * sd_in, (sd_in, sd_out))

    def test_a_fast_translation_is_followed_without_visible_lag(self):
        raw = {i: _kps(0.0, cx=200.0 + 15.0 * i) for i in range(50)}
        out = _smoothed(raw)
        lag = np.abs(out[49] - raw[49].astype(np.float64)).max()
        self.assertLess(lag, 3.0, lag)

    def test_a_registered_cut_resets_the_filter(self):
        left, right = _kps(0.0, cx=300.0), _kps(0.0, cx=900.0)
        raw = {i: (left if i < 20 else right) for i in range(40)}
        with_cut = _smoothed(raw, cuts={20})
        without = _smoothed(raw, cuts=())
        np.testing.assert_allclose(with_cut[20], right.astype(np.float64), atol=1e-3)
        # without the reset the filter drags 10% of the old position into the new shot
        self.assertGreater(np.abs(without[20] - right).max(), 30.0)


if __name__ == '__main__':
    unittest.main()


class TargetRankIsStableAcrossAngles(unittest.TestCase):
    """Several captures of one person (front, 3/4, profile) are ONE display rank,
    and capturing another angle of an existing person leaves everyone's rank alone."""

    def setUp(self):
        import api
        import roop.globals as g
        self.api, self.g = api, g
        self.saved = {k: list(getattr(g, k, []) or [])
                      for k in ('TARGET_FACES', 'TARGET_FACE_GROUP',
                                'TARGET_FACE_PERSON_IDS', 'TARGET_REFERENCE_FACE_IDS')}

    def tearDown(self):
        for k, v in self.saved.items():
            getattr(self.g, k)[:] = v

    def _ranks(self, people):
        class _F:
            def __init__(self, x):
                self.bbox = np.array([x, 1, x + 4, 8], np.float32)
                self.embedding = np.ones(512, np.float32)
                self.kps = None

            def __getitem__(self, key):
                return getattr(self, key)

        names = sorted(set(people))
        self.g.TARGET_FACES[:] = [_F(10 * i) for i in range(len(people))]
        self.g.TARGET_FACE_GROUP[:] = [names.index(p) for p in people]
        self.g.TARGET_FACE_PERSON_IDS[:] = list(people)
        return self.api._target_groups_ranked()

    def test_all_angles_of_one_person_share_a_rank(self):
        ranks = self._ranks(['pA', 'pB', 'pA', 'pB', 'pA'])
        self.assertEqual(ranks, [0, 1, 0, 1, 0])

    def test_a_new_angle_of_an_existing_person_does_not_move_anyone(self):
        before = self._ranks(['pA', 'pB'])
        after = self._ranks(['pA', 'pB', 'pA'])
        self.assertEqual(after[:2], before)
        self.assertEqual(self._ranks(['pA', 'pB', 'pA']), after)   # and it is repeatable
