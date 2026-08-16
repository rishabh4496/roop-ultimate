"""The expression-restore geometry, tested without the models.

Expression_LivePortrait keeps its maths in module-level functions precisely so
this suite can run: no 537 MB download, no GPU, no ONNX session. What is asserted
here is the part that would be silently wrong rather than loudly broken — a sign
error in the rotation, a strength that does not reduce to a no-op, or a transfer
that quietly moves the head instead of only the mouth.
"""

import contextlib
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.processors.Expression_LivePortrait import (  # noqa: E402
    EYE_INDICES, LIP_INDICES, NUM_BINS, blend_expression, concat_feat,
    driving_keypoints, get_rotation_matrix, headpose_pred_to_degree,
    transform_keypoint,
)

RNG = np.random.default_rng(11)


@contextlib.contextmanager
def _env(**pairs):
    """Set env vars for the duration of a test and put them back."""
    old = {k: os.environ.get(k) for k in pairs}
    os.environ.update({k: str(v) for k, v in pairs.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Named:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    """Enough of an InferenceSession to exercise the call plumbing offline.

    Records every image it was fed, so a test can assert WHICH crop reached
    which model — feeding the target crop where the swapped one belongs applies
    the expression backwards and still produces a plausible-looking face.
    """

    def __init__(self, in_names, outputs, out_names=("output",), delay=0.0,
                 raises=None):
        self._in = list(in_names)
        self._out_names = list(out_names)
        self._outputs = outputs
        self._delay = delay
        self._raises = raises
        self.seen = []

    def get_inputs(self):
        return [_Named(n) for n in self._in]

    def get_outputs(self):
        return [_Named(n) for n in self._out_names]

    def run(self, _out, feeds):
        if self._delay:
            time.sleep(self._delay)
        self.seen.append(feeds)
        if self._raises is not None:
            raise self._raises
        return self._outputs


def _fake_motion_outputs(tag):
    """The seven declared outputs, with `tag` written into the translation so a
    test can tell one motion session's result from another's."""
    return [np.zeros((1, 66), np.float32),      # pitch
            np.zeros((1, 66), np.float32),      # yaw
            np.zeros((1, 66), np.float32),      # roll
            np.full((1, 3), tag, np.float32),   # t
            np.zeros((1, 63), np.float32),      # exp
            np.ones((1, 1), np.float32),        # scale
            np.zeros((1, 63), np.float32)]      # kp


def _fake_sessions(motion2=False, appearance_raises=None, motion2_delay=0.0):
    s = {
        "appearance": _FakeSession(["img"], [np.zeros((1, 32, 16, 64, 64), np.float32)],
                                   raises=appearance_raises),
        "motion": _FakeSession(["img"], _fake_motion_outputs(1.0)),
        "stitching": _FakeSession(["input"], [np.zeros((1, 65), np.float32)]),
        "warping": _FakeSession(["feature_3d", "kp_driving", "kp_source"],
                                [np.zeros((1, 3, 512, 512), np.float32)],
                                out_names=["out"]),
    }
    if motion2:
        s["motion2"] = _FakeSession(["img"], _fake_motion_outputs(2.0),
                                    delay=motion2_delay)
    return s


class TestPoolingContract(unittest.TestCase):
    """How the restorer excludes concurrent threads, without loading models.

    ProcessMgr decides whether to hold the global GPU lock from
    `self_excluding`. If those two ever disagree the failure is not a wrong
    pixel — it is either a stage silently running single-file (the bug this
    exists to prevent) or two threads sharing one TensorRT context, which
    corrupts the CUDA context.
    """

    def _restorer(self):
        from roop.processors.Expression_LivePortrait import Expression_LivePortrait
        return Expression_LivePortrait()

    def test_unpooled_by_default_and_leases_the_single_set(self):
        r = self._restorer()
        r.sessions = {"warping": object()}
        self.assertFalse(r.pooled)
        with r._leased() as sessions:
            self.assertIs(sessions, r.sessions)

    def test_skips_the_global_lock_even_without_a_pool(self):
        """_leased() is the exclusion, pooled or not — so the global lock is
        redundant either way, and holding it would serialise this stage against
        unrelated ones."""
        r = self._restorer()
        self.assertFalse(r.pooled)
        self.assertTrue(r.self_excluding)

    def test_global_lock_can_be_forced_back_on(self):
        r = self._restorer()
        with _env(ROOP_EXPR_GLOBAL_LOCK='1'):
            self.assertFalse(r.self_excluding)

    def test_unpooled_lease_is_exclusive(self):
        """The property the dropped global lock now rests on: two threads cannot
        be inside the single session set at the same time."""
        import threading
        r = self._restorer()
        r.sessions = {"warping": object()}
        overlapped = []
        inside = threading.Event()
        released = threading.Event()

        def hold():
            with r._leased():
                inside.set()
                released.wait(2.0)

        t = threading.Thread(target=hold)
        t.start()
        self.assertTrue(inside.wait(2.0))
        second = threading.Thread(
            target=lambda: overlapped.append(True) if r._lock.acquire(timeout=0.2)
            else None)
        second.start()
        second.join()
        released.set()
        t.join()
        self.assertEqual(overlapped, [], "a second thread entered the lease")

    def test_pooled_leases_distinct_sets_and_returns_them(self):
        from roop.session_pool import SessionPool
        r = self._restorer()
        r.pool = SessionPool(lambda i: {"slot": i}, 2)
        self.assertTrue(r.pooled)
        with r._leased() as a:
            with r._leased() as b:
                self.assertNotEqual(a["slot"], b["slot"])
        # Both must be back in the queue, or the pool deadlocks on reuse.
        with r._leased():
            with r._leased():
                pass

    def test_release_drops_the_pool(self):
        from roop.session_pool import SessionPool
        r = self._restorer()
        r.pool = SessionPool(lambda i: {"slot": i}, 2)
        r.Release()
        self.assertIsNone(r.pool)
        self.assertFalse(r.pooled)


class TestLockSplit(unittest.TestCase):
    """prepare/infer/finish exist so ProcessMgr can hold the global GPU lock
    around `infer` alone. The two ends must therefore be GPU-free and must
    degrade to the untouched crop, because ProcessMgr calls them unconditionally
    — including on the path where infer() declined or failed inside the lock."""

    def _restorer(self):
        from roop.processors.Expression_LivePortrait import Expression_LivePortrait
        return Expression_LivePortrait()

    def test_prepare_needs_no_sessions_and_yields_two_net_inputs(self):
        from roop.processors.Expression_LivePortrait import INPUT_SIZE
        r = self._restorer()                       # never Initialize()d: no models
        crop = np.zeros((512, 512, 3), np.uint8)
        src, drv = r.prepare(crop, crop)
        for t in (src, drv):
            self.assertEqual(t.shape, (1, 3, INPUT_SIZE, INPUT_SIZE))
            self.assertEqual(t.dtype, np.float32)

    def test_prepare_declines_missing_crops(self):
        r = self._restorer()
        self.assertIsNone(r.prepare(None, np.zeros((8, 8, 3), np.uint8)))
        self.assertIsNone(r.prepare(np.zeros((8, 8, 3), np.uint8), None))

    def test_infer_declines_without_running_anything(self):
        r = self._restorer()
        self.assertIsNone(r.infer(None, 1.0))
        self.assertIsNone(r.infer(("x", "y"), 0.0))   # strength 0 = no-op

    def test_finish_returns_the_input_when_infer_declined(self):
        r = self._restorer()
        crop = RNG.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = r.finish(None, crop)
        np.testing.assert_array_equal(out, crop)

    def test_finish_resizes_the_net_output_back_to_the_crop(self):
        r = self._restorer()
        crop = np.zeros((97, 61, 3), np.uint8)        # deliberately not 512
        raw = RNG.random((1, 3, 512, 512)).astype(np.float32)
        out = r.finish(raw, crop)
        self.assertEqual(out.shape, crop.shape)

    def test_non_finite_output_falls_back_to_the_input(self):
        """A NaN from the warp must cost the expression, never the frame."""
        r = self._restorer()
        crop = RNG.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        raw = np.full((1, 3, 512, 512), np.nan, np.float32)
        np.testing.assert_array_equal(r.finish(raw, crop), crop)


class TestOverlappedFrontHalf(unittest.TestCase):
    """appearance / motion(swapped) / motion(target) are issued together.

    Overlapping them must not change WHICH crop reaches which model. The two
    motion calls are indistinguishable at the type level — both take a 256x256
    tensor and return the same seven outputs — so a swapped pair produces a
    perfectly plausible face with the expression applied backwards, which no
    crash and no shape check would catch.
    """

    def _restorer(self, sessions, workers=2):
        import concurrent.futures
        from roop.processors.Expression_LivePortrait import Expression_LivePortrait
        r = Expression_LivePortrait()
        r.sessions = sessions
        r._exec = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self.addCleanup(r._exec.shutdown, True)
        return r

    def _inputs(self):
        src = np.zeros((1, 3, 256, 256), np.float32)
        drv = np.ones((1, 3, 256, 256), np.float32)
        return src, drv

    def test_sequential_and_overlapped_agree(self):
        src, drv = self._inputs()
        seq = self._restorer(_fake_sessions())
        seq._exec = None
        par = self._restorer(_fake_sessions(motion2=True))

        f_a, m_sa, m_da = seq._front_half(seq.sessions, src, drv)
        f_b, m_sb, m_db = par._front_half(par.sessions, src, drv)

        np.testing.assert_array_equal(f_a, f_b)
        for a, b in zip(m_sa, m_sb):
            np.testing.assert_array_equal(a, b)
        # Only the driving EXPRESSION is read downstream, and the two motion
        # sessions hold identical weights, so these must agree too.
        np.testing.assert_array_equal(m_da[4], m_db[4])

    def test_each_model_gets_the_crop_it_is_supposed_to(self):
        src, drv = self._inputs()
        r = self._restorer(_fake_sessions(motion2=True))
        r._front_half(r.sessions, src, drv)
        np.testing.assert_array_equal(r.sessions["appearance"].seen[0]["img"], src)
        np.testing.assert_array_equal(r.sessions["motion"].seen[0]["img"], src)
        np.testing.assert_array_equal(r.sessions["motion2"].seen[0]["img"], drv)

    def test_without_a_second_motion_session_both_motions_run_in_series(self):
        """Level 1 costs no extra VRAM: one motion session, used twice."""
        src, drv = self._inputs()
        r = self._restorer(_fake_sessions())          # no motion2
        _, _, m_d = r._front_half(r.sessions, src, drv)
        self.assertEqual(len(r.sessions["motion"].seen), 2)
        np.testing.assert_array_equal(r.sessions["motion"].seen[1]["img"], drv)
        self.assertEqual(m_d[3][0, 0], 1.0)           # came from the one session

    def test_a_failure_still_joins_the_helper_threads(self):
        """The lease returns the session set to the pool when _front_half exits.
        A helper thread still running inside it would then hand one TensorRT
        context to two threads — so every future must be joined, including on
        the path where another one raised."""
        src, drv = self._inputs()
        sessions = _fake_sessions(motion2=True,
                                  appearance_raises=RuntimeError("boom"),
                                  motion2_delay=0.05)
        r = self._restorer(sessions)
        with self.assertRaises(RuntimeError):
            r._front_half(sessions, src, drv)
        self.assertEqual(len(sessions["motion2"].seen), 1,
                         "returned before the driving motion thread finished")


class TestDeviceChaining(unittest.TestCase):
    """feature_3d stays in GPU memory between the two models that share it."""

    def _restorer(self):
        from roop.processors.Expression_LivePortrait import Expression_LivePortrait
        return Expression_LivePortrait()

    def test_off_by_default_until_initialize_proves_both_ends_are_on_a_gpu(self):
        self.assertFalse(self._restorer()._chain)

    def test_plain_path_feeds_the_warping_inputs_by_name(self):
        """Positional binding is the documented trap here: warping_spade
        declares kp_driving BEFORE kp_source."""
        r = self._restorer()
        sessions = _fake_sessions()
        x_s, x_d = _kp(), _kp()
        r._warp(sessions, np.zeros((1, 32, 16, 64, 64), np.float32), x_s, x_d)
        feeds = sessions["warping"].seen[0]
        np.testing.assert_array_equal(feeds["kp_source"], x_s)
        np.testing.assert_array_equal(feeds["kp_driving"], x_d)

    def test_a_chained_failure_falls_back_and_stays_fallen_back(self):
        """One bad io_binding must cost neither the face nor the rest of the
        run: retry it unchained, then stop trying."""
        r = self._restorer()
        r.sessions = _fake_sessions()
        r._chain = True
        calls = []

        def _once(*a, **k):
            calls.append(r._chain)
            if r._chain:
                raise RuntimeError("no device chaining here")
            return "restored"

        r._infer_once = _once
        out = r.infer((np.zeros((1, 3, 256, 256), np.float32),) * 2, 1.0)
        self.assertEqual(out, "restored")
        self.assertEqual(calls, [True, False])
        self.assertFalse(r._chain)


def _kp(n=21):
    return RNG.normal(size=(1, n, 3)).astype(np.float32)


class TestHeadposeDecoding(unittest.TestCase):
    """The pose heads emit a 66-bin distribution, not an angle."""

    def test_range_matches_the_bin_scheme(self):
        lo = headpose_pred_to_degree(np.eye(NUM_BINS)[0] * 50)[0]
        hi = headpose_pred_to_degree(np.eye(NUM_BINS)[NUM_BINS - 1] * 50)[0]
        self.assertAlmostEqual(lo, -97.5, places=3)
        self.assertAlmostEqual(hi, NUM_BINS * 3 - 3 - 97.5, places=3)

    def test_monotonic_in_the_argmax_bin(self):
        angles = [headpose_pred_to_degree(np.eye(NUM_BINS)[i] * 50)[0]
                  for i in range(0, NUM_BINS, 8)]
        self.assertEqual(angles, sorted(angles))

    def test_softmax_is_overflow_safe(self):
        """Raw logits can be large; a naive exp() would return NaN and poison
        the rotation matrix."""
        out = headpose_pred_to_degree(np.full((1, NUM_BINS), 10000.0))
        self.assertTrue(np.isfinite(out).all())


class TestRotationMatrix(unittest.TestCase):
    def test_identity_at_zero(self):
        np.testing.assert_allclose(get_rotation_matrix(0, 0, 0)[0],
                                   np.eye(3), atol=1e-6)

    def test_orthonormal_with_unit_determinant(self):
        for angles in ((30, -20, 15), (-75, 60, -40), (10, 90, 0)):
            r = get_rotation_matrix(*angles)[0]
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-5)
            self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=5)

    def test_preserves_lengths(self):
        pts = _kp()[0]
        r = get_rotation_matrix(25, -35, 12)[0]
        np.testing.assert_allclose(np.linalg.norm(pts @ r, axis=1),
                                   np.linalg.norm(pts, axis=1), atol=1e-5)


class TestKeypointTransform(unittest.TestCase):
    def test_drops_the_z_translation(self):
        kp, exp = _kp(), np.zeros((1, 21, 3), np.float32)
        t = np.array([[0.1, 0.2, 999.0]], np.float32)
        out = transform_keypoint(kp, exp, np.ones((1, 1), np.float32), t,
                                 get_rotation_matrix(0, 0, 0))
        np.testing.assert_allclose(out[0, :, 2], kp[0, :, 2], atol=1e-5)

    def test_scale_and_translation_apply(self):
        kp = _kp()
        zeros = np.zeros_like(kp)
        base = transform_keypoint(kp, zeros, np.ones((1, 1), np.float32),
                                  np.zeros((1, 3), np.float32),
                                  get_rotation_matrix(0, 0, 0))
        scaled = transform_keypoint(kp, zeros, np.full((1, 1), 2.0, np.float32),
                                    np.zeros((1, 3), np.float32),
                                    get_rotation_matrix(0, 0, 0))
        np.testing.assert_allclose(scaled, base * 2, atol=1e-5)


class TestExpressionBlend(unittest.TestCase):
    def test_zero_strength_is_an_exact_no_op(self):
        """The setting's off position must not perturb a single value."""
        src, dst = _kp(), _kp()
        np.testing.assert_array_equal(blend_expression(src, dst, 0.0), src)

    def test_full_strength_adopts_the_target_expression(self):
        src, dst = _kp(), _kp()
        np.testing.assert_allclose(blend_expression(src, dst, 1.0), dst, atol=1e-6)

    def test_above_one_exaggerates_past_the_target(self):
        src, dst = _kp(), _kp()
        out = blend_expression(src, dst, 2.0)
        np.testing.assert_allclose(out, src + 2 * (dst - src), atol=1e-6)

    def test_region_gating_touches_only_its_indices(self):
        src, dst = _kp(), _kp()
        for region, idx in (("lips", LIP_INDICES), ("eyes", EYE_INDICES)):
            out = blend_expression(src, dst, 1.0, region)
            moved = {i for i in range(src.shape[1])
                     if not np.allclose(out[0, i], src[0, i])}
            self.assertTrue(moved.issubset(set(idx)), f"{region} moved {moved - set(idx)}")
            self.assertTrue(moved, f"{region} moved nothing at all")


class TestDrivingKeypoints(unittest.TestCase):
    """The property the whole design rests on: expression moves, pose cannot."""

    def _setup(self):
        kp, exp_s, exp_d = _kp(), _kp(), _kp()
        scale = np.full((1, 1), 1.7, np.float32)
        t = np.array([[0.3, -0.2, 5.0]], np.float32)
        rot = get_rotation_matrix(21, -33, 9)
        x_s = transform_keypoint(kp, exp_s, scale, t, rot)
        return kp, exp_s, exp_d, scale, t, rot, x_s

    def test_zero_strength_returns_the_source_exactly(self):
        _, exp_s, exp_d, scale, _, _, x_s = self._setup()
        np.testing.assert_allclose(
            driving_keypoints(x_s, scale, exp_s, exp_d, 0.0), x_s, atol=1e-6)

    def test_matches_the_full_transform_it_short_circuits(self):
        """driving_keypoints skips rebuilding the rotation. It must agree with
        the long form for any strength, or the shortcut is not equivalent."""
        kp, exp_s, exp_d, scale, t, rot, x_s = self._setup()
        for strength in (0.25, 0.5, 1.0, 1.75):
            mixed = blend_expression(exp_s, exp_d, strength)
            expected = transform_keypoint(kp, mixed, scale, t, rot)
            got = driving_keypoints(x_s, scale, exp_s, exp_d, strength)
            np.testing.assert_allclose(got, expected, atol=1e-5,
                                       err_msg=f"strength={strength}")

    def test_head_pose_cannot_drift(self):
        """Rotation and translation cancel in the difference, so the centroid
        shift comes only from the expression delta — never from pose."""
        _, exp_s, exp_d, scale, _, _, x_s = self._setup()
        x_d = driving_keypoints(x_s, scale, exp_s, exp_d, 1.0)
        expected_shift = (scale.reshape(-1, 1, 1) * (exp_d - exp_s)).mean(axis=1)
        np.testing.assert_allclose((x_d - x_s).mean(axis=1), expected_shift, atol=1e-5)

    def test_region_restriction_survives_into_the_keypoints(self):
        _, exp_s, exp_d, scale, _, _, x_s = self._setup()
        x_d = driving_keypoints(x_s, scale, exp_s, exp_d, 1.0, region="lips")
        moved = {i for i in range(x_s.shape[1])
                 if not np.allclose(x_d[0, i], x_s[0, i], atol=1e-6)}
        self.assertTrue(moved.issubset(set(LIP_INDICES)))


class TestStitchingInput(unittest.TestCase):
    def test_concat_feat_shape_and_order(self):
        a, b = _kp(), _kp()
        out = concat_feat(a, b)
        self.assertEqual(out.shape, (1, 21 * 3 * 2))
        np.testing.assert_allclose(out[0, :63], a.reshape(-1), atol=1e-6)
        np.testing.assert_allclose(out[0, 63:], b.reshape(-1), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
