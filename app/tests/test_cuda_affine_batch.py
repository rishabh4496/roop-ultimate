"""CudaAffineBatch: device-side alignment, inversion, masks and compositing.

What is pinned here, and why each number is what it is
------------------------------------------------------

* MATRICES and GAUSSIAN MASKS are exact: ``invert_affine``, ``similarity_from_landmarks``,
  ``gaussian_blur`` and ``box_mask`` reproduce ``cv2.invertAffineTransform``,
  ``roop.face_util.estimate_norm`` (skimage's SimilarityTransform) and
  ``cv2.GaussianBlur`` to float rounding, so those tests use tight tolerances.
* RESAMPLING is not: ``grid_sample`` against ``cv2.warpAffine`` has a floor of ~5e-3
  (measured on real faces; OpenCV quantises sample coordinates to 1/32 px), so the
  warp test only guards against a CONVENTION error (a half-pixel shift, a transposed
  matrix, a swapped axis), each of which is an order of magnitude larger than the
  floor on a smooth image.  A per-pixel 1e-4 against OpenCV is not achievable and
  is deliberately not asserted.
* The mask math runs on CPU tensors; only the processor tests need CUDA.
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch                                                       # noqa: E402

from roop.optimized_processor import (                            # noqa: E402
    CudaAffineBatch, CudaIOBinding, GpuFaceSwapProcessor,
    dynamic_batch_model_bytes)
from roop.optimized_prepass import FaceObservation, FrameAnalysis  # noqa: E402

CUDA = torch.cuda.is_available()


def _random_affine(rng, n, shear=False):
    ms = []
    for _ in range(n):
        s = rng.uniform(0.4, 2.5)
        th = rng.uniform(-1.2, 1.2)
        c, si = np.cos(th) * s, np.sin(th) * s
        lin = np.array([[c, -si], [si, c]])
        if shear:
            lin = lin @ np.array([[1.0, rng.uniform(-0.3, 0.3)], [0.0, 1.0]])
        ms.append(np.hstack([lin, rng.uniform(-80, 300, (2, 1))]))
    return np.stack(ms)


def _smooth_image(h, w, seed=0):
    """A natural-ish smooth RGB image in [0, 1] (random but band-limited)."""
    rng = np.random.RandomState(seed)
    img = cv2.GaussianBlur(rng.rand(h, w, 3).astype(np.float32), (0, 0), 6)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.clip(0.6 * img + 0.2 * (xx / w)[..., None] + 0.2 * (yy / h)[..., None], 0, 1).astype(np.float32)


class TestInvertAffine(unittest.TestCase):
    def test_matches_cv2_float64(self):
        m = _random_affine(np.random.RandomState(0), 64, shear=True)
        got = CudaAffineBatch.invert_affine(torch.from_numpy(m)).numpy()
        want = np.stack([cv2.invertAffineTransform(x) for x in m])
        self.assertLess(np.abs(got - want).max(), 1e-10)

    def test_matches_cv2_float32(self):
        m = _random_affine(np.random.RandomState(1), 64).astype(np.float32)
        got = CudaAffineBatch.invert_affine(torch.from_numpy(m)).numpy()
        want = np.stack([cv2.invertAffineTransform(x) for x in m])
        scale = max(1.0, float(np.abs(want).max()))
        self.assertLess(np.abs(got - want).max() / scale, 1e-5)

    def test_singular_matrix_is_zeros_like_cv2_not_nan(self):
        m = np.array([[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0]]], np.float32)   # det == 0
        got = CudaAffineBatch.invert_affine(torch.from_numpy(m)).numpy()
        self.assertTrue(np.isfinite(got).all())
        np.testing.assert_array_equal(got, cv2.invertAffineTransform(m[0])[None])

    def test_round_trip_is_identity(self):
        m = torch.from_numpy(_random_affine(np.random.RandomState(2), 8))
        inv = CudaAffineBatch.invert_affine(m)
        pt = torch.tensor([[17.0], [-4.0], [1.0]], dtype=torch.float64)
        for a, b in zip(m, inv):
            mapped = a @ pt
            back = b @ torch.cat([mapped, torch.ones(1, 1, dtype=torch.float64)])
            np.testing.assert_allclose(back.numpy(), pt[:2].numpy(), atol=1e-9)

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            CudaAffineBatch.invert_affine(torch.zeros(3, 3, 3))


class TestSimilarityFromLandmarks(unittest.TestCase):
    TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], np.float64)

    def _faces(self, n, seed=0, noise=1.5):
        rng = np.random.RandomState(seed)
        out = []
        for m in _random_affine(rng, n):
            inv = cv2.invertAffineTransform(m)
            pts = self.TEMPLATE @ inv[:, :2].T + inv[:, 2]
            out.append(pts + rng.randn(5, 2) * noise)
        return np.stack(out)

    def test_matches_skimage_similarity_transform(self):
        from skimage import transform as trans
        faces = self._faces(40)
        got = CudaAffineBatch.similarity_from_landmarks(
            torch.from_numpy(faces), torch.from_numpy(self.TEMPLATE)).numpy()
        for f, g in zip(faces, got):
            t = trans.SimilarityTransform()
            t.estimate(f, self.TEMPLATE)
            self.assertLess(np.abs(g - t.params[:2]).max(), 1e-9)

    def test_matches_the_repos_estimate_norm_at_every_crop_size(self):
        try:
            from roop.face_util import estimate_norm, swap_template_points
        except Exception as exc:                                   # pragma: no cover
            self.skipTest(f"roop.face_util unavailable: {exc}")
        faces = self._faces(12, seed=3).astype(np.float32)
        for size in (128, 256, 512):
            tmpl = swap_template_points(size)
            got = CudaAffineBatch.similarity_from_landmarks(
                torch.from_numpy(faces), torch.from_numpy(np.asarray(tmpl, np.float32))).numpy()
            want = np.stack([estimate_norm(f, size) for f in faces])
            self.assertLess(np.abs(got - want).max(), 1e-3, f"size={size}")
            # relative to the matrix's own scale this is float32 rounding, ~1e-6
            self.assertLess(np.abs(got - want).max() / np.abs(want).max(), 1e-5, f"size={size}")

    def test_batched_equals_one_at_a_time(self):
        faces = torch.from_numpy(self._faces(6, seed=4)).float()
        batch = CudaAffineBatch.similarity_from_landmarks(faces, self.TEMPLATE.astype(np.float32))
        for i in range(6):
            one = CudaAffineBatch.similarity_from_landmarks(faces[i:i + 1], self.TEMPLATE.astype(np.float32))
            np.testing.assert_array_equal(batch[i].numpy(), one[0].numpy())

    def test_recovers_an_exact_transform(self):
        m = _random_affine(np.random.RandomState(5), 5)
        pts = np.stack([self.TEMPLATE @ cv2.invertAffineTransform(x)[:, :2].T
                        + cv2.invertAffineTransform(x)[:, 2] for x in m])
        got = CudaAffineBatch.similarity_from_landmarks(
            torch.from_numpy(pts), torch.from_numpy(self.TEMPLATE)).numpy()
        np.testing.assert_allclose(got, m, atol=1e-9)

    def test_rejects_mismatched_template(self):
        with self.assertRaises(ValueError):
            CudaAffineBatch.similarity_from_landmarks(
                torch.zeros(2, 5, 2), torch.zeros(4, 2))


class TestGaussianBlur(unittest.TestCase):
    def _check(self, ksize, sigma, shape=(3, 64, 80), tol=2e-6):
        rng = np.random.RandomState(ksize * 7 + int(sigma * 10))
        img = rng.rand(*shape).astype(np.float32)
        got = CudaAffineBatch.gaussian_blur(torch.from_numpy(img)[None], ksize, sigma)[0].numpy()
        want = np.stack([cv2.GaussianBlur(img[c], (ksize, ksize) if ksize else (0, 0), sigma)
                         for c in range(shape[0])])
        self.assertLess(np.abs(got - want).max(), tol, f"ksize={ksize} sigma={sigma}")

    def test_production_antialias_kernel_3x3_sigma0(self):
        """procmgr_masking.blur_area's `cv2.GaussianBlur(m, (3, 3), 0)`: a 0.25/0.5/0.25 kernel."""
        self._check(3, 0.0)

    def test_other_small_fixed_kernels(self):
        for k in (1, 5, 7):
            self._check(k, 0.0)

    def test_large_kernels_use_the_sigma_formula(self):
        for k in (11, 15, 31):
            self._check(k, 0.0)

    def test_ksize_9_uses_opencvs_fixed_kernel(self):
        """OpenCV 4.11 fixes a 9-tap kernel 2e-3 away from the sigma formula."""
        self._check(9, 0.0)

    def test_kernel_table_matches_the_installed_opencv_for_every_odd_size(self):
        """The version-dependent fixed-kernel table is checked against cv2 itself."""
        for k in range(1, 32, 2):
            want = cv2.getGaussianKernel(k, 0, ktype=cv2.CV_64F).ravel()
            got = CudaAffineBatch.gaussian_kernel_1d(k, 0.0, dtype=torch.float64).numpy()
            self.assertLess(np.abs(got - want).max(), 1e-12,
                            f"ksize={k}: the installed OpenCV's Gaussian table differs from "
                            "CudaAffineBatch._CV_SMALL_KERNELS - update the table")

    def test_explicit_sigma_and_derived_ksize(self):
        self._check(0, 3.5)
        self._check(11, 2.0)

    def test_constant_image_is_preserved(self):
        x = torch.full((2, 1, 40, 40), 0.37)
        np.testing.assert_allclose(
            CudaAffineBatch.gaussian_blur(x, 0, 4.0).numpy(), 0.37, atol=1e-6)

    def test_oversized_kernel_is_an_error_not_a_silent_wrong_answer(self):
        with self.assertRaises(ValueError):
            CudaAffineBatch.gaussian_blur(torch.zeros(1, 1, 16, 16), 33, 0.0)

    def test_needs_a_kernel_or_a_sigma(self):
        with self.assertRaises(ValueError):
            CudaAffineBatch.gaussian_blur(torch.zeros(1, 1, 16, 16), 0, 0.0)


class TestBoxMask(unittest.TestCase):
    @staticmethod
    def _reference(size, blur, padding=(0, 0, 0, 0)):
        blur_amount = int(size * 0.5 * blur)
        area = max(blur_amount // 2, 1)
        m = np.ones((size, size), np.float32)
        m[:max(area, int(size * padding[0] / 100)), :] = 0
        m[-max(area, int(size * padding[2] / 100)):, :] = 0
        m[:, :max(area, int(size * padding[3] / 100))] = 0
        m[:, -max(area, int(size * padding[1] / 100)):] = 0
        if blur_amount > 0:
            m = cv2.GaussianBlur(m, (0, 0), blur_amount * 0.25)
        return m

    def test_matches_the_cv2_recipe(self):
        for size, blur, pad in ((128, 0.3, (0, 0, 0, 0)), (256, 0.5, (0, 0, 0, 0)),
                                (128, 0.3, (10, 0, 5, 0)), (512, 0.15, (0, 8, 0, 8))):
            got = CudaAffineBatch.box_mask(size, blur, pad)[0, 0].numpy()
            self.assertLess(np.abs(got - self._reference(size, blur, pad)).max(), 2e-6,
                            f"size={size} blur={blur} pad={pad}")

    def test_shape_range_and_falloff(self):
        m = CudaAffineBatch.box_mask(128, 0.3)
        self.assertEqual(tuple(m.shape), (1, 1, 128, 128))
        self.assertGreaterEqual(float(m.min()), 0.0)
        self.assertLessEqual(float(m.max()), 1.0 + 1e-6)
        self.assertAlmostEqual(float(m[0, 0, 64, 64]), 1.0, places=4)      # centre kept
        self.assertLess(float(m[0, 0, 0, 0]), 1e-2)                        # corner (~5e-3: the Gaussian tail) gone
        row = m[0, 0, 64]
        self.assertTrue(bool((row[:20].diff() >= -1e-7).all()), "monotone ramp in from the edge")

    def test_is_cached_per_configuration(self):
        a = CudaAffineBatch.box_mask(96, 0.25)
        self.assertIs(a, CudaAffineBatch.box_mask(96, 0.25))
        self.assertIsNot(a, CudaAffineBatch.box_mask(96, 0.4))

    def test_zero_blur_is_a_hard_box_one_pixel_in(self):
        m = CudaAffineBatch.box_mask(32, 0.0)[0, 0].numpy()
        self.assertEqual(m[0, 5], 0.0)
        self.assertEqual(m[1, 5], 1.0)


class TestWarpFramesConvention(unittest.TestCase):
    """Guards the sampling CONVENTION, not 1e-4 parity (see the module docstring)."""

    def test_warp_matches_cv2_cubic_replicate_within_the_resampling_floor(self):
        h, w, size = 240, 320, 128
        img = _smooth_image(h, w)
        for m in (np.array([[0.9, 0.25, -60.0], [-0.25, 0.9, 20.0]], np.float32),
                  np.array([[1.4, 0.0, -90.0], [0.0, 1.4, -40.0]], np.float32)):
            ref = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
            frame = torch.from_numpy(img).permute(2, 0, 1)[None]
            got = CudaAffineBatch.warp_frames(frame, torch.from_numpy(m)[None], size)[0]
            got = got.permute(1, 2, 0).numpy()
            err = np.abs(got - ref)
            self.assertLess(float(err.mean()), 1e-3)
            self.assertLess(float(err.max()), 2e-2)

    def test_half_pixel_shift_would_be_caught(self):
        """A half-pixel convention error is far above the floor the test allows."""
        h, w, size = 240, 320, 128
        img = _smooth_image(h, w, seed=1)
        m = np.array([[1.0, 0.0, -50.0], [0.0, 1.0, -30.0]], np.float32)
        shifted = m.copy()
        shifted[:, 2] += 0.5
        ref = cv2.warpAffine(img, shifted, (size, size), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        frame = torch.from_numpy(img).permute(2, 0, 1)[None]
        got = CudaAffineBatch.warp_frames(frame, torch.from_numpy(m)[None], size)[0]
        self.assertGreater(float(np.abs(got.permute(1, 2, 0).numpy() - ref).mean()), 1e-3)


class TestPasteFaces(unittest.TestCase):
    S = 64

    def _setup(self, faces=1):
        frames = torch.rand(2, 3, self.S, self.S) * 0.5
        patches = torch.ones(faces, 3, self.S, self.S)               # solid white "swap"
        ident = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).repeat(faces, 1, 1)
        return frames, patches, ident

    def test_default_feathers_the_edge(self):
        frames, patches, ident = self._setup()
        out = CudaAffineBatch.paste_faces(frames, patches, torch.tensor([0]), ident)
        self.assertTrue(torch.equal(out[1], frames[1]), "an untouched frame is untouched")
        # A HARD seam replaces the edge pixel outright: |patch - frame| >= 0.5 here (patch 1.0,
        # frame <= 0.5).  The feather leaves a fraction: this crop is only 64 px, so the outermost
        # row keeps ~15% alpha (~0.1); at 128-512 px the same recipe leaves ~0.5%.
        self.assertLess(float((out[0, :, 0, :] - frames[0, :, 0, :]).abs().max()), 0.25,
                        "crop edge keeps the original frame: no hard seam")
        self.assertGreater(float(out[0, 0, 32, 32]), 0.99, "centre is the swapped patch")
        alpha = CudaAffineBatch.box_mask(self.S, 0.3)[0, 0]
        want = patches[0, 0] * alpha + frames[0, 0] * (1 - alpha)
        self.assertLess(float((out[0, 0] - want).abs().max()), 2e-3)

    def test_feather_false_keeps_the_legacy_hard_square(self):
        frames, patches, ident = self._setup()
        out = CudaAffineBatch.paste_faces(frames, patches, torch.tensor([0]), ident, feather=False)
        self.assertGreater(float(out[0, 0, 0, 0]), 0.99, "edge pixels are fully replaced")

    def test_explicit_mask_is_used_as_given(self):
        frames, patches, ident = self._setup()
        mask = torch.zeros(1, self.S, self.S)
        out = CudaAffineBatch.paste_faces(frames, patches, torch.tensor([0]), ident, masks=mask)
        self.assertTrue(torch.allclose(out, frames, atol=1e-6))

    def test_occlusion_matte_multiplies_in_and_zero_removes_the_swap(self):
        frames, patches, ident = self._setup()
        out = CudaAffineBatch.paste_faces(
            frames, patches, torch.tensor([0]), ident,
            occlusion_masks=torch.zeros(1, 1, self.S, self.S))
        self.assertTrue(torch.allclose(out, frames, atol=1e-6))
        half = CudaAffineBatch.paste_faces(
            frames, patches, torch.tensor([0]), ident,
            occlusion_masks=torch.full((1, 1, self.S, self.S), 0.5))
        full = CudaAffineBatch.paste_faces(frames, patches, torch.tensor([0]), ident)
        self.assertLess(float((half[0, 0, 32, 32] - (0.5 * full[0, 0, 32, 32] + 0.5 * frames[0, 0, 32, 32])).abs()), 2e-3)

    def test_occlusion_matte_of_another_size_is_resampled(self):
        frames, patches, ident = self._setup()
        out = CudaAffineBatch.paste_faces(
            frames, patches, torch.tensor([0]), ident,
            occlusion_masks=torch.zeros(1, 32, 32))
        self.assertTrue(torch.allclose(out, frames, atol=1e-6))

    def test_two_faces_in_one_frame_both_land_and_other_frames_are_untouched(self):
        frames, patches, ident = self._setup(faces=2)
        shift = ident.clone()
        shift[1, 0, 2] = -40.0           # second face's crop covers a different region
        out = CudaAffineBatch.paste_faces(frames, patches, torch.tensor([0, 0]), shift)
        self.assertTrue(torch.equal(out[1], frames[1]))
        self.assertGreater(float(out[0, 0, 32, 32]), 0.99)
        # input is not mutated
        self.assertFalse(torch.equal(out[0], frames[0]))


_INSWAPPER = os.path.join(os.path.dirname(__file__), '..', 'models', 'inswapper_128.onnx')


@unittest.skipUnless(os.path.exists(_INSWAPPER), "models/inswapper_128.onnx not present")
class TestDynamicBatchModel(unittest.TestCase):
    """`dynamic_batch_model_bytes`: a swap model that accepts PRE-BOUND batched outputs.

    `_relax_batch_dim` alone leaves 237 stale batch-1 `value_info` annotations, so ORT
    reports the output as [1, 3, 128, 128] and rejects a bound [B, 3, 128, 128] buffer.
    """

    @classmethod
    def setUpClass(cls):
        import onnxruntime as ort
        cls.ort = ort
        cls.data = dynamic_batch_model_bytes(_INSWAPPER)

    def test_output_batch_axis_is_symbolic(self):
        s = self.ort.InferenceSession(self.data, providers=["CPUExecutionProvider"])
        self.assertEqual(s.get_outputs()[0].shape, ["N", 3, 128, 128])
        self.assertEqual([i.shape[0] for i in s.get_inputs()], ["N", "N"])

    def test_batched_rows_equal_batch_one_rows_on_cpu(self):
        ref = self.ort.InferenceSession(_INSWAPPER, providers=["CPUExecutionProvider"])
        dyn = self.ort.InferenceSession(self.data, providers=["CPUExecutionProvider"])
        rng = np.random.RandomState(0)
        target = rng.rand(3, 3, 128, 128).astype(np.float32)
        source = rng.randn(3, 512).astype(np.float32)
        source /= np.linalg.norm(source, axis=1, keepdims=True)
        batched = dyn.run(None, {"target": target, "source": source})[0]
        for i in range(3):
            one = ref.run(None, {"target": target[i:i + 1], "source": source[i:i + 1]})[0]
            self.assertLess(float(np.abs(batched[i] - one[0]).max()), 1e-4)

    @unittest.skipUnless(CUDA, "CUDA required")
    def test_pre_bound_gpu_outputs_work_for_any_batch_size(self):
        s = self.ort.InferenceSession(self.data, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if s.get_providers()[0] != "CUDAExecutionProvider":
            self.skipTest("CUDA execution provider not active")
        binding = CudaIOBinding(s, 0)
        ref = self.ort.InferenceSession(_INSWAPPER, providers=["CPUExecutionProvider"])
        rng = np.random.RandomState(1)
        for n in (1, 2, 3, 5):
            target = torch.from_numpy(rng.rand(n, 3, 128, 128).astype(np.float32)).cuda()
            source = torch.from_numpy(rng.randn(n, 512).astype(np.float32)).cuda()
            source = source / source.norm(dim=1, keepdim=True)
            out = binding.run_gpu({"target": target, "source": source},
                                  output_shapes={"output": (n, 3, 128, 128)})[0]
            self.assertEqual(tuple(out.shape), (n, 3, 128, 128))
            for i in range(n):
                one = ref.run(None, {"target": target[i:i + 1].cpu().numpy(),
                                     "source": source[i:i + 1].cpu().numpy()})[0]
                # CUDA EP convolutions are TF32 by default: ~1.5e-2, not bit-equal
                self.assertLess(float(np.abs(out[i].cpu().numpy() - one[0]).max()), 5e-2)


@unittest.skipUnless(CUDA, "CUDA required (GpuFaceSwapProcessor and CudaFrameBridge are CUDA-only)")
class TestGpuFaceSwapProcessor(unittest.TestCase):
    """The whole processor, with an identity 'swapper': swapping a face to itself must
    reproduce the original frame (only resampling error remains)."""

    class _Runner:
        device_id = 0

        def __init__(self, fn=None):
            self.calls = []
            self.fn = fn or (lambda feeds: feeds["target"])

        def run_gpu(self, feeds, batch_size=None, pad_to_batch=True):
            self.calls.append({k: tuple(v.shape) for k, v in feeds.items()})
            return [self.fn(feeds)]

    TEMPLATE = TestSimilarityFromLandmarks.TEMPLATE.astype(np.float32)

    def _scene(self, faces_per_frame=(1, 2)):
        rng = np.random.RandomState(11)
        frames, analyses = [], []
        for fi, n in enumerate(faces_per_frame):
            img = (_smooth_image(240, 320, seed=fi) * 255).astype(np.uint8)
            frames.append(np.ascontiguousarray(img[:, :, ::-1]))
            obs = []
            for k in range(n):
                scale = rng.uniform(0.9, 1.3)
                m = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0]], np.float64)
                m[:, 2] = (160 - 60 * k, 110)       # face placement in the frame
                # landmarks such that the fitted alignment maps them onto the template
                pts = (self.TEMPLATE.astype(np.float64) @ cv2.invertAffineTransform(
                    np.array([[scale, 0, 0], [0, scale, 0]]))[:, :2].T) + m[:, 2] * 0
                pts = pts + np.array([160 - 60 * k, 110]) - pts.mean(0)
                fit = CudaAffineBatch.similarity_from_landmarks(
                    torch.from_numpy(pts[None]), torch.from_numpy(self.TEMPLATE.astype(np.float64)))[0].numpy()
                obs.append(FaceObservation(track_id=k, bbox=np.array([0, 0, 1, 1], np.float32),
                                           landmarks=pts.astype(np.float32), matrix=fit.astype(np.float32), score=1.0))
            analyses.append(FrameAnalysis(frame_index=fi, faces=obs, detection_run=True))
        return frames, analyses

    def _processor(self, runner=None, **kw):
        return GpuFaceSwapProcessor(
            runner or self._Runner(), np.zeros((1, 512), np.float32), 128,
            model_channel_order="bgr", mask_output_index=None, **kw)

    def test_identity_swap_reproduces_the_frame(self):
        frames, analyses = self._scene()
        runner = self._Runner()
        out = self._processor(runner)(frames, analyses)
        self.assertEqual(runner.calls[0]["target"], (3, 3, 128, 128), "all 3 faces in ONE batch")
        self.assertEqual(runner.calls[0]["source"], (3, 512), "identity broadcast to [B, 512]")
        for src, dst in zip(frames, out):
            err = np.abs(src.astype(int) - dst.astype(int))
            self.assertLess(float(err.mean()), 0.6)
            self.assertLessEqual(int(err.max()), 24)

    def test_gpu_computed_matrices_equal_uploaded_matrices(self):
        frames, analyses = self._scene()
        a = self._processor()(frames, analyses)
        b = self._processor(alignment_template=self.TEMPLATE)(frames, analyses)
        for x, y in zip(a, b):
            self.assertLessEqual(int(np.abs(x.astype(int) - y.astype(int)).max()), 1)

    def test_swap_actually_changes_the_face_inside_the_feather(self):
        frames, analyses = self._scene(faces_per_frame=(1,))
        runner = self._Runner(lambda feeds: torch.ones_like(feeds["target"]))
        out = self._processor(runner)(frames, analyses)[0]
        centre = out[110, 160].astype(int)
        self.assertGreater(int(centre.min()), 240, "the swapped (white) patch lands on the face")
        self.assertTrue(np.array_equal(out[0, 0], frames[0][0, 0]), "far-away pixels are untouched")

    def test_occlusion_provider_keeps_the_original_where_it_says_so(self):
        frames, analyses = self._scene(faces_per_frame=(1,))
        runner = self._Runner(lambda feeds: torch.ones_like(feeds["target"]))
        seen = {}

        def occluder(aligned_target):
            seen["shape"] = tuple(aligned_target.shape)
            return torch.zeros(aligned_target.shape[0], 1, *aligned_target.shape[2:],
                               device=aligned_target.device)
        out = self._processor(runner, occlusion_provider=occluder)(frames, analyses)[0]
        self.assertEqual(seen["shape"], (1, 3, 128, 128))
        self.assertLessEqual(int(np.abs(out.astype(int) - frames[0].astype(int)).max()), 1)

    def test_frames_without_faces_are_returned_untouched(self):
        frames, _ = self._scene()
        out = self._processor()(frames, [FrameAnalysis(frame_index=i) for i in range(2)])
        for a, b in zip(frames, out):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
