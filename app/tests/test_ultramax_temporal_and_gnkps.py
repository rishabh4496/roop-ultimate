"""UltraMax / restorer upgrades: bypass ramp, luminance-only sharpen, temporal shimmer, batching, seam colour modes.

Each class names the brief item it answers and the production site it drives:

  TestBypassRamp                  (a) smart_restorer_bypass.bypass_weight / blend_bypass, and the two processors'
                                      own ``Run`` gates (Enhance_CodeFormer, Enhance_RestoreFormerPPlus)
  TestLumaOnlySharpen             enhance_common.luma_only_sharpen[_tensor], the UltraMax post-sharpen
  TestTemporalShimmer             (b) enhancer_stabilizer.TemporalEnhancerBlend (the production flow-registered EMA)
  TestUltraMaxBatchedTensorRT     (c) the real codeformer.fp16 graph under TensorRT, B=4 vs B=1 (``gpu`` marker)
  TestSeamColourModes             blender.skin_lab_match(mode=reinhard|adain|cdf)
  TestGnKpsPremise                the SCRFD GN-kps weights the brief names are not shipped (and why)
"""
import importlib.util
import json
import os
import sys
import unittest
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(TESTS)
ROOT = os.path.dirname(APP)
for _p in (APP, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from roop import blender                                                      # noqa: E402
from roop import smart_restorer_bypass as sb                                  # noqa: E402
from roop.enhancer_stabilizer import DEFAULT_BLEND, TemporalEnhancerBlend     # noqa: E402
from roop.processors import enhance_common as ec                              # noqa: E402


# ───────────────────────────── (a) smooth 256 -> 384 px bypass ramp ──────────────────────────────────────────
class TestBypassRamp(unittest.TestCase):
    HEIGHTS = np.arange(200, 450, 1.0)

    def test_weight_endpoints_and_midpoint(self):
        self.assertEqual(sb.ramp_weight(200.0), 1.0)
        self.assertEqual(sb.ramp_weight(256.0), 1.0)
        self.assertAlmostEqual(sb.ramp_weight(320.0), 0.5, places=6)
        self.assertEqual(sb.ramp_weight(384.0), 0.0)
        self.assertEqual(sb.ramp_weight(600.0), 0.0)
        self.assertEqual(sb.ramp_weight(None), 0.0)

    def test_weight_has_no_step_anywhere(self):
        """Adjacent pixel heights differ by at most the ramp's own slope (1/128), over the whole range."""
        w = np.array([sb.ramp_weight(h) for h in self.HEIGHTS])
        self.assertLessEqual(float(np.abs(np.diff(w)).max()), 1.0 / 128.0 + 1e-9)
        self.assertTrue(np.all(np.diff(w) <= 1e-12), "the weight must only ever fall as the face grows")

    def test_the_legacy_step_is_the_defect_the_ramp_removes(self):
        """The 320 px step jumped the decision by 1.0 between two adjacent heights."""
        face = lambda h: SimpleNamespace(bbox=[0, 0, h, h])                # noqa: E731
        proc = SimpleNamespace(processorname="ultramax")
        a = sb.should_bypass_restorer(face(320), "realswap", proc, enabled=True)
        b = sb.should_bypass_restorer(face(321), "realswap", proc, enabled=True)
        self.assertTrue(a)
        self.assertFalse(b)                                                # the step: 1.0 between 320 and 321 px
        wa = sb.bypass_weight(face(320), "realswap", proc, enabled=True)
        wb = sb.bypass_weight(face(321), "realswap", proc, enabled=True)
        self.assertLess(abs(wa - wb), 0.01)

    def test_output_is_continuous_across_the_ramp(self):
        """Blend a restorer crop with its DoG crop at every height: the image moves by <= 1/128 of their gap."""
        rng = np.random.default_rng(3)
        restored = cv2.GaussianBlur(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8), (0, 0), 1.2)
        dog = sb.apply_dog_unsharp_mask(restored)
        gap = float(np.abs(restored.astype(np.float32) - dog.astype(np.float32)).mean())
        self.assertGreater(gap, 0.5, "fixture: the DoG crop must actually differ from the restorer's")
        weights = [sb.ramp_weight(h) for h in self.HEIGHTS]
        # The blend itself, before 8-bit rounding: exactly one ramp slope per pixel of face height.
        exact = [w * dog.astype(np.float32) + (1.0 - w) * restored.astype(np.float32) for w in weights]
        exact_steps = [float(np.abs(exact[i + 1] - exact[i]).mean()) for i in range(len(exact) - 1)]
        self.assertLessEqual(max(exact_steps), gap / 128.0 + 1e-3)
        # The uint8 output adds rounding. At w = 0.5 exactly, half the pixels are on a .5 tie and round the same
        # way, which moves the mean by 0.25 grey level: quantisation, 12x smaller than the old step (= gap).
        outs = [sb.blend_bypass(restored, 1, dog, w)[0].astype(np.float32) for w in weights]
        steps = [float(np.abs(outs[i + 1] - outs[i]).mean()) for i in range(len(outs) - 1)]
        self.assertLessEqual(max(steps), gap / 128.0 + 0.30,
                             f"largest frame-to-frame step {max(steps):.4f} vs one ramp slope {gap / 128.0:.4f}")
        self.assertLess(max(steps), 0.15 * gap, "the ramp must be far smaller than the 320 px step it replaces")
        np.testing.assert_array_equal(outs[0], dog.astype(np.float32))          # <= 256 px: the DoG crop exactly
        np.testing.assert_array_equal(outs[-1], restored.astype(np.float32))    # >= 384 px: the restorer exactly

    def test_gates_other_than_height_still_force_zero(self):
        face = SimpleNamespace(bbox=[0, 0, 300, 300])
        ultra = SimpleNamespace(processorname="ultramax")
        self.assertGreater(sb.bypass_weight(face, "realswap", ultra, enabled=True), 0.0)
        self.assertEqual(sb.bypass_weight(face, "realswap", ultra, enabled=False), 0.0)
        self.assertEqual(sb.bypass_weight(face, "inswapper", ultra, enabled=True), 0.0)
        self.assertEqual(sb.bypass_weight(face, "realswap", SimpleNamespace(processorname="GPEN-BFR-512"),
                                          enabled=True), 0.0)
        self.assertEqual(sb.bypass_weight(None, "realswap", ultra, enabled=True), 0.0)

    def test_blend_bypass_resizes_to_an_enlarged_restorer_output(self):
        restored = np.full((128, 128, 3), 200, np.uint8)                   # a 2x restorer output
        dog = np.full((64, 64, 3), 100, np.uint8)
        out, scale = sb.blend_bypass(restored, 2, dog, 0.25)
        self.assertEqual(out.shape, (128, 128, 3))
        self.assertEqual(scale, 2)
        self.assertEqual(int(out[10, 10, 0]), 175)                          # 0.25*100 + 0.75*200

    # The two processors that gate inside their own Run: drive the real method with the network stubbed.
    def _drive(self, cls, height):
        import roop.globals as g
        proc = object.__new__(cls)
        calls = []
        restored = np.full((64, 64, 3), 200, np.uint8)

        def _net(_faceset, _face, _frame):
            calls.append(1)
            return restored, 1
        proc._run_network = _net
        saved = (getattr(g, "CFG", None), getattr(g, "swap_model", None))
        g.CFG = SimpleNamespace(auto_bypass_restorer_native_swap=True)
        g.swap_model = "realswap"
        try:
            frame = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
            out, scale = proc.Run(None, SimpleNamespace(bbox=[0, 0, height, height]), frame)
        finally:
            g.CFG, g.swap_model = saved
        return out, scale, len(calls), frame, restored

    def _check_processor(self, cls):
        out, _, n, frame, _ = self._drive(cls, 200)                          # <= 256 px: network never runs
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(out, sb.apply_dog_unsharp_mask(frame))
        out, _, n, _, restored = self._drive(cls, 500)                       # >= 384 px: network alone
        self.assertEqual(n, 1)
        np.testing.assert_array_equal(out, restored)
        out, _, n, frame, restored = self._drive(cls, 320)                   # mid-ramp: half and half
        self.assertEqual(n, 1)
        expect = np.clip(np.rint(0.5 * sb.apply_dog_unsharp_mask(frame).astype(np.float32)
                                 + 0.5 * restored.astype(np.float32)), 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(out, expect)

    def test_codeformer_run_gate(self):
        from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
        self._check_processor(Enhance_CodeFormer)

    def test_restoreformer_pp_run_gate(self):
        from roop.processors.Enhance_RestoreFormerPPlus import Enhance_RestoreFormerPPlus
        self._check_processor(Enhance_RestoreFormerPPlus)

    def test_manager_blends_only_for_processors_without_their_own_gate(self):
        """UltraMax has no gate of its own, so ProcessMgr mixes; CodeFormer / RF++ already did - never twice."""
        from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
        from roop.processors.Enhance_RestoreFormerPPlus import Enhance_RestoreFormerPPlus
        from roop.processors.Enhance_UltraMax import Enhance_UltraMax
        self.assertTrue(sb.manager_blends(object.__new__(Enhance_UltraMax), 0.5))
        self.assertFalse(sb.manager_blends(object.__new__(Enhance_CodeFormer), 0.5))
        self.assertFalse(sb.manager_blends(object.__new__(Enhance_RestoreFormerPPlus), 0.5))
        self.assertFalse(sb.manager_blends(object.__new__(Enhance_UltraMax), 0.0))
        self.assertFalse(sb.manager_blends(object.__new__(Enhance_UltraMax), 1.0))


# ───────────────────────────── luminance-only sharpening keeps chroma locked ─────────────────────────────────
def _chroma(img_bgr):
    """(B - Y, R - Y) in float: what a luminance-only edit must not change."""
    f = img_bgr.astype(np.float32)
    y = f[..., 0] * 0.114 + f[..., 1] * 0.587 + f[..., 2] * 0.299
    return np.stack([f[..., 0] - y, f[..., 2] - y], -1)


def _isoluminant_edge():
    """Left half teal, right half rose, equal luminance: a pure CHROMA edge, the case a per-channel unsharp breaks."""
    img = np.zeros((96, 96, 3), np.uint8)
    img[:, :48] = (150, 110, 80)                                            # B, G, R  (luma 105.6)
    img[:, 48:] = (40, 86, 170)                                             # B, G, R  (luma 105.9)
    return img


class TestLumaOnlySharpen(unittest.TestCase):
    GAIN = 0.18                                                            # UltraMax's shipped structure sharpen

    @staticmethod
    def _per_channel(img, gain, sigma=0.8):
        base = img.astype(np.float32)
        return np.clip(base + gain * (base - cv2.GaussianBlur(base, (0, 0), sigma)), 0, 255).astype(np.uint8)

    def test_a_chroma_edge_is_not_overshot(self):
        img = _isoluminant_edge()
        before = _chroma(img)
        legacy = np.abs(_chroma(self._per_channel(img, self.GAIN)) - before).max()
        lumaonly = np.abs(_chroma(ec.luma_only_sharpen(img, self.GAIN)) - before).max()
        self.assertGreater(legacy, 3.0, "fixture: the per-channel unsharp must visibly move chroma on this edge")
        self.assertLessEqual(lumaonly, 1.0, f"luma-only sharpen moved chroma by {lumaonly:.2f} levels")

    def test_a_luminance_edge_is_still_sharpened(self):
        img = np.zeros((64, 64, 3), np.uint8)
        img[:, :32] = 90
        img[:, 32:] = 170
        out = ec.luma_only_sharpen(img, 0.5).astype(np.int16)
        # undershoot on the dark side of the edge and overshoot on the bright side: what an unsharp mask does
        self.assertLess(int(out[32, 31, 0]), 90)
        self.assertGreater(int(out[32, 32, 0]), 170)
        # and on a grey edge it is the per-channel operator (all three channels share one delta)
        legacy = self._per_channel(img, 0.5).astype(np.int16)
        self.assertLessEqual(int(np.abs(out - legacy).max()), 1)

    def test_zero_gain_returns_the_input_object(self):
        img = _isoluminant_edge()
        self.assertIs(ec.luma_only_sharpen(img, 0.0), img)

    @pytest.mark.gpu
    def test_cuda_tensor_path_matches_the_host_path(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            self.skipTest("no CUDA device")
        from roop.processors.Enhance_UltraMax import Enhance_UltraMax
        rng = np.random.default_rng(5)
        img = cv2.GaussianBlur(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8), (0, 0), 1.5)
        cpu = ec.luma_only_sharpen(img, self.GAIN).astype(np.float32)
        t = torch.from_numpy(img).to("cuda", dtype=torch.float32)
        gpu = ec.luma_only_sharpen_tensor(t, self.GAIN, Enhance_UltraMax._blur_gpu).cpu().numpy()
        self.assertLessEqual(float(np.abs(cpu - gpu).max()), 2.0)


# ───────────────────────────── (b) temporal shimmer through the production EMA ───────────────────────────────
def _face_texture(seed=1, size=192):
    rng = np.random.default_rng(seed)
    chans = []
    for c in range(3):
        n = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), 3.0)
        n = (n - n.min()) / max(float(n.max() - n.min()), 1e-6)
        chans.append(100.0 + 60.0 * n + 5.0 * c)
    return np.stack(chans, -1).astype(np.float32)


def _restorer_redraw(base, seed, sigma=7.0):
    """The same face with an independent high-frequency texture draw each frame - what a restorer does."""
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.standard_normal(base.shape[:2]).astype(np.float32), (0, 0), 0.7)
    n *= sigma / max(float(n.std()), 1e-6)
    return np.clip(base + n[..., None] + 0.5, 0, 255).astype(np.uint8)


def _hf_variance(frames):
    """Mean over pixels of the TEMPORAL VARIANCE of the high-pass band (frame-to-frame restoration shimmer)."""
    hp = [f.astype(np.float32) - cv2.GaussianBlur(f.astype(np.float32), (0, 0), 1.5) for f in frames]
    return float(np.mean(np.var(np.stack(hp), axis=0)))


class TestTemporalShimmer(unittest.TestCase):
    N, SKIP = 40, 8

    def _reduction(self, blend):
        base = _face_texture()
        frames = [_restorer_redraw(base, 700 + i) for i in range(self.N)]
        eng = TemporalEnhancerBlend(blend=blend)
        outs = [eng.blend_crop(f, track_id=1, frame_index=i) for i, f in enumerate(frames)]
        return 1.0 - _hf_variance(outs[self.SKIP:]) / _hf_variance(frames[self.SKIP:]), eng

    def test_the_default_blend_is_the_briefs_0_20(self):
        self.assertEqual(DEFAULT_BLEND, 0.20)
        self.assertEqual(TemporalEnhancerBlend().blend, 0.20)

    def test_variance_reduction_at_the_default_is_the_ema_closed_form_not_40_percent(self):
        """An EMA that keeps weight w of its history leaves variance (1 - w) / (1 + w) of white shimmer:
        33.3% less at w = 0.20. The brief's '>40%' is unreachable at that setting; this pins what is true."""
        w = DEFAULT_BLEND
        expected = 1.0 - (1.0 - w) / (1.0 + w)
        got, eng = self._reduction(w)
        self.assertAlmostEqual(got, expected, delta=0.06, msg=f"measured {got:.3f}, closed form {expected:.3f}")
        self.assertLess(got, 0.40)
        self.assertEqual(eng.stats()["applied"], self.N - 1)               # the stage ran on every frame but the seed

    def test_forty_percent_needs_a_history_weight_of_at_least_0_25(self):
        need = 0.25
        self.assertAlmostEqual(1.0 - (1.0 - need) / (1.0 + need), 0.40, places=9)
        got, _ = self._reduction(0.30)
        self.assertGreater(got, 0.40, f"blend 0.30 reduced HF variance by only {got:.3f}")

    def test_a_shot_cut_resets_instead_of_ghosting(self):
        eng = TemporalEnhancerBlend(blend=DEFAULT_BLEND)
        a = _face_texture(seed=1)
        b = _face_texture(seed=99)
        for i in range(6):
            eng.blend_crop(_restorer_redraw(a, i), 1, i)
        first_of_new_shot = _restorer_redraw(b, 500)
        out = eng.blend_crop(first_of_new_shot, 1, 6, cut=True)
        self.assertIs(out, first_of_new_shot)
        self.assertEqual(eng.stats()["reset_shot"], 1)


# ───────────────────────────── (c) batched UltraMax under TensorRT ───────────────────────────────────────────
def _load_probe():
    spec = importlib.util.spec_from_file_location("ultramax_batch_probe", os.path.join(TESTS, "ultramax_batch_probe.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.gpu
class TestUltraMaxBatchedTensorRT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ort = pytest.importorskip("onnxruntime")
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            raise unittest.SkipTest("TensorRT is not available on this machine")
        cls.probe = _load_probe()
        if not os.path.isfile(os.path.join(APP, cls.probe.MODEL_REL)):
            raise unittest.SkipTest("codeformer.fp16.onnx is not present")
        import roop.globals as g
        cls._saved = (os.getcwd(), getattr(g, "CFG", None), getattr(g, "execution_providers", None))
        try:
            cls.probe.init_environment()
            if not any("Tensorrt" in str(p) for p in g.execution_providers):
                raise unittest.SkipTest("TensorRT is not among the decoded execution providers")
            cls.sess = cls.probe.build_session("shipped")
        except unittest.SkipTest:
            cls._restore()
            raise
        except Exception:
            cls._restore()
            raise

    @classmethod
    def _restore(cls):
        import roop.globals as g
        os.chdir(cls._saved[0])
        g.CFG, g.execution_providers = cls._saved[1], cls._saved[2]

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_saved"):
            cls._restore()

    def test_the_session_really_runs_on_tensorrt(self):
        self.assertEqual(self.sess.get_providers()[0], "TensorrtExecutionProvider")

    def test_shipped_profile_admits_a_batch_of_four(self):
        from roop import trt_shape_profile as tsp
        prof = tsp.resolve_profile("ultramax", os.path.join(APP, self.probe.MODEL_REL))
        lo = int(prof.min_shapes.split(":")[1].split("x")[0])
        hi = int(prof.max_shapes.split(":")[1].split("x")[0])
        self.assertEqual(lo, 1)
        self.assertGreaterEqual(hi, 8)

    def test_b4_matches_four_b1_runs(self):
        r = self.probe.measure(self.sess, 4, iters=3, warm=2)
        self.assertTrue(r["finite"])
        self.assertLessEqual(r["max_levels_u8"], 2, f"B=4 differs from B=1 by {r['max_levels_u8']} levels")
        self.assertLessEqual(r["frac_pixels_differing_u8"], 0.001)
        # proof the TensorRT engine executed (a CPU fallback would be seconds per face, not tens of ms)
        self.assertLess(r["batched_ms_per_face"], 250.0)


# ───────────────────────────── seam colour: reinhard | adain | cdf ───────────────────────────────────────────
def _skin_pair(seed=0, size=96):
    """A swapped crop and a plate crop over the same skin mask; the plate is darker, warmer and BIMODAL in L."""
    rng = np.random.default_rng(seed)
    skin = np.zeros((size, size), bool)
    skin[8:-8, 8:-8] = True
    noise = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), 2.0)
    content = np.zeros((size, size, 3), np.float32)
    content[...] = (150.0, 165.0, 190.0)
    content += noise[..., None] * 45.0                    # a textured swap, wider than the plate's own grain
    plate = np.zeros_like(content)
    lit = (np.arange(size)[None, :] < size // 2)
    plate[...] = np.where(lit[..., None], np.float32((120, 140, 175)), np.float32((95, 112, 150)))
    plate += noise[..., None] * 20.0
    return np.clip(content, 0, 255), np.clip(plate, 0, 255), skin


def _lab_skin(img, skin):
    return blender._to_lab(img)[skin]


class TestSeamColourModes(unittest.TestCase):
    def test_default_mode_is_reinhard_and_unchanged(self):
        content, plate, skin = _skin_pair()
        default, info = blender.skin_lab_match(content, plate, skin)
        explicit, _ = blender.skin_lab_match(content, plate, skin, mode="reinhard")
        np.testing.assert_array_equal(default, explicit)
        self.assertEqual(info["mode"], "reinhard")
        self.assertTrue(info["applied"])

    def test_mode_names_normalise_and_unknown_falls_back(self):
        self.assertEqual(blender.normalize_color_mode("CDF Histogram"), "cdf")
        self.assertEqual(blender.normalize_color_mode("cdf-histogram"), "cdf")
        self.assertEqual(blender.normalize_color_mode("AdaIN"), "adain")
        self.assertEqual(blender.normalize_color_mode("nonsense"), "reinhard")
        self.assertEqual(blender.normalize_color_mode(None), "reinhard")

    def test_adain_matches_mean_and_sigma_more_closely_than_reinhard(self):
        content, plate, skin = _skin_pair()
        ref = _lab_skin(plate, skin)
        before = _lab_skin(content, skin)
        self.assertGreater(ref.std(0)[0] / before.std(0)[0], 1.3,
                           "fixture: the sigma ratio must exceed Reinhard's 1.15 clamp for the two to differ")
        gaps = {}
        for mode in ("reinhard", "adain"):
            out, info = blender.skin_lab_match(content, plate, skin, mode=mode)
            self.assertTrue(info["applied"])
            gaps[mode] = abs(float(_lab_skin(out, skin).std(0)[0]) - float(ref.std(0)[0]))
        self.assertLess(gaps["adain"], gaps["reinhard"],
                        f"AdaIN L-sigma gap {gaps['adain']:.3f} vs Reinhard {gaps['reinhard']:.3f}")

    def test_cdf_matches_the_whole_distribution_not_two_moments(self):
        content, plate, skin = _skin_pair()
        ref = _lab_skin(plate, skin)
        before = _lab_skin(content, skin)
        qs = np.linspace(5, 95, 19)

        def quantile_gap(x):
            return float(np.abs(np.percentile(x[:, 0], qs) - np.percentile(ref[:, 0], qs)).mean())

        out, info = blender.skin_lab_match(content, plate, skin, mode="cdf")
        self.assertTrue(info["applied"])
        self.assertEqual(info["mode"], "cdf")
        rein, _ = blender.skin_lab_match(content, plate, skin, mode="reinhard")
        self.assertLess(quantile_gap(_lab_skin(out, skin)), quantile_gap(before))
        # a bimodal plate stays bimodal: CDF closes the quantile gap further than a two-moment match does
        self.assertLess(quantile_gap(_lab_skin(out, skin)), quantile_gap(_lab_skin(rein, skin)))

    def test_pixels_outside_the_skin_weight_are_left_alone(self):
        content, plate, skin = _skin_pair()
        for mode in blender.COLOR_MODES:
            out, _ = blender.skin_lab_match(content, plate, skin, mode=mode)
            corner = np.abs(out[:3, :3] - content[:3, :3]).max()
            self.assertLess(float(corner), 1.0, f"{mode} recoloured the crop corner by {corner:.2f}")

    def test_strength_zero_and_too_small_a_mask_are_no_ops(self):
        content, plate, skin = _skin_pair()
        for mode in blender.COLOR_MODES:
            out, info = blender.skin_lab_match(content, plate, skin, strength=0.0, mode=mode)
            self.assertFalse(info["applied"])
            np.testing.assert_array_equal(out, content)
            tiny = np.zeros_like(skin)
            tiny[40:44, 40:44] = True
            out, info = blender.skin_lab_match(content, plate, tiny, mode=mode)
            self.assertFalse(info["applied"])

    def test_the_setting_reaches_the_renderer(self):
        """Settings default, globals default and the call site all carry the mode (a bound control nobody reads
        is the failure this repo has been bitten by)."""
        import roop.globals as g
        self.assertEqual(g.skin_lab_match_mode, "reinhard")
        src = open(os.path.join(APP, "roop", "procmgr_masking.py"), encoding="utf-8").read()
        self.assertIn("mode=getattr(roop.globals, 'skin_lab_match_mode'", src)
        schema = json.load(open(os.path.join(APP, "settings.schema.json"), encoding="utf-8"))
        text = json.dumps(schema)
        self.assertIn("skin_lab_match_mode", text)


# ───────────────────────────── the SCRFD GN-kps weights the brief names ──────────────────────────────────────
class TestGnKpsPremise(unittest.TestCase):
    """GN-kps 10g is an optional pinned offline asset. Its InsightFace research
    terms remain explicit; registration does not grant a commercial licence.
    Buffalo's existing SCRFD remains available for backward compatibility."""

    def test_gnkps_is_registered_as_an_optional_offline_asset(self):
        from roop.model_integrity import lookup
        entry = lookup('scrfd_10g_gnkps.onnx')
        self.assertIsNotNone(entry)
        self.assertEqual(len(entry.sha256), 64)
        self.assertTrue(entry.offline_only)
        self.assertFalse(entry.required)
        self.assertIn('InsightFace', entry.license)

    def test_the_detector_that_exists_is_scrfd_10g_from_buffalo(self):
        self.assertTrue(os.path.isfile(os.path.join(APP, "models", "buffalo_l", "det_10g.onnx")))

    def test_notice_records_the_insightface_terms(self):
        notice = open(os.path.join(ROOT, "NOTICE.md"), encoding="utf-8").read()
        self.assertIn("InsightFace", notice)
        self.assertIn("non-commercial", notice.lower())


if __name__ == "__main__":
    unittest.main()
