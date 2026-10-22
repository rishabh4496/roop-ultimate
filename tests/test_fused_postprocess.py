"""app/core/fused_postprocess.py: the one-kernel Stage 3 post-processing must equal the unfused maths it replaces.

Needs a CUDA device and Triton; without them every test skips WITH the reason (a skip is "unverified", not "passed").
The kernel is judged against ``reference_postprocess`` (the same maths as plain torch ops), and the maths in turn against
OpenCV (LAB conversion, Gaussian with reflect borders) so the oracle is not just the kernel agreeing with its twin.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")
from app.core import fused_postprocess as fp  # noqa: E402

pytestmark = pytest.mark.gpu
if not (torch.cuda.is_available() and fp.HAVE_TRITON):
    pytest.skip("needs CUDA and triton (pip install triton-windows)", allow_module_level=True)

DEV = "cuda"


def _case(B=2, H=96, W=112, seed=0, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g).to(DEV)
    sw = (0.25 + 0.5 * r(B, 3, H, W)).to(dtype)
    rs = (sw.float() + 0.12 * (r(B, 3, H, W) - 0.5)).clamp(0, 1).to(dtype)
    m = torch.zeros(B, 1, H, W)
    m[:, :, H // 6:5 * H // 6, W // 6:5 * W // 6] = 1.0
    tgt = (r(B, 3, H, W) * 255).to(torch.uint8)
    return sw.contiguous(), rs.contiguous(), m.to(DEV).to(dtype).contiguous(), tgt


def _lab(sw, tgt):
    sm, ss = fp.lab_stats(sw)
    tm, ts = fp.lab_stats(tgt.float() / 255.0)
    return fp.reinhard_params(sm, ss, tm, ts)


def _close(a, b, tol_levels):
    d = (a.float() - b.float()).abs()
    return float(d.max()) <= tol_levels, float(d.max())


class TestMathAgainstOpenCV:
    def test_lab_matches_cv2_on_float_input(self):
        import cv2
        rgb = np.random.default_rng(1).random((40, 50, 3)).astype(np.float32)
        ref = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        mine = fp.rgb_to_lab(torch.from_numpy(rgb).permute(2, 0, 1)[None]).permute(0, 2, 3, 1)[0].numpy()
        # OpenCV's float RGB->Lab is a gamma LUT with interpolation (its a/b come out on a 1/16 grid), so it is the APPROXIMATE
        # one: max 0.36 / mean 0.1 LAB units off the exact CIE formulas used here (L spans 0..100, 1 unit ~ 1 JND).
        assert np.abs(mine - ref).max() < 0.5 and np.abs(mine - ref).mean() < 0.15

    def test_lab_round_trip(self):
        rgb = torch.rand(1, 3, 30, 30, device=DEV)
        assert float((fp.lab_to_rgb(fp.rgb_to_lab(rgb)) - rgb).abs().max()) < 2e-4

    @pytest.mark.parametrize("sigma", [1.0, 2.0, 3.0])
    def test_blur_matches_cv2_gaussian_with_reflect_borders(self, sigma):
        import cv2
        x = np.random.default_rng(2).random((60, 70)).astype(np.float32)
        r, _ = fp.gaussian_taps(sigma)
        ref = cv2.sepFilter2D(x, -1, (k := cv2.getGaussianKernel(2 * r + 1, sigma, cv2.CV_32F)), k, borderType=cv2.BORDER_REFLECT)
        mine = fp.blur_reference(torch.from_numpy(x)[None, None].to(DEV), sigma)[0, 0].cpu().numpy()
        assert np.abs(mine - ref).max() < 1e-5


class TestKernelEqualsReference:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_everything_but_grain(self, dtype):
        sw, rs, m, tgt = _case(dtype=dtype)
        lab = _lab(sw.float(), tgt)
        got = fp.fused_postprocess(sw, rs, m, tgt, lab=lab)
        ref = fp.reference_postprocess(sw, rs, m, tgt, lab=lab)
        ok, worst = _close(got.float() / 255.0, ref, 1.5 / 255.0)                 # uint8 rounding
        assert ok, f"{worst * 255:.2f} levels"

    def test_no_colour_transfer_and_float_target(self):
        sw, rs, m, tgt = _case(seed=3)
        tgt_f = tgt.float() / 255.0
        got = fp.fused_postprocess(sw, rs, m, tgt_f)
        ref = fp.reference_postprocess(sw, rs, m, tgt_f)
        assert _close(got, ref, 2e-4)[0]

    @pytest.mark.parametrize("H,W", [(64, 64), (97, 113), (130, 70), (48, 200)])
    def test_odd_shapes_and_borders(self, H, W):
        sw, rs, m, tgt = _case(B=1, H=H, W=W, seed=H)
        got = fp.fused_postprocess(sw, rs, m, tgt.float() / 255.0)
        ref = fp.reference_postprocess(sw, rs, m, tgt.float() / 255.0)
        assert _close(got, ref, 2e-4)[0]

    def test_bgr_order(self):
        sw, rs, m, tgt = _case(seed=4)
        lab = _lab(sw, tgt)
        got = fp.fused_postprocess(sw, rs, m, tgt.float() / 255.0, lab=lab, bgr=True)
        ref = fp.reference_postprocess(sw, rs, m, tgt.float() / 255.0, lab=lab, bgr=True)
        assert _close(got, ref, 3e-4)[0]
        assert not _close(got, fp.reference_postprocess(sw, rs, m, tgt.float() / 255.0, lab=lab, bgr=False), 3e-4)[0]

    @pytest.mark.parametrize("sigma,feather", [(1.0, 1.0), (2.0, 2.5), (2.5, 1.5)])
    def test_sigmas(self, sigma, feather):
        sw, rs, m, tgt = _case(seed=5)
        got = fp.fused_postprocess(sw, rs, m, tgt.float() / 255.0, blur_sigma=sigma, feather_sigma=feather, detail_weight=0.6)
        ref = fp.reference_postprocess(sw, rs, m, tgt.float() / 255.0, blur_sigma=sigma, feather_sigma=feather, detail_weight=0.6)
        assert _close(got, ref, 3e-4)[0]

    def test_alpha_zero_leaves_the_target_untouched(self):
        sw, rs, _, tgt = _case(seed=6)
        zero = torch.zeros(2, 1, *sw.shape[-2:], device=DEV)
        assert torch.equal(fp.fused_postprocess(sw, rs, zero, tgt, lab=_lab(sw, tgt)), tgt)

    def test_composite_into_a_strided_view_of_the_frame_touches_only_the_roi(self):
        sw, rs, m, tgt = _case(B=1, H=80, W=90, seed=7)
        frame = (torch.rand(1, 3, 240, 320, device=DEV) * 255).to(torch.uint8)
        before = frame.clone()
        y0, x0 = 50, 70
        roi_in = frame[:, :, y0:y0 + 80, x0:x0 + 90].clone()
        dest = frame[:, :, y0:y0 + 80, x0:x0 + 90]
        fp.fused_postprocess(sw, rs, m, frame[:, :, y0:y0 + 80, x0:x0 + 90], out=dest)
        outside = torch.ones_like(frame, dtype=torch.bool)
        outside[:, :, y0:y0 + 80, x0:x0 + 90] = False
        assert torch.equal(frame[outside], before[outside])
        ref = fp.reference_postprocess(sw, rs, m, roi_in)
        assert _close(dest.float() / 255.0, ref, 1.5 / 255.0)[0]


class TestGrain:
    def _run(self, sigma, seed=11):
        sw, rs, _, tgt = _case(B=1, H=128, W=128, seed=9)
        ones = torch.ones(1, 1, 128, 128, device=DEV)
        tgt_f = torch.full((1, 3, 128, 128), 0.5, device=DEV)
        gs = torch.full((1,), float(sigma), device=DEV)
        flat = torch.full_like(sw, 0.5)
        return fp.fused_postprocess(flat, flat, ones, tgt_f, grain_sigma=gs, seed=seed, blur_sigma=1.0, feather_sigma=1.0)

    def test_grain_has_the_requested_sigma_zero_mean_and_is_luminance_only(self):
        out = self._run(4.0)[:, :, 20:-20, 20:-20]
        noise = (out - 0.5) * 255.0
        assert abs(float(noise.mean())) < 0.1
        assert abs(float(noise.std()) - 4.0) < 0.2
        assert torch.equal(noise[:, 0], noise[:, 1]) and torch.equal(noise[:, 1], noise[:, 2])

    def test_repeatable_and_seed_dependent_and_off_at_zero(self):
        a, b, c = self._run(3.0, 1), self._run(3.0, 1), self._run(3.0, 2)
        assert torch.equal(a, b) and not torch.equal(a, c)
        assert float((self._run(0.0) - 0.5).abs().max()) < 1e-6

    def test_estimator_and_deficit(self):
        g = torch.Generator(device="cpu").manual_seed(0)
        roi = (0.5 + 0.05 * torch.randn(2, 3, 120, 120, generator=g)).clamp(0, 1).to(DEV)        # sigma ~ 12.75 per channel
        est = fp.estimate_noise_sigma(roi)
        assert torch.allclose(est, torch.full((2,), 12.75 * 0.67, device=DEV), rtol=0.1)           # luma of 3 independent channels
        assert float(fp.grain_deficit(torch.tensor(5.0), torch.tensor(2.0))) == pytest.approx(21 ** 0.5)
        assert float(fp.grain_deficit(torch.tensor(1.0), torch.tensor(3.0))) == 0.0


class TestStatsEMA:
    def test_first_frame_is_raw_then_blends_and_a_cut_restarts(self):
        ema = fp.LabStatsEMA(alpha=0.5, cut_threshold=18.0)
        m0, s0 = torch.tensor([50.0, 0, 0]), torch.tensor([10.0, 5, 5])
        assert torch.equal(ema.update(1, m0, s0)[0], m0)
        assert torch.allclose(ema.update(1, m0 + 4, s0)[0], m0 + 2)
        assert torch.equal(ema.update(1, m0 + 60, s0)[0], m0 + 60)                    # cut: restart, no blend
        assert torch.equal(ema.update(2, m0, s0)[0], m0)                              # per-face-id
        ema.reset()
        assert ema.state == {}


class TestRefusals:
    def test_bad_inputs_are_refused_not_miscomputed(self):
        sw, rs, m, tgt = _case(B=1, H=64, W=64)
        with pytest.raises(ValueError):
            fp.fused_postprocess(sw.cpu(), rs, m, tgt)
        with pytest.raises(ValueError):
            fp.fused_postprocess(sw[:, :2].contiguous(), rs, m, tgt)
        with pytest.raises(ValueError):
            fp.fused_postprocess(sw, rs, m, tgt, blur_sigma=3.0)                      # halo 9: more than a 32 px tile can carry
        with pytest.raises(ValueError):
            fp.fused_postprocess(sw, rs, m, tgt, out=torch.empty(1, 3, 64, 64, device=DEV))   # dtype mismatch with target


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())
