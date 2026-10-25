"""app/core/fused_postprocess.py without a GPU: the keep-original matte, the dispatcher and the torch fallback.

Runs on CPU. The Triton kernel's parity with this same maths is ``tests/test_fused_postprocess.py`` (needs CUDA).
The manual-mask tests pin WHICH WAY the brush works, because the brief that asked for it got the sign wrong: production's
painted (exclude) region KEEPS the original pixels and the include brush only cancels an exclusion (ProcessMgr, "Apply manual
mask in canonical face-crop space"); neither can turn the swap ON where XSeg refused it.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")
from app.core import fused_postprocess as fp  # noqa: E402


def _case(B=2, H=72, W=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    sw = 0.25 + 0.5 * torch.rand(B, 3, H, W, generator=g)
    rs = (sw + 0.12 * (torch.rand(B, 3, H, W, generator=g) - 0.5)).clamp(0, 1)
    m = torch.zeros(B, 1, H, W)
    m[:, :, H // 6:5 * H // 6, W // 6:5 * W // 6] = 1.0
    tgt = torch.rand(B, 3, H, W, generator=g)
    return sw, rs, m, tgt


def _box(B, H, W, y0, y1, x0, x1):
    k = torch.zeros(B, 1, H, W)
    k[:, :, y0:y1, x0:x1] = 1.0
    return k


class TestManualKeepMask:
    def test_exclude_keeps_original_include_cancels_it_overrides_survive_include(self):
        B, H, W = 1, 40, 40
        exc, inc = _box(B, H, W, 5, 25, 5, 25), _box(B, H, W, 15, 35, 15, 35)
        mouth = _box(B, H, W, 30, 38, 30, 38)
        keep = fp.manual_keep_mask(exc, inc)
        assert keep[0, 0, 10, 10] == 1.0                    # painted, not erased: original stays
        assert keep[0, 0, 20, 20] == 0.0                    # painted AND included: exclusion cancelled
        assert keep[0, 0, 30, 30] == 0.0                    # include alone excludes nothing
        both = fp.manual_keep_mask(exc, inc, mouth)
        assert both[0, 0, 33, 33] == 1.0                    # the mouth override is not cancelled by the include brush
        assert fp.manual_keep_mask() is None and fp.manual_keep_mask(None, inc) is None
        assert torch.equal(fp.manual_keep_mask(None, None, mouth), mouth)

    def test_keep_equals_production_crop_space_composite(self):
        """ProcessMgr: fake' = fake*(1-c) + aligned*c, then composite with alpha. With aligned == plate that IS alpha*(1-c)."""
        sw, rs, m, tgt = _case()
        c = fp.manual_keep_mask(_box(2, 72, 80, 20, 50, 25, 60), _box(2, 72, 80, 35, 50, 45, 60))
        face = fp.reference_postprocess(sw, rs, torch.ones_like(m), tgt)                 # alpha == 1 everywhere: out is the face
        alpha = fp.blur_reference(m, 2.0).clamp(0, 1)
        production = tgt * (1 - alpha) + (face * (1 - c) + tgt * c) * alpha
        ours = fp.reference_postprocess(sw, rs, m, tgt, keep_mask=c)
        assert float((production - ours).abs().max()) < 1e-6

    def test_the_briefs_sign_would_swap_where_production_keeps_the_original(self):
        sw, rs, m, tgt = _case(seed=1)
        paint = _box(2, 72, 80, 20, 50, 25, 60)                                          # the user painted this region out
        inside = (slice(None), slice(None), slice(30, 40), slice(35, 50))
        ours = fp.reference_postprocess(sw, rs, m, tgt, keep_mask=paint)
        assert torch.equal(ours[inside], tgt[inside])                                    # original kept
        alpha_brief = (fp.blur_reference(m, 2.0) + paint - 0.0).clamp(0, 1)              # xseg*feather + paint - erase
        face = fp.reference_postprocess(sw, rs, torch.ones_like(m), tgt)
        brief = tgt + alpha_brief * (face - tgt)
        assert float((brief - tgt)[inside].abs().max()) > 0.05                            # ... the brief's formula swaps it


class TestLowAndDetailWeight:
    def test_low_zero_detail_one_returns_the_restorer(self):
        sw, rs, m, tgt = _case(seed=2)
        out = fp.reference_postprocess(sw, rs, torch.ones_like(m), tgt, low_weight=0.0, detail_weight=1.0)
        assert float((out - rs).abs().max()) < 1e-5

    def test_default_low_weight_is_the_shipped_maths(self):
        sw, rs, m, tgt = _case(seed=3)
        a = fp.reference_postprocess(sw, rs, m, tgt)
        b = fp.reference_postprocess(sw, rs, m, tgt, low_weight=1.0)
        assert torch.equal(a, b)
        assert not torch.equal(a, fp.reference_postprocess(sw, rs, m, tgt, low_weight=0.8))


class TestDispatcherAndFallback:
    def test_backend_selection_is_explicit(self):
        cpu = torch.zeros(1)
        assert fp.select_backend(cpu) == "torch" and fp.select_backend(cpu, "torch") == "torch"
        with pytest.raises(RuntimeError):
            fp.select_backend(cpu, "triton")                  # forced, cannot run: refuse, do not silently fall back
        with pytest.raises(ValueError):
            fp.select_backend(cpu, "inductor")

    @pytest.mark.parametrize("target_u8", [True, False])
    def test_fallback_matches_reference_and_target_layout(self, target_u8):
        sw, rs, m, tgt = _case(seed=4)
        c = _box(2, 72, 80, 10, 30, 10, 30)
        t = (tgt * 255).to(torch.uint8) if target_u8 else tgt
        got = fp.postprocess(sw, rs, m, t, keep_mask=c, low_weight=0.8, detail_weight=1.0)
        ref = fp.reference_postprocess(sw, rs, m, t, keep_mask=c, low_weight=0.8, detail_weight=1.0)
        assert got.dtype == t.dtype and got.shape == t.shape
        err = (got.float() / (255.0 if target_u8 else 1.0) - ref).abs().max()
        assert float(err) <= (0.5 / 255.0 + 1e-6 if target_u8 else 1e-6)

    def test_fallback_composites_into_a_view_and_touches_only_the_roi(self):
        sw, rs, m, _ = _case(B=1, H=40, W=50, seed=5)
        frame = (torch.rand(1, 3, 120, 160) * 255).to(torch.uint8)
        before = frame.clone()
        dest = frame[:, :, 30:70, 40:90]
        fp.postprocess(sw, rs, m, dest.clone(), out=dest)
        outside = torch.ones_like(frame, dtype=torch.bool)
        outside[:, :, 30:70, 40:90] = False
        assert torch.equal(frame[outside], before[outside]) and not torch.equal(frame, before)

    def test_fallback_grain_has_the_requested_sigma_and_zero_mean(self):
        flat = torch.full((1, 3, 160, 160), 0.5)
        ones = torch.ones(1, 1, 160, 160)
        out = fp.postprocess(flat, flat, ones, flat.clone(), blur_sigma=1.0, feather_sigma=1.0,
                             grain_sigma=torch.tensor([4.0]), seed=3)[:, :, 20:-20, 20:-20]
        noise = (out - 0.5) * 255.0
        assert abs(float(noise.mean())) < 0.15 and abs(float(noise.std()) - 4.0) < 0.25
        again = fp.postprocess(flat, flat, ones, flat.clone(), blur_sigma=1.0, feather_sigma=1.0,
                               grain_sigma=torch.tensor([4.0]), seed=3)[:, :, 20:-20, 20:-20]
        assert torch.equal(out, again)                        # repeatable per seed

    def test_keep_wrong_shape_is_refused_by_the_kernel_wrapper(self):
        sw, rs, m, tgt = _case(B=1)
        if not (torch.cuda.is_available() and fp.HAVE_TRITON):
            pytest.skip("kernel wrapper needs CUDA + triton (unverified here)")
        with pytest.raises(ValueError):
            fp.fused_postprocess(sw.cuda(), rs.cuda(), m.cuda(), tgt.cuda(), keep_mask=torch.zeros(1, 1, 3, 3, device="cuda"))


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())
