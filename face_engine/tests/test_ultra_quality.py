"""Ultra Restore: frequency split, scale routing, region weights, LAB lock, TensorRT GPEN.

Synthetic tests build faces with a KNOWN answer: a swap ``S`` with the right
lighting but soft detail, a restoration ``R`` with full detail but drifted
lighting and colour, and the ideal ``T`` = S's lighting + R's detail. GPU
tests use insightface's t1.jpg (six faces): 640x360 -> every face bypassed,
1920x1080 -> GPEN-512, 3840x2160 -> GPEN-1024. The real-footage numbers are
in face_engine/enhancers/ultra_engine.py's docstring.
"""
from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kornia")

from face_engine.enhancers import (
    FrequencySplitBlender,
    IrisStabilizer,
    Route,
    ScaleAwareEnhancerRouter,
    SemanticRegionalRestorer,
    gaussian_kernel_size,
    lab_lock,
    linear_blend,
    paste_aware_sigma,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------- synthetic faces
def _smooth(shape: tuple[int, ...], scale: int, gen: torch.Generator) -> torch.Tensor:
    """Low-frequency field: coarse noise upsampled (lighting / colour drift)."""
    coarse = torch.rand((shape[0], shape[1], scale, scale), generator=gen)
    return torch.nn.functional.interpolate(coarse, size=shape[-2:], mode="bicubic",
                                           align_corners=False)


def _case(seed: int, size: int = 128) -> dict[str, torch.Tensor]:
    """S (right light, soft detail), R (detail, wrong light + cast), T (ideal)."""
    from kornia.filters import gaussian_blur2d

    g = torch.Generator().manual_seed(seed)
    shape = (1, 3, size, size)
    light_s = 60 + 120 * _smooth(shape, 4, g)                        # swap's lighting
    drift = 30 * (_smooth(shape, 3, g) - 0.5) + torch.tensor([6.0, -4.0, 9.0]).view(1, 3, 1, 1)
    grain = torch.randn((1, 1, size, size), generator=g) * 9.0          # pores
    texture = grain - gaussian_blur2d(grain, (17, 17), (2.5, 2.5))      # zero-mean high band
    s = light_s + gaussian_blur2d(texture, (9, 9), (2.0, 2.0))          # soft detail
    r = light_s + drift + texture                                       # drifted light, full detail
    t = light_s + texture                                               # ideal
    return {k: v.clamp(0, 255).to(DEV) for k, v in {"S": s, "R": r, "T": t}.items()}


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(255 ** 2 / max(mse, 1e-12))


def _band_ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """SSIM of the high bands (offset to mid-grey, kornia's SSIM)."""
    from kornia.filters import gaussian_blur2d
    from kornia.metrics import ssim

    def hi(x: torch.Tensor) -> torch.Tensor:
        return ((x - gaussian_blur2d(x, (17, 17), (2.5, 2.5))) / 255.0 + 0.5).clamp(0, 1)

    return float(ssim(hi(a), hi(b), 7).mean())


def _low(x: torch.Tensor) -> torch.Tensor:
    from kornia.filters import gaussian_blur2d

    return gaussian_blur2d(x, (33, 33), (6.0, 6.0))


# ---------------------------------------------------------------------------- frequency split
def test_split_is_exact_and_the_brief_formula_holds() -> None:
    c = _case(0)
    blender = FrequencySplitBlender(sigma=2.5, kernel_size=9)
    lo, hi = blender.split(c["R"])
    assert torch.equal(lo + hi, c["R"].float()) or float((lo + hi - c["R"]).abs().max()) < 1e-4
    fused = blender.blend(c["S"], c["R"], texture_boost=0.7)
    ls = blender.low(c["S"].float())
    manual = (ls + 0.7 * (c["R"].float() - blender.low(c["R"].float()))).clamp(0, 255)
    assert float((fused - manual).abs().max()) < 1e-4
    unit = FrequencySplitBlender(max_value=1.0).blend(c["S"] / 255, c["R"] / 255 * 3)
    assert float(unit.min()) >= 0.0 and float(unit.max()) <= 1.0     # clamped to [0, 1]


def test_crossfade_endpoints_are_the_swap_and_the_full_restorer_detail() -> None:
    c = _case(1)
    b = FrequencySplitBlender(swap_detail="complement")
    assert float((b.blend(c["S"], c["R"], texture_boost=0.0) - c["S"]).abs().max()) < 1e-3
    full = b.blend(c["S"], c["R"], texture_boost=1.0)
    brief = FrequencySplitBlender(swap_detail=0.0).blend(c["S"], c["R"], texture_boost=1.0)
    assert float((full - brief).abs().max()) < 1e-3


def test_200_iterations_split_beats_linear_blend_without_moving_macro_luminance() -> None:
    """The brief's comparison: naive linear blend vs frequency-separated blend."""
    blender = FrequencySplitBlender(sigma=2.5)
    rows = []
    for seed in range(200):
        c = _case(seed)
        split = blender.blend(c["S"], c["R"], texture_boost=1.0)
        lin = linear_blend(c["S"], c["R"], 0.8)
        rows.append((_psnr(split, c["T"]), _psnr(lin, c["T"]),
                     _band_ssim(split, c["T"]), _band_ssim(lin, c["T"]),
                     float((_low(split) - _low(c["S"])).abs().mean()),
                     float((_low(lin) - _low(c["S"])).abs().mean())))
    a = np.array(rows)
    assert (a[:, 0] > a[:, 1]).all()                      # PSNR vs the ideal, every case
    assert (a[:, 2] > a[:, 3]).all()                      # fine-detail SSIM, every case
    assert np.median(a[:, 4]) < 0.5 and np.median(a[:, 5]) > 5   # macro luminance (levels)
    print(f"\n200 cases: PSNR split {np.median(a[:, 0]):.2f} vs linear {np.median(a[:, 1]):.2f} dB; "
          f"detail SSIM {np.median(a[:, 2]):.3f} vs {np.median(a[:, 3]):.3f}; "
          f"macro shift {np.median(a[:, 4]):.2f} vs {np.median(a[:, 5]):.2f} levels")


def test_the_briefs_9_tap_kernel_is_not_a_sigma_2_5_gaussian() -> None:
    """k=9 truncates sigma 2.5 at 1.6 sigma: it splits like a SMALLER sigma (less of a
    24 px wave reaches the detail band); the default 2*ceil(3 sigma)+1 = 17 matches
    the analytic Gaussian."""
    assert gaussian_kernel_size(2.5) == 17
    x = torch.arange(256, dtype=torch.float32, device=DEV)
    wave = (128 + 60 * torch.sin(2 * math.pi * x / 24)).view(1, 1, 1, -1).expand(1, 3, 256, 256)
    band9 = float(FrequencySplitBlender(kernel_size=9).split(wave)[1][..., 32:-32].abs().mean())
    band17 = float(FrequencySplitBlender().split(wave)[1][..., 32:-32].abs().mean())
    ideal = 60 * (1 - math.exp(-2 * (math.pi * 2.5 / 24) ** 2)) * 2 / math.pi  # true Gaussian
    assert abs(band17 - ideal) < 0.05 * ideal
    assert band9 < 0.8 * ideal


def test_paste_aware_sigma_is_constant_on_screen() -> None:
    ratio = torch.tensor([0.25, 0.5, 1.0])
    s = paste_aware_sigma(ratio, screen_sigma=1.25)
    assert torch.allclose(s * ratio, torch.full((3,), 1.25))
    assert float(paste_aware_sigma(torch.tensor([0.01]))) == 8.0      # clamped


# ---------------------------------------------------------------------------- router
def _boxes(diagonals: list[float]) -> torch.Tensor:
    side = torch.tensor(diagonals) / math.sqrt(2)
    return torch.stack([torch.zeros_like(side), torch.zeros_like(side), side, side], 1).to(DEV)


def test_router_thresholds_and_telemetry() -> None:
    seen = []
    router = ScaleAwareEnhancerRouter(telemetry=seen.append)
    plan = router.plan(_boxes([40, 119.9, 120, 200, 349.9, 350, 900]))
    assert plan.routes.tolist() == [0, 0, 1, 1, 1, 2, 2]
    assert plan.counts == {Route.BYPASS: 2, Route.GPEN_512: 3, Route.GPEN_1024: 2}
    assert seen[-1]["routes"] == {"bypass": 2, "gpen_512": 3, "gpen_1024": 2}
    assert 0.0 < plan.saving < 1.0 and seen[-1]["saving"] == round(plan.saving, 4)
    assert router.stats.faces["bypass"] == 2 and router.stats.batches == 1


def test_small_faces_skip_the_network_entirely() -> None:
    """< 120 px diagonal: output is the input frame, bit for bit, and no network runs."""
    from face_engine.enhancers import UltraRestorer

    class Exploding:
        def restore(self, *a: Any, **k: Any) -> Any:
            raise AssertionError("a bypassed face reached the restorer")

    frames = torch.rand(1, 3, 360, 640, device=DEV) * 255
    kps = torch.tensor([[[300, 170], [330, 170], [315, 185], [303, 200], [327, 200]]],
                       dtype=torch.float32, device=DEV)
    out = UltraRestorer(Exploding()).process(frames, kps, _boxes([80.0]))
    assert torch.equal(out.frames, frames) and out.plan.counts == {Route.BYPASS: 1}
    assert out.restored.tolist() == [False] and out.plan.saving == 1.0


@cuda
def test_roi_paste_equals_the_full_frame_paste() -> None:
    from face_engine.enhancers import paste_roi
    from face_engine.pipeline.aligner import (
        similarity_matrices_cuda,
        warp_face_inverse_cuda,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    frames = torch.rand((2, 3, 540, 960), device="cuda", generator=g) * 255
    kps = torch.tensor([[[400, 250], [460, 250], [430, 285], [405, 320], [455, 320]],
                        [[700, 100], [780, 110], [735, 150], [705, 190], [770, 195]],
                        [[930, 500], [990, 500], [960, 530], [935, 560], [985, 560]]],
                       dtype=torch.float32, device="cuda")       # the last one runs off the frame
    m = similarity_matrices_cuda(kps, 512, "ffhq_512")
    crops = torch.rand((3, 3, 512, 512), device="cuda", generator=g) * 255
    mask = torch.rand((3, 1, 512, 512), device="cuda", generator=g)
    fi = torch.tensor([0, 1, 1], device="cuda")
    full = warp_face_inverse_cuda(frames, crops, m, mask, frame_index=fi)
    roi = paste_roi(frames.clone(), crops, m, mask, fi)
    # kornia normalizes the grid by the output size, so a smaller canvas rounds
    # differently: 0.03 / 255 measured, far under the 0.71 / 255 render noise floor.
    assert float((full - roi).abs().max()) < 0.1


# ---------------------------------------------------------------------------- region weights
def test_label_weights_follow_the_region_table() -> None:
    sem = SemanticRegionalRestorer(feather_kernel=1)
    labels = torch.zeros((1, 512, 512), dtype=torch.long, device=DEV)
    for cls, (y0, y1) in {1: (0, 100), 4: (100, 200), 11: (200, 300), 12: (300, 400),
                          17: (400, 512)}.items():
        labels[:, y0:y1] = cls
    w = sem.label_weights(labels, 1024)[0, 0]
    at = lambda y: round(float(w[y, 500]), 3)
    assert (at(100), at(300), at(500), at(700), at(900)) == (0.7, 0.95, 0.15, 0.7, 0.0)


def test_zero_weight_regions_return_the_swap_exactly() -> None:
    c = _case(2, 256)
    sem = SemanticRegionalRestorer()
    hair = torch.full((1, 512, 512), 17, dtype=torch.long, device=DEV)  # hair: weight 0
    fused = sem.fuse(c["S"], c["R"], labels=hair).fused
    assert float((fused - c["S"]).abs().max()) < 1e-3


def test_inner_mouth_keeps_the_swap_detail_the_brief_formula_would_erase() -> None:
    """Swap and restorer each carry their OWN sharp mouth detail (teeth edges vs GPEN's
    synthetic teeth). At weight 0.15 the crossfade keeps mostly the swap's; the
    brief's ``L_S + 0.15 H_R`` deletes it."""
    from kornia.filters import gaussian_blur2d

    g = torch.Generator().manual_seed(5)
    light = 90 + 60 * _smooth((1, 3, 256, 256), 4, g)

    def sharp() -> torch.Tensor:
        n = torch.randn((1, 1, 256, 256), generator=g) * 12
        return n - gaussian_blur2d(n, (17, 17), (2.5, 2.5))

    swap_teeth, gpen_teeth = sharp(), sharp()
    s = (light + swap_teeth).clamp(0, 255).to(DEV)
    r = (light + gpen_teeth).clamp(0, 255).to(DEV)
    mouth = torch.full((1, 512, 512), 11, dtype=torch.long, device=DEV)
    ours = SemanticRegionalRestorer().fuse(s, r, labels=mouth).fused
    brief = FrequencySplitBlender().blend(s, r, texture_boost=0.15)
    split = FrequencySplitBlender()

    def corr(x: torch.Tensor, detail: torch.Tensor) -> float:
        hx = (x - split.low(x)).mean(1).flatten()
        hd = (detail.to(DEV) - split.low(detail.to(DEV).expand(1, 3, 256, 256)).mean(1,
              keepdim=True)).flatten()
        return float(torch.corrcoef(torch.stack([hx, hd]))[0, 1])

    assert corr(ours, swap_teeth) > 0.9                  # the swap's teeth survive
    assert corr(brief, swap_teeth) < 0.3                 # the brief's form erases them


def test_landmark_weights_place_eyes_mouth_and_skin() -> None:
    from face_engine.pipeline.aligner import template_tensor

    kps = template_tensor(512, "ffhq_512", device=DEV)[None]
    w = SemanticRegionalRestorer().landmark_weights(kps, 512)[0, 0]
    le, re, _, lm, rm = kps[0].round().long().tolist()
    mouth = [(lm[0] + rm[0]) // 2, (lm[1] + rm[1]) // 2]
    cheek = [(le[0] + lm[0]) // 2 - 10, (le[1] + lm[1]) // 2]
    val = lambda p: round(float(w[p[1], p[0]]), 2)
    assert val(le) == val(re) == 0.95 and val(mouth) == 0.15 and val(cheek) == 0.70
    assert val([5, 5]) == 0.0


# ---------------------------------------------------------------------------- colour lock
def test_lab_lock_matches_mean_and_std_and_can_spare_luma_contrast() -> None:
    from face_engine.processors.color import bgr_to_lab

    c = _case(4, 128)
    locked = lab_lock(c["R"], c["S"])
    a, b = bgr_to_lab(locked), bgr_to_lab(c["S"])
    assert torch.allclose(a.mean((2, 3)), b.mean((2, 3)), atol=0.3)
    assert torch.allclose(a.std((2, 3)), b.std((2, 3)), rtol=0.03, atol=0.2)
    spare = bgr_to_lab(lab_lock(c["R"], c["S"], match_luma_std=False))
    r = bgr_to_lab(c["R"])
    assert abs(float(spare[:, 0].std()) - float(r[:, 0].std())) < 0.3   # L* spread kept


# ---------------------------------------------------------------------------- iris stabilizer
def test_iris_stabilizer_damps_flicker_follows_motion_and_resets_on_saccades() -> None:
    size = 64
    g = torch.Generator().manual_seed(0)
    eye_w = torch.zeros((1, 1, size, size), device=DEV)
    eye_w[..., 16:48, 16:48] = 1.0
    kps = torch.tensor([[[24, 32], [40, 32], [32, 40], [26, 50], [38, 50]]],
                       dtype=torch.float32, device=DEV)
    stab = IrisStabilizer(beta=0.5, reset_px=6.0)
    raw, out = [], []
    for _ in range(20):  # a still eye whose restored texture flickers frame to frame
        h = torch.randn((1, 3, size, size), generator=g).to(DEV) * 10
        raw.append(h)
        out.append(stab(h, eye_w, kps, ["a"]))
    t_raw = torch.stack(raw)[5:, ..., 20:44, 20:44].std(0).mean()
    t_out = torch.stack(out)[5:, ..., 20:44, 20:44].std(0).mean()
    assert float(t_out) < 0.7 * float(t_raw)
    before = stab.resets
    stab(raw[0], eye_w, kps + 20.0, ["a"])                    # a 20 px jump: history dropped
    assert stab.resets == before + 1
    # A pattern moving 2 px/frame with its landmarks is carried, not smeared.
    s2 = IrisStabilizer(beta=0.5)
    pattern = torch.zeros((1, 3, size, size), device=DEV)
    pattern[..., 30:34, 30:34] = 50
    for step in range(4):
        moved = torch.roll(pattern, 2 * step, dims=3)
        res = s2(moved, eye_w, kps + torch.tensor([2.0 * step, 0], device=DEV), ["b"])
    peak = int(res[0, 0, 32].argmax())
    assert 36 <= peak <= 39                                   # at the moved position (30 + 6)


# ---------------------------------------------------------------------------- GPU (TensorRT)
@pytest.fixture(scope="module")
def gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    import cv2
    import insightface

    from face_engine.core.config import EngineConfig, Provider
    from face_engine.core.execution import ExecutionEngine
    from face_engine.enhancers import UltraRestoreEngine
    from face_engine.models.zoo import build_default_registry
    from face_engine.pipeline.detector import GPUSCRFDDetector

    reg = build_default_registry()
    engine = ExecutionEngine(EngineConfig(providers=[Provider.TENSORRT, Provider.CUDA]))
    image = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    det = GPUSCRFDDetector(engine, reg.ensure("scrfd_10g_bnkps"))
    frames = {}
    for w, h in ((1920, 1080), (3840, 2160)):
        f = torch.from_numpy(cv2.resize(image, (w, h))).cuda().permute(2, 0, 1)[None].float()
        d = det.detect_cuda(f)
        frames[w] = (f, d.kps, d.boxes)
    paths = {m: reg.ensure(m) for m in ("gpen_bfr_512", "gpen_bfr_1024")}
    return {"engine": engine, "reg": reg, "frames": frames,
            "ure": UltraRestoreEngine(engine, paths)}


def _need_trt(gpu: dict[str, Any], model: str) -> None:
    if not gpu["ure"].uses_tensorrt(model):
        pytest.skip(f"no compiled {model} fp16 engine (tools/compile_engines.py --models {model})")


@cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("width", "route"), [(1920, Route.GPEN_512), (3840, Route.GPEN_1024)])
def test_end_to_end_routes_restores_and_stays_on_the_gpu(gpu: dict[str, Any], width: int,
                                                        route: Route) -> None:
    from face_engine.enhancers import UltraRestorer
    from face_engine.pipeline.masker import GPUMasker

    # Without a compiled engine the restorer falls back to an in-process ONNX Runtime /
    # TensorRT build of the GPEN network: 400+ s for GPEN-512 on an RTX 4070 (150 s for 1024),
    # which reads as a hung test run. Check the compiled engine WITHOUT building anything
    # (`_need_trt` goes through `uses_tensorrt`, whose fallback is that very build).
    from face_engine.core.trt_compiler import aot_engine
    model = "gpen_bfr_512" if route == Route.GPEN_512 else "gpen_bfr_1024"
    if aot_engine(gpu["ure"].model_paths[model], gpu["ure"].precision, gpu["engine"].config) is None:
        pytest.skip(f"no compiled {model} fp16 engine; run "
                    f"`python tools/compile_engines.py --models {model}` once (minutes)")

    frame, kps, boxes = gpu["frames"][width]
    parser = GPUMasker(gpu["engine"], bisenet_path=gpu["reg"].ensure("bisenet_resnet34"))
    out = UltraRestorer(gpu["ure"], parser=parser).process(frame, kps, boxes, reference=frame)
    assert out.plan.counts == {route: 6}
    assert out.frames.device == frame.device and bool(out.restored.all())
    assert bool(torch.isfinite(out.frames).all()) and float((out.frames - frame).abs().max()) > 5


@cuda
@pytest.mark.gpu
def test_engine_buffers_are_bound_once_and_batches_equal_single_faces(gpu: dict[str, Any]) -> None:
    from face_engine.pipeline.aligner import similarity_matrices_cuda, warp_face_cuda

    _need_trt(gpu, "gpen_bfr_512")
    frame, kps, _ = gpu["frames"][1920]
    ure = gpu["ure"]
    m = similarity_matrices_cuda(kps[:4], 512, "ffhq_512")
    crops = warp_face_cuda(frame, m, 512, padding_mode="border").clamp(0, 255)
    bound = ure._bind("gpen_bfr_512")
    ptr = bound.inputs.data_ptr()
    batch, ok = ure.restore("gpen_bfr_512", crops)
    assert bool(ok.all()) and bound.inputs.data_ptr() == ptr        # no rebinding
    for i in range(4):
        one, _ = ure.restore("gpen_bfr_512", crops[i:i + 1])
        assert float((one[0] - batch[i]).abs().max()) < 1e-3


@cuda
@pytest.mark.gpu
def test_split_keeps_identity_that_linear_blending_loses(gpu: dict[str, Any]) -> None:
    """t1 at 4K through GPEN-1024: identity of the pasted face to the input face."""
    from face_engine.enhancers import UltraRestorer
    from face_engine.processors.color import ColorMode
    from face_engine.processors.enhancer import BatchedFaceEnhancer
    from face_engine.processors.swapper import GPUIdentityEncoder

    _need_trt(gpu, "gpen_bfr_1024")
    frame, kps, boxes = gpu["frames"][3840]
    arc = GPUIdentityEncoder(gpu["engine"], gpu["reg"].ensure("arcface_w600k_r50"))
    linear = BatchedFaceEnhancer(gpu["engine"], "gpen_bfr_1024", gpu["reg"].ensure("gpen_bfr_1024"),
                                 color=ColorMode.LAB_MEAN)
    e_in = arc.embed(frame, kps)
    lin = linear.enhance(frame, kps, alpha=0.8).frames
    split = UltraRestorer(gpu["ure"]).process(frame, kps, boxes, reference=frame).frames
    id_lin = (arc.embed(lin, kps) * e_in).sum(1)
    id_split = (arc.embed(split, kps) * e_in).sum(1)
    print(f"\nidentity to input: linear {id_lin.tolist()}\n                   split  {id_split.tolist()}")
    assert bool((id_split > id_lin).all())


def _latency(fn: Any, n: int = 50) -> float:
    for _ in range(5):
        fn()
    times = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def _one_face_ms(gpu: dict[str, Any], width: int) -> tuple[float, float]:
    """(end-to-end ms per face, network-only ms per face) on the 4070."""
    from face_engine.enhancers import UltraRestorer
    from face_engine.pipeline.aligner import similarity_matrices_cuda, warp_face_cuda
    from face_engine.pipeline.masker import GPUMasker

    frame, kps, boxes = gpu["frames"][width]
    model, size = ("gpen_bfr_512", 512) if width == 1920 else ("gpen_bfr_1024", 1024)
    _need_trt(gpu, model)
    parser = GPUMasker(gpu["engine"], bisenet_path=gpu["reg"].ensure("bisenet_resnet34"))
    ur = UltraRestorer(gpu["ure"], parser=parser)
    k, b = kps[:1], boxes[:1]
    total = _latency(lambda: ur.process(frame, k, b, reference=frame))
    crop = warp_face_cuda(frame, similarity_matrices_cuda(k, size, "ffhq_512"), size)
    network = _latency(lambda: gpu["ure"].restore(model, crop))
    return total, network


def _rtx4070() -> None:
    if "4070" not in torch.cuda.get_device_name(0):
        pytest.skip("latency budgets are for an RTX 4070")


@cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("width", "overhead_ms"), [(1920, 14.0), (3840, 16.0)])
def test_everything_around_the_network_fits_its_budget(gpu: dict[str, Any], width: int,
                                                       overhead_ms: float) -> None:
    """Routing, crop, LAB lock, BiSeNet, region weights, split and paste per face.

    A regression guard at the measured cost + ~25% (2026-09-28: 11.3 ms at 1080p /
    GPEN-512, 13.1 ms at 4K / GPEN-1024; with sync barriers: LAB lock 2.5-3.0,
    ROI paste 2.3 (kornia's warp syncs 4x per call), weights + split 1.9-3.1,
    BiSeNet 1.7, crop warp 1.3-1.6). Before the ROI paste and 128 px colour
    statistics it was 12.1 / 19.5 ms."""
    _rtx4070()
    total, network = _one_face_ms(gpu, width)
    print(f"\n{width}: end to end {total:.2f} ms/face, network {network:.2f}, "
          f"rest {total - network:.2f}")
    assert total - network < overhead_ms


@cuda
@pytest.mark.gpu
@pytest.mark.xfail(strict=True, reason="the brief's < 3.5 / < 9.0 ms: GPEN-512 / GPEN-1024 "
                   "networks alone cost 17.8 / 32.3 ms on an RTX 4070 (TensorRT FP16)")
@pytest.mark.parametrize(("width", "target_ms"), [(1920, 3.5), (3840, 9.0)])
def test_briefs_end_to_end_latency_targets(gpu: dict[str, Any], width: int,
                                           target_ms: float) -> None:
    _rtx4070()
    total, _ = _one_face_ms(gpu, width)
    assert total < target_ms
