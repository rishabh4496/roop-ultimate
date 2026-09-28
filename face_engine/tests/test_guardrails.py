"""Stage 8 guardrails: degenerate geometry, zero-face pass-through, pose policy, A/V sync.

The processor cases inject landmarks in place of detection on the insightface
sample photo, through the real models of the Balanced and Cinema presets.
Before the guard, degenerate / NaN / inf landmarks raised ``linalg.inv:
singular`` inside kornia and one NaN face turned the whole frame NaN.
"""
from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from face_engine.core.cuda_streams import CUDAStreamPipeline
from face_engine.core.guardrails import (
    DENSE_MAX_YAW_DEG,
    check_av_sync,
    choose_alignment_points,
    estimate_yaw_5pt,
    safe_affine,
    valid_landmarks,
)
from face_engine.media.tools import find_tool
from face_engine.pipeline.aligner import (
    similarity_matrices_cuda,
    warp_face_cuda,
    warp_face_inverse_cuda,
)
from face_engine.pipeline.detector import GPUDetections
from face_engine.server.processing import (
    PRESETS,
    FramePlan,
    FrameStats,
    GpuFrameProcessor,
    ProcessorConfig,
    RenderParams,
    model_paths,
    required_models,
)
from face_engine.tests import media_fixtures as mf

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
FRONTAL = np.array([[38.3, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]])


# ---------------------------------------------------------------------------- pure
def test_safe_affine_replaces_only_unusable_matrices() -> None:
    good = torch.tensor([[2.0, 0.1, 5.0], [-0.1, 2.0, 7.0]])
    bad = [torch.full((2, 3), float("nan")), torch.full((2, 3), float("inf")),
           torch.zeros(2, 3), torch.tensor([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0]])]  # det 0
    m, ok = safe_affine(torch.stack([good, *bad]))
    assert ok.tolist() == [True, False, False, False, False]
    assert torch.equal(m[0], good)
    assert torch.isfinite(m).all()
    assert torch.equal(m[1:, :, :2], torch.eye(2).expand(4, 2, 2))


def test_valid_landmarks_rejects_non_finite_and_collapsed_points() -> None:
    cases = np.stack([FRONTAL, np.full((5, 2), 500.0), np.full((5, 2), np.nan),
                      FRONTAL + np.array([[np.inf, 0]] + [[0, 0]] * 4), FRONTAL - 1e6])
    expected = [True, False, False, False, True]  # far away is still geometry
    assert valid_landmarks(cases).tolist() == expected
    assert valid_landmarks(torch.as_tensor(cases)).tolist() == expected


def test_yaw_estimate_is_zero_frontal_signed_turned_and_saturates_collapsed() -> None:
    assert abs(estimate_yaw_5pt(FRONTAL[None])[0]) < 5
    turned = FRONTAL.copy()
    turned[2, 0] += 12  # nose towards the right eye
    mirrored = FRONTAL.copy()
    mirrored[2, 0] -= 12
    yaw_r, yaw_l = estimate_yaw_5pt(np.stack([turned, mirrored]))
    assert yaw_r > 20 and yaw_l < -20 and abs(yaw_r + yaw_l) < 1.0
    profile = FRONTAL.copy()
    profile[1] = profile[0] + [1.0, 0.0]  # eyes overlap, nose beside them
    assert abs(estimate_yaw_5pt(profile[None])[0]) > DENSE_MAX_YAW_DEG


def test_alignment_policy_falls_back_to_five_points() -> None:
    dense = np.random.default_rng(0).uniform(30, 90, (203, 2))
    assert choose_alignment_points(FRONTAL).source == "5-point"
    assert choose_alignment_points(FRONTAL, dense, 0.34).source == "5-point"
    assert choose_alignment_points(FRONTAL, dense, None).source == "5-point"
    assert choose_alignment_points(FRONTAL, dense, 0.9, yaw_deg=80).source == "5-point"
    assert choose_alignment_points(FRONTAL, dense, 0.9, yaw_deg=-76).source == "5-point"
    assert choose_alignment_points(FRONTAL, np.full((203, 2), np.nan), 0.9).source == "5-point"
    chosen = choose_alignment_points(FRONTAL, dense, 0.35, yaw_deg=75)
    assert chosen.source == "dense" and chosen.points.shape == (203, 2)


# ---------------------------------------------------------------------------- aligner
@cuda
def test_warps_survive_singular_matrices_and_keep_the_good_face_exact() -> None:
    frame = torch.rand(1, 3, 240, 320, device="cuda") * 255
    good = torch.as_tensor(FRONTAL[None] + [100, 60], dtype=torch.float32, device="cuda")
    kps = torch.cat([good, torch.full((1, 5, 2), float("nan"), device="cuda"),
                     torch.full((1, 5, 2), 50.0, device="cuda")])
    m = similarity_matrices_cuda(kps, 256)
    crops = warp_face_cuda(frame, m, 256)  # raised linalg.inv before the guard
    assert crops.shape == (3, 3, 256, 256) and torch.isfinite(crops).all()
    swapped = 255 - crops
    out = warp_face_inverse_cuda(frame, swapped, m)
    solo = warp_face_inverse_cuda(frame, swapped[:1], m[:1])
    assert torch.isfinite(out).all()
    # The bad faces paste nothing; the good one matches its solo paste to
    # kornia's batched-grid rounding (0.0068 levels measured).
    assert (out - solo).abs().max() <= 0.01


# ---------------------------------------------------------------------------- processor
@pytest.fixture(scope="module")
def sample() -> tuple[np.ndarray, np.ndarray]:
    import cv2
    import insightface

    from face_engine.core.config import EngineConfig, Provider
    from face_engine.core.execution import ExecutionEngine
    from face_engine.models.zoo import build_default_registry
    from face_engine.pipeline.detector import SCRFDDetector
    from face_engine.processors.swapper import IdentityEncoder

    image = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    registry = build_default_registry()
    with ExecutionEngine(EngineConfig(providers=[Provider.CUDA, Provider.CPU])) as engine:
        faces = SCRFDDetector(engine, registry.ensure("scrfd_10g_bnkps")).detect(image)
        emb = IdentityEncoder(engine, registry.ensure("arcface_w600k_r50")).embed(image, faces[0]).embedding
    return image, emb


@cuda
@pytest.mark.gpu
@pytest.mark.parametrize("preset", ["balanced", "cinema"])
def test_processor_survives_unusable_landmarks(sample: tuple[np.ndarray, np.ndarray],
                                               preset: str) -> None:
    from face_engine.models.zoo import build_default_registry

    image, emb = sample
    params = RenderParams(**PRESETS[preset]["params"])
    config = ProcessorConfig(params, model_paths(required_models(params),
                                                 build_default_registry()), {"s": emb})
    processor = GpuFrameProcessor(config)
    h, w = image.shape[:2]
    frame = torch.from_numpy(image).cuda().permute(2, 0, 1)[None].contiguous()
    good = FRONTAL * 1.5 + [w / 2 - 80, h / 2 - 100]
    cases = {"good": [good], "nan": [np.full((5, 2), np.nan)], "inf": [np.full((5, 2), np.inf)],
             "collapsed": [np.full((5, 2), 100.0)], "good+nan": [good, np.full((5, 2), np.nan)],
             "good+other": [good, FRONTAL * 1.5 + [60, 60]],
             "half_out": [good - [w / 2, 0]], "fully_out": [good + [10 * w, 0]]}
    results = {}
    try:
        for name, kps_list in cases.items():
            kps = torch.as_tensor(np.stack(kps_list), dtype=torch.float32, device="cuda")
            n = kps.shape[0]
            det = GPUDetections(torch.zeros(n, 4, device="cuda"), kps, torch.ones(n, device="cuda"),
                                torch.zeros(n, dtype=torch.long, device="cuda"), (h, w), 1)
            processor._faces = lambda f, det=det: det  # type: ignore[method-assign]
            out, stats = processor.process_tensor(frame)
            assert torch.isfinite(out).all(), name
            results[name] = (out, stats)
    finally:
        processor.close()
    for name in ("nan", "inf", "collapsed"):
        out, stats = results[name]
        assert stats.swapped == 0 and torch.equal(out, frame.float()), name
    assert results["good"][1].swapped == 1
    # A batch of 2 is not bit-identical to a batch of 1 (batched engines pick
    # other kernels: up to 1.06 levels on Balanced, 2026-09-28), so the control
    # for "the NaN face leaks nothing" is the good face beside a VALID face.
    ys, xs = slice(int(h / 2 - 140), int(h / 2 + 60)), slice(int(w / 2 - 120), int(w / 2 + 80))
    leak = (results["good+nan"][0] - results["good+other"][0])[..., ys, xs].abs().max()
    assert leak <= 1e-3
    assert results["good+nan"][1].swapped == 1
    assert torch.equal(results["fully_out"][0], frame.float())


# ---------------------------------------------------------------------------- zero faces
class NoFaces:
    def infer(self, frame: torch.Tensor) -> FramePlan:
        return FramePlan(None, [], FrameStats())

    def composite(self, plan: FramePlan) -> None:
        return GpuFrameProcessor.composite(plan)


@cuda
@pytest.mark.gpu
def test_zero_face_frames_pass_through_bit_exact(tmp_path: Path) -> None:
    clip = tmp_path / "src.mp4"
    subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=320x240:rate=25,trim=end_frame=20", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    a = CUDAStreamPipeline().run(clip, tmp_path / "a.mp4", NoFaces())
    b = CUDAStreamPipeline().run(clip, tmp_path / "b.mp4", None)  # float round trip
    assert a.passthrough_frames == 20 and b.passthrough_frames == 0
    import av

    with av.open(str(tmp_path / "a.mp4")) as ca, av.open(str(tmp_path / "b.mp4")) as cb:
        for fa, fb in zip(ca.decode(video=0), cb.decode(video=0), strict=True):
            assert np.array_equal(fa.to_ndarray(format="bgr24"), fb.to_ndarray(format="bgr24"))


# ---------------------------------------------------------------------------- A/V sync
@cuda
@pytest.mark.gpu
def test_ntsc_render_keeps_exact_rate_and_audio(tmp_path: Path) -> None:
    clip = mf.ntsc_clip(tmp_path / "ntsc.mp4", seconds=3)  # 24000/1001, AAC audio
    out = tmp_path / "out.mp4"
    stats = CUDAStreamPipeline().run(clip, out, None)
    report = check_av_sync(out, clip, stats.frames_total)
    assert report.ok, report.problems
    assert report.fps == Fraction(24000, 1001)
    assert report.audio_duration is not None


@cuda
@pytest.mark.gpu
def test_vfr_render_uses_the_average_rate(tmp_path: Path) -> None:
    clip = mf.vfr_clip(tmp_path / "vfr.mp4")
    out = tmp_path / "out.mp4"
    stats = CUDAStreamPipeline().run(clip, out, None)
    report = check_av_sync(out, clip, stats.frames_total)
    assert report.ok, report.problems


@cuda
@pytest.mark.gpu
def test_av_check_reports_a_short_render(tmp_path: Path) -> None:
    clip = mf.ntsc_clip(tmp_path / "ntsc.mp4", seconds=3)
    out = tmp_path / "short.mp4"
    stats = CUDAStreamPipeline().run(clip, out, None, max_frames=30)
    report = check_av_sync(out, clip, stats.frames_total + 10)
    assert not report.ok and any("video frames" in p for p in report.problems)
