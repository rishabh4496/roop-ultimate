"""GPUSCRFDDetector (tight canvas, TensorRT FP16) and TemporalFaceTracker.

GPU tests use the insightface sample photo t1.jpg (six faces) and a 1080p
frame made from it; InsightFace's own SCRFD implementation over the same model
file is the landmark reference.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from face_engine.pipeline.detector import GPUDetections, GPUSCRFDDetector
from face_engine.pipeline.tracker import TemporalFaceTracker, TemporalTrackerConfig
from face_engine.tests.test_gpu_pipeline import no_host_copies  # noqa: F401 - fixture

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
CUDA0 = torch.device("cuda", 0)


# ---------------------------------------------------------------------------- tracker
def _dets(boxes: list[list[float]], kps: np.ndarray | None = None,
          scores: list[float] | None = None) -> GPUDetections:
    b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if kps is None:  # 5 points spread over each box
        rel = torch.tensor([[.3, .4], [.7, .4], [.5, .55], [.35, .75], [.65, .75]])
        k = b[:, None, :2] + rel[None] * (b[:, None, 2:] - b[:, None, :2])
    else:
        k = torch.as_tensor(kps, dtype=torch.float32)
    s = torch.tensor(scores or [0.9] * b.shape[0])
    return GPUDetections(b, k, s, torch.zeros(b.shape[0], dtype=torch.int64), (1000, 1000), 1)


def test_ema_weights_the_new_measurement_by_alpha() -> None:
    t = TemporalFaceTracker(TemporalTrackerConfig(alpha=0.75))
    t.update(_dets([[100, 100, 200, 200]]))
    out = t.update(_dets([[110, 100, 210, 200]]))
    assert out.track_ids == [0] and out.predicted == [False]
    assert torch.allclose(out.detections.boxes[0], torch.tensor([107.5, 100, 207.5, 200]))


def test_unsmoothed_mode_passes_boxes_through() -> None:
    t = TemporalFaceTracker(TemporalTrackerConfig(smooth=False))
    t.update(_dets([[100, 100, 200, 200]]))
    out = t.update(_dets([[110, 100, 210, 200]]))
    assert torch.equal(out.detections.boxes[0], torch.tensor([110.0, 100, 210, 200]))


def test_missed_face_is_carried_on_its_motion_for_three_frames() -> None:
    t = TemporalFaceTracker(TemporalTrackerConfig(smooth=False, max_missed=3))
    t.update(_dets([[100, 100, 200, 200]]))
    t.update(_dets([[110, 100, 210, 200]]))  # moving +10 px / frame
    xs = []
    for _ in range(3):
        out = t.update(_dets([]))
        assert out.predicted == [True] and out.track_ids == [0]
        xs.append(float(out.detections.boxes[0, 0]))
    assert xs == [120.0, 130.0, 140.0]
    assert t.update(_dets([])).track_ids == []  # 4th miss: dropped


def test_tracks_keep_ids_through_crossing_order_and_new_faces_get_new_ids() -> None:
    t = TemporalFaceTracker(TemporalTrackerConfig(smooth=False))
    first = t.update(_dets([[100, 100, 200, 200], [600, 100, 700, 200]]))
    # the detector lists them in the other order; a third face appears
    second = t.update(_dets([[605, 100, 705, 200], [102, 100, 202, 200],
                             [300, 500, 380, 580]]))
    assert first.track_ids == [0, 1]
    assert second.track_ids == [0, 1, 2]
    assert float(second.detections.boxes[0, 0]) == 102.0  # track 0 is still the left face


def test_a_far_detection_is_not_matched_to_an_old_track() -> None:
    t = TemporalFaceTracker(TemporalTrackerConfig(smooth=False, max_missed=0))
    t.update(_dets([[100, 100, 200, 200]]))
    out = t.update(_dets([[700, 700, 800, 800]]))
    assert out.track_ids == [1]


# ---------------------------------------------------------------------------- detector
@pytest.fixture(scope="module")
def setup() -> dict[str, Any]:
    import cv2
    import insightface

    from face_engine.core.config import EngineConfig, Provider
    from face_engine.core.execution import ExecutionEngine
    from face_engine.models.zoo import build_default_registry

    image = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    path = build_default_registry().ensure("scrfd_10g_bnkps")
    engine = ExecutionEngine(EngineConfig(providers=[Provider.TENSORRT, Provider.CUDA]))
    frame = torch.from_numpy(cv2.resize(image, (1920, 1080))).to(CUDA0).permute(2, 0, 1)[None]
    return {"image": image, "path": path, "engine": engine,
            "det": GPUSCRFDDetector(engine, path), "frame1080": frame}


@cuda
@pytest.mark.gpu
def test_canvas_keeps_aspect_and_pads_to_32(setup: dict[str, Any]) -> None:
    det = setup["det"]
    assert det.canvas_size(1080, 1920)[3:] == (384, 640)
    assert det.canvas_size(1920, 1080)[3:] == (640, 384)
    assert det.canvas_size(2160, 3840)[3:] == (384, 640)
    assert det.canvas_size(200, 1920)[3:] == (384, 640)  # the engine profile's minimum side
    canvas, scale = det.preprocess_cuda(setup["frame1080"])
    assert canvas.shape == (1, 3, 384, 640) and abs(scale - 640 / 1920) < 1e-9
    assert float(canvas.min()) >= -1.0 and float(canvas.max()) <= 1.0


@cuda
@pytest.mark.gpu
def test_everything_stays_on_cuda0(setup: dict[str, Any], no_host_copies: list[str]) -> None:  # noqa: F811
    det = setup["det"]
    frame = setup["frame1080"]
    canvas, _ = det.preprocess_cuda(frame)
    assert canvas.device == CUDA0
    found = det.detect_cuda(frame)
    for t in (found.boxes, found.kps, found.scores, found.frame_index):
        assert t.device == CUDA0
    assert len(found) == 6 and no_host_copies == []


@cuda
@pytest.mark.gpu
def test_rgb01_input_equals_bgr255(setup: dict[str, Any]) -> None:
    from face_engine.pipeline.detector import GPUSCRFDDetector as D

    frame = setup["frame1080"]
    a = setup["det"].detect_cuda(frame)
    b = D(setup["engine"], setup["path"], input_format="rgb01").detect_cuda(
        frame.flip(1).float() / 255.0)
    assert len(a) == len(b)
    assert float((a.kps - b.kps).abs().max()) < 0.5


@cuda
@pytest.mark.gpu
def test_landmarks_match_insightface_and_the_arcface_template(setup: dict[str, Any]) -> None:
    import onnxruntime as ort
    from insightface.model_zoo.scrfd import SCRFD

    from face_engine.pipeline.aligner import TEMPLATES, estimate_similarity_transform

    image, path = setup["image"], setup["path"]
    ref = SCRFD(str(path), session=ort.InferenceSession(str(path),
                                                       providers=["CUDAExecutionProvider"]))
    ref.prepare(0, input_size=(640, 640), det_thresh=0.5)
    ref_boxes, ref_kps = ref.detect(image, input_size=(640, 640))
    found = setup["det"].detect_cuda(torch.from_numpy(image).to(CUDA0).permute(2, 0, 1)[None])
    boxes, kps = found.boxes.cpu().numpy(), found.kps.cpu().numpy()
    assert len(kps) == len(ref_kps) == 6
    template = TEMPLATES["arcface_112"] * 112

    def residual(points: np.ndarray) -> float:
        m = estimate_similarity_transform(points, template)
        return float(np.sqrt((((points @ m[:, :2].T + m[:, 2]) - template) ** 2).sum(1).mean()))

    for rb, rk in zip(ref_boxes, ref_kps):
        j = int(np.argmin(np.abs(boxes - rb[:4]).sum(1)))
        size = float(np.sqrt((rb[2] - rb[0]) * (rb[3] - rb[1])))
        assert np.linalg.norm(kps[j] - rk, axis=1).mean() / size < 0.015
        # Same fit to ArcFace's canonical points as the reference detector's
        # (the residual itself is mostly head pose: 2-9 px at 112 on t1).
        assert abs(residual(kps[j]) - residual(rk)) < 0.5


@cuda
@pytest.mark.gpu
def test_1080p_latency_on_rtx_4070(setup: dict[str, Any]) -> None:
    det, frame = setup["det"], setup["frame1080"]
    if "4070" not in torch.cuda.get_device_name(0):
        pytest.skip("the 2 ms target is for an RTX 4070")
    if not det.uses_tensorrt_engine:
        pytest.skip("no compiled scrfd_10g_bnkps engine (tools/download_scrfd.py --engine)")
    for _ in range(20):
        det.detect_cuda(frame)
    times = []
    for _ in range(200):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        det.detect_cuda(frame)  # preprocess + TensorRT + decode + NMS
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    median = statistics.median(times)
    assert median <= 2.0, f"median {median:.2f} ms (p95 {np.percentile(times, 95):.2f} ms)"
