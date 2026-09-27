"""Stage 2, CUDA-resident: detection stride, kornia warps and the GPU masker.

Every test needs a CUDA device. Real images are the photos shipped inside
``insightface`` (``t1.jpg``: six faces; ``mask_blue.jpg``: a surgical-mask
texture used as an occluder), as in ``test_pipeline.py``. The headline test,
:func:`test_pipeline_never_leaves_cuda0`, runs detection -> tracking ->
alignment -> masking -> paste-back with every host-copy API of
``torch.Tensor`` patched to raise.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kornia")

from face_engine.core.config import EngineConfig, Provider
from face_engine.core.execution import ExecutionEngine
from face_engine.models.zoo import build_default_registry
from face_engine.pipeline.aligner import (
    crop_valid_mask,
    crop_valid_mask_cuda,
    estimate_similarity_transform,
    estimate_similarity_transform_cuda,
    similarity_is_valid,
    similarity_matrices_cuda,
    template_points,
    warp_face_by_translation,
    warp_face_cuda,
    warp_face_inverse,
    warp_face_inverse_cuda,
)
from face_engine.pipeline.detector import (
    GPUDetections,
    SCRFDDetector,
    YOLOFaceDetector,
)
from face_engine.pipeline.masker import (
    CompositeMasker,
    GPUMasker,
    MaskerConfig,
)
from face_engine.pipeline.tracker import (
    LucasKanadeTracker,
    SceneCutDetector,
    StridedFaceTracker,
    TrackerConfig,
    to_gray,
)

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")]

CUDA0 = torch.device("cuda", 0)


def _to_cuda(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).to(CUDA0).permute(2, 0, 1)[None].float()


def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    return tensor.clamp(0, 255).round().byte().permute(1, 2, 0).cpu().numpy()


def _shift(image: np.ndarray, dx: float, dy: float, angle: float = 0.0,
           scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += (dx, dy)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT_101), m


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def images() -> dict[str, np.ndarray]:
    import insightface

    root = Path(insightface.__file__).parent / "data" / "images"
    return {"t1": cv2.imread(str(root / "t1.jpg")),
            "texture": cv2.imread(str(root / "mask_blue.jpg"))}


@pytest.fixture(scope="module")
def registry():  # type: ignore[no-untyped-def]
    return build_default_registry()


@pytest.fixture(scope="module")
def engine() -> Iterator[ExecutionEngine]:
    eng = ExecutionEngine(EngineConfig(providers=[Provider.CUDA, Provider.CPU], strict=True))
    yield eng
    eng.close()


@pytest.fixture(scope="module")
def scrfd(engine: ExecutionEngine, registry) -> SCRFDDetector:  # type: ignore[no-untyped-def]
    return SCRFDDetector(engine, registry.ensure("scrfd_10g_bnkps", show_progress=False))


@pytest.fixture(scope="module")
def models(registry) -> tuple[Path, Path]:  # type: ignore[no-untyped-def]
    return (registry.ensure("xseg_3", show_progress=False),
            registry.ensure("bisenet_resnet34", show_progress=False))


@pytest.fixture(scope="module")
def t1_gpu(scrfd: SCRFDDetector, images: dict[str, np.ndarray]) -> GPUDetections:
    return scrfd.detect_cuda(_to_cuda(images["t1"]))


# ------------------------------------------------------------------ alignment
def test_closed_form_similarity_equals_umeyama() -> None:
    rng = np.random.default_rng(0)
    dst = template_points(256)
    src = rng.uniform(100, 900, size=(64, 5, 2))
    got = estimate_similarity_transform_cuda(torch.as_tensor(src, device=CUDA0),
                                             torch.as_tensor(dst, device=CUDA0)).cpu().numpy()
    for i in range(len(src)):
        np.testing.assert_allclose(got[i], estimate_similarity_transform(src[i], dst),
                                   rtol=1e-7, atol=1e-7)
    # Mirrored landmarks: a rotation (positive determinant), never a reflection.
    mirrored = dst.copy()
    mirrored[:, 0] = 256 - mirrored[:, 0]
    m = estimate_similarity_transform_cuda(torch.as_tensor(mirrored[None], device=CUDA0),
                                           torch.as_tensor(dst, device=CUDA0))[0]
    assert float(torch.linalg.det(m[:, :2])) > 0
    # Degenerate input flags itself instead of raising.
    same = torch.full((1, 5, 2), 50.0, dtype=torch.float64, device=CUDA0)
    assert not bool(similarity_is_valid(
        estimate_similarity_transform_cuda(same, torch.as_tensor(dst, device=CUDA0)))[0])


@pytest.mark.parametrize("size", [256, 512, 1024])
def test_warp_face_cuda_matches_opencv(images: dict[str, np.ndarray],
                                       t1_gpu: GPUDetections, size: int) -> None:
    frame = images["t1"]
    matrices = similarity_matrices_cuda(t1_gpu.kps, size)
    crops = warp_face_cuda(_to_cuda(frame), matrices, size, padding_mode="border")
    assert crops.device == CUDA0 and crops.shape == (len(t1_gpu), 3, size, size)
    for i in range(len(t1_gpu)):
        ref = warp_face_by_translation(frame, matrices[i].double().cpu().numpy(), size,
                                       antialias=False)
        diff = np.abs(_to_uint8(crops[i]).astype(int) - ref)
        assert diff.mean() < 0.6 and diff.max() <= 2


def test_reflection_padding_only_changes_the_out_of_frame_area(
        images: dict[str, np.ndarray], t1_gpu: GPUDetections) -> None:
    frame = images["t1"][:, 60:]  # cuts the left-most face
    kps = t1_gpu.kps - torch.tensor([60.0, 0.0], device=CUDA0)
    matrices = similarity_matrices_cuda(kps, 256)
    reflect = warp_face_cuda(_to_cuda(frame), matrices, 256)
    border = warp_face_cuda(_to_cuda(frame), matrices, 256, padding_mode="border")
    valid = crop_valid_mask_cuda(frame.shape[:2], matrices, 256)
    inside = valid >= 1.0
    assert bool((valid < 1).any()), "the fixture must include a face cut by the frame edge"
    assert float(((reflect - border).abs() * inside).max()) < 1e-3


def test_valid_mask_matches_the_cpu_warp(images: dict[str, np.ndarray],
                                         t1_gpu: GPUDetections) -> None:
    frame = images["t1"][:, 60:]
    kps = t1_gpu.kps - torch.tensor([60.0, 0.0], device=CUDA0)
    matrices = similarity_matrices_cuda(kps, 256)
    got = crop_valid_mask_cuda(frame.shape[:2], matrices, 256)
    for i in range(len(kps)):
        ref = crop_valid_mask(frame.shape, matrices[i].double().cpu().numpy(), 256)
        assert np.abs(got[i, 0].cpu().numpy() - ref).mean() < 0.01


def test_supersampling_is_the_antialias(images: dict[str, np.ndarray],
                                        t1_gpu: GPUDetections) -> None:
    """A 3x-upscaled frame into a 112 crop shrinks the face ~5x: the supersampled
    crop must be closer to the CPU path's Gaussian-prefiltered crop than a
    plain bilinear warp is."""
    big = cv2.resize(images["t1"], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    kps = t1_gpu.kps * 3
    matrices = similarity_matrices_cuda(kps, 112, "arcface_112")
    frame = _to_cuda(big)
    plain = warp_face_cuda(frame, matrices, 112, padding_mode="border")
    smooth = warp_face_cuda(frame, matrices, 112, padding_mode="border", antialias=True)
    for i in range(len(kps)):
        ref = warp_face_by_translation(big, matrices[i].double().cpu().numpy(), 112).astype(float)
        err_plain = np.abs(_to_uint8(plain[i]) - ref).mean()
        err_smooth = np.abs(_to_uint8(smooth[i]) - ref).mean()
        assert err_smooth < 0.7 * err_plain


def test_inverse_warp_matches_the_cpu_paste(images: dict[str, np.ndarray],
                                            t1_gpu: GPUDetections) -> None:
    frame = images["t1"]
    matrices = similarity_matrices_cuda(t1_gpu.kps, 256)
    red = np.zeros((256, 256, 3), np.uint8)
    red[:, :, 2] = 255
    masker = GPUMasker.__new__(GPUMasker)
    masker.config, masker._box_cache = MaskerConfig(), {}
    mask = masker.box(256, CUDA0)
    ref = frame
    for i in range(len(t1_gpu)):
        ref = warp_face_inverse(ref, red, matrices[i].double().cpu().numpy(),
                                mask[0, 0].cpu().numpy())
    crops = _to_cuda(red).expand(len(t1_gpu), -1, -1, -1)
    got = warp_face_inverse_cuda(_to_cuda(frame), crops, matrices,
                                 mask.expand(len(t1_gpu), -1, -1, -1))
    assert got.device == CUDA0
    diff = np.abs(_to_uint8(got[0]).astype(int) - ref)
    assert diff.mean() < 0.1 and diff.max() <= 3


def test_inverse_warp_routes_faces_to_their_frames(images: dict[str, np.ndarray],
                                                   t1_gpu: GPUDetections) -> None:
    frames = _to_cuda(images["t1"]).expand(2, -1, -1, -1).contiguous()
    matrices = similarity_matrices_cuda(t1_gpu.kps[:2], 256)
    white = torch.full((2, 3, 256, 256), 255.0, device=CUDA0)
    out = warp_face_inverse_cuda(frames, white, matrices,
                                 frame_index=torch.tensor([1, 1], device=CUDA0))
    assert torch.equal(out[0], frames[0])  # frame 0 untouched
    assert float((out[1] - frames[1]).abs().sum()) > 0


# ------------------------------------------------------------------ detection
def test_detect_cuda_matches_the_host_detector(scrfd: SCRFDDetector,
                                               images: dict[str, np.ndarray],
                                               t1_gpu: GPUDetections) -> None:
    host = scrfd.detect(images["t1"])
    got = t1_gpu.to_faces()[0]
    assert t1_gpu.boxes.device == CUDA0 and len(got) == len(host) == 6
    for a in host:
        b = max(got, key=lambda f: _iou(a.bbox, f.bbox))
        assert _iou(a.bbox, b.bbox) > 0.97
        assert np.abs(a.kps - b.kps).max() < 1.0
        assert abs(a.score - b.score) < 0.01


def test_yoloface_cuda_matches_the_host_detector(engine: ExecutionEngine, registry,  # type: ignore[no-untyped-def]
                                                 images: dict[str, np.ndarray]) -> None:
    yolo = YOLOFaceDetector(engine, registry.ensure("yoloface_8n", show_progress=False))
    host = yolo.detect(images["t1"])
    got = yolo.detect_cuda(_to_cuda(images["t1"])).to_faces()[0]
    assert len(got) == len(host) == 6
    for a in host:
        b = max(got, key=lambda f: _iou(a.bbox, f.bbox))
        assert _iou(a.bbox, b.bbox) > 0.97


def test_batched_nms_keeps_frames_apart(scrfd: SCRFDDetector,
                                        images: dict[str, np.ndarray]) -> None:
    frame = _to_cuda(images["t1"])
    batch = scrfd.detect_cuda(torch.cat([frame, frame, torch.zeros_like(frame)]))
    assert torch.bincount(batch.frame_index, minlength=3).tolist() == [6, 6, 0]
    assert bool((batch.frame_index[1:] >= batch.frame_index[:-1]).all())
    assert len(scrfd.detect_cuda(torch.zeros((1, 3, 64, 64), device=CUDA0))) == 0


# ------------------------------------------------------------------ tracking
def test_lucas_kanade_recovers_a_known_motion(images: dict[str, np.ndarray],
                                              t1_gpu: GPUDetections) -> None:
    frame = images["t1"]
    moved, m = _shift(frame, 6.4, -3.7, angle=2.0, scale=1.03)
    lk = LucasKanadeTracker()
    prev, cur = lk.pyramid(to_gray(_to_cuda(frame))), lk.pyramid(to_gray(_to_cuda(moved)))
    points = t1_gpu.kps.reshape(-1, 2)
    truth = points.cpu().numpy() @ m[:, :2].T + m[:, 2]
    forward, backward = lk.track_forward_backward(prev, cur, points)
    err = np.linalg.norm(forward.cpu().numpy() - truth, axis=1)
    assert np.median(err) < 0.25 and err.max() < 1.0
    assert float((backward - points).norm(dim=1).max()) < 1.0
    lk.cuda_graphs = False  # the captured graph computes what eager LK computes
    eager, _ = lk.track_forward_backward(prev, cur, points)
    assert float((eager - forward).abs().max()) < 1e-3


def test_scene_cut_fires_on_a_new_shot_only(images: dict[str, np.ndarray]) -> None:
    frame = images["t1"]
    other = cv2.resize(images["texture"], (frame.shape[1], frame.shape[0]))
    cuts = SceneCutDetector()
    assert float(cuts.score(_to_cuda(frame))) == 0.0  # first frame
    assert float(cuts.score(_to_cuda(_shift(frame, 5, 3)[0]))) < 0.1  # camera motion
    assert float(cuts.score(_to_cuda(other))) > 0.5  # a different shot


def test_stride_schedule_ids_and_forced_detection(scrfd: SCRFDDetector,
                                                  images: dict[str, np.ndarray]) -> None:
    frame = images["t1"]
    other = cv2.resize(images["texture"], (frame.shape[1], frame.shape[0]))
    clip = [_shift(frame, 1.5 * i, 0.7 * i)[0] for i in range(8)] + [other, other, frame]
    tracker = StridedFaceTracker(scrfd, TrackerConfig(detection_stride=3))
    outs = [tracker.update(_to_cuda(f)) for f in clip]
    # A shot with no faces detects every frame ("empty"); cuts force detection.
    assert [o.source for o in outs] == ["detect", "track", "track", "detect", "track", "track",
                                        "detect", "track", "cut", "empty", "cut"]
    assert tracker.stats.detections == 6 and tracker.stats.tracked == 5
    assert tracker.stats.cuts == 2
    first = outs[0].track_ids.tolist()
    for o in outs[1:8]:  # the same six people, followed
        assert sorted(o.track_ids.tolist()) == sorted(first)
    assert len(outs[8]) == len(outs[9]) == 0
    assert len(outs[10]) == 6
    assert set(outs[10].track_ids.tolist()).isdisjoint(first)  # after a cut: new ids
    # Tracked landmarks agree with the detector on the same frame.
    for i in (1, 2, 4, 5, 7):
        truth = scrfd.detect_cuda(_to_cuda(clip[i]))
        tracked = outs[i].detections
        for k in range(len(truth)):
            j = int(torch.cdist(truth.kps[k:k + 1].flatten(1), tracked.kps.flatten(1)).argmin())
            size = float((truth.boxes[k, 2:] - truth.boxes[k, :2]).prod().sqrt())
            err = float((tracked.kps[j] - truth.kps[k]).norm(dim=1).mean()) / size
            assert err < 0.02


# ------------------------------------------------------------------ masking
def test_gpu_masker_matches_the_cpu_masker(engine: ExecutionEngine, models: tuple[Path, Path],
                                           images: dict[str, np.ndarray],
                                           t1_gpu: GPUDetections) -> None:
    xseg, bisenet = models
    for size in (256, 512):
        config = MaskerConfig(crop_size=size)
        result = GPUMasker(engine, xseg, bisenet, config).generate(_to_cuda(images["t1"]),
                                                                   t1_gpu.kps)
        assert result.status == "ok" and result.mask.shape == (6, 1, size, size)
        assert set(result.layers) == {"box", "valid", "xseg", "regions"}
        with CompositeMasker(engine, xseg, bisenet, config) as cpu:
            for i, face in enumerate(t1_gpu.to_faces()[0]):
                ref = cpu.generate(images["t1"], face).crop_mask
                assert np.abs(result.mask[i, 0].cpu().numpy() - ref).mean() < 0.005


def test_gpu_xseg_keeps_face_and_drops_the_occluder(engine: ExecutionEngine,
                                                    models: tuple[Path, Path],
                                                    images: dict[str, np.ndarray],
                                                    t1_gpu: GPUDetections) -> None:
    """xseg_3 has the polarity measured for xseg: HIGH = visible face, so it is
    used as-is, not inverted (inverting would keep the occluder)."""
    masker = GPUMasker(engine, models[0], None)
    tp = template_points(256)
    y0 = int(tp[2, 1] + 8)
    x0, x1 = int(tp[3, 0] - 30), int(tp[4, 0] + 30)
    crops = warp_face_cuda(_to_cuda(images["t1"]), similarity_matrices_cuda(t1_gpu.kps, 256), 256)
    occluded = crops.clone()
    occluded[:, :, y0:, x0:x1] = _to_cuda(cv2.resize(images["texture"], (x1 - x0, 256 - y0)))
    clean, occ = masker.xseg(crops[:4]), masker.xseg(occluded[:4])
    eyes = (slice(int(tp[0, 1]) - 8, int(tp[0, 1]) + 8), slice(int(tp[0, 0]) - 8, int(tp[1, 0]) + 8))
    band = (slice(y0 + 5, y0 + 40), slice(x0 + 5, x1 - 5))
    for i in range(4):
        assert float(clean[i, 0][eyes].mean()) > 0.6 and float(clean[i, 0][band].mean()) > 0.6
        assert float(occ[i, 0][eyes].mean()) > 0.6
        assert float(occ[i, 0][band].mean()) < 0.15


def test_a_failing_layer_degrades(engine: ExecutionEngine, images: dict[str, np.ndarray],
                                  t1_gpu: GPUDetections, tmp_path: Path) -> None:
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"not a model")
    result = GPUMasker(engine, broken, None).generate(_to_cuda(images["t1"]), t1_gpu.kps)
    assert result.failed_layers == ("xseg",) and result.status == "degraded"
    assert float(result.mask.max()) > 0.9  # box x valid still applies


# ------------------------------------------------------------------ residency
_HOST_COPIES = ("cpu", "numpy", "tolist", "__array__")


@pytest.fixture
def no_host_copies(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Make every Tensor -> host copy raise. Scalar reads (``item``/``bool``/
    ``float``) stay allowed: they are control-flow decisions, not data."""
    calls: list[str] = []

    def guard(name: str, original: Any) -> Any:
        def wrapped(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            if self.is_cuda:
                calls.append(name)
                raise AssertionError(f"Tensor.{name}() copied a CUDA tensor to the host")
            return original(self, *args, **kwargs)
        return wrapped

    for name in _HOST_COPIES:
        monkeypatch.setattr(torch.Tensor, name, guard(name, getattr(torch.Tensor, name)))
    original_to = torch.Tensor.to

    def to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = original_to(self, *args, **kwargs)
        if self.is_cuda and not out.is_cuda:
            calls.append("to")
            raise AssertionError("Tensor.to() moved a CUDA tensor to the host")
        return out

    monkeypatch.setattr(torch.Tensor, "to", to)
    yield calls


def test_pipeline_never_leaves_cuda0(scrfd: SCRFDDetector, engine: ExecutionEngine,
                                     models: tuple[Path, Path], images: dict[str, np.ndarray],
                                     no_host_copies: list[str]) -> None:
    """Upload once; detection (stride 3: detect + tracked frames), alignment at
    256/512/1024, the tri-layer mask and the paste-back all stay on cuda:0."""
    frames = [_to_cuda(_shift(images["t1"], 2.0 * i, 1.0 * i)[0]) for i in range(4)]
    masker = GPUMasker(engine, *models)
    tracker = StridedFaceTracker(scrfd, TrackerConfig(detection_stride=3))
    sources = []
    for frame in frames:
        assert frame.device == CUDA0
        out = tracker.update(frame)
        sources.append(out.source)
        faces = out.detections
        for t in (faces.boxes, faces.kps, faces.scores, faces.frame_index, out.track_ids):
            assert t.device == CUDA0
        result = masker.generate(frame, faces.kps)
        for t in (result.mask, result.crops, result.matrices, result.labels,
                  *result.layers.values()):
            assert t.device == CUDA0
        assert result.status == "ok" and result.mask.shape == (6, 1, 256, 256)
        for size in (512, 1024):
            crops = warp_face_cuda(frame, similarity_matrices_cuda(faces.kps, size), size)
            assert crops.device == CUDA0 and crops.shape[-1] == size
        pasted = warp_face_inverse_cuda(frame, 255.0 - result.crops, result.matrices,
                                        result.mask)
        assert pasted.device == CUDA0
    assert sources == ["detect", "track", "track", "detect"]
    assert no_host_copies == []
