"""Angle resilience: AngleResilientSCRFD, RobustByteTracker, ProfileGuardedAligner.

Orientation tests rotate a 1080p frame made from insightface's t1.jpg (six
faces) by 0 / 90 / 180 / 270 degrees; the ground truth is the upright
detection mapped through the same rotation. The profile turn is simulated:
detections whose score falls to 0.22 (the tracker) and landmarks compressed
horizontally about the nose (the aligner).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from face_engine.pipeline.aligner import (
    ProfileGuardedAligner,
    estimate_similarity_transform_cuda,
    profile_guarded_similarity_cuda,
    template_tensor,
)
from face_engine.pipeline.detector import (
    AngleResilientSCRFD,
    DualDetections,
    GPUDetections,
    unrotate_boxes_cuda,
    unrotate_points_cuda,
)
from face_engine.pipeline.tracker import (
    ByteTrackConfig,
    RobustByteTracker,
)
from face_engine.tests.test_gpu_pipeline import (
    no_host_copies,  # noqa: F401 - fixture
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
CUDA0 = torch.device("cuda", 0)


def rotate_points(points: torch.Tensor, k: int, height: int, width: int) -> torch.Tensor:
    """Forward counterpart of ``unrotate_points_cuda`` (upright -> rotated frame)."""
    x, y = points[..., 0], points[..., 1]
    k %= 4
    if k == 0:
        return points.clone()
    if k == 1:
        return torch.stack([y, width - x], -1)
    if k == 2:
        return torch.stack([width - x, height - y], -1)
    return torch.stack([height - y, x], -1)


# ---------------------------------------------------------------------------- geometry
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_unrotate_inverts_torch_rot90_on_a_non_square_image(k: int) -> None:
    h, w = 7, 12
    for (y, x) in [(0, 0), (2, 9), (6, 11), (3, 4)]:
        image = torch.zeros(1, 1, h, w)
        image[..., y, x] = 1
        rot = torch.rot90(image, k, dims=[2, 3])
        ry, rx = (int(v) for v in (rot[0, 0] == 1).nonzero()[0])
        centre = torch.tensor([[rx + 0.5, ry + 0.5]])  # pixel centre, continuous coords
        back = unrotate_points_cuda(centre, k, h, w)
        assert torch.allclose(back, torch.tensor([[x + 0.5, y + 0.5]]))
        assert torch.allclose(rotate_points(back, k, h, w), centre)


def test_unrotate_boxes_takes_per_row_angles() -> None:
    boxes = torch.tensor([[10.0, 20.0, 30.0, 60.0]] * 4)
    k = torch.tensor([0.0, 1.0, 2.0, 3.0])
    out = unrotate_boxes_cuda(boxes, k, 100.0, 200.0)
    for i in range(4):
        corners = torch.tensor([[10.0, 20.0], [30.0, 60.0]])
        back = unrotate_points_cuda(corners, i, 100.0, 200.0)
        assert torch.allclose(out[i], torch.cat([back.amin(0), back.amax(0)]))


# ---------------------------------------------------------------------------- aligner
def _frontal_kps(n: int = 1, size: float = 120.0) -> torch.Tensor:
    """ArcFace-like frontal landmarks, ~``size`` px face, rolled 0."""
    tpl = torch.tensor([[38.29, 51.70], [73.53, 51.50], [56.03, 71.74],
                        [41.55, 92.37], [70.73, 92.20]], dtype=torch.float64)
    return ((tpl - 56.0) * (size / 112.0) + torch.tensor([640.0, 360.0], dtype=torch.float64)
            ).expand(n, 5, 2).clone()


def _compress(kps: torch.Tensor, k: float) -> torch.Tensor:
    """Simulated yaw: squeeze x about the nose by ``k = cos(yaw)``."""
    out = kps.clone()
    nose_x = kps[:, 2:3, 0]
    out[..., 0] = nose_x + (kps[..., 0] - nose_x) * k
    return out


def test_guarded_similarity_equals_umeyama_when_the_guard_does_not_fire() -> None:
    rng = np.random.default_rng(3)
    kps = _frontal_kps(8) + torch.as_tensor(rng.normal(0, 2, (8, 5, 2)))
    theta = torch.as_tensor(rng.uniform(-np.pi, np.pi, 8))
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    kps = (kps - 640.0) @ rot.transpose(1, 2) + 640.0      # any roll, incl. upside down
    dst = template_tensor(256, device="cpu", dtype=torch.float64)
    guarded, ratio, clamped = profile_guarded_similarity_cuda(kps, dst, 0.15)
    plain = estimate_similarity_transform_cuda(kps, dst).float()
    assert not bool(clamped.any()) and float(ratio.min()) > 0.5
    assert torch.allclose(guarded, plain, atol=1e-4)
    no_guard, _, _ = profile_guarded_similarity_cuda(kps, dst, 0.0)
    assert torch.allclose(no_guard, plain, atol=1e-4)


def test_profile_guard_fires_only_at_steep_yaw_and_holds_the_crop_scale() -> None:
    dst = template_tensor(256, device="cpu", dtype=torch.float64)
    base = _frontal_kps()
    yaws = np.arange(0, 90, 1.0)
    rows = []
    for yaw in yaws:
        kps = _compress(base, float(np.cos(np.deg2rad(yaw))))
        g, ratio, clamped = profile_guarded_similarity_cuda(kps, dst, 0.15)
        u, _, _ = profile_guarded_similarity_cuda(kps, dst, 0.0)
        scale = lambda m: float(torch.linalg.det(m[0, :, :2].double()).sqrt())
        rows.append((yaw, float(ratio[0]), bool(clamped[0]), scale(g), scale(u)))
    ratio = np.array([r[1] for r in rows])
    fired = np.array([r[2] for r in rows])
    guarded = np.array([r[3] for r in rows])
    plain = np.array([r[4] for r in rows])
    first = int(np.argmax(fired))
    # The guard is a steep-profile guard: it does not fire before ~75 deg.
    assert fired.any() and yaws[first] >= 75, (yaws[first], ratio[:: 10])
    assert np.array_equal(fired, ratio < 0.15)
    assert np.array_equal(guarded[~fired], plain[~fired])
    # Past the threshold the unguarded zoom swings back toward frontal; the
    # guarded one does not fall below its value at the threshold.
    assert plain[-1] < plain[first] - 1e-3
    assert guarded[first:].min() >= guarded[first] - 1e-9
    # And it is continuous at the threshold (no zoom jump when it engages).
    assert abs(guarded[first] - guarded[first - 1]) < 0.02 * guarded[first - 1]


def test_fully_collapsed_profile_stays_a_usable_matrix() -> None:
    dst = template_tensor(256, device="cpu", dtype=torch.float64)
    kps = _compress(_frontal_kps(), 0.0)          # every point on one vertical line
    m, _, clamped = profile_guarded_similarity_cuda(kps, dst, 0.15)
    assert bool(clamped[0]) and bool(torch.isfinite(m).all())
    det = float(torch.linalg.det(m[0, :, :2]))
    assert det > 0


@cuda
@pytest.mark.gpu
def test_aligner_crops_on_cuda_through_kornia() -> None:
    frame = torch.rand(1, 3, 720, 1280, device=CUDA0) * 255
    kps = torch.cat([_frontal_kps(), _compress(_frontal_kps(), 0.05)]).to(CUDA0).float()
    out = ProfileGuardedAligner().align(frame, kps)
    assert out.crops.shape == (2, 3, 256, 256) and out.crops.device == CUDA0
    assert out.clamped.tolist() == [False, True] and bool(out.valid.all())
    assert bool(torch.isfinite(out.crops).all())
    # The frontal face lands on the ArcFace-256 template.
    m = out.matrices[0]
    mapped = kps[0] @ m[:, :2].T + m[:, 2]
    assert float((mapped - template_tensor(256, device=CUDA0)).norm(dim=-1).max()) < 1.5


# ---------------------------------------------------------------------------- tracker
def _dual(boxes: list[list[float]], scores: list[float], high: float = 0.5) -> DualDetections:
    b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    rel = torch.tensor([[.3, .4], [.7, .4], [.5, .55], [.35, .75], [.65, .75]])
    k = b[:, None, :2] + rel[None] * (b[:, None, 2:] - b[:, None, :2])
    s = torch.tensor(scores, dtype=torch.float32)
    idx = torch.zeros(len(scores), dtype=torch.int64)
    hi = s >= high
    mk = lambda m: GPUDetections(b[m], k[m], s[m], idx[m], (1080, 1920), 1)
    return DualDetections(mk(hi), mk(~hi))


def _profile_turn(tracker: RobustByteTracker) -> list[list[int]]:
    """A face drifting right 4 px/frame while turning to profile: score
    0.90 -> 0.22 for frames 3-8 (the box narrows as the head turns), back up."""
    scores = [0.90, 0.85, 0.60, 0.35, 0.28, 0.22, 0.22, 0.24, 0.31, 0.72]
    ids = []
    for i, sc in enumerate(scores):
        x = 800 + 4 * i
        narrow = 12 if 3 <= i <= 8 else 0
        out = tracker.update(_dual([[x + narrow, 300, x + 200 - narrow, 560]], [sc]))
        ids.append(out.track_ids)
    return ids


def test_bytetrack_keeps_the_id_through_a_profile_turn_at_score_022() -> None:
    t = RobustByteTracker()
    ids = _profile_turn(t)
    assert ids == [[0]] * 10
    assert t.stats.low_matches == 6 and t.stats.new_tracks == 1 and t.stats.lost_events == 0


def test_without_the_second_association_the_same_turn_loses_the_face() -> None:
    """Control arm (mutation proof): single-threshold tracking on the same input."""
    t = RobustByteTracker(ByteTrackConfig(low_association=False, emit_lost=False))
    ids = _profile_turn(t)
    assert ids[:3] == [[0]] * 3
    assert all(i == [] for i in ids[3:9])         # six frames with no face
    assert ids[9] == [1]                          # and a NEW identity afterwards


def test_low_detections_never_start_tracks() -> None:
    t = RobustByteTracker()
    for _ in range(5):
        out = t.update(_dual([[100, 100, 200, 200]], [0.45]))
        assert out.track_ids == []
    assert t.stats.new_tracks == 0


def test_lost_track_is_predicted_on_its_velocity_then_dropped_after_5_frames() -> None:
    t = RobustByteTracker()
    for i in range(12):                           # learn a +10 px/frame motion
        t.update(_dual([[100 + 10 * i, 100, 200 + 10 * i, 200]], [0.9]))
    last_x = 100 + 10 * 11
    xs = []
    for j in range(5):
        out = t.update(_dual([], []))
        assert out.track_ids == [0] and out.predicted == [True]
        xs.append(float(out.detections.boxes[0, 0]))
    steps = np.diff([last_x] + xs)
    assert np.all(steps > 7) and np.all(steps < 13), steps     # still moving ~10 px/frame
    kps0 = out.detections.kps[0, 0]
    assert abs(float(kps0[0]) - (xs[-1] + 0.3 * 100)) < 3      # landmarks moved with it
    assert t.update(_dual([], [])).track_ids == []            # 6th miss: dropped
    assert t.stats.removed == 1


def test_motion_compensated_ema_does_not_trail_a_moving_face() -> None:
    def lag(compensated: bool) -> float:
        t = RobustByteTracker(ByteTrackConfig(motion_compensated_ema=compensated))
        err = []
        for i in range(30):
            x = 100 + 20 * i                       # fast: 20 px/frame
            out = t.update(_dual([[x, 100, x + 100, 200]], [0.9]))
            err.append(abs(float(out.detections.kps[0, 0, 0]) - (x + 30)))
        return float(np.mean(err[10:]))
    plain, compensated = lag(False), lag(True)
    assert plain > 5.0                             # 20 * (1 - 0.75) / 0.75 = 6.7 px behind
    assert compensated < 1.0


def test_on_lost_asks_the_detector_to_sweep() -> None:
    calls = []
    t = RobustByteTracker(on_lost=lambda: calls.append(1))
    t.update(_dual([[100, 100, 200, 200]], [0.9]))
    t.update(_dual([], []))
    assert calls == [1]


# ---------------------------------------------------------------------------- detector (GPU)
@pytest.fixture(scope="module")
def setup() -> dict[str, Any]:
    import cv2
    import insightface

    from face_engine.core.config import EngineConfig, Provider
    from face_engine.core.execution import ExecutionEngine
    from face_engine.models.zoo import build_default_registry

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    image = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    path = build_default_registry().ensure("scrfd_10g_bnkps")
    engine = ExecutionEngine(EngineConfig(providers=[Provider.TENSORRT, Provider.CUDA]))
    frame = torch.from_numpy(cv2.resize(image, (1920, 1080))).to(CUDA0).permute(2, 0, 1)[None]
    reference = AngleResilientSCRFD(engine, path).detect_dual(frame).high
    assert len(reference) == 6
    return {"path": path, "engine": engine, "frame": frame, "ref": reference}


def _landmark_error(found: GPUDetections, ref: GPUDetections, k: int,
                    hw: tuple[int, int]) -> np.ndarray:
    """Per reference face: min mean landmark distance / face size, in the rotated frame."""
    truth = rotate_points(ref.kps, k, *hw)
    size = (ref.boxes[:, 2:] - ref.boxes[:, :2]).prod(1).sqrt()
    if len(found) == 0:
        return np.full(len(ref), np.inf)
    d = (truth[:, None] - found.kps[None]).norm(dim=-1).mean(-1)
    return (d.min(1).values / size).cpu().numpy()


@cuda
@pytest.mark.gpu
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_recall_is_100_percent_at_every_orientation(setup: dict[str, Any], k: int) -> None:
    det = AngleResilientSCRFD(setup["engine"], setup["path"])
    frame = torch.rot90(setup["frame"], k, dims=[2, 3]).contiguous()
    hw = tuple(setup["frame"].shape[-2:])
    for i in range(12):  # the sweep frame, the preferred-angle hold, and past it
        out = det.detect_dual(frame)
        err = _landmark_error(out.high, setup["ref"], k, hw)
        assert len(out.high) == 6, (k, i, len(out.high))
        assert err.max() < 0.01, (k, i, err)       # the landmarks, not just the boxes
        assert out.tilted == 0
    # Upright content never sweeps; rotated content sweeps once, then holds the
    # angle for 10 frames and re-sweeps when the hold expires (frame 11).
    expected = 0 if k == 0 else 2
    assert det.stats.sweeps == expected, det.stats
    assert out.angle == (360 - 90 * k) % 360      # the pass that undoes the rotation


@cuda
@pytest.mark.gpu
def test_faces_lying_in_a_landscape_frame(setup: dict[str, Any]) -> None:
    """Sideways content pasted into a 1080p landscape frame (not a portrait frame)."""
    frame = torch.zeros_like(setup["frame"])
    side = torch.rot90(setup["frame"], 1, dims=[2, 3])          # 1920 x 1080
    small = torch.nn.functional.interpolate(side.float(), size=(1080, 608), mode="bilinear",
                                            antialias=True).to(frame.dtype)
    frame[..., :, 656:1264] = small
    det = AngleResilientSCRFD(setup["engine"], setup["path"])
    out = det.detect_dual(frame)
    assert len(out.high) == 6 and out.swept and out.tilted == 0


@cuda
@pytest.mark.gpu
def test_batch_of_three_equals_three_single_inferences(setup: dict[str, Any]) -> None:
    """The sweep batch's (h, w, image, anchor) output order is undone correctly."""
    det = AngleResilientSCRFD(setup["engine"], setup["path"])
    resized, _ = det._resized(setup["frame"])
    base = torch.full((1, 3, 640, 640), -1.0, device=CUDA0)
    base[..., 128:128 + 360, :] = resized
    batch = torch.cat([torch.rot90(base, k, dims=[2, 3]) for k in (1, 2, 3)])
    together = det._infer(batch)
    for i in range(3):
        single = det._infer(batch[i:i + 1].contiguous())
        for a, b in zip(together, single):
            assert float((a[i] - b[0]).abs().max()) < 2e-2


@cuda
@pytest.mark.gpu
def test_onnxruntime_fallback_sweeps_too(setup: dict[str, Any]) -> None:
    from face_engine.core.config import EngineConfig, Provider
    from face_engine.core.execution import ExecutionEngine

    engine = ExecutionEngine(EngineConfig(providers=[Provider.CUDA]))
    det = AngleResilientSCRFD(engine, setup["path"])
    assert not det.uses_tensorrt_engine
    frame = torch.rot90(setup["frame"], 2, dims=[2, 3]).contiguous()
    out = det.detect_dual(frame)
    err = _landmark_error(out.high, setup["ref"], 2, tuple(setup["frame"].shape[-2:]))
    assert len(out.high) == 6 and out.swept and err.max() < 0.01


@cuda
@pytest.mark.gpu
def test_detection_never_copies_image_data_to_the_host(setup: dict[str, Any],
                                                      no_host_copies: list[str]) -> None:  # noqa: F811
    det = AngleResilientSCRFD(setup["engine"], setup["path"])
    frame = torch.rot90(setup["frame"], 1, dims=[2, 3]).contiguous()
    for _ in range(3):
        out = det.detect_dual(frame)
    for t in (out.high.boxes, out.high.kps, out.high.scores, out.low.boxes):
        assert t.device == CUDA0
    assert len(out.high) == 6 and no_host_copies == []


@cuda
@pytest.mark.gpu
def test_empty_scene_stops_sweeping_every_frame(setup: dict[str, Any]) -> None:
    det = AngleResilientSCRFD(setup["engine"], setup["path"], empty_after=5, empty_rescan=10)
    blank = torch.zeros_like(setup["frame"])
    for _ in range(35):
        det.detect_dual(blank)
    # 5 frames to register the empty scene, then one sweep per 10 frames
    assert det.scene_empty and det.stats.sweeps == 5 + 3, det.stats
    out = det.detect_dual(setup["frame"])        # a face appears: back to normal
    assert len(out.high) == 6 and not det.scene_empty


@cuda
@pytest.mark.gpu
def test_a_face_that_rolls_over_mid_clip_keeps_its_track(setup: dict[str, Any]) -> None:
    """Upright for 5 frames, then the content turns 90 deg: the sweep finds the
    faces at once, and the tracker keeps (or re-acquires within the lost
    window) all six."""
    det = AngleResilientSCRFD(setup["engine"], setup["path"])
    tracker = RobustByteTracker(on_lost=det.note_track_lost)
    frame = setup["frame"]
    square = torch.zeros((1, 3, 1920, 1920), dtype=frame.dtype, device=CUDA0)
    square[..., 420:1500, :] = frame
    for i in range(12):
        f = square if i < 5 else torch.rot90(square, 3, dims=[2, 3]).contiguous()
        out = det.detect_dual(f)
        tracks = tracker.update(out)
        assert len(out.high) == 6, (i, len(out.high))
    assert len([p for p in tracks.predicted if not p]) == 6


# ---------------------------------------------------------------------------- throughput
@cuda
@pytest.mark.gpu
def test_sustained_upright_throughput_on_rtx_4070(setup: dict[str, Any]) -> None:
    """600 frames of moving 1080p upright video: detect + track + align matrices."""
    if "4070" not in torch.cuda.get_device_name(0):
        pytest.skip("the 55 fps target is for an RTX 4070")
    det = AngleResilientSCRFD(setup["engine"], setup["path"])
    if not det.uses_tensorrt_engine:
        pytest.skip("no compiled scrfd_10g_bnkps engine (tools/download_scrfd.py --engine)")
    tracker = RobustByteTracker(on_lost=det.note_track_lost)
    aligner = ProfileGuardedAligner()
    base = setup["frame"]

    def shifted(s: int) -> torch.Tensor:  # content moved s px right, black edge, no wrap
        f = torch.zeros_like(base)
        f[..., s:] = base[..., :base.shape[-1] - s]
        return f

    # Ping-pong 3 px/frame over 0..87 px: smooth motion, no teleport, no wrap.
    frames = [shifted(3 * i) for i in range(30)] + [shifted(3 * i) for i in range(29, -1, -1)]
    det.prepare(1080, 1920)
    for f in frames[:20]:  # warm-up
        tracker.update(det.detect_dual(f))
    tracker.reset()
    det.stats.sweeps = 0
    ids = set()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(600):
        out = tracker.update(det.detect_dual(frames[i % len(frames)]))
        aligner.estimate(out.detections.kps)
        ids.update(out.track_ids)
    torch.cuda.synchronize()
    fps = 600 / (time.perf_counter() - t0)
    print(f"\nupright 1080p detect+track+align: {fps:.1f} fps, sweeps {det.stats.sweeps}, "
          f"track ids {sorted(ids)}")
    assert det.stats.sweeps == 0
    assert sorted(ids) == list(range(6))          # six faces, six identities, 600 frames
    assert fps >= 55.0, f"{fps:.1f} fps"
