"""Stage 2 vision pipeline: detection, alignment, composite masking.

Real-image tests use the photos shipped inside the ``insightface`` package
(``t1.jpg``: six frontal-ish faces; ``mask_blue.jpg``: a surgical-mask
texture used as a real occluder), so no private footage is needed. Models
come from the registry (``FACE_ENGINE_MODELS_DIR`` or ``./.cache/models``)
and are downloaded + verified on first use.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from face_engine.core.config import EngineConfig, Provider
from face_engine.core.execution import ExecutionEngine
from face_engine.models.zoo import build_default_registry
from face_engine.pipeline.aligner import (
    AlignmentError,
    align_face,
    crop_valid_mask,
    estimate_similarity_transform,
    paste_mask_to_canvas,
    template_points,
    transform_points,
    warp_face_by_translation,
    warp_face_inverse,
)
from face_engine.pipeline.detector import (
    Face,
    Normalization,
    SCRFDDetector,
    YOLOFaceDetector,
    batched_nms,
    letterbox,
    normalize,
)
from face_engine.pipeline.masker import (
    BoxMaskConfig,
    CompositeMasker,
    FaceRegion,
    MaskerConfig,
    RegionConfig,
    box_mask,
)


def _insightface_images() -> Path:
    import insightface

    return Path(insightface.__file__).parent / "data" / "images"


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def images() -> dict[str, np.ndarray]:
    root = _insightface_images()
    return {"t1": cv2.imread(str(root / "t1.jpg")), "texture": cv2.imread(str(root / "mask_blue.jpg"))}


@pytest.fixture(scope="module")
def registry():  # type: ignore[no-untyped-def]
    return build_default_registry()


@pytest.fixture(scope="module")
def engine() -> ExecutionEngine:
    eng = ExecutionEngine(EngineConfig(providers=[Provider.CUDA, Provider.CPU]))
    yield eng
    eng.close()


@pytest.fixture(scope="module")
def scrfd(engine: ExecutionEngine, registry) -> SCRFDDetector:  # type: ignore[no-untyped-def]
    return SCRFDDetector(engine, registry.ensure("scrfd_10g_bnkps", show_progress=False))


@pytest.fixture(scope="module")
def yolo(engine: ExecutionEngine, registry) -> YOLOFaceDetector:  # type: ignore[no-untyped-def]
    return YOLOFaceDetector(engine, registry.ensure("yoloface_8n", show_progress=False))


@pytest.fixture(scope="module")
def t1_faces(scrfd: SCRFDDetector, images: dict[str, np.ndarray]) -> list[Face]:
    return scrfd.detect(images["t1"])


@pytest.fixture(scope="module")
def masker(engine: ExecutionEngine, registry) -> CompositeMasker:  # type: ignore[no-untyped-def]
    m = CompositeMasker(engine, registry.ensure("xseg", show_progress=False),
                        registry.ensure("bisenet_resnet34", show_progress=False))
    yield m
    m.close()


# ------------------------------------------------------------------ detector: pure
def test_letterbox_maps_points_back_exactly() -> None:
    frame = np.zeros((360, 1000, 3), np.uint8)
    canvas, info = letterbox(frame, 640)
    assert canvas.shape == (640, 640, 3)
    pts = np.array([[0.0, 0.0], [999.0, 359.0], [500.0, 180.0]], np.float32)
    on_canvas = np.stack([pts[:, 0] * info.scale + info.pad_x,
                          pts[:, 1] * info.scale + info.pad_y], axis=1)
    np.testing.assert_allclose(info.to_frame(on_canvas), pts, atol=1e-3)
    assert info.pad_x == 0 and info.pad_y > 0  # wide frame: padded top/bottom


def test_normalization_modes() -> None:
    canvas = np.full((4, 4, 3), [0, 128, 255], np.uint8)  # B, G, R
    sym = normalize(canvas, Normalization.SYMMETRIC_128, swap_rb=True)
    assert sym.shape == (1, 3, 4, 4)
    np.testing.assert_allclose(sym[0, :, 0, 0], [(255 - 127.5) / 128, 0.5 / 128, -127.5 / 128])
    unit = normalize(canvas, Normalization.UNIT, swap_rb=False)
    np.testing.assert_allclose(unit[0, :, 0, 0], [0.0, 128 / 255, 1.0], rtol=1e-6)
    imagenet = normalize(canvas, Normalization.IMAGENET, swap_rb=True)
    np.testing.assert_allclose(imagenet[0, 0, 0, 0], (1.0 - 0.485) / 0.229, rtol=1e-5)


def test_batched_nms_suppresses_within_a_frame_only() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = batched_nms(boxes, scores, np.array([0, 0, 1]), 0.4)
    assert sorted(keep.tolist()) == [0, 2]


# ------------------------------------------------------------------ detector: models
@pytest.mark.gpu
def test_scrfd_matches_insightface_reference(scrfd: SCRFDDetector, registry,  # type: ignore[no-untyped-def]
                                             images: dict[str, np.ndarray]) -> None:
    from insightface.model_zoo.scrfd import SCRFD

    ref = SCRFD(str(registry.local_path("scrfd_10g_bnkps")))
    ref.prepare(-1, input_size=(640, 640), det_thresh=0.5, nms_thresh=0.4)
    ref_boxes, ref_kps = ref.detect(images["t1"], input_size=(640, 640))
    ours = scrfd.detect(images["t1"])
    assert len(ours) == len(ref_boxes) == 6
    for face in ours:
        ious = [_iou(face.bbox, b) for b in ref_boxes]
        j = int(np.argmax(ious))
        assert ious[j] > 0.9
        # Landmark agreement within 3% of face width (letterbox placement differs).
        assert np.abs(face.kps - ref_kps[j]).max() < 0.03 * face.width
        assert face.frame_size == images["t1"].shape[:2]
        assert 0.5 <= face.score <= 1.0


@pytest.mark.gpu
def test_yoloface_agrees_with_scrfd(yolo: YOLOFaceDetector, t1_faces: list[Face],
                                    images: dict[str, np.ndarray]) -> None:
    faces = yolo.detect(images["t1"])
    assert yolo.input_size == 640
    assert len(faces) == 6
    for face in faces:
        best = max(t1_faces, key=lambda f: _iou(face.bbox, f.bbox))
        assert _iou(face.bbox, best.bbox) > 0.5
        assert np.linalg.norm(face.kps - best.kps, axis=1).mean() < 0.1 * best.width


@pytest.mark.gpu
def test_detect_batch_equals_single_frame_detection(scrfd: SCRFDDetector,
                                                    images: dict[str, np.ndarray]) -> None:
    flipped = images["t1"][:, ::-1].copy()
    batch = scrfd.detect_batch([images["t1"], flipped, None])
    assert [len(r) for r in batch] == [6, len(scrfd.detect(flipped)), 0]
    for a, b in zip(batch[0], scrfd.detect(images["t1"])):
        np.testing.assert_allclose(a.bbox, b.bbox, atol=1e-4)


@pytest.mark.gpu
def test_face_cut_by_the_frame_edge(scrfd: SCRFDDetector, t1_faces: list[Face],
                                    images: dict[str, np.ndarray]) -> None:
    face = t1_faces[0]
    cut = int(face.center[0])  # slice through the middle of the face
    frame = np.ascontiguousarray(images["t1"][:, cut:])
    faces = scrfd.detect(frame)
    h, w = frame.shape[:2]
    for f in faces:
        assert 0 <= f.bbox[0] <= f.bbox[2] <= w and 0 <= f.bbox[1] <= f.bbox[3] <= h


@pytest.mark.gpu
@pytest.mark.parametrize("frame", [
    None, np.zeros((0, 0, 3), np.uint8), np.zeros((1, 1, 3), np.uint8),
    np.zeros((64, 64), np.uint8), np.zeros((64, 64, 4), np.uint8),
    np.full((64, 64, 3), np.nan, np.float32), np.zeros((8, 8, 2), np.uint8),
], ids=["none", "empty", "1px", "gray", "bgra", "nan", "2ch"])
def test_detector_survives_bad_frames(scrfd: SCRFDDetector, yolo: YOLOFaceDetector,
                                      frame: np.ndarray | None) -> None:
    assert scrfd.detect(frame) == []
    assert yolo.detect(frame) == []


@pytest.mark.gpu
def test_detector_accepts_gray_and_bgra_faces(scrfd: SCRFDDetector,
                                              images: dict[str, np.ndarray]) -> None:
    gray = cv2.cvtColor(images["t1"], cv2.COLOR_BGR2GRAY)
    bgra = cv2.cvtColor(images["t1"], cv2.COLOR_BGR2BGRA)
    assert len(scrfd.detect(bgra)) == 6
    assert len(scrfd.detect(gray)) >= 5


# ------------------------------------------------------------------ aligner
def test_similarity_transform_recovers_a_known_transform() -> None:
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 500, (5, 2))
    angle, scale, t = 0.7, 0.37, np.array([12.0, -40.0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = scale * src @ rot.T + t
    m = estimate_similarity_transform(src, dst)
    np.testing.assert_allclose(m[:, :2], scale * rot, atol=1e-10)
    np.testing.assert_allclose(m[:, 2], t, atol=1e-8)


def test_similarity_transform_matches_skimage() -> None:
    from skimage.transform import SimilarityTransform

    rng = np.random.default_rng(1)
    dst = template_points(256)
    for _ in range(20):
        src = dst * rng.uniform(0.2, 3) + rng.normal(0, 4, (5, 2)) + rng.uniform(0, 900, 2)
        ref = SimilarityTransform()
        ref.estimate(src, dst)
        np.testing.assert_allclose(estimate_similarity_transform(src, dst), ref.params[:2],
                                   atol=1e-9)


def test_mirrored_landmarks_give_a_rotation_and_a_large_residual() -> None:
    dst = template_points(256)
    mirrored = dst.copy()
    mirrored[:, 0] = 256 - mirrored[:, 0]
    m = estimate_similarity_transform(mirrored, dst)
    assert np.linalg.det(m[:, :2]) > 0  # never a reflection
    frame = np.zeros((300, 300, 3), np.uint8)
    good = align_face(frame, dst, 256)
    bad = align_face(frame, mirrored, 256)
    assert good is not None and bad is not None
    assert good.fit_error < 1e-6 and bad.fit_error > 0.05


def test_degenerate_landmarks() -> None:
    with pytest.raises(AlignmentError):
        estimate_similarity_transform(np.ones((5, 2)), template_points(256))
    with pytest.raises(AlignmentError):
        estimate_similarity_transform(np.full((5, 2), np.nan), template_points(256))
    frame = np.zeros((100, 100, 3), np.uint8)
    assert align_face(frame, np.ones((5, 2)), 256) is None
    assert align_face(frame, template_points(256), 300) is None  # no canonical 300px template


def test_templates_are_canonical_for_the_standard_sizes() -> None:
    np.testing.assert_allclose(template_points(112)[0], [38.2946, 51.6963])
    np.testing.assert_allclose(template_points(256)[0], [(38.2946 + 8) * 2, 51.6963 * 2])
    np.testing.assert_allclose(template_points(512)[0], [0.37691676 * 512, 0.46864664 * 512])


def test_roi_warp_equals_full_frame_warp(images: dict[str, np.ndarray],
                                         t1_faces: list[Face]) -> None:
    frame = images["t1"]
    for face in t1_faces:
        m = estimate_similarity_transform(face.kps, template_points(256))
        ours = warp_face_by_translation(frame, m, 256, antialias=False)
        full = cv2.warpAffine(frame, m, (256, 256), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        assert np.array_equal(ours, full)


def test_paste_back_round_trip(images: dict[str, np.ndarray], t1_faces: list[Face]) -> None:
    frame = images["t1"]
    face = t1_faces[0]
    aligned = align_face(frame, face.kps, 256)
    assert aligned is not None
    pasted = warp_face_inverse(frame, aligned.crop, aligned.matrix, box_mask(256, BoxMaskConfig()))
    x0, y0, x1, y1 = face.bbox.astype(int)
    diff = np.abs(pasted[y0:y1, x0:x1].astype(int) - frame[y0:y1, x0:x1])
    assert diff.mean() < 3.0  # resampling twice, nothing else
    far = np.abs(pasted.astype(int) - frame)
    far[max(y0 - 80, 0):y1 + 80, max(x0 - 80, 0):x1 + 80] = 0
    assert far.max() == 0  # nothing outside the face region changes


def test_paste_back_changes_only_where_the_mask_is_on(images: dict[str, np.ndarray],
                                                     t1_faces: list[Face]) -> None:
    frame = images["t1"]
    aligned = align_face(frame, t1_faces[1].kps, 256)
    assert aligned is not None
    red = np.zeros_like(aligned.crop)
    red[:, :, 2] = 255
    mask = box_mask(256, BoxMaskConfig(blur=0.0, padding_top=0.3, padding_bottom=0.3,
                                       padding_left=0.3, padding_right=0.3))
    pasted = warp_face_inverse(frame, red, aligned.matrix, mask)
    canvas = paste_mask_to_canvas(mask, aligned.matrix, frame.shape)
    changed = np.any(pasted != frame, axis=2)
    assert not np.any(changed & (canvas == 0))
    centre = transform_points(np.array([[128.0, 128.0]]), cv2.invertAffineTransform(aligned.matrix))[0]
    cx, cy = np.round(centre).astype(int)
    assert pasted[cy, cx, 2] == 255 and pasted[cy, cx, 0] == 0


def test_uint8_paste_matches_the_float_path(images: dict[str, np.ndarray],
                                            t1_faces: list[Face]) -> None:
    frame = images["t1"]
    crop = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    m = estimate_similarity_transform(t1_faces[0].kps, template_points(256))
    mask = box_mask(256, BoxMaskConfig())
    fast = warp_face_inverse(frame, crop, m, mask)
    slow = warp_face_inverse(frame.astype(np.float32), crop.astype(np.float32), m, mask)
    assert fast.dtype == np.uint8 and slow.dtype == np.float32
    assert np.abs(fast.astype(np.float64) - np.clip(np.rint(slow), 0, 255)).max() <= 1


def test_crops_partially_and_fully_outside_the_frame() -> None:
    frame = np.full((200, 200, 3), 100, np.uint8)
    dst = template_points(256)
    half_out = dst * 0.5 + np.array([-60.0, 20.0])  # face centre near x=4
    aligned = align_face(frame, half_out, 256)
    assert aligned is not None and not aligned.fully_inside
    assert 0.2 < aligned.valid.mean() < 0.9
    pasted = warp_face_inverse(frame, np.zeros_like(aligned.crop), aligned.matrix)
    assert pasted.shape == frame.shape

    far_away = dst * 0.5 + np.array([5000.0, 5000.0])
    lost = align_face(frame, far_away, 256)
    assert lost is not None and lost.valid.max() == 0 and lost.crop.max() == 0
    assert np.array_equal(warp_face_inverse(frame, lost.crop, lost.matrix), frame)
    assert paste_mask_to_canvas(np.ones((256, 256)), lost.matrix, frame.shape).max() == 0
    assert crop_valid_mask(frame.shape, lost.matrix, 256).max() == 0


@pytest.mark.gpu
@pytest.mark.parametrize("tf32", [False, True], ids=["fp32", "tf32-on"])
def test_gpu_warps_match_opencv(images: dict[str, np.ndarray], t1_faces: list[Face],
                                tf32: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("kornia")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    # roop-ultimate's core.py enables TF32 process-wide; the warps must not care.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", tf32)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", tf32)
    from face_engine.pipeline.aligner import (
        frames_to_tensor,
        tensor_to_frames,
        warp_face_gpu,
        warp_face_inverse_gpu,
    )

    frame = images["t1"]
    matrices = np.stack([estimate_similarity_transform(f.kps, template_points(256))
                         for f in t1_faces])
    frames_t = frames_to_tensor([frame] * len(t1_faces))
    crops = warp_face_gpu(frames_t, matrices, 256)
    assert crops.device.type == "cuda" and crops.shape == (len(t1_faces), 3, 256, 256)
    for i, m in enumerate(matrices):
        ref = warp_face_by_translation(frame, m, 256, antialias=False)
        got = tensor_to_frames(crops[i:i + 1])[0]
        assert np.abs(got.astype(int) - ref).mean() < 0.6
        assert np.abs(got.astype(int) - ref).max() <= 2

    mask = box_mask(256, BoxMaskConfig())
    red = np.zeros((256, 256, 3), np.uint8)
    red[:, :, 2] = 255
    ref = warp_face_inverse(frame, red, matrices[0], mask)
    got = warp_face_inverse_gpu(frames_to_tensor(frame), frames_to_tensor(red), matrices[0],
                                torch.from_numpy(mask)[None, None].cuda())
    assert np.abs(tensor_to_frames(got)[0].astype(int) - ref).max() <= 2
    assert torch.backends.cuda.matmul.allow_tf32 is tf32  # restored after the warp


# ------------------------------------------------------------------ masker
def test_box_mask_is_zero_on_the_border_and_feathered() -> None:
    m = box_mask(256, BoxMaskConfig(blur=0.3))
    assert m.shape == (256, 256) and m.dtype == np.float32
    assert m[0].max() == m[-1].max() == m[:, 0].max() == m[:, -1].max() == 0
    assert m[128, 128] == pytest.approx(1.0, abs=1e-4)
    row = m[128, :128]
    assert np.all(np.diff(row) >= -1e-6)  # monotone ramp toward the centre
    hard = box_mask(256, BoxMaskConfig(blur=0.0, padding_top=0.25))
    assert hard[:63].max() == 0 and hard[70:250, 10:250].min() == 1


@pytest.mark.gpu
def test_xseg_keeps_face_and_drops_the_occluder(masker: CompositeMasker, t1_faces: list[Face],
                                                images: dict[str, np.ndarray]) -> None:
    tp = template_points(256)
    y0 = int(tp[2, 1] + 8)
    x0, x1 = int(tp[3, 0] - 30), int(tp[4, 0] + 30)
    eyes = (slice(int(tp[0, 1]) - 8, int(tp[0, 1]) + 8), slice(int(tp[0, 0]) - 8, int(tp[1, 0]) + 8))
    band = (slice(y0 + 5, y0 + 40), slice(x0 + 5, x1 - 5))
    for face in t1_faces[:4]:
        aligned = align_face(images["t1"], face.kps, 256)
        assert aligned is not None
        occluded = aligned.crop.copy()
        occluded[y0:, x0:x1] = cv2.resize(images["texture"], (x1 - x0, 256 - y0))
        clean_p, occ_p = masker.xseg(aligned.crop), masker.xseg(occluded)
        assert clean_p[eyes].mean() > 0.6 and clean_p[band].mean() > 0.6
        assert occ_p[eyes].mean() > 0.6
        assert occ_p[band].mean() < 0.15  # the occluder is masked OUT (not inverted)


@pytest.mark.gpu
def test_regions_select_what_they_name(engine: ExecutionEngine, registry,  # type: ignore[no-untyped-def]
                                       t1_faces: list[Face], images: dict[str, np.ndarray]) -> None:
    config = MaskerConfig(regions=RegionConfig(regions=frozenset({FaceRegion.NOSE}),
                                               feather_sigma=0.0))
    with CompositeMasker(engine, None, registry.local_path("bisenet_resnet34"), config) as nose:
        for face in t1_faces:
            result = nose.generate(images["t1"], face)
            assert result.status == "ok" and result.labels is not None
            region = result.layers["regions"]
            ys, xs = np.nonzero(region > 0.5)
            nose_kp = transform_points(face.kps[2:3], result.aligned.matrix)[0]
            assert len(xs) > 0
            # Nose region lies around the nose-tip landmark.
            assert abs(xs.mean() - nose_kp[0]) < 0.08 * 256 and abs(ys.mean() - nose_kp[1]) < 0.12 * 256
            # The upper lip sits above the lower lip on the parser crop.
            lab = result.labels
            assert np.nonzero(lab == 12)[0].mean() < np.nonzero(lab == 13)[0].mean()


@pytest.mark.gpu
def test_composite_is_the_product_of_its_layers(masker: CompositeMasker, t1_faces: list[Face],
                                                images: dict[str, np.ndarray]) -> None:
    frame = images["t1"]
    face = t1_faces[2]
    result = masker.generate(frame, face)
    assert result.status == "ok" and result.failed_layers == ()
    assert set(result.layers) == {"box", "xseg", "regions", "valid"}
    product = np.prod(np.stack(list(result.layers.values())), axis=0)
    np.testing.assert_allclose(result.crop_mask, product, atol=1e-6)
    assert result.canvas_mask.shape == frame.shape[:2]
    # Canvas mask is on over the face and off far away from it.
    cx, cy = face.kps[2].astype(int)
    assert result.canvas_mask[cy, cx] > 0.9
    x0, y0, x1, y1 = face.bbox.astype(int)
    outside = result.canvas_mask.copy()
    outside[max(y0 - 100, 0):y1 + 100, max(x0 - 100, 0):x1 + 100] = 0
    assert outside.max() == 0
    # Pixels the parser calls hair or background are excluded (away from the
    # feathered boundary).
    parser = align_face(frame, face.kps, 512, "ffhq_512")
    assert parser is not None and result.aligned is not None
    to_swap = (np.vstack([result.aligned.matrix, [0, 0, 1]])
               @ np.vstack([cv2.invertAffineTransform(parser.matrix), [0, 0, 1]]))[:2]
    for cls in (0, 17):  # background, hair
        region = (result.labels == cls).astype(np.uint8)
        region = cv2.erode(region, np.ones((15, 15), np.uint8))
        in_swap = cv2.warpAffine(region, to_swap, (256, 256), flags=cv2.INTER_NEAREST)
        assert in_swap.sum() > 0
        assert result.crop_mask[in_swap > 0].max() < 0.05
    assert result.as_tensor("cpu").shape == (1, 1, 256, 256)


@pytest.mark.gpu
def test_masker_never_raises(masker: CompositeMasker, t1_faces: list[Face],
                             images: dict[str, np.ndarray]) -> None:
    frame = images["t1"]
    face = t1_faces[0]
    # Face cut by the frame edge.
    cut = np.ascontiguousarray(frame[:, int(face.center[0]):])
    shifted = face.kps - np.array([int(face.center[0]), 0], np.float32)
    edge = masker.generate(cut, shifted)
    assert edge.status == "ok" and edge.layers["valid"].mean() < 0.9
    assert edge.canvas_mask.shape == cut.shape[:2]
    # Sharp profile: eyes collapsed together, nose past them.
    profile = face.kps.copy()
    profile[1] = profile[0] + [4.0, 0.0]
    profile[4] = profile[3] + [4.0, 0.0]
    profile[2, 0] = profile[0, 0] - 25
    assert masker.generate(frame, profile).status == "ok"
    mirrored = face.kps[[1, 0, 2, 4, 3]]
    assert masker.generate(frame, mirrored).status == "ok"
    for bad in (np.full((5, 2), np.nan), np.ones((5, 2)), np.zeros((3, 2))):
        assert masker.generate(frame, bad).status == "no_face"
    assert masker.generate(None, face).status == "no_face"  # type: ignore[arg-type]
    far = face.kps + 10_000
    lost = masker.generate(frame, far)
    assert lost.canvas_mask.max() == 0


@pytest.mark.gpu
def test_a_failing_layer_degrades_instead_of_raising(engine: ExecutionEngine, registry,  # type: ignore[no-untyped-def]
                                                     t1_faces: list[Face], tmp_path: Path,
                                                     images: dict[str, np.ndarray]) -> None:
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"not an onnx model")
    with CompositeMasker(engine, broken, registry.local_path("bisenet_resnet34")) as m:
        result = m.generate(images["t1"], t1_faces[0])
    assert result.status == "degraded" and result.failed_layers == ("xseg",)
    assert "xseg" not in result.layers and result.crop_mask.max() > 0.9
