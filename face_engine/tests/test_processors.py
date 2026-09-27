"""Stage 3: swapping, restoration, expression restoration.

Two kinds of test:

* Mock-tensor tests replace the ONNX session with an identity network (output
  = input) or a scripted one. They pin the I/O conventions end to end — RGB
  order, mean/std, [-1,1] denormalization, the polyphase Pixel Boost layout,
  the weight blend — without model files. An identity network round-tripping
  a crop to itself is only possible if every convention is inverted exactly.
* Model tests use the real networks on the insightface sample photo and
  check OUTCOMES: the swapped face must be recognised as the source by
  ArcFace, restoration must keep colour, expression restore must move the
  expression toward the target.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pytest

from face_engine.core.config import EngineConfig
from face_engine.core.execution import ExecutionEngine
from face_engine.core.registry import ModelUnavailableError
from face_engine.models.zoo import build_default_registry
from face_engine.processors.enhancer import (
    ColorMode,
    FaceEnhancer,
    face_region_mask,
    transfer_color,
)
from face_engine.processors.expression import (
    EYE_INDICES,
    LIP_INDICES,
    ExpressionRestorer,
    ExpressionWeights,
    driving_keypoints,
    rotation_matrix,
    transform_keypoints,
)
from face_engine.processors.swapper import (
    FaceSwapper,
    Identity,
    IdentityEncoder,
    SwapError,
    explode_pixel_boost,
    implode_pixel_boost,
)

# ------------------------------------------------------------------ mock sessions


@dataclass
class _Node:
    name: str


@dataclass
class _FakeOrt:
    inputs: list[str]
    fn: Any
    calls: list[dict[str, np.ndarray]] = field(default_factory=list)

    def get_inputs(self) -> list[_Node]:
        return [_Node(n) for n in self.inputs]

    def run(self, _names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append(feeds)
        return self.fn(feeds)


@dataclass
class _FakeHandle:
    session: _FakeOrt

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(self.session.inputs)


class _FakeEngine:
    def __init__(self, sessions: dict[str, _FakeOrt]) -> None:
        self.sessions = sessions
        self.requests: list[tuple[str, Any]] = []

    def get_session(self, path: Any, **kwargs: Any) -> _FakeHandle:
        self.requests.append((str(path), kwargs.get("trt_fp16")))
        return _FakeHandle(self.sessions[str(path)])


def _face_frame(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A textured frame and plausible 5-point landmarks for a ~200px face."""
    rng = np.random.default_rng(seed)
    frame = cv2.GaussianBlur(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), (0, 0), 2)
    kps = np.array([[280, 220], [360, 220], [320, 265], [290, 310], [350, 310]], np.float32)
    return frame, kps


def _identity_swapper(model: str) -> tuple[FaceSwapper, _FakeOrt]:
    """A swapper whose network returns its target input unchanged."""
    fake = _FakeOrt(["target", "source"], lambda f: [f["target"].copy()])
    swapper = FaceSwapper(_FakeEngine({"m.onnx": fake}), model, "m.onnx")  # type: ignore[arg-type]
    return swapper, fake


def _unit_identity(seed: int = 1) -> Identity:
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return Identity(v / np.linalg.norm(v), 20.0)


# ------------------------------------------------------------------ swapper: mock tensors
@pytest.mark.parametrize("factor", [1, 2, 3, 4])
def test_pixel_boost_implode_explode_round_trip(factor: int) -> None:
    crop = np.random.default_rng(factor).integers(0, 256, (64 * factor, 64 * factor, 3), np.uint8)
    tiles = implode_pixel_boost(crop, 64, factor)
    assert tiles.shape == (factor * factor, 64, 64, 3)
    # Tile (a, b) is the polyphase component: every factor-th pixel from (a, b).
    for a in range(factor):
        for b in range(factor):
            assert np.array_equal(tiles[a * factor + b], crop[a::factor, b::factor])
    assert np.array_equal(explode_pixel_boost(tiles, 64, factor), crop)


@pytest.mark.parametrize("model", ["hyperswap_1a_256", "inswapper_128"])
@pytest.mark.parametrize("boost_factor", [1, 2, 4])
def test_identity_network_round_trips_the_target_crop(model: str, boost_factor: int) -> None:
    swapper, fake = _identity_swapper(model)
    if model == "inswapper_128":
        swapper._emap = np.eye(512, dtype=np.float32)  # skip reading the real graph
    frame, kps = _face_frame()
    boost = swapper.spec.size * boost_factor
    result = swapper.swap(frame, kps, _unit_identity(), pixel_boost=boost)
    assert result.crop.shape == (boost, boost, 3) and result.tiles == boost_factor ** 2
    assert len(fake.calls) == boost_factor ** 2
    # Normalize -> identity net -> denormalize must be lossless up to rounding.
    assert np.abs(result.crop.astype(int) - result.target_crop).max() <= 1
    blob = fake.calls[0]["target"]
    assert blob.shape == (1, 3, swapper.spec.size, swapper.spec.size) and blob.dtype == np.float32
    lo, hi = (-1.0, 1.0) if swapper.spec.denormalize else (0.0, 1.0)
    assert lo - 1e-6 <= blob.min() and blob.max() <= hi + 1e-6
    # RGB order: channel 0 of the blob is the crop's RED channel (BGR index 2).
    tile0 = implode_pixel_boost(result.target_crop, swapper.spec.size, boost_factor)[0]
    red = (tile0[:, :, 2] / 255.0 - swapper.spec.mean[0]) / swapper.spec.std[0]
    np.testing.assert_allclose(blob[0, 0], red, atol=1e-5)


def test_weight_blends_swap_and_target() -> None:
    fake = _FakeOrt(["source", "target"], lambda f: [np.ones_like(f["target"])])  # white face
    swapper = FaceSwapper(_FakeEngine({"m.onnx": fake}), "hyperswap_1b_256", "m.onnx")  # type: ignore[arg-type]
    frame, kps = _face_frame()
    full = swapper.swap(frame, kps, _unit_identity(), weight=1.0)
    none = swapper.swap(frame, kps, _unit_identity(), weight=0.0)
    half = swapper.swap(frame, kps, _unit_identity(), weight=0.5)
    assert full.crop.min() == 255
    assert np.array_equal(none.crop, none.target_crop)
    expected = np.rint(0.5 * 255 + 0.5 * half.target_crop.astype(np.float32))
    assert np.abs(half.crop.astype(np.float32) - expected).max() <= 1


def test_latent_per_embedding_mode() -> None:
    identity = _unit_identity()
    hyper, _ = _identity_swapper("hyperswap_1c_256")
    np.testing.assert_allclose(hyper.latent(identity)[0], identity.embedding)
    ins, _ = _identity_swapper("inswapper_128")
    emap = np.random.default_rng(3).normal(size=(512, 512)).astype(np.float32)
    ins._emap = emap
    latent = ins.latent(identity)[0]
    expected = identity.embedding @ emap
    np.testing.assert_allclose(latent, expected / np.linalg.norm(expected), rtol=1e-5)
    assert np.linalg.norm(latent) == pytest.approx(1.0, abs=1e-5)


def test_swap_sessions_are_fp32_and_inputs_are_bound_by_name() -> None:
    seen: list[set[str]] = []
    fake = _FakeOrt(["source", "target"],
                    lambda f: seen.append(set(f)) or [np.zeros((1, 3, 256, 256), np.float32)])
    engine = _FakeEngine({"m.onnx": fake})
    swapper = FaceSwapper(engine, "hyperswap_1a_256", "m.onnx")  # type: ignore[arg-type]
    frame, kps = _face_frame()
    identity = _unit_identity()
    swapper.swap(frame, kps, identity)
    assert seen == [{"source", "target"}]
    assert fake.calls[0]["source"].shape == (1, 512)
    np.testing.assert_allclose(fake.calls[0]["source"][0], identity.embedding)
    assert engine.requests and all(fp16 is False for _, fp16 in engine.requests)


def test_swapper_input_validation() -> None:
    swapper, _ = _identity_swapper("hyperswap_1a_256")
    frame, kps = _face_frame()
    with pytest.raises(ValueError):
        swapper.swap(frame, kps, _unit_identity(), pixel_boost=300)
    with pytest.raises(ValueError):
        swapper.swap(frame, kps, _unit_identity(), weight=1.5)
    with pytest.raises(SwapError):
        swapper.swap(frame, np.full((5, 2), np.nan), _unit_identity())
    with pytest.raises(SwapError):
        swapper.swap(None, kps, _unit_identity())  # type: ignore[arg-type]
    with pytest.raises(ModelUnavailableError):
        FaceSwapper(_FakeEngine({}), "alphaface_256", "a.onnx")  # type: ignore[arg-type]
    nan = _FakeOrt(["source", "target"], lambda f: [np.full_like(f["target"], np.nan)])
    broken = FaceSwapper(_FakeEngine({"m.onnx": nan}), "hyperswap_1a_256", "m.onnx")  # type: ignore[arg-type]
    with pytest.raises(SwapError, match="non-finite"):
        broken.swap(frame, kps, _unit_identity())


# ------------------------------------------------------------------ enhancer: mock tensors
def _identity_enhancer(fn: Any = None) -> tuple[FaceEnhancer, _FakeOrt]:
    fake = _FakeOrt(["input"], fn or (lambda f: [f["input"].copy()]))
    return FaceEnhancer(_FakeEngine({"e.onnx": fake}), "gpen_bfr_512", "e.onnx"), fake  # type: ignore[arg-type]


def test_enhancer_identity_network_leaves_the_face_unchanged() -> None:
    enhancer, fake = _identity_enhancer()
    frame, kps = _face_frame()
    result = enhancer.enhance(frame, kps, color=ColorMode.NONE)
    assert result.status == "ok"
    blob = fake.calls[0]["input"]
    assert blob.shape == (1, 3, 512, 512) and -1.0 <= blob.min() and blob.max() <= 1.0
    assert np.abs(result.crop_out.astype(int) - result.crop_in).max() <= 1
    # Crop -> paste -> crop resamples twice; nothing else may change.
    assert np.abs(result.frame.astype(int) - frame).mean() < 1.0


@pytest.mark.parametrize("output", ["nan", "flat"])
def test_enhancer_rejects_broken_output(output: str) -> None:
    value = np.nan if output == "nan" else 0.1
    enhancer, _ = _identity_enhancer(lambda f: [np.full_like(f["input"], value)])
    frame, kps = _face_frame()
    result = enhancer.enhance(frame, kps)
    assert result.status == "rejected" and np.array_equal(result.frame, frame)


def test_enhancer_alpha_and_neighbour_protection() -> None:
    # Brightened but still textured (a flat output would be rejected as collapsed).
    enhancer, _ = _identity_enhancer(lambda f: [np.clip(f["input"] + 0.8, -1, 1)])
    frame, kps = _face_frame()
    zero = enhancer.enhance(frame, kps, alpha=0.0, color=ColorMode.NONE)
    assert np.abs(zero.frame.astype(int) - frame).mean() < 1.0
    full = enhancer.enhance(frame, kps, alpha=1.0, color=ColorMode.NONE)
    assert full.status == "ok"
    nose = kps[2].astype(int)
    patch = (slice(nose[1] - 5, nose[1] + 5), slice(nose[0] - 5, nose[0] + 5))
    assert full.frame[patch].astype(int).mean() > frame[patch].astype(int).mean() + 80
    # Outside the face ellipse (e.g. a neighbour beside the head) stays untouched.
    assert np.array_equal(full.frame[:, :150], frame[:, :150])
    with pytest.raises(ValueError):
        enhancer.enhance(frame, kps, alpha=2.0)


def test_color_transfer_modes() -> None:
    rng = np.random.default_rng(0)
    ref = np.clip(rng.normal([90, 120, 170], 12, (128, 128, 3)), 0, 255).astype(np.uint8)
    img = np.clip(rng.normal([140, 150, 150], 25, (128, 128, 3)), 0, 255).astype(np.uint8)
    lab = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    mean = transfer_color(img, ref, ColorMode.LAB_MEAN)
    np.testing.assert_allclose(lab(mean).mean(0), lab(ref).mean(0), atol=1.0)
    reinhard = transfer_color(img, ref, ColorMode.REINHARD)
    np.testing.assert_allclose(lab(reinhard).std(0), lab(ref).std(0), rtol=0.1)
    chroma = transfer_color(img, ref, ColorMode.KEEP_CHROMA)
    assert np.abs(lab(chroma)[:, 0] - lab(img)[:, 0]).mean() < 1.5  # keeps restored L
    assert np.array_equal(transfer_color(img, ref, ColorMode.NONE), img)


# ------------------------------------------------------------------ expression: math
def test_zero_weights_are_an_exact_no_op_and_rotation_cancels() -> None:
    rng = np.random.default_rng(0)
    kp, exp_s, exp_d = (rng.normal(0, 0.1, (1, 21, 3)).astype(np.float32) for _ in range(3))
    scale, t = np.array([[1.3]], np.float32), np.array([[0.1, -0.2, 0.5]], np.float32)
    x_s = transform_keypoints(kp, exp_s, scale, t, rotation_matrix(np.array([10.0]),
                                                                  np.array([-20.0]),
                                                                  np.array([5.0])))
    zero = driving_keypoints(x_s, scale, exp_s, exp_d, ExpressionWeights(0, 0, 0).per_keypoint())
    assert np.array_equal(zero, x_s)
    w = ExpressionWeights(lips=1.0, eyes=0.0, other=0.5).per_keypoint()
    moved = driving_keypoints(x_s, scale, exp_s, exp_d, w) - x_s
    # The delta is independent of pose: only the expression difference moves.
    np.testing.assert_allclose(moved, (exp_d - exp_s) * w.reshape(1, 21, 1) * 1.3, atol=1e-6)
    assert np.all(moved[0, list(EYE_INDICES)] == 0)
    assert w[list(LIP_INDICES)].tolist() == [1.0] * 6


def test_expression_failure_returns_the_input() -> None:
    def boom(_f: dict[str, np.ndarray]) -> list[np.ndarray]:
        raise RuntimeError("simulated session failure")

    sessions = {k: _FakeOrt(["img"], boom) for k in ("a.onnx", "m.onnx", "w.onnx")}
    restorer = ExpressionRestorer(_FakeEngine(sessions),  # type: ignore[arg-type]
                                  {"appearance": "a.onnx", "motion": "m.onnx", "warping": "w.onnx"},
                                  blink=False)
    restorer._warping_path = restorer.paths["warping"]  # skip the graph rewrite
    crop = np.zeros((512, 512, 3), np.uint8)
    assert restorer.restore(crop, crop) is crop
    assert restorer.restore(crop, crop, ExpressionWeights(0, 0, 0), blink=False) is crop


# ------------------------------------------------------------------ real models
@pytest.fixture(scope="module")
def stack():  # type: ignore[no-untyped-def]
    from pathlib import Path

    import insightface

    from face_engine.pipeline.detector import SCRFDDetector

    registry = build_default_registry()
    engine = ExecutionEngine(EngineConfig())
    image = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    detector = SCRFDDetector(engine, registry.ensure("scrfd_10g_bnkps", show_progress=False))
    encoder = IdentityEncoder(engine, registry.ensure("arcface_w600k_r50", show_progress=False))
    faces = detector.detect(image)
    identities = [encoder.embed(image, f) for f in faces]
    yield {"registry": registry, "engine": engine, "image": image, "detector": detector,
           "encoder": encoder, "faces": faces, "ids": identities}
    engine.close()


PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 4)]


def _nearest(faces: list[Any], face: Any) -> Any:
    return min(faces, key=lambda f: float(np.linalg.norm(f.center - face.center)))


@pytest.mark.gpu
def test_source_embeddings_are_unit_and_distinct(stack) -> None:  # type: ignore[no-untyped-def]
    ids = stack["ids"]
    assert len(ids) == 6
    for identity in ids:
        assert identity.embedding.shape == (512,)
        assert np.linalg.norm(identity.embedding) == pytest.approx(1.0, abs=1e-5)
    sims = np.array([[a.similarity(b) for b in ids] for a in ids])
    assert np.all(np.diag(sims) > 0.999) and (sims - np.eye(6) * 2).max() < 0.35


@pytest.mark.gpu
@pytest.mark.parametrize("model,boost", [("hyperswap_1a_256", 256), ("hyperswap_1a_256", 512),
                                         ("inswapper_128", 128), ("inswapper_128", 256)])
def test_end_to_end_identity_transfer(stack, model: str, boost: int) -> None:  # type: ignore[no-untyped-def]
    swapper = FaceSwapper(stack["engine"], model,
                          stack["registry"].ensure(model, show_progress=False))
    image, faces, ids, enc = stack["image"], stack["faces"], stack["ids"], stack["encoder"]
    feather = cv2.GaussianBlur(np.pad(np.ones((200, 200), np.float32), 28), (0, 0), 8)
    for si, ti in PAIRS:
        result = swapper.swap(image, faces[ti], ids[si], pixel_boost=boost)
        out = swapper.paste(image, result, feather)
        swapped = enc.embed(out, _nearest(stack["detector"].detect(out), faces[ti]))
        # Measured 2026-09-27: hyperswap 0.72-0.79 to the source, 0.08-0.15 to
        # the target; inswapper 0.81-0.86 / 0.05-0.16.
        assert swapped.similarity(ids[si]) > 0.6
        assert swapped.similarity(ids[ti]) < 0.3
    if swapper.session.primary_provider == "TensorrtExecutionProvider":
        opts = swapper.session.session.get_provider_options()["TensorrtExecutionProvider"]
        assert opts["trt_fp16_enable"] == "0"


@pytest.mark.gpu
@pytest.mark.parametrize("model", ["gpen_bfr_1024", "restoreformer_plus_plus"])
def test_restoration_keeps_identity_and_colour(stack, model: str) -> None:  # type: ignore[no-untyped-def]
    enhancer = FaceEnhancer(stack["engine"], model,
                            stack["registry"].ensure(model, show_progress=False))
    image, faces, ids, enc = stack["image"], stack["faces"], stack["ids"], stack["encoder"]
    region = face_region_mask(enhancer.size) > 0
    for i in (0, 2, 4):
        result = enhancer.enhance(image, faces[i], reference_frame=image)
        assert result.status == "ok"
        lab_in = cv2.cvtColor(result.crop_in, cv2.COLOR_BGR2LAB)[region].astype(float).mean(0)
        lab_out = cv2.cvtColor(result.crop_out, cv2.COLOR_BGR2LAB)[region].astype(float).mean(0)
        assert np.abs(lab_out - lab_in).max() < 1.5  # LAB_MEAN colour lock
        assert enc.embed(result.frame, faces[i]).similarity(ids[i]) > 0.6
    if enhancer.session.primary_provider == "TensorrtExecutionProvider":
        opts = enhancer.session.session.get_provider_options()["TensorrtExecutionProvider"]
        assert opts["trt_fp16_enable"] == "0"


@pytest.fixture(scope="module")
def restorer(stack):  # type: ignore[no-untyped-def]
    return ExpressionRestorer.from_registry(stack["engine"], stack["registry"])


@pytest.mark.gpu
def test_motion_extractor_is_not_constant(stack, restorer) -> None:  # type: ignore[no-untyped-def]
    """Regression: under TensorRT FP16 the motion net returned one constant for every face."""
    swapper = FaceSwapper(stack["engine"], "hyperswap_1a_256",
                          stack["registry"].ensure("hyperswap_1a_256", show_progress=False))
    crops = [swapper.swap(stack["image"], f, stack["ids"][0]).target_crop for f in stack["faces"][:3]]
    coeffs = [restorer.expression_coefficients(c) for c in crops]
    assert not np.allclose(coeffs[0], coeffs[1]) and not np.allclose(coeffs[1], coeffs[2])


@pytest.mark.gpu
def test_expression_moves_toward_the_target(stack, restorer) -> None:  # type: ignore[no-untyped-def]
    swapper = FaceSwapper(stack["engine"], "hyperswap_1a_256",
                          stack["registry"].ensure("hyperswap_1a_256", show_progress=False))
    lips = list(LIP_INDICES)
    before, after, id_blink = [], [], []
    for si, ti in PAIRS:
        res = swapper.swap(stack["image"], stack["faces"][ti], stack["ids"][si], pixel_boost=512)
        target = restorer.expression_coefficients(res.target_crop)
        restored = restorer.restore(res.crop, res.target_crop)
        assert restored.shape == res.crop.shape and restored.dtype == np.uint8
        before.append(np.linalg.norm(restorer.expression_coefficients(res.crop)[lips] - target[lips]))
        after.append(np.linalg.norm(restorer.expression_coefficients(restored)[lips] - target[lips]))
        blink_only = restorer.restore(res.crop, res.target_crop, ExpressionWeights(0, 0, 0), True)
        a = dataclasses.replace(res, crop=blink_only)
        out = swapper.paste(stack["image"], a)
        id_blink.append(stack["encoder"].embed(out, stack["faces"][ti]).similarity(stack["ids"][si]))
        # Zero weights and no blink: bit-exact no-op.
        assert restorer.restore(res.crop, res.target_crop, ExpressionWeights(0, 0, 0),
                                blink=False) is res.crop
    assert np.mean(after) < np.mean(before)
    assert np.mean(id_blink) > 0.6
