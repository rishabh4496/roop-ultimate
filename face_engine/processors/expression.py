"""Expression restoration with LivePortrait: put the target's expression back on the swap.

Swap networks regress toward their training set's mean expression and
restorers push further toward a neutral prior, so a wide smile, a frown or a
half-closed lid comes out compressed. LivePortrait separates a face into
canonical keypoints, head pose and an *expression deformation*; the last term
is read off the ORIGINAL target crop and re-applied to the swapped one.

Both crops are the same aligned crop of the same frame, so pose, scale and
translation already agree and only the expression is transferred
(ported from roop-ultimate ``Expression_LivePortrait``)::

    x_s = scale_s * (kp_s @ R_s + exp_s) + t_s
    x_d = x_s + scale_s * w * (exp_target - exp_s)      (w per keypoint)

Rotation cancels in the difference, so a noisy pose estimate cannot move the
head; ``w = 0`` is an exact no-op.

Controls (:class:`ExpressionWeights`) map to LivePortrait's own retargeting
keypoint groups: ``lips`` (6, 12, 14, 17, 19, 20), ``eyes`` (11, 13, 15, 16,
18: eyeball direction AND lids) and ``other`` (brows, cheeks, jaw: the rest).
Defaults follow roop-ultimate's measurements: following the target's eye
keypoints (gaze) made eye direction WORSE, while **blink sync** — LivePortrait's
eye-retargeting MLP driven by the lid opening measured on both crops —
carried the target's blinks on 97% of frames instead of 33% at an identity
cost of 0.006. So ``eyes=0`` and ``blink=True``.

Engineering notes carried over from the app:

* Inputs are bound BY NAME: ``warping_spade`` declares
  ``(feature_3d, kp_driving, kp_source)`` — driving before source — and a
  positional feed silently applied the expression backwards.
* ``warping_spade`` samples a 5-D volume with GridSample, which neither the
  CUDA EP nor TensorRT executes; :mod:`face_engine.utils.gridsample5d`
  rewrites those nodes into gathers (237.8 -> 34.0 ms per face on a 4070,
  and CUDA-only machines can run it at all). If the rewrite fails, the stock
  model runs on TensorRT or CPU, never CUDA.
* Any failure returns the input crop unchanged: an expression tweak must
  never cost a frame.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from face_engine.core.config import Provider
from face_engine.core.execution import ExecutionEngine, ManagedSession
from face_engine.core.registry import ModelRegistry
from face_engine.pipeline.aligner import (
    AlignmentError,
    estimate_similarity_transform,
    template_points,
    warp_face_by_translation,
    warp_face_inverse,
)
from face_engine.pipeline.detector import Face, as_bgr

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

INPUT_SIZE = 256
LANDMARK_SIZE = 224
NUM_BINS = 66
LIP_INDICES = (6, 12, 14, 17, 19, 20)
EYE_INDICES = (11, 13, 15, 16, 18)
MODELS = ("appearance", "motion", "warping", "stitching", "eye", "landmark")

# TensorRT FP16 per model, measured 2026-09-27 (RTX 4070, vs a CPU FP32
# reference): the motion extractor returns a CONSTANT under FP16 (identical
# coefficients for different faces, relative error 9.07) and the landmark
# net is 8.7% off, so both are pinned to FP32 (2.0 -> 3.7 ms and 2.1 -> 2.2 ms).
# appearance, stitching and eye match at FP16. warping is configurable.
_FP16_OK = {"appearance": None, "motion": False, "landmark": False, "stitching": None,
            "eye": None}


@dataclass(frozen=True)
class ExpressionWeights:
    """How far each keypoint group moves toward the target (0 = keep the swap's).

    Values above 1 exaggerate past the target (for expressions the swap
    compressed rather than removed).
    """

    lips: float = 1.0
    eyes: float = 0.0
    other: float = 1.0

    def per_keypoint(self, n: int = 21) -> np.ndarray:
        w = np.full(n, self.other, np.float32)
        for i in LIP_INDICES:
            w[i] = self.lips
        for i in EYE_INDICES:
            w[i] = self.eyes
        return w

    @property
    def active(self) -> bool:
        return any(v != 0.0 for v in (self.lips, self.eyes, self.other))


# ------------------------------------------------------------------ keypoint math
def headpose_to_degrees(pred: np.ndarray) -> np.ndarray:
    """66-bin pose distribution -> degrees (LivePortrait's conversion)."""
    pred = np.asarray(pred, np.float32).reshape(-1, NUM_BINS)
    e = np.exp(pred - pred.max(axis=1, keepdims=True))
    prob = e / e.sum(axis=1, keepdims=True)
    return (prob * np.arange(NUM_BINS, dtype=np.float32)).sum(axis=1) * 3.0 - 97.5


def rotation_matrix(pitch: np.ndarray, yaw: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """Row-vector rotation, so keypoints compose as ``kp @ R``; ``(B, 3, 3)``."""
    p, y, r = (np.radians(np.asarray(a, np.float32).reshape(-1)) for a in (pitch, yaw, roll))
    n = p.shape[0]
    z, o = np.zeros(n, np.float32), np.ones(n, np.float32)
    rx = np.stack([o, z, z, z, np.cos(p), -np.sin(p), z, np.sin(p), np.cos(p)], 1).reshape(n, 3, 3)
    ry = np.stack([np.cos(y), z, np.sin(y), z, o, z, -np.sin(y), z, np.cos(y)], 1).reshape(n, 3, 3)
    rz = np.stack([np.cos(r), -np.sin(r), z, np.sin(r), np.cos(r), z, z, z, o], 1).reshape(n, 3, 3)
    return np.transpose(rz @ ry @ rx, (0, 2, 1))


def transform_keypoints(kp: np.ndarray, exp: np.ndarray, scale: np.ndarray, t: np.ndarray,
                        rot: np.ndarray) -> np.ndarray:
    """``scale * (kp @ R + exp) + t`` with t_z dropped, as LivePortrait does."""
    b = kp.shape[0]
    t = np.asarray(t, np.float32).reshape(b, 3).copy()
    t[:, 2] = 0.0
    out = (kp.reshape(b, -1, 3) @ rot + exp.reshape(b, -1, 3)) * scale.reshape(-1, 1, 1)
    return (out + t[:, None, :]).astype(np.float32)


def driving_keypoints(x_s: np.ndarray, scale_s: np.ndarray, exp_s: np.ndarray,
                      exp_d: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``x_s + scale_s * w * (exp_d - exp_s)``; rotation and translation cancel."""
    n = x_s.shape[1]
    delta = (exp_d - exp_s).reshape(x_s.shape) * weights.reshape(1, n, 1)
    return (x_s + delta * scale_s.reshape(-1, 1, 1)).astype(np.float32)


def eye_close_ratio(pts203: np.ndarray) -> np.ndarray:
    """(left, right) lid gap over eye width from LivePortrait's 203 landmarks."""
    p = np.asarray(pts203, np.float32).reshape(-1, 2)

    def ratio(a: int, b: int, c: int, d: int) -> float:
        return float(np.linalg.norm(p[a] - p[b]) / (np.linalg.norm(p[c] - p[d]) + 1e-6))

    return np.array([ratio(6, 18, 0, 12), ratio(30, 42, 24, 36)], np.float32)


def _to_input(bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])


@dataclass(frozen=True)
class _Motion:
    pitch: np.ndarray
    yaw: np.ndarray
    roll: np.ndarray
    t: np.ndarray
    exp: np.ndarray
    scale: np.ndarray
    kp: np.ndarray


class ExpressionRestorer:
    """LivePortrait expression transfer between two aligned crops of the same face.

    Args:
        engine: Session provider.
        model_paths: ``{key: path}`` for :data:`MODELS` (``eye`` and
            ``landmark`` are only needed for blink sync).
        weights: Default :class:`ExpressionWeights`.
        blink: Default for blink sync.
        stitching: Apply LivePortrait's stitching correction (keeps the crop
            border aligned with the untouched surroundings).
        warping_fp16: TensorRT precision of the warping generator (the
            dominant cost). See the README for the measured trade-off.
    """

    def __init__(self, engine: ExecutionEngine, model_paths: dict[str, Path | str],
                 weights: ExpressionWeights | None = None, blink: bool = True,
                 stitching: bool = True, warping_fp16: bool = True) -> None:
        missing = {"appearance", "motion", "warping"} - set(model_paths)
        if missing:
            raise KeyError(f"missing LivePortrait models: {sorted(missing)}")
        self.engine = engine
        self.paths = {k: Path(v) for k, v in model_paths.items()}
        self.weights = weights or ExpressionWeights()
        self.blink = blink
        self.stitching = stitching
        self.warping_fp16 = warping_fp16
        self._warping_path: Path | None = None

    @classmethod
    def from_registry(cls, engine: ExecutionEngine, registry: ModelRegistry,
                      **kwargs: object) -> ExpressionRestorer:
        """Fetch/verify the LivePortrait models from the zoo."""
        paths = {k: registry.ensure(f"liveportrait_{k}", show_progress=False) for k in MODELS}
        return cls(engine, paths, **kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ sessions
    def _session(self, key: str) -> ManagedSession:
        if key == "warping":
            return self._warping_session()
        return self.engine.get_session(self.paths[key], trt_fp16=_FP16_OK[key])

    def _warping_session(self) -> ManagedSession:
        if self._warping_path is None:
            from face_engine.utils.gridsample5d import ensure_patched_model

            self._warping_path = Path(ensure_patched_model(str(self.paths["warping"]),
                                                           verbose=False))
        if self._warping_path != self.paths["warping"]:
            return self.engine.get_session(self._warping_path, trt_fp16=self.warping_fp16)
        # Stock model: its 5-D GridSample must not land on the CUDA EP.
        return self.engine.get_session(self._warping_path, trt_fp16=self.warping_fp16,
                                       providers=[Provider.TENSORRT, Provider.CPU])

    def _run(self, key: str, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run by input NAME; a missing name raises instead of silently misbinding."""
        handle = self._session(key)
        expected = set(handle.input_names)
        missing = expected - set(feeds)
        if missing:
            raise KeyError(f"{key}: model expects {sorted(expected)}, missing {sorted(missing)}")
        return handle.session.run(None, {k: v for k, v in feeds.items() if k in expected})

    def _motion(self, img: np.ndarray) -> _Motion:
        pitch, yaw, roll, t, exp, scale, kp = self._run("motion", {"img": img})[:7]
        return _Motion(headpose_to_degrees(pitch), headpose_to_degrees(yaw),
                       headpose_to_degrees(roll), np.asarray(t, np.float32).reshape(1, 3),
                       np.asarray(exp, np.float32).reshape(1, -1, 3),
                       np.asarray(scale, np.float32).reshape(1, 1),
                       np.asarray(kp, np.float32).reshape(1, -1, 3))

    def expression_coefficients(self, crop: np.ndarray) -> np.ndarray:
        """``(21, 3)`` LivePortrait expression deformation of an aligned crop."""
        return self._motion(_to_input(crop, INPUT_SIZE)).exp.reshape(-1, 3)

    def _eye_ratio(self, crop: np.ndarray) -> np.ndarray:
        out = self._run("landmark", {"input": _to_input(crop, LANDMARK_SIZE)})
        return eye_close_ratio(np.asarray(out[2], np.float32).reshape(-1, 2) * LANDMARK_SIZE)

    # ------------------------------------------------------------------ restore
    def restore(self, swapped_crop: np.ndarray, target_crop: np.ndarray,
                weights: ExpressionWeights | None = None, blink: bool | None = None) -> np.ndarray:
        """``swapped_crop`` with ``target_crop``'s expression; same size as the input.

        Both must be the SAME aligned crop (same matrix) of the same frame.
        Returns the input unchanged on any failure.
        """
        weights = weights or self.weights
        blink = self.blink if blink is None else blink
        if swapped_crop is None or target_crop is None or not (weights.active or blink):
            return swapped_crop
        try:
            return self._restore(swapped_crop, target_crop, weights, blink)
        except Exception as exc:  # noqa: BLE001 - never cost a frame
            logger.warning("expression restore failed, keeping the swap: %s", exc)
            return swapped_crop

    def _restore(self, swapped: np.ndarray, target: np.ndarray, weights: ExpressionWeights,
                 blink: bool) -> np.ndarray:
        src_in, drv_in = _to_input(swapped, INPUT_SIZE), _to_input(target, INPUT_SIZE)
        feature = self._run("appearance", {"img": src_in})[0]
        m_s, m_d = self._motion(src_in), self._motion(drv_in)
        rot = rotation_matrix(m_s.pitch, m_s.yaw, m_s.roll)
        x_s = transform_keypoints(m_s.kp, m_s.exp, m_s.scale, m_s.t, rot)
        n = x_s.shape[1]
        w = weights.per_keypoint(n)
        x_d = driving_keypoints(x_s, m_s.scale, m_s.exp, m_d.exp, w)

        if blink and "eye" in self.paths and "landmark" in self.paths:
            ratio_s, ratio_t = self._eye_ratio(swapped), self._eye_ratio(target)
            # Feed the MLP the keypoints AFTER the eye delta and the opening they
            # imply, so a gaze weight and blink sync do not move the lids twice
            # (roop-ultimate measured that double move closing open eyes).
            w_eye = float(np.clip(np.mean([w[i] for i in EYE_INDICES]), 0.0, 1.0))
            kp_in = x_d if w_eye > 0 else x_s
            ratio_in = ratio_s + w_eye * (ratio_t - ratio_s)
            eye_input = np.concatenate([kp_in.reshape(1, -1),
                                        np.array([[ratio_in[0], ratio_in[1], ratio_t[0]]],
                                                 np.float32)], axis=1)
            x_d = x_d + np.asarray(self._run("eye", {"input": eye_input})[0],
                                   np.float32).reshape(x_d.shape)

        if self.stitching and "stitching" in self.paths:
            delta = np.asarray(self._run("stitching", {"input": np.concatenate(
                [x_s.reshape(1, -1), x_d.reshape(1, -1)], axis=1)})[0], np.float32).reshape(-1)
            if delta.size >= n * 3 + 2:
                x_d = x_d + delta[:n * 3].reshape(1, n, 3)
                x_d[:, :, :2] += delta[n * 3:n * 3 + 2].reshape(1, 1, 2)

        out = self._run("warping", {"feature_3d": np.asarray(feature, np.float32),
                                    "kp_source": np.ascontiguousarray(x_s),
                                    "kp_driving": np.ascontiguousarray(x_d)})[0]
        img = np.asarray(out, np.float32)[0].transpose(1, 2, 0)
        if not np.all(np.isfinite(img)):
            raise RuntimeError("warping produced non-finite output")
        bgr = cv2.cvtColor(np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
        h, w_ = swapped.shape[:2]
        if bgr.shape[:2] != (h, w_):
            bgr = cv2.resize(bgr, (w_, h), interpolation=cv2.INTER_CUBIC if h > 512
                             else cv2.INTER_AREA)
        return bgr

    def restore_in_frame(self, frame: np.ndarray, original_frame: np.ndarray,
                         face: Face | np.ndarray, mask: np.ndarray | None = None,
                         crop_size: int = 512, template: str = "arcface_128",
                         weights: ExpressionWeights | None = None,
                         blink: bool | None = None) -> np.ndarray:
        """Restore the expression of one face in a (swapped/enhanced) frame.

        Cuts the same crop from ``frame`` and ``original_frame``, restores, and
        pastes back with ``mask`` (``(S, S)`` any size; default: none, i.e.
        the whole crop — pass the composite face mask in real use).
        """
        bgr, orig = as_bgr(frame), as_bgr(original_frame)
        if bgr is None or orig is None:
            return frame
        kps = face.kps if isinstance(face, Face) else np.asarray(face, np.float64)
        try:
            matrix = estimate_similarity_transform(kps, template_points(crop_size, template))
        except AlignmentError:
            return bgr.copy()
        swapped_crop = warp_face_by_translation(bgr, matrix, crop_size)
        target_crop = warp_face_by_translation(orig, matrix, crop_size)
        restored = self.restore(swapped_crop, target_crop, weights, blink)
        if restored is swapped_crop:
            return bgr.copy()
        if mask is not None and mask.shape[:2] != (crop_size, crop_size):
            mask = cv2.resize(np.asarray(mask, np.float32), (crop_size, crop_size))
        return warp_face_inverse(bgr, restored, matrix, mask)


# ---------------------------------------------------------------------------- CUDA, batched
def headpose_to_degrees_cuda(pred: torch.Tensor) -> torch.Tensor:
    """``(N, 66)`` bins -> ``(N,)`` degrees (:func:`headpose_to_degrees` on the GPU)."""
    import torch

    prob = torch.softmax(pred.float().reshape(-1, NUM_BINS), dim=1)
    bins = torch.arange(NUM_BINS, device=pred.device, dtype=torch.float32)
    return (prob * bins).sum(dim=1) * 3.0 - 97.5


def rotation_matrix_cuda(pitch: torch.Tensor, yaw: torch.Tensor,
                         roll: torch.Tensor) -> torch.Tensor:
    """Row-vector rotations ``(N, 3, 3)`` (:func:`rotation_matrix` on the GPU)."""
    import torch

    p, y, r = (torch.deg2rad(a.float().reshape(-1)) for a in (pitch, yaw, roll))
    z, o = torch.zeros_like(p), torch.ones_like(p)
    rx = torch.stack([o, z, z, z, p.cos(), -p.sin(), z, p.sin(), p.cos()], 1).view(-1, 3, 3)
    ry = torch.stack([y.cos(), z, y.sin(), z, o, z, -y.sin(), z, y.cos()], 1).view(-1, 3, 3)
    rz = torch.stack([r.cos(), -r.sin(), z, r.sin(), r.cos(), z, z, z, o], 1).view(-1, 3, 3)
    return (rz @ ry @ rx).transpose(1, 2)


def eye_close_ratio_cuda(pts203: torch.Tensor) -> torch.Tensor:
    """``(N, 203, 2)`` landmarks -> ``(N, 2)`` lid gap / eye width (left, right)."""
    def ratio(a: int, b: int, c: int, d: int) -> torch.Tensor:
        return ((pts203[:, a] - pts203[:, b]).norm(dim=1)
                / ((pts203[:, c] - pts203[:, d]).norm(dim=1) + 1e-6))

    import torch

    return torch.stack([ratio(6, 18, 0, 12), ratio(30, 42, 24, 36)], dim=1)


class BatchedExpressionRestorer:
    """:class:`ExpressionRestorer` over batches of faces, on the GPU.

    Same model, same maths, same safeguards (inputs bound by name, the
    gridsample5d-patched warping generator, FP32 for the motion extractor and
    landmark net, blink sync on and gaze follow off by default). What moves:

    * the keypoint algebra runs as batched torch ops;
    * appearance, motion, eye and stitching nets run once per batch (verified
      dynamic-batch copies); the landmark net and the warping generator fix
      their batch inside the graph (:mod:`face_engine.utils.onnx_batch`) and
      run once per face;
    * a face whose warped output is non-finite keeps its input crop
      (``torch.where``, no host read), so an expression tweak never costs a
      frame.

    Retargeted terms: LivePortrait's expression deformation (lips, brows,
    cheeks, jaw; eyeballs only with ``eyes > 0``) and, via the eye MLP, lid
    opening. Head pose (yaw/pitch/roll) is NOT transferred: both crops are the
    same aligned crop of the same frame, so the pose already matches, and in
    ``x_d - x_s`` the rotation cancels (a noisy pose estimate cannot move the
    head).
    """

    def __init__(self, engine: ExecutionEngine, model_paths: dict[str, Path | str],
                 weights: ExpressionWeights | None = None, blink: bool = True,
                 stitching: bool = True, warping_fp16: bool = True, max_batch: int = 8,
                 batching: bool = True) -> None:
        from face_engine.utils.onnx_batch import batched_model

        self.host = ExpressionRestorer(engine, model_paths, weights, blink, stitching,
                                       warping_fp16)
        self.engine = engine
        self.max_batch = max_batch
        # Calls that raised and returned their input unchanged. Benchmarks and
        # tests must read this: a swallowed failure looks like a fast success.
        self.failures = 0
        self._batched: dict[str, Path] = {}
        for key in ("appearance", "motion", "eye", "stitching") if batching else ():
            if key in self.host.paths:
                path = batched_model(self.host.paths[key])
                if path is not None:
                    self._batched[key] = path

    @classmethod
    def from_registry(cls, engine: ExecutionEngine, registry: ModelRegistry,
                      **kwargs: Any) -> BatchedExpressionRestorer:
        paths = {k: registry.ensure(f"liveportrait_{k}", show_progress=False) for k in MODELS}
        return cls(engine, paths, **kwargs)

    @property
    def weights(self) -> ExpressionWeights:
        return self.host.weights

    def _session(self, key: str) -> tuple[ManagedSession, bool]:
        if key in self._batched:
            from face_engine.utils.onnx_batch import batch_shape_profile

            path = self._batched[key]
            profile = batch_shape_profile(path, max_batch=self.max_batch)
            return self.engine.get_session(path, shape_profile=profile,
                                           trt_fp16=_FP16_OK[key]), True
        return self.host._session(key), False

    def _run(self, key: str, feeds: dict[str, torch.Tensor],
             shapes: dict[str, tuple[int, ...]] | None = None) -> dict[str, torch.Tensor]:
        """Bound BY NAME; batched where the model batches, else one row at a time."""
        import torch

        handle, batched = self._session(key)
        expected = set(handle.input_names)
        missing = expected - set(feeds)
        if missing:
            raise KeyError(f"{key}: model expects {sorted(expected)}, missing {sorted(missing)}")
        n = next(iter(feeds.values())).shape[0]
        step = self.max_batch if batched else 1
        parts: list[dict[str, torch.Tensor]] = []
        for start in range(0, n, step):
            stop = min(start + step, n)
            out_shapes = None
            if shapes is not None:
                out_shapes = {k: (stop - start, *v) for k, v in shapes.items()}
            parts.append(handle.run_binding(
                {k: feeds[k][start:stop].contiguous() for k in expected},
                output_shapes=out_shapes))
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

    @staticmethod
    def _to_input(crops: torch.Tensor, size: int) -> torch.Tensor:
        from face_engine.processors.swapper import resample_crops

        return (resample_crops(crops.float(), size).clamp(0, 255).flip(1) / 255.0).contiguous()

    def _motion(self, img: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self._run("motion", {"img": img})
        n = img.shape[0]
        return {"pitch": headpose_to_degrees_cuda(out["pitch"]),
                "yaw": headpose_to_degrees_cuda(out["yaw"]),
                "roll": headpose_to_degrees_cuda(out["roll"]),
                "t": out["t"].float().reshape(n, 3), "exp": out["exp"].float().reshape(n, -1, 3),
                "scale": out["scale"].float().reshape(n, 1), "kp": out["kp"].float().reshape(n, -1, 3)}

    def expression_coefficients(self, crops: torch.Tensor) -> torch.Tensor:
        """``(N, 21, 3)`` expression deformation of aligned crops."""
        return self._motion(self._to_input(crops, INPUT_SIZE))["exp"]

    def _eye_ratio(self, crops: torch.Tensor) -> torch.Tensor:
        handle, _ = self._session("landmark")
        names = handle.output_names
        out = self._run("landmark", {"input": self._to_input(crops, LANDMARK_SIZE)})
        pts = out[names[2]].float().reshape(crops.shape[0], -1, 2) * LANDMARK_SIZE
        return eye_close_ratio_cuda(pts)

    def restore(self, swapped: torch.Tensor, target: torch.Tensor,
                weights: ExpressionWeights | None = None,
                blink: bool | None = None) -> torch.Tensor:
        """``(N, 3, S, S)`` swapped crops with the matching target crops' expression.

        Both batches must be the SAME aligned crops (same matrices) of the same
        frames. Returns the input unchanged where the network output is
        unusable, and entirely on any exception.
        """
        weights = weights or self.host.weights
        blink = self.host.blink if blink is None else blink
        if swapped.shape[0] == 0 or not (weights.active or blink):
            return swapped
        try:
            return self._restore(swapped.float(), target.float(), weights, blink)
        except Exception as exc:  # noqa: BLE001 - never cost a frame
            self.failures += 1
            logger.warning("batched expression restore failed, keeping the swap: %s", exc)
            return swapped

    def _restore(self, swapped: torch.Tensor, target: torch.Tensor, weights: ExpressionWeights,
                 blink: bool) -> torch.Tensor:
        import torch

        from face_engine.processors.swapper import resample_crops

        n = swapped.shape[0]
        src_in, drv_in = self._to_input(swapped, INPUT_SIZE), self._to_input(target, INPUT_SIZE)
        feature = self._run("appearance", {"img": src_in})
        feature = next(iter(feature.values())).float()
        both = self._motion(torch.cat([src_in, drv_in]))
        m_s = {k: v[:n] for k, v in both.items()}
        m_d = {k: v[n:] for k, v in both.items()}
        rot = rotation_matrix_cuda(m_s["pitch"], m_s["yaw"], m_s["roll"])
        t = m_s["t"].clone()
        t[:, 2] = 0.0
        x_s = (m_s["kp"] @ rot + m_s["exp"]) * m_s["scale"].view(-1, 1, 1) + t[:, None, :]
        k = x_s.shape[1]
        w = torch.as_tensor(weights.per_keypoint(k), device=x_s.device).view(1, k, 1)
        x_d = x_s + (m_d["exp"] - m_s["exp"]) * w * m_s["scale"].view(-1, 1, 1)

        if blink and "eye" in self.host.paths and "landmark" in self.host.paths:
            ratios = self._eye_ratio(torch.cat([swapped, target]))
            ratio_s, ratio_t = ratios[:n], ratios[n:]
            w_eye = float(np.clip(np.mean([weights.per_keypoint(k)[i] for i in EYE_INDICES]),
                                  0.0, 1.0))
            kp_in = x_d if w_eye > 0 else x_s
            ratio_in = ratio_s + w_eye * (ratio_t - ratio_s)
            eye_input = torch.cat([kp_in.reshape(n, -1), ratio_in, ratio_t[:, :1]], dim=1)
            delta = next(iter(self._run("eye", {"input": eye_input}).values())).float()
            x_d = x_d + delta.reshape(x_d.shape)

        if self.host.stitching and "stitching" in self.host.paths:
            delta = next(iter(self._run("stitching", {"input": torch.cat(
                [x_s.reshape(n, -1), x_d.reshape(n, -1)], dim=1)}).values())).float()
            if delta.shape[1] >= k * 3 + 2:
                x_d = x_d + delta[:, :k * 3].reshape(n, k, 3)
                x_d[:, :, :2] = x_d[:, :, :2] + delta[:, k * 3:k * 3 + 2].reshape(n, 1, 2)

        warped = self._run("warping", {"feature_3d": feature, "kp_source": x_s.contiguous(),
                                       "kp_driving": x_d.contiguous()})
        img = next(iter(warped.values())).float()
        ok = torch.isfinite(img).flatten(1).all(1)
        bgr = (img.nan_to_num(0.0).clamp(0, 1) * 255.0).flip(1)
        bgr = resample_crops(bgr, swapped.shape[-1])
        return torch.where(ok.view(-1, 1, 1, 1), bgr, swapped)
