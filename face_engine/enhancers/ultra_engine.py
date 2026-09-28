"""Ultra Restore: routed GPEN restoration, LAB lock, region-weighted frequency-split fusion.

:class:`UltraRestoreEngine` runs GPEN-BFR-512 / 1024 from the AOT TensorRT
engines (``tools/compile_engines.py``: ``gpen_bfr_{512,1024}_sm{SM}_fp16_b1.engine``,
the encoder's final linear and the style pixel-norm pinned to FP32; unpinned,
1024 produced NaN and 512 failed fidelity at 0.035 of range). Inputs and
outputs are bound ONCE to pre-allocated device buffers
(``context.set_tensor_address``); a call copies the crop in, enqueues
``execute_async_v3`` on the current torch stream and copies the result out.

Batching: GPEN's StyleGAN2 modulated convolutions fix the batch inside the
graph (``face_engine.utils.onnx_batch``), so the engines are batch 1. A batch
of up to ``max_batch`` faces (default 4) runs as consecutive enqueues on one
stream through the same bound buffers, without a host sync between faces.

:class:`UltraRestorer` is the per-frame pipeline:

1. :class:`~face_engine.enhancers.scale_router.ScaleAwareEnhancerRouter`:
   bypass / 512 / 1024 per face;
2. per route: ``ffhq_512``-template crops ``S`` of the SWAPPED frame at the
   model's size; GPEN -> ``R``; an output guard (non-finite or flat faces are
   dropped: nothing pasted);
3. LAB lock of ``R`` to the ORIGINAL target crop (:func:`lab_lock`);
4. region weights (BiSeNet via :class:`~face_engine.pipeline.masker.GPUMasker`
   when given, else the 5-landmark map) and the frequency-split crossfade
   at crop sigma 2.5;
5. paste back through the face ellipse x valid-area mask.

Measured 2026-09-28 (Love / Weeds / Monica, 20 frames each, 78 faces routed
to 512 / 1024, the footage face as S; identity = ArcFace cosine of the pasted
face to S, texture = skin high-band std on cheek/forehead discs vs the plate,
macro = |8 px low band - S's|):

==================================  ===============  =======  ======  =======
arm                                 identity med/p10  texture  macro   ms/face
==================================  ===============  =======  ======  =======
linear alpha 0.8 (the Ultra preset)  0.916 / 0.693    232%     1.12    27.3
split, sigma 2.5 (default)           0.991 / 0.959    160%     0.03    32.2
split, paste-aware sigma             0.994 / 0.922    166%     0.03    31.5
split, LAB lock without L* std       0.989 / 0.945    182%     0.04    33.1
split, no LAB lock                   0.989 / 0.944    181%     0.04    29.8
the brief's literal formula, k=9     0.991 / 0.962    172%     0.03    31.9
==================================  ===============  =======  ======  =======

Paste-aware sigma is an option, not the default: +6 points of texture for
-0.04 identity at p10 (a larger sigma on small faces hands the restorer more
of the mid band). The LAB lock's MEAN part is a no-op in the output (the
split keeps only R's high band, which a constant shift does not reach; "L*
mean only" == "no lock" above); its std part is what helps (p10 0.945 ->
0.959, texture 182% -> 160%).

Per face, the network is 85-90% of the pass (RTX 4070): 512 = 18.9 ms +
lab_lock 1.8 + BiSeNet 1.6 + weights/split 1.6; 1024 = 33.5 + 3.0 + 1.6 + 2.9.

Everything is torch / kornia on the frames' device. Host reads: the route
codes (which networks run), the paste-aware kernel size (one scalar), and
nothing else.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from face_engine.enhancers.frequency import paste_aware_sigma
from face_engine.enhancers.scale_router import (
    ROUTE_MODEL,
    Route,
    ScaleAwareEnhancerRouter,
)
from face_engine.enhancers.semantic_fusion import (
    IrisStabilizer,
    RegionWeights,
    SemanticRegionalRestorer,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

TEMPLATE = "ffhq_512"
MODEL_SIZE = {"gpen_bfr_512": 512, "gpen_bfr_1024": 1024}
COLLAPSE_STD = 2.0  # levels: a restored face flatter than this is a failed pass
STATS_SIZE = 128    # LAB-lock statistics resolution


# ---------------------------------------------------------------------------- colour lock
def lab_lock(image: torch.Tensor, reference: torch.Tensor, region: torch.Tensor | None = None,
             *, match_luma_std: bool = True, stats_size: int | None = None) -> torch.Tensor:
    """Match per-channel L*, a*, b* mean and std of ``image`` to ``reference``.

    ``image`` ``(N, 3, H, W)`` BGR ``[0, 255]``; ``reference`` the same faces
    on the same template at ANY size. Statistics inside ``region``
    ``(N or 1, 1, h, w)`` (any size; default: all), computed on
    ``stats_size``-square copies of both (None = ``image``'s size): moments
    need no 1024 px, and the reference then only has to be cut at that size.
    ``match_luma_std=False`` matches L*'s mean only. Colour: kornia CIE
    L*a*b*, D65.
    """
    import torch
    import torch.nn.functional as F

    from face_engine.processors.color import _masked_moments, bgr_to_lab, lab_to_bgr

    img = image.float()
    n, _, h, _ = img.shape
    side = stats_size or h

    def at(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == (side, side):
            return x
        return F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False,
                             antialias=x.shape[-1] > side)

    lab = bgr_to_lab(img)
    small = lab if side == h else bgr_to_lab(at(img))
    ref = bgr_to_lab(at(reference.float()))
    weight = (torch.ones((1, 1, side, side), device=img.device) if region is None
              else at(region.to(img.device, img.dtype))).expand(n, 1, side, side)
    mu_i, sd_i = _masked_moments(small, weight)
    mu_r, sd_r = _masked_moments(ref, weight)
    ratio = sd_r / sd_i.clamp_min(1e-3)
    if not match_luma_std:
        ratio = torch.cat([torch.ones_like(ratio[:, :1]), ratio[:, 1:]], 1)
    out = (lab - mu_i) * ratio + mu_r
    enough = (weight > 0.5).flatten(1).sum(1) >= 16
    return torch.where(enough.view(n, 1, 1, 1), lab_to_bgr(out), img)


def paste_roi(canvas: torch.Tensor, crops: torch.Tensor, matrix: torch.Tensor,
              mask: torch.Tensor, frame_index: torch.Tensor | None = None) -> torch.Tensor:
    """:func:`~face_engine.pipeline.aligner.warp_face_inverse_cuda` over each
    face's own footprint instead of the whole frame, IN PLACE on ``canvas``.

    The full-frame paste warps every face to ``H x W`` (8.3 Mpx per face at
    4K). Here each crop is warped only onto the box its corners cover (+2 px),
    same premultiplied compositing, faces in order. One host read: the boxes
    (and frame indices) of all faces.
    """
    import torch

    from face_engine.pipeline.aligner import invert_affine_cuda, warp_face_inverse_cuda

    n, s = crops.shape[0], crops.shape[-1]
    if n == 0:
        return canvas
    b, _, h, w = canvas.shape
    m = matrix.float()
    inv = invert_affine_cuda(m)
    corners = torch.tensor([[0.0, 0.0], [s, 0.0], [0.0, s], [s, s]], device=m.device)
    pts = corners[None] @ inv[:, :, :2].transpose(1, 2) + inv[:, None, :, 2]   # (N, 4, 2)
    lo = (pts.amin(1) - 2).floor().nan_to_num(0.0)
    hi = (pts.amax(1) + 2).ceil().nan_to_num(0.0)
    if frame_index is not None:
        fi = frame_index.to(m.device).float()
    elif n == b:
        fi = torch.arange(n, device=m.device).float()
    else:
        fi = torch.zeros(n, device=m.device)
    table = torch.cat([lo, hi, fi[:, None]], 1).cpu().tolist()   # the one host read
    for i, (x0, y0, x1, y1, f) in enumerate(table):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 <= x0 or y1 <= y0:
            continue
        f = int(f)
        roi_m = m[i:i + 1].clone()
        shift = torch.tensor([float(x0), float(y0)], device=m.device)
        roi_m[:, :, 2] = roi_m[:, :, 2] + roi_m[:, :, :2] @ shift
        sub = canvas[f:f + 1, :, y0:y1, x0:x1]
        canvas[f:f + 1, :, y0:y1, x0:x1] = warp_face_inverse_cuda(sub, crops[i:i + 1], roi_m,
                                                                  mask[i:i + 1])
    return canvas


# ---------------------------------------------------------------------------- engine
@dataclass
class _Bound:
    runner: Any
    size: int
    trt: bool
    inputs: Any = None           # pre-allocated input buffer (TensorRT)
    outputs: dict[str, Any] = field(default_factory=dict)
    first: str = ""


class UltraRestoreEngine:
    """GPEN-512 / GPEN-1024 restoration with pre-bound TensorRT buffers.

    Args:
        engine: :class:`~face_engine.core.execution.ExecutionEngine` (its
            config decides whether TensorRT engines may be used).
        model_paths: zoo name (``gpen_bfr_512`` / ``gpen_bfr_1024``) -> ONNX file.
        precision: AOT engine precision; without a compiled engine the model
            runs through ONNX Runtime (FP32).
        max_batch: Faces per :meth:`restore` call accepted as one batch (1..4).
    """

    def __init__(self, engine: Any, model_paths: dict[str, Path | str], *,
                 precision: str = "fp16", max_batch: int = 4) -> None:
        if not 1 <= max_batch <= 4:
            raise ValueError("max_batch must be 1..4")
        unknown = set(model_paths) - set(MODEL_SIZE)
        if unknown:
            raise KeyError(f"unknown restorers {sorted(unknown)}; known {sorted(MODEL_SIZE)}")
        self.engine = engine
        self.model_paths = {k: Path(v) for k, v in model_paths.items()}
        self.precision = precision
        self.max_batch = max_batch
        self._bound: dict[str, _Bound] = {}

    def uses_tensorrt(self, model: str) -> bool:
        return self._bind(model).trt

    def _bind(self, model: str) -> _Bound:
        if model in self._bound:
            return self._bound[model]
        import torch

        from face_engine.core.trt_compiler import aot_engine

        size = MODEL_SIZE[model]
        runner = aot_engine(self.model_paths[model], self.precision, self.engine.config)
        if runner is None:
            logger.info("%s: no compiled %s engine; ONNX Runtime", model, self.precision)
            bound = _Bound(self.engine.get_session(self.model_paths[model]), size, False)
            bound.first = bound.runner.output_names[0]
            self._bound[model] = bound
            return bound
        ctx = runner.context
        name_in = runner.input_names[0]
        dt_in = runner._dtypes[name_in]
        inputs = torch.empty((1, 3, size, size), dtype=dt_in, device=runner.device)
        ctx.set_input_shape(name_in, tuple(inputs.shape))
        ctx.set_tensor_address(name_in, inputs.data_ptr())
        outputs = {}
        for name in runner.output_names:
            out = torch.empty(tuple(ctx.get_tensor_shape(name)), dtype=runner._dtypes[name],
                              device=runner.device)
            ctx.set_tensor_address(name, out.data_ptr())
            outputs[name] = out
        bound = _Bound(runner, size, True, inputs, outputs, runner.output_names[0])
        self._bound[model] = bound
        return bound

    def _run_one(self, bound: _Bound, blob: torch.Tensor) -> torch.Tensor:
        """One ``(1, 3, S, S)`` normalized face -> raw output (a new tensor)."""
        import torch

        if not bound.trt:
            s = bound.size
            out = bound.runner.run_binding({bound.runner.input_names[0]: blob},
                                           output_shapes={bound.first: (1, 3, s, s)},
                                           unreturned_outputs=bound.runner.output_names[1:])
            return out[bound.first]
        bound.inputs.copy_(blob)
        stream = torch.cuda.current_stream(bound.runner.device)
        if not bound.runner.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"{bound.runner.path.name}: execute_async_v3 failed")
        return bound.outputs[bound.first].clone()  # the buffer is reused by the next face

    def restore(self, model: str, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(N, 3, S, S)`` BGR ``[0, 255]`` crops at the model's size -> ``(restored, ok)``.

        ``ok`` is False for a non-finite or collapsed (flat) output; that face's
        ``restored`` is its input.
        """
        import torch

        bound = self._bind(model)
        s = bound.size
        if crops.shape[-2:] != (s, s):
            raise ValueError(f"{model} takes {s}x{s} crops, got {tuple(crops.shape[-2:])}")
        x = crops.float().clamp(0, 255)
        blob = x.flip(1) / 127.5 - 1.0  # RGB, [-1, 1]
        raws = []
        for start in range(0, blob.shape[0], self.max_batch):
            for i in range(start, min(start + self.max_batch, blob.shape[0])):
                raws.append(self._run_one(bound, blob[i:i + 1]))
        raw = torch.cat(raws).float()
        finite = torch.isfinite(raw).flatten(1).all(1)
        restored = ((raw.nan_to_num(0.0).clamp(-1, 1) + 1.0) * 127.5).flip(1)
        std = restored.flatten(1).std(1)
        ok = finite & (std >= COLLAPSE_STD)
        return torch.where(ok.view(-1, 1, 1, 1), restored, x), ok


# ---------------------------------------------------------------------------- pipeline
@dataclass
class UltraResult:
    """Attributes:
        frames: ``(B, 3, H, W)`` output frames (float BGR ``[0, 255]``).
        plan: The router's :class:`~face_engine.enhancers.scale_router.RoutingPlan`.
        restored: ``(N,)`` bool on the device: restored and pasted (False =
            bypassed, rejected, or degenerate landmarks).
        sigma: ``(N,)`` crop-space split sigma used (0 for bypassed faces).
    """

    frames: Any
    plan: Any
    restored: Any
    sigma: Any


class UltraRestorer:
    """The Ultra Restore pass over swapped frames (see the module docstring).

    Args:
        restore_engine: :class:`UltraRestoreEngine` holding the routed models.
        router: :class:`ScaleAwareEnhancerRouter`.
        regional: :class:`SemanticRegionalRestorer` (weights, split filter).
        parser: Optional :class:`~face_engine.pipeline.masker.GPUMasker` with a
            BiSeNet path: region weights from the parse; else from landmarks.
        sigma_mode: ``"fixed"`` (default: ``regional.blender.sigma`` crop
            pixels, the brief's 2.5) or ``"paste_aware"`` (``screen_sigma``
            frame pixels; measured worse on identity, see the module docstring).
        lock_luma_std: :func:`lab_lock`'s ``match_luma_std``.
        iris: Optional :class:`IrisStabilizer` (needs ``track_ids``).
    """

    def __init__(self, restore_engine: UltraRestoreEngine,
                 router: ScaleAwareEnhancerRouter | None = None,
                 regional: SemanticRegionalRestorer | None = None, parser: Any = None, *,
                 sigma_mode: str = "fixed", screen_sigma: float = 1.25,
                 lock_luma_std: bool = True, iris: IrisStabilizer | None = None,
                 paste_blur: float = 0.3) -> None:
        if sigma_mode not in ("paste_aware", "fixed"):
            raise ValueError("sigma_mode must be 'paste_aware' or 'fixed'")
        self.engine = restore_engine
        self.router = router or ScaleAwareEnhancerRouter()
        self.regional = regional or SemanticRegionalRestorer(RegionWeights())
        self.parser = parser
        self.sigma_mode = sigma_mode
        self.screen_sigma = float(screen_sigma)
        self.lock_luma_std = lock_luma_std
        self.iris = iris
        self.paste_blur = paste_blur
        self._masks: dict[tuple[int, str], tuple[Any, Any]] = {}

    def _paste_masks(self, size: int, device: Any) -> tuple[Any, Any]:
        """(paste weight, colour-statistics region) for a ``size`` crop, cached."""
        import torch

        from face_engine.processors.enhancer import face_region_mask, paste_mask

        key = (size, str(device))
        if key not in self._masks:
            self._masks[key] = (
                torch.as_tensor(paste_mask(size, self.paste_blur), device=device)[None, None],
                torch.as_tensor((face_region_mask(size) > 0).astype("float32"),
                                device=device)[None, None])
        return self._masks[key]

    def process(self, frames: torch.Tensor, kps: torch.Tensor, boxes: torch.Tensor, *,
                reference: torch.Tensor | None = None, frame_index: torch.Tensor | None = None,
                track_ids: list[Any] | None = None) -> UltraResult:
        """Restore the faces ``kps`` ``(N, 5, 2)`` / ``boxes`` ``(N, 4)`` in ``frames``.

        ``frames`` hold the SWAPPED faces; ``reference`` the original targets
        (the LAB lock's reference; default ``frames``). ``frame_index`` ``(N,)``
        when ``B > 1``; ``track_ids`` for the iris stabilizer.
        """
        import torch
        import torch.nn.functional as F

        from face_engine.pipeline.aligner import (
            crop_valid_mask_cuda,
            similarity_is_valid,
            similarity_matrices_cuda,
            transform_points_cuda,
            warp_face_cuda,
        )

        out = (frames if frames.ndim == 4 else frames[None]).float().clone()  # pasted in place
        if reference is frames:
            reference = None  # the colour reference IS the input: reuse its crops
        ref_frames = None if reference is None else (
            reference if reference.ndim == 4 else reference[None]).float()
        n = kps.shape[0]
        dev = out.device
        restored_flag = torch.zeros(n, dtype=torch.bool, device=dev)
        sigmas = torch.zeros(n, device=dev)
        plan = self.router.plan(boxes.to(dev))
        for route in (Route.GPEN_512, Route.GPEN_1024):
            idx = plan.indices.get(route)
            if idx is None:
                continue
            model = ROUTE_MODEL[route]
            size = MODEL_SIZE[model]
            fi = None if frame_index is None else frame_index.to(dev)[idx]
            m = similarity_matrices_cuda(kps.to(dev, torch.float32)[idx], size, TEMPLATE)
            crops = warp_face_cuda(out, m, size, frame_index=fi, padding_mode="border",
                                   antialias=True, mode="bicubic").clamp(0, 255)
            valid = crop_valid_mask_cuda(out.shape[-2:], m, size)
            restored, ok = self.engine.restore(model, crops)
            ok = ok & similarity_is_valid(m) & (valid.flatten(1).amax(1) > 0)
            paste, region = self._paste_masks(size, dev)
            # Statistics only: the reference is cut straight at STATS_SIZE.
            ref_crops = crops if ref_frames is None else warp_face_cuda(
                ref_frames, similarity_matrices_cuda(kps.to(dev, torch.float32)[idx],
                                                     STATS_SIZE, TEMPLATE),
                STATS_SIZE, frame_index=fi, padding_mode="border").clamp(0, 255)
            restored = lab_lock(restored, ref_crops, region * (valid > 0.99).float(),
                                match_luma_std=self.lock_luma_std, stats_size=STATS_SIZE)
            # frame px per crop px = 1 / similarity scale
            scale = (m[:, 0, 0] * m[:, 1, 1] - m[:, 0, 1] * m[:, 1, 0]).abs().sqrt()
            if self.sigma_mode == "paste_aware":
                sigma = paste_aware_sigma(1.0 / scale.clamp_min(1e-6), self.screen_sigma)
            else:
                sigma = torch.full_like(scale, self.regional.blender.sigma)
            kps_crop = transform_points_cuda(kps.to(dev, torch.float32)[idx], m)
            labels = None
            if self.parser is not None and self.parser.bisenet_path is not None:
                parse_in = crops if size == 512 else F.interpolate(
                    crops, size=(512, 512), mode="bilinear", antialias=True, align_corners=False)
                labels = self.parser.parse(parse_in)
            if self.iris is not None and track_ids is not None:
                restored = self._stabilize_eyes(crops, restored, sigma, labels, kps_crop,
                                                [track_ids[i] for i in idx.tolist()])
            fused = self.regional.fuse(crops, restored, labels=labels,
                                       kps_crop=None if labels is not None else kps_crop,
                                       sigma=sigma).fused
            weight = paste * valid * ok.float().view(-1, 1, 1, 1)
            out = paste_roi(out, fused, m, weight, fi)
            restored_flag[idx] = ok
            sigmas[idx] = sigma
        return UltraResult(out, plan, restored_flag, sigmas)

    def _stabilize_eyes(self, crops: torch.Tensor, restored: torch.Tensor, sigma: torch.Tensor,
                        labels: Any, kps_crop: torch.Tensor, ids: list[Any]) -> torch.Tensor:
        """Replace ``restored``'s eye-region high band with the stabilized one."""
        from face_engine.enhancers.semantic_fusion import L_EYE, R_EYE

        blender = self.regional.blender
        low, high = blender.split(restored, sigma)
        size = restored.shape[-1]
        if labels is not None:
            import torch.nn.functional as F

            eyes = ((labels == L_EYE) | (labels == R_EYE)).float()[:, None]
            eyes = F.interpolate(eyes, size=(size, size), mode="bilinear", align_corners=False)
        else:
            only_eyes = SemanticRegionalRestorer(RegionWeights(skin=0.0, eyes=1.0, lips=0.0,
                                                               inner_mouth=0.0))
            eyes = only_eyes.landmark_weights(kps_crop, size)
        return low + self.iris(high, eyes, kps_crop, ids)
