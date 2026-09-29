"""Concrete `BaseFaceSwapper`s over the SWAP_MODELS spec table.

The per-model numbers (file, resolution, mean/std, output range, alignment
template, how the identity vector is prepared) are NOT restated here: they are
read from `roop.processors.FaceSwapInsightFace.SWAP_MODELS`, the table the
production processor loads from. Two copies of those numbers is how realswap's
secondary net once got a differently scaled image (see procmgr_tiling.to_blob).

Only embedding-source models fit the contract, because `pre_process` takes an
embedding: `normed` (hyperswap), `normed_emap` (inswapper family) and
`converted_*` (hififace, ghost, simswap). Image-source models (blendswap,
uniface) and cscs need a source IMAGE and are refused at construction.

Identity prep matches FaceSwapInsightFace._compute_latent exactly; the test
suite checks it against that method on the same vector.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .swapper_base import (BaseFaceSwapper, TensorSpec, inspect_onnx_topology,
                           resolve_providers, validate_tensor, validate_tensor_torch)

EMBEDDING_DIM = 512
_EMBEDDING_MODES = ("normed", "normed_emap", "converted_raw", "converted_norm")


def swap_spec(spec_key: str) -> dict:
    """The SWAP_MODELS entry for `spec_key` (KeyError when absent)."""
    from roop.processors.FaceSwapInsightFace import SWAP_MODELS
    return SWAP_MODELS[spec_key]


def models_dir() -> str:
    from roop.utilities import resolve_relative_path
    return resolve_relative_path('../models')


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


# ── torch warps, for frames already in VRAM ─────────────────────────────────────

def warp_affine_torch(img, M, out_hw, border: str = "replicate"):
    """cv2.warpAffine(img, M, (w, h)) for an (H,W,C) or (H,W) tensor.

    M maps SOURCE pixels to OUTPUT pixels, as in cv2. Bilinear; `border`
    'replicate' (cv2.BORDER_REPLICATE) or 'zeros' (BORDER_CONSTANT 0). The
    result keeps the input's dtype (uint8 is rounded).
    """
    import torch
    import torch.nn.functional as F
    h_out, w_out = out_hw
    squeeze = img.dim() == 2
    x = img[..., None] if squeeze else img
    H, W = x.shape[:2]
    inv = np.linalg.inv(np.vstack([np.asarray(M, np.float64).reshape(2, 3), [0, 0, 1]]))
    # output pixel (u, v) -> source pixel inv @ (u, v, 1) -> grid_sample's [-1, 1]
    to_norm_src = np.array([[2.0 / W, 0, 1.0 / W - 1], [0, 2.0 / H, 1.0 / H - 1], [0, 0, 1]])
    from_norm_out = np.array([[w_out / 2.0, 0, w_out / 2.0 - 0.5],
                              [0, h_out / 2.0, h_out / 2.0 - 0.5], [0, 0, 1]])
    theta = (to_norm_src @ inv @ from_norm_out)[:2]
    # The grid by elementwise fp32 math, NOT F.affine_grid: that is a matmul,
    # and roop/core.py enables TF32 matmul globally -- measured, the crop warp
    # went from mean 0.11 to 0.75 levels (max 3) off cv2 under it.
    ys = (torch.arange(h_out, device=x.device, dtype=torch.float32) * 2 + 1) / h_out - 1
    xs = (torch.arange(w_out, device=x.device, dtype=torch.float32) * 2 + 1) / w_out - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    t = [[float(v) for v in row] for row in theta]
    grid = torch.stack([t[0][0] * gx + t[0][1] * gy + t[0][2],
                        t[1][0] * gx + t[1][1] * gy + t[1][2]], dim=-1)[None]
    src = x.permute(2, 0, 1)[None].to(torch.float32)
    out = F.grid_sample(src, grid, mode="bilinear",
                        padding_mode="border" if border == "replicate" else "zeros",
                        align_corners=False)[0].permute(1, 2, 0)
    if img.dtype == torch.uint8:
        out = out.round().clamp(0, 255).to(torch.uint8)
    else:
        out = out.to(img.dtype)
    return out[..., 0] if squeeze else out


def to_crop_torch(swap_crop, denormalize: bool, device):
    """Model output [1,3,S,S] or [3,S,S] (array or tensor) -> uint8 BGR (S,S,3) tensor."""
    import torch
    t = torch.as_tensor(swap_crop, device=device, dtype=torch.float32)
    if t.dim() == 4:
        t = t[0]
    t = t.permute(1, 2, 0)
    if denormalize:
        t = (t + 1.0) / 2.0
    return (t * 255.0).round().clamp(0, 255).flip(-1).to(torch.uint8)


def paste_back_torch(crop, M, target_frame, mask):
    """BaseFaceSwapper.paste_back on tensors (the frame stays on its device)."""
    import torch
    from .swapper_base import paste_roi
    out = target_frame.clone()
    roi = paste_roi(M, int(crop.shape[0]), target_frame.shape[:2])
    if roi is None:
        return out
    x0, y0, x1, y1 = roi
    inv = np.linalg.inv(np.vstack([np.asarray(M, np.float64).reshape(2, 3), [0, 0, 1]]))[:2]
    inv[:, 2] -= (x0, y0)            # the footprint only, as BaseFaceSwapper.paste_back
    hw = (y1 - y0, x1 - x0)
    pasted = warp_affine_torch(crop.to(torch.float32), inv, hw, border="replicate")
    alpha = warp_affine_torch(torch.as_tensor(mask, device=target_frame.device,
                                              dtype=torch.float32), inv, hw, border="zeros")[..., None]
    region = target_frame[y0:y1, x0:x1].to(torch.float32)
    out[y0:y1, x0:x1] = (alpha * pasted + (1.0 - alpha) * region).round().clamp(0, 255).to(torch.uint8)
    return out


class OnnxSpecSwapper(BaseFaceSwapper):
    """A single-network ONNX swapper driven by one SWAP_MODELS entry."""

    spec_key: str = ""
    # Subclasses narrow this so the registry cannot bind, say, a ghost spec to
    # the HiFiFace class.
    accepted_spec_keys: Sequence[str] = ()

    def __init__(self, spec_key: Optional[str] = None) -> None:
        super().__init__()
        key = spec_key or self.spec_key
        if self.accepted_spec_keys and key not in self.accepted_spec_keys:
            raise ValueError(f"{type(self).__name__} does not serve spec {key!r} "
                             f"(accepts {list(self.accepted_spec_keys)})")
        spec = swap_spec(key)
        mode = spec.get("embedding", "normed_emap")
        if mode not in _EMBEDDING_MODES:
            raise ValueError(f"spec {key!r} takes a source {mode!r}, not an embedding; "
                             "it cannot implement pre_process(target_crop, source_embedding)")
        if spec.get("secondary"):
            raise ValueError(f"spec {key!r} is a composite of two nets; use FaceSwapInsightFace")
        self.spec_key = key
        self.spec = spec
        self.embedding_mode = mode
        self.model_name = key
        self.model_output_size = int(spec["output_size"])
        self.model_mean = tuple(spec["mean"])
        self.model_standard_deviation = tuple(spec["standard_deviation"])
        self.model_denormalize = bool(spec["denormalize"])
        self.model_template = spec.get("template", "arcface")
        self.image_input_name = "target"
        self.embed_input_name = "source"
        self.emap: Optional[np.ndarray] = None
        self.converter = None
        self.converter_input = "input"
        self.providers: list = []
        # TemporalMatrixStabilizer or None; used by `align` (HiFiFace sets it).
        self.stabilizer = None
        # Converter output per source identity. The crossface MLP is per face,
        # not per frame; production caches it on the Face object, this caches it
        # on the embedding's bytes because the contract hands us only the vector.
        self._latent_cache: "OrderedDict[bytes, np.ndarray]" = OrderedDict()
        self._latent_lock = threading.Lock()

    # -- paths ---------------------------------------------------------------

    def default_model_path(self) -> str:
        return os.path.join(models_dir(), self.spec["file"])

    def default_converter_path(self) -> Optional[str]:
        name = self.spec.get("converter_file")
        return os.path.join(models_dir(), name) if name else None

    # -- 1. session ----------------------------------------------------------

    def initialize_session(self, model_path: str, execution_provider: str, **kwargs) -> None:
        """Load `model_path` on `execution_provider` after checking its graph.

        kwargs:
            session_options  an onnxruntime.SessionOptions (default: the app's)
            converter_path   crossface converter for converted_* models
                             (default: models/<spec converter_file>)
            download         True fetches missing files from the spec URLs
            precision        'fp16' (default) | 'fp32' -- TensorRT only; see
                             inference_engine for what it costs and saves
            strict           raise if the session is not on the requested
                             provider (default: print it and carry on)

        'tensorrt' / 'cuda' build an OptimizedInferenceSession (provider chain,
        fixed batch-1 TRT profile, persistent device IO binding); `infer` then
        also takes torch CUDA tensors and runs zero-copy. Anything else ('cpu',
        a full provider list) is a plain onnxruntime session, as in Stage 1.
        """
        import onnxruntime

        model_path = model_path or self.default_model_path()
        converter_path = kwargs.get("converter_path") or self.default_converter_path()
        if kwargs.get("download"):
            from roop.utilities import conditional_download
            urls = [self.spec["url"]] + ([self.spec["converter_url"]]
                                         if self.spec.get("converter_url") else [])
            conditional_download(os.path.dirname(model_path), urls)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"{self.spec_key}: model not found at {model_path}")

        inputs, outputs = inspect_onnx_topology(model_path)
        self._check_topology(inputs, outputs)

        session_options = kwargs.get("session_options")
        if session_options is None:
            from roop.utilities import get_onnx_session_options
            session_options = get_onnx_session_options()
        self.engine = None
        name = execution_provider.lower() if isinstance(execution_provider, str) else None
        name = {"tensorrtexecutionprovider": "tensorrt", "trt": "tensorrt",
                "cudaexecutionprovider": "cuda"}.get(name, name)
        if name in ("tensorrt", "cuda"):
            from .inference_engine import OptimizedInferenceSession
            self.engine = OptimizedInferenceSession(
                model_path, name, kwargs.get("precision", "fp16"),
                device_id=int(kwargs.get("device_id", 0)), strict=bool(kwargs.get("strict", False)),
                session_options=session_options)
            self.session = self.engine.session
            self.providers = self.engine.providers
        else:
            self.providers = resolve_providers(execution_provider)
            self.session = onnxruntime.InferenceSession(model_path, session_options,
                                                        providers=self.providers)

        if self.embedding_mode == "normed_emap":
            import onnx
            from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace
            self.emap = FaceSwapInsightFace._find_emap(onnx.load(model_path).graph)
        if self.embedding_mode.startswith("converted_"):
            if not converter_path or not os.path.isfile(converter_path):
                raise FileNotFoundError(f"{self.spec_key}: crossface converter not found "
                                        f"at {converter_path}")
            c_in, c_out = inspect_onnx_topology(converter_path)
            if (len(c_in) != 1 or c_in[0].rank != 2 or c_in[0].shape[1] != EMBEDDING_DIM
                    or not c_out or c_out[0].rank != 2 or c_out[0].shape[1] != EMBEDDING_DIM):
                raise ValueError(f"{self.spec_key}: converter must map [?,512] -> [?,512], "
                                 f"got {[s.shape for s in c_in]} -> {[s.shape for s in c_out]}")
            # CPU on purpose, as in production: once per identity, not per frame.
            self.converter = onnxruntime.InferenceSession(
                converter_path, session_options, providers=["CPUExecutionProvider"])
            self.converter_input = c_in[0].name
        with self._latent_lock:
            self._latent_cache.clear()

    def _check_topology(self, inputs: Sequence[TensorSpec], outputs: Sequence[TensorSpec]) -> None:
        """Resolve target/source inputs by rank and insist on the static geometry.

        Rank 4 is the image, rank 2 the identity (as FaceSwapInsightFace
        resolves them -- hyperswap lists `source` first). Channel, spatial and
        embedding dims must be STATIC and equal to the spec; only batch may be
        symbolic (hififace declares 'batch_size', hyperswap a fixed 1).
        """
        size = self.model_output_size
        images = [s for s in inputs if s.rank == 4]
        vectors = [s for s in inputs if s.rank == 2]
        if len(images) != 1 or len(vectors) != 1 or len(inputs) != 2:
            raise ValueError(f"{self.spec_key}: expected one [B,3,{size},{size}] image and one "
                             f"[B,{EMBEDDING_DIM}] input, got {[(s.name, s.shape) for s in inputs]}")
        target, source = images[0], vectors[0]
        for spec, axes, want in ((target, (1, 2, 3), (3, size, size)),
                                 (source, (1,), (EMBEDDING_DIM,))):
            if np.dtype(spec.dtype) != np.float32:
                raise ValueError(f"{self.spec_key}: input {spec.name} is {np.dtype(spec.dtype)}, "
                                 "expected float32")
            for axis, value in zip(axes, want):
                if not spec.is_static(axis) or spec.shape[axis] != value:
                    raise ValueError(f"{self.spec_key}: input {spec.name} axis {axis} is "
                                     f"{spec.shape[axis]!r}, expected static {value} ({spec.shape})")
        if not outputs or outputs[0].rank != 4 or tuple(outputs[0].shape[1:]) != (3, size, size):
            raise ValueError(f"{self.spec_key}: output[0] must be [B,3,{size},{size}], "
                             f"got {[(s.name, s.shape) for s in outputs]}")
        self.image_input_name = target.name
        self.embed_input_name = source.name
        self.input_specs = {target.name: target, source.name: source}
        self.output_specs = tuple(outputs)

    # -- 2. pre_process --------------------------------------------------------

    def pre_process(self, target_crop: np.ndarray, source_embedding: np.ndarray) -> Dict[str, Any]:
        """(S,S,3) BGR uint8 aligned crop + RAW 512-d ArcFace embedding -> feed.

        Pass `face.embedding`, not `face.normed_embedding`: the crossface
        converters are fed the raw vector, and the normed modes normalize here.
        """
        self._require_session()
        size = self.model_output_size
        crop = np.asarray(target_crop)
        if crop.dtype != np.uint8 or crop.shape != (size, size, 3):
            raise ValueError(f"{self.spec_key}: target_crop must be uint8 ({size},{size},3) "
                             f"aligned on '{self.model_template}', got {crop.dtype} {crop.shape}")
        from roop.procmgr_tiling import to_blob
        blob = to_blob(crop, self.model_mean, self.model_standard_deviation)
        latent = self.prepare_latent(source_embedding)
        return {
            self.image_input_name: validate_tensor(
                blob, self.input_specs[self.image_input_name], batch=1),
            self.embed_input_name: validate_tensor(
                latent, self.input_specs[self.embed_input_name], batch=1),
        }

    def prepare_latent(self, source_embedding: np.ndarray) -> np.ndarray:
        """Raw embedding -> the (1,512) float32 identity this model consumes."""
        emb = np.asarray(source_embedding)
        if not np.issubdtype(emb.dtype, np.floating) or emb.size != EMBEDDING_DIM:
            raise ValueError(f"{self.spec_key}: source_embedding must be {EMBEDDING_DIM} floats, "
                             f"got {emb.dtype} {emb.shape}")
        emb = emb.reshape(1, EMBEDDING_DIM).astype(np.float32)
        if not np.isfinite(emb).all():
            raise ValueError(f"{self.spec_key}: source_embedding is not finite")
        mode = self.embedding_mode
        if mode.startswith("converted_"):
            key = emb.tobytes()
            with self._latent_lock:
                cached = self._latent_cache.get(key)
            if cached is not None:
                return cached
            converted = self.converter.run(None, {self.converter_input: emb})[0].ravel()
            if mode == "converted_norm":
                converted = converted / np.linalg.norm(converted)
            latent = np.ascontiguousarray(converted.reshape(1, -1).astype(np.float32))
            with self._latent_lock:
                self._latent_cache[key] = latent
                while len(self._latent_cache) > 64:
                    self._latent_cache.popitem(last=False)
            return latent
        # A float32 norm, as insightface's normed_embedding computes it, so the
        # hyperswap latent is bit-identical to the production one.
        norm = np.linalg.norm(emb)
        if float(norm) <= 1e-12:
            raise ValueError(f"{self.spec_key}: source_embedding has zero norm")
        latent = emb / norm
        if mode == "normed_emap" and self.emap is not None:
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)
        return np.ascontiguousarray(latent.astype(np.float32))

    # -- 3. infer ------------------------------------------------------------

    def infer(self, inputs: Dict[str, Any]) -> np.ndarray:
        self._require_session()
        missing = set(self.input_specs) - set(inputs)
        if missing:
            raise ValueError(f"{self.spec_key}: feed is missing {sorted(missing)}")
        batch = int(inputs[self.image_input_name].shape[0])
        if _is_torch(inputs[self.image_input_name]):
            # Zero-copy path: CUDA tensors in, the engine's persistent CUDA
            # output buffers out (cloned: the next call overwrites them).
            if getattr(self, "engine", None) is None or not self.engine.on_device:
                raise RuntimeError(f"{self.spec_key}: torch inputs need a 'cuda' or 'tensorrt' session")
            feed = {name: validate_tensor_torch(inputs[name], spec, batch=batch)
                    for name, spec in self.input_specs.items()}
            named = self.engine.run_binding(feed, return_all=True, clone=True)
            outs = [named[n] for n in self.engine.output_names]
        else:
            feed = {name: validate_tensor(inputs[name], spec, batch=batch)
                    for name, spec in self.input_specs.items()}
            outs = self.session.run(None, feed)
        image = outs[0]
        mask = None
        if len(outs) > 1:
            m = outs[1]
            if m.ndim == 4 and m.shape[1] == 1 and m.shape[-2:] == image.shape[-2:]:
                mask = m
        self._mask_tls.mask = mask
        return image

    # -- 4. post_process -------------------------------------------------------

    def post_process(self, swap_crop: np.ndarray, affine_matrix: np.ndarray,
                     target_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        crop = self.to_crop(swap_crop)
        if mask is not None:
            mask = np.asarray(mask, dtype=np.float32)
            if mask.ndim == 4:
                mask = mask[0]
        return self.paste_back(crop, affine_matrix, target_frame, mask)

    # -- helpers ---------------------------------------------------------------

    def align(self, frame, kps, track_id=None, frame_index=None):
        """(crop, M) for this model's template.

        Without a stabilizer (or without a `track_id`) this is the render's own
        align_crop, unchanged. With one, M is smoothed FIRST and the crop is
        warped with the smoothed M, so post_process pastes with the same matrix
        the crop was taken with (see temporal_stabilizer's module note).
        A CUDA torch frame is warped in torch and returns a tensor crop.
        """
        from roop.face_util import align_crop, estimate_norm
        pts = kps.detach().cpu().numpy() if _is_torch(kps) else kps
        pts = np.asarray(pts, dtype=np.float32).reshape(5, 2)
        size = self.model_output_size
        stabilizer = getattr(self, "stabilizer", None)
        if not _is_torch(frame) and (stabilizer is None or track_id is None):
            return align_crop(frame, pts, size, self.model_template)
        M = estimate_norm(pts, size, self.model_template).astype(np.float32)
        if stabilizer is not None and track_id is not None:
            M = stabilizer.update(M, track_id=track_id, frame_index=frame_index)
        if _is_torch(frame):
            return warp_affine_torch(frame, M, (size, size), border="replicate"), M
        import cv2
        return cv2.warpAffine(frame, M, (size, size), borderMode=cv2.BORDER_REPLICATE), M

    def _require_session(self) -> None:
        if self.session is None:
            raise RuntimeError(f"{self.spec_key}: initialize_session() has not been called")

    def release(self) -> None:
        super().release()
        self.converter = None
        self.emap = None
        with self._latent_lock:
            self._latent_cache.clear()


class HiFiFaceSwapper(OnnxSpecSwapper):
    """HifiFace (unofficial) 256: mtcnn_512 alignment, [-1,1] in/out,
    identity = crossface_hififace(raw embedding), L2-normalized. Batch-dynamic.

    post_process is the Stage 2 compositor: composite mask (hull x box x
    occluder x parser x the model's own mask, dilated + normalized blur) ->
    Reinhard LAB colour match inside that mask -> inverse-affine paste. The
    stabilizer, when set, acts in `align` (it must: see temporal_stabilizer).
    """
    spec_key = "hififace"
    accepted_spec_keys = ("hififace",)

    def __init__(self, spec_key: Optional[str] = None, *, blur_amount: float = 0.3,
                 padding=(0, 0, 0, 0), color_match: bool = True,
                 parser_session=None, occluder_session=None, stabilizer=None) -> None:
        super().__init__(spec_key)
        self.blur_amount = float(blur_amount)
        self.padding = tuple(padding)
        self.color_match = bool(color_match)
        self.parser_session = parser_session
        self.occluder_session = occluder_session
        self.stabilizer = stabilizer
        self.last_composite_mask = None     # the last call's mask, for inspection

    def initialize_session(self, model_path: str, execution_provider: str, **kwargs) -> None:
        """As the base, plus optional mask models and the stabilizer:

            occluder_path   XSeg / face_occluder ONNX (HIGH on visible face)
            parser_path     BiSeNet 19-class ONNX (e.g. models/resnet18.onnx)
            mask_provider   provider for those two (default: execution_provider)
            stabilize       True, or a smoothing factor, attaches a
                            TemporalMatrixStabilizer sized to this crop
        """
        super().initialize_session(model_path, execution_provider, **kwargs)
        import onnxruntime
        providers = resolve_providers(kwargs.get("mask_provider", execution_provider))
        for attr, key in (("occluder_session", "occluder_path"), ("parser_session", "parser_path")):
            path = kwargs.get(key)
            if path:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"{key}: {path}")
                setattr(self, attr, onnxruntime.InferenceSession(path, providers=providers))
        stabilize = kwargs.get("stabilize")
        if stabilize:
            from .temporal_stabilizer import TemporalMatrixStabilizer
            factor = 0.9 if stabilize is True else float(stabilize)
            self.stabilizer = TemporalMatrixStabilizer(factor, crop_size=self.model_output_size)

    def post_process(self, swap_crop, affine_matrix, target_frame, mask,
                     *, landmarks=None, color_match: Optional[bool] = None):
        """Swapped crop -> the frame, through mask fusion and colour matching.

        mask       the swap model's own face mask (e.g. `last_mask`), or None
        landmarks  the target's landmarks in FRAME coordinates (68 / 106 / any
                   N x 2); mapped into the crop by `affine_matrix`. None
                   leaves the hull out (box x models only).
        """
        from .mask_engine import generate_composite_mask
        from .color_matcher import match_color_reinhard
        do_color = self.color_match if color_match is None else bool(color_match)
        size = self.model_output_size
        torch_mode = _is_torch(target_frame)
        M = affine_matrix.detach().cpu().numpy() if _is_torch(affine_matrix) else affine_matrix
        M = np.asarray(M, dtype=np.float32).reshape(2, 3)

        if torch_mode:
            crop = to_crop_torch(swap_crop, self.model_denormalize, target_frame.device)
            target_crop = warp_affine_torch(target_frame, M, (size, size), border="replicate")
        else:
            import cv2
            crop = self.to_crop(swap_crop)
            target_crop = cv2.warpAffine(np.asarray(target_frame), M, (size, size),
                                         borderMode=cv2.BORDER_REPLICATE)
        if tuple(crop.shape[:2]) != (size, size):
            raise ValueError(f"swap crop {tuple(crop.shape)} is not {size}x{size}")

        lm = None
        if landmarks is not None:
            pts = landmarks.detach().cpu().numpy() if _is_torch(landmarks) else landmarks
            pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
            lm = pts @ M[:, :2].T + M[:, 2]
        model_mask = None
        if mask is not None:
            model_mask = mask.squeeze() if _is_torch(mask) else np.asarray(mask, np.float32).squeeze()

        composite = generate_composite_mask(
            target_crop, lm, parser_session=self.parser_session,
            occluder_session=self.occluder_session, blur_amount=self.blur_amount,
            padding=self.padding, model_mask=model_mask)
        self.last_composite_mask = composite
        if do_color:
            crop = match_color_reinhard(crop, target_crop, composite)
        if torch_mode:
            return paste_back_torch(crop, M, target_frame, composite)
        return self.paste_back(crop, M, target_frame, composite)


class HyperSwapSwapper(OnnxSpecSwapper):
    """HyperSwap 1a/1b/1c 256: arcface alignment, [-1,1] in/out, identity =
    normed embedding (no emap). The exports are fixed at batch 1."""
    spec_key = "hyperswap"
    accepted_spec_keys = ("hyperswap", "hyperswap_1b", "hyperswap_1c")
