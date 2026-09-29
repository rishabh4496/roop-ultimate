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
                           resolve_providers, validate_tensor)

EMBEDDING_DIM = 512
_EMBEDDING_MODES = ("normed", "normed_emap", "converted_raw", "converted_norm")


def swap_spec(spec_key: str) -> dict:
    """The SWAP_MODELS entry for `spec_key` (KeyError when absent)."""
    from roop.processors.FaceSwapInsightFace import SWAP_MODELS
    return SWAP_MODELS[spec_key]


def models_dir() -> str:
    from roop.utilities import resolve_relative_path
    return resolve_relative_path('../models')


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
        batch = int(np.shape(inputs[self.image_input_name])[0])
        feed = {name: validate_tensor(inputs[name], spec, batch=batch)
                for name, spec in self.input_specs.items()}
        outs = self.session.run(None, feed)
        image = outs[0]
        mask = None
        if len(outs) > 1:
            m = np.asarray(outs[1])
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

    def align(self, frame: np.ndarray, kps: np.ndarray):
        """(crop, M) for this model's template, via the render's own align_crop."""
        from roop.face_util import align_crop
        return align_crop(frame, np.asarray(kps, dtype=np.float32).reshape(5, 2),
                          self.model_output_size, self.model_template)

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
    identity = crossface_hififace(raw embedding), L2-normalized. Batch-dynamic."""
    spec_key = "hififace"
    accepted_spec_keys = ("hififace",)


class HyperSwapSwapper(OnnxSpecSwapper):
    """HyperSwap 1a/1b/1c 256: arcface alignment, [-1,1] in/out, identity =
    normed embedding (no emap). The exports are fixed at batch 1."""
    spec_key = "hyperswap"
    accepted_spec_keys = ("hyperswap", "hyperswap_1b", "hyperswap_1c")
