"""Recognition inference engine: a verified ONNX Runtime session + preprocessing.

Binds ``roop.recognition_registry`` (which file, which normalisation) to an
``onnxruntime.InferenceSession`` tuned for the GPU it lands on, and turns an aligned
112x112 crop into a unit-length identity embedding.

Provider fallback chain (starts at the requested device and only ever goes DOWN):

    TensorRT -> CUDA -> DirectML / CoreML -> CPU

THE PART THAT MATTERS. ONNX Runtime does not raise when an execution provider fails: it
logs, drops the provider and hands back a WORKING session on the next one, and it can do
that during the FIRST INFERENCE as well as at construction (a rejected provider option,
a missing cuDNN/TensorRT DLL and a CPU-only wheel all look like this). So a bare
``try: InferenceSession(...) except: CPU`` -- the obvious reading of "graceful fallback"
-- reports success while running on the wrong device. Every tier here is therefore
built, warmed up with one real inference, and then CHECKED: the tier counts only if the
provider it was built for is still first in ``session.get_providers()`` afterwards.
Otherwise the engine records why and moves down. Where it ends up is on the object
(``active_providers``, ``fallback_log``, ``degraded``) and printed once, never implied.

Hardware profiles (compute capability read from torch when it is importable):

    Ada 8.9+ : CUDA EP cudnn EXHAUSTIVE, arena kNextPowerOfTwo, per-session arena cap sized
               from FREE VRAM; TensorRT FP16 with a fixed (1,3,112,112) profile.
    Ampere   : CUDA EP cudnn HEURISTIC, arena kSameAsRequested.
    other    : CUDA EP cudnn HEURISTIC, arena kNextPowerOfTwo.

TensorRT is not offered below 7 GiB of VRAM (the app's own rule for the 6 GB laptop tier);
a TensorRT request there starts at CUDA and the log line says so.

Not measured here: CoreML (no Mac available) -- it is bound with ORT's default options on
purpose, because a rejected option would drop the session to CPU; the post-warm-up check
above is what makes that safe either way.
"""

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from roop.degrade import swallowed as _swallowed
from roop.recognition_registry import RecognitionModelSpec, get_model_spec, resolve_model_path

_TRT = "TensorrtExecutionProvider"
_CUDA = "CUDAExecutionProvider"
_DML = "DmlExecutionProvider"
_COREML = "CoreMLExecutionProvider"
_CPU = "CPUExecutionProvider"

_DEVICES = {
    "auto": "auto", "tensorrt": "tensorrt", "trt": "tensorrt",
    "cuda": "cuda", "gpu": "cuda",
    "directml": "directml", "dml": "directml",
    "coreml": "coreml", "cpu": "cpu",
}

# TensorRT is not admitted below this (the app's 6 GB laptop tier has it stripped).
_TRT_MIN_VRAM = 7 * 1024 ** 3
# Per-session CUDA arena cap on Ada: 40% of the memory free right now, within these bounds.
_ARENA_FRACTION, _ARENA_MIN, _ARENA_MAX = 0.4, 1 << 30, 4 << 30

_ALGOS = ("DEFAULT", "HEURISTIC", "EXHAUSTIVE")


def _physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def _gpu_info(gpu_id: int) -> Optional[Dict[str, Any]]:
    """Name / compute capability / VRAM of one CUDA device, or None without torch+CUDA."""
    try:
        import torch
        if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
            return None
        major, minor = torch.cuda.get_device_capability(gpu_id)
        free, total = torch.cuda.mem_get_info(gpu_id)
        if (major, minor) >= (8, 9):
            arch = "ada"
        elif major == 8:
            arch = "ampere"
        else:
            arch = "generic"
        return {"name": torch.cuda.get_device_name(gpu_id), "capability": (major, minor),
                "total_bytes": int(total), "free_bytes": int(free), "arch": arch}
    except Exception as exc:
        _swallowed("roop/recognition_engine.py:_gpu_info", exc, "no GPU tuning; generic profile")
        return None


class RecognitionInferenceEngine:
    """One recogniser: resolved file, verified session, preprocessing, embedding."""

    def __init__(self, model_name: str, models_dir: str, device: str = "auto",
                 gpu_id: int = 0, *, trt_fp16: bool = True,
                 cudnn_algo: Optional[str] = None, strict: bool = False) -> None:
        """
        device     -- 'auto' | 'tensorrt' | 'cuda' | 'directml' | 'coreml' | 'cpu'. 'auto'
                      picks the best provider this machine offers (TensorRT only on Ada).
        trt_fp16   -- TensorRT FP16. Embeddings differ slightly from FP32; do not compare
                      vectors made with different settings against each other.
        cudnn_algo -- override the per-architecture default ('DEFAULT'|'HEURISTIC'|'EXHAUSTIVE').
        strict     -- raise instead of degrading to CPU when a GPU provider was requested.
        """
        key = _DEVICES.get(str(device).strip().lower())
        if key is None:
            raise ValueError(f"Unknown device '{device}'. Valid options: {sorted(set(_DEVICES))}")
        if cudnn_algo is not None and str(cudnn_algo).upper() not in _ALGOS:
            raise ValueError(f"cudnn_algo must be one of {_ALGOS}, got '{cudnn_algo}'")

        self.model_name = model_name
        self.models_dir = models_dir
        self.device = key
        self.gpu_id = int(gpu_id)
        self.trt_fp16 = bool(trt_fp16)
        self.cudnn_algo = str(cudnn_algo).upper() if cudnn_algo else None
        self.strict = bool(strict)

        self.spec: RecognitionModelSpec = get_model_spec(model_name)
        self.model_path: str = resolve_model_path(model_name, models_dir)
        self.gpu: Optional[Dict[str, Any]] = None

        self.fallback_log: List[str] = []
        self.active_providers: List[str] = []
        self.degraded = False
        self.invalid_outputs = 0
        self._lock = threading.Lock()

        self._mean = np.asarray(self.spec.mean, dtype=np.float32)
        self._std = np.asarray(self.spec.std, dtype=np.float32)

        self.session: ort.InferenceSession = self._build_verified_session()
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name

    # ------------------------------------------------------------------ providers

    def _note(self, message: str) -> None:
        self.fallback_log.append(message)
        print(f"[Recognition] {self.model_name}: {message}")

    def _cuda_options(self) -> Dict[str, Any]:
        arch = self.gpu["arch"] if self.gpu else "generic"
        algo = self.cudnn_algo or ("EXHAUSTIVE" if arch == "ada" else "HEURISTIC")
        options: Dict[str, Any] = {
            "device_id": self.gpu_id,
            "cudnn_conv_algo_search": algo,
            "do_copy_in_default_stream": True,
            "arena_extend_strategy": "kSameAsRequested" if arch == "ampere" else "kNextPowerOfTwo",
        }
        if arch == "ada" and self.gpu:
            # "Dynamic" cap: a recogniser session needs a fraction of a GB; the cap stops
            # the power-of-two arena from claiming the card out from under the swapper.
            options["gpu_mem_limit"] = int(min(_ARENA_MAX, max(
                _ARENA_MIN, self.gpu["free_bytes"] * _ARENA_FRACTION)))
        return options

    def _fixed_profile(self) -> Dict[str, str]:
        """(1,3,H,W) min/opt/max for every input whose graph leaves an axis free.

        A fully static graph (SFace) needs no profile and gets none -- a profile there
        would only change the engine cache identity.
        """
        try:
            from roop.trt_shape_profile import graph_inputs
            inputs = graph_inputs(self.model_path)
        except Exception as exc:
            _swallowed("roop/recognition_engine.py:_fixed_profile", exc, "no TensorRT profile")
            return {}
        height, width = self.spec.input_size
        shapes = []
        for item in inputs:
            if not item.dims or ":" in item.name:       # ORT splits "name:shape" on the first ':'
                return {}
            if item.dynamic:
                shapes.append(f"{item.name}:1x3x{height}x{width}")
        if not shapes:
            return {}
        joined = ",".join(shapes)
        return {"trt_profile_min_shapes": joined, "trt_profile_opt_shapes": joined,
                "trt_profile_max_shapes": joined}

    def _trt_options(self) -> Dict[str, Any]:
        cap = "%d%d" % self.gpu["capability"] if self.gpu else "xx"
        label = "recognition_%s_%s_sm%s" % (self.model_name, "fp16" if self.trt_fp16 else "fp32", cap)
        profile = self._fixed_profile()
        if profile:
            label += "_b1"
        cache = os.path.join(self.models_dir, "trt_cache", label)
        os.makedirs(cache, exist_ok=True)
        # Python bools, never "1"/"0": ORT 1.23 rejects those for TRT bool options and a
        # rejected option silently drops BOTH TensorRT and CUDA to a CPU-only session.
        options: Dict[str, Any] = {
            "device_id": self.gpu_id,
            "trt_fp16_enable": self.trt_fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": cache,
        }
        options.update(profile)
        return options

    def _tiers(self) -> List[Tuple[str, List[Any], str]]:
        """(label, providers, provider that must still lead after warm-up), best first."""
        available = set(ort.get_available_providers())
        on = self.device
        tiers: List[Tuple[str, List[Any], str]] = []

        wants_cuda_family = on in ("auto", "tensorrt", "cuda")
        if wants_cuda_family and _CUDA in available:
            from roop.gpu_preflight import register_gpu_runtime_dirs
            register_gpu_runtime_dirs()                  # before any GPU session exists
            try:
                import torch  # noqa: F401  (capability probe; absent torch only loses tuning)
                torch_present = True
            except ImportError:
                torch_present = False
            self.gpu = _gpu_info(self.gpu_id) if torch_present else None
            if torch_present and self.gpu is None:
                self._note(f"torch sees no CUDA device {self.gpu_id}; skipping CUDA tiers")
                wants_cuda_family = False

        if wants_cuda_family and _CUDA in available:
            arch = self.gpu["arch"] if self.gpu else "generic"
            trt_asked = on == "tensorrt" or (on == "auto" and arch == "ada")
            if trt_asked and _TRT in available:
                if self.gpu and self.gpu["total_bytes"] < _TRT_MIN_VRAM:
                    self._note("TensorRT not offered below 7 GiB VRAM; starting at CUDA")
                else:
                    tiers.append(("TensorRT", [(_TRT, self._trt_options()),
                                               (_CUDA, self._cuda_options()), _CPU], _TRT))
            elif on == "tensorrt":
                self._note("TensorrtExecutionProvider not available in this onnxruntime; starting at CUDA")
            tiers.append(("CUDA", [(_CUDA, self._cuda_options()), _CPU], _CUDA))
        elif wants_cuda_family:
            self._note("CUDAExecutionProvider not available in this onnxruntime")

        if on in ("auto", "tensorrt", "cuda", "directml") and _DML in available:
            tiers.append(("DirectML", [(_DML, {"device_id": self.gpu_id}), _CPU], _DML))
        if on in ("auto", "tensorrt", "cuda", "coreml") and _COREML in available:
            tiers.append(("CoreML", [_COREML, _CPU], _COREML))
        tiers.append(("CPU", [_CPU], _CPU))
        return tiers

    def _session_options(self, label: str) -> ort.SessionOptions:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if label == "CPU":
            # Thread counts are SESSION options, not CPU-provider options.
            so.intra_op_num_threads = _physical_cores()
            so.inter_op_num_threads = 1
        else:
            # The few nodes left on the CPU provider must not add to a pipeline that
            # already runs a thread per worker.
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
        # DirectML requires the memory pattern optimisation off.
        so.enable_mem_pattern = label != "DirectML"
        return so

    def _warmup_blob(self) -> np.ndarray:
        height, width = self.spec.input_size
        return np.zeros((1, 3, height, width), dtype=np.float32)

    def _build_verified_session(self) -> ort.InferenceSession:
        tiers = self._tiers()
        wanted_gpu = self.device in ("tensorrt", "cuda", "directml", "coreml")
        for position, (label, providers, lead) in enumerate(tiers):
            last = position == len(tiers) - 1
            try:
                session = ort.InferenceSession(self.model_path, sess_options=self._session_options(label),
                                               providers=providers)
                name = session.get_inputs()[0].name
                out = session.run(None, {name: self._warmup_blob()})[0]
            except Exception as exc:
                if last:
                    raise
                self._note(f"{label} failed ({type(exc).__name__}: {str(exc).splitlines()[0][:200]}); falling back")
                continue
            active = list(session.get_providers())     # AFTER the first inference: ORT drops EPs there
            if not last and (not active or active[0] != lead):
                self._note(f"{label} requested but active providers are {active}; falling back")
                continue
            if int(np.asarray(out).size) != self.spec.output_dim:
                raise ValueError(
                    f"{self.model_name}: model returned {np.asarray(out).size} values but the "
                    f"registry says output_dim={self.spec.output_dim}")
            self.active_providers = active
            self.degraded = wanted_gpu and (not active or active[0] == _CPU)
            if self.degraded:
                self._note(f"running on CPU only (requested device='{self.device}'); "
                           f"reasons: {self.fallback_log or 'none recorded'}")
                if self.strict:
                    raise RuntimeError(
                        f"{self.model_name}: device='{self.device}' requested but the session "
                        f"is CPU-only. {self.fallback_log}")
            else:
                print(f"[Recognition] {self.model_name}: active = {active}")
            return session
        raise RuntimeError("unreachable: the CPU tier always terminates the chain")  # pragma: no cover

    # ------------------------------------------------------------- preprocessing

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Aligned BGR crop (H, W, 3) -> float32 C-contiguous NCHW (1, 3, H, W) tensor."""
        if face_crop is None or face_crop.ndim != 3 or face_crop.shape[2] != 3:
            raise ValueError("face_crop must be an (H, W, 3) BGR image, got %s" %
                             (None if face_crop is None else (face_crop.shape,)))
        target_h, target_w = self.spec.input_size
        h, w = face_crop.shape[:2]
        if (h, w) != (target_h, target_w):
            shrinking = h * w > target_h * target_w
            face_crop = cv2.resize(face_crop, (target_w, target_h),
                                   interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
        if self.spec.color_space == "RGB":
            face_crop = face_crop[:, :, ::-1]
        data = (face_crop.astype(np.float32) - self._mean) / self._std
        return np.ascontiguousarray(data.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)

    # ------------------------------------------------------------------ embedding

    def compute_embedding(self, face_crop: np.ndarray) -> Tuple[np.ndarray, float]:
        """(unit-length embedding, quality score) for one aligned face crop.

        quality_score is the raw output norm for models whose norm encodes quality
        (spec.extract_quality_score, i.e. MagFace) and 1.0 otherwise. A non-finite or
        zero-norm output returns (zeros, 0.0) and increments ``invalid_outputs``: a
        zero vector must never pass for "a face that matched nobody".
        """
        blob = self.preprocess(face_crop)
        raw = self.session.run([self.output_name], {self.input_name: blob})[0]
        z = np.asarray(raw, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(z))
        if not np.isfinite(norm) or norm <= 1e-12:
            with self._lock:
                self.invalid_outputs += 1
            return np.zeros_like(z), 0.0
        return z / norm, (norm if self.spec.extract_quality_score else 1.0)

    def describe(self) -> Dict[str, Any]:
        """JSON-safe summary of what is actually running."""
        return {
            "model": self.model_name, "path": self.model_path, "requested_device": self.device,
            "active_providers": list(self.active_providers), "degraded": self.degraded,
            "gpu": None if self.gpu is None else {k: self.gpu[k] for k in ("name", "capability", "arch")},
            "fallback_log": list(self.fallback_log),
        }
