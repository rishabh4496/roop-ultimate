"""TensorRT-backed ONNX Runtime session wrapper.

A thin, explicit way to open one ONNX model on the TensorRT execution
provider with an on-disk engine + timing cache, falling back to CUDA and
then CPU in that order.  Compare :mod:`roop.optimized_trt_engine`, which is
the *strict* variant (TensorRT only, persistent I/O-binding buffers) used by
the high-throughput renderer; this wrapper is the permissive one for callers
that would rather run slower than fail.

Two things this module does that the bare ``ort.InferenceSession`` call does
not, both learned the hard way (see ``roop/predictor.py``):

* it registers the packaged TensorRT / CUDA DLL directories first.  Without
  that, ORT on Windows drops the TensorRT provider *silently* and returns a
  working CPU session -- a 4 ms model reports as 210 ms and nothing raises.
* it reads back ``get_providers()`` after construction and logs a warning if
  TensorRT is not the provider that actually bound.  That list is the only
  tell.
"""

import gc
import logging
import os
from typing import Any, Dict, List, Optional

import onnxruntime as ort

logger = logging.getLogger("roop.trt")

# Name ORT reports for the provider we asked for, when it really bound.
_TRT_EP = "TensorrtExecutionProvider"


def _prepare_runtime() -> None:
    """Put the packaged TensorRT/CUDA DLLs on the loader path (idempotent)."""
    try:
        from roop.trt_session_builder import prepare_tensorrt_runtime
    except Exception:  # pragma: no cover - builder is optional at import time
        return
    try:
        prepare_tensorrt_runtime()
    except Exception as exc:  # pragma: no cover - never fatal, only slower
        logger.debug("TensorRT runtime prep skipped: %s", exc)


class TensorRTInferenceSession:
    """Manages an ONNX Runtime InferenceSession configured for TensorRT acceleration."""

    def __init__(
        self,
        model_path: str,
        device_id: int = 0,
        workspace_size_gb: int = 4,
        cache_dir: str = "./models/trt_cache",
        enable_fp16: bool = True,
        dynamic_shape_profile: Optional[Dict[str, str]] = None,
    ):
        self.model_path = os.path.abspath(model_path)
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        trt_options: Dict[str, Any] = {
            "device_id": device_id,
            "trt_max_workspace_size": workspace_size_gb * 1024 * 1024 * 1024,
            "trt_fp16_enable": enable_fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": self.cache_dir,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": self.cache_dir,
            "trt_builder_optimization_level": 3,
            "trt_force_sequential_engine_build": True,
        }

        if dynamic_shape_profile:
            trt_options["trt_profile_min_shapes"] = dynamic_shape_profile.get("min", "")
            trt_options["trt_profile_opt_shapes"] = dynamic_shape_profile.get("opt", "")
            trt_options["trt_profile_max_shapes"] = dynamic_shape_profile.get("max", "")

        cuda_options: Dict[str, Any] = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }

        providers = [
            (_TRT_EP, trt_options),
            ("CUDAExecutionProvider", cuda_options),
            ("CPUExecutionProvider", {}),
        ]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True

        _prepare_runtime()
        logger.info("Initializing session for %s with TensorRT EP...", os.path.basename(model_path))
        self.session = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)
        self.active_providers = self.session.get_providers()
        logger.info("Session bound with providers: %s", self.active_providers)
        if self.active_providers[:1] != [_TRT_EP]:
            # ORT does not raise when a requested provider fails to load; the
            # session it hands back is valid and merely slow.
            logger.warning(
                "%s: TensorRT did not bind (active: %s); running on %s",
                os.path.basename(model_path), self.active_providers,
                self.active_providers[0] if self.active_providers else "nothing")

    @property
    def is_tensorrt(self) -> bool:
        """True only when TensorRT is the provider that actually bound."""
        return bool(self.active_providers) and self.active_providers[0] == _TRT_EP

    def run(self, output_names: Optional[List[str]], input_feed: Dict[str, Any]) -> List[Any]:
        return self.session.run(output_names, input_feed)

    def get_session(self) -> ort.InferenceSession:
        return self.session

    def __del__(self):
        # Explicit teardown ensures TensorRT engine serialization writes to disk cleanly
        if hasattr(self, "session"):
            del self.session
        gc.collect()


__all__ = ["TensorRTInferenceSession"]
