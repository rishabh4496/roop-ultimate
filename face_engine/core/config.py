"""Typed configuration for the face engine runtime.

Every knob the execution layer reads lives on :class:`EngineConfig`. Paths are
kept relative in the defaults and resolved against the current working
directory (or an environment override) at use time, so no machine-specific
drive letter is ever baked into the package.

Environment overrides
---------------------
``FACE_ENGINE_CACHE_DIR``   root for TensorRT engines and verification sidecars
``FACE_ENGINE_MODELS_DIR``  where model files are stored / looked up
``FACE_ENGINE_DEVICE_ID``   CUDA device ordinal used by both GPU providers
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Provider(str, Enum):
    """ONNX Runtime execution providers the engine knows how to configure.

    The declaration order is the preference order: the engine tries TensorRT,
    then CUDA, then CPU.
    """

    TENSORRT = "TensorrtExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    CPU = "CPUExecutionProvider"


DEFAULT_PROVIDER_ORDER: list[Provider] = [Provider.TENSORRT, Provider.CUDA, Provider.CPU]

_FOUR_GIB = 4 * 1024 ** 3


class TensorRTOptions(BaseModel):
    """Provider options passed verbatim to ``TensorrtExecutionProvider``."""

    model_config = ConfigDict(frozen=True)

    trt_max_workspace_size: int = Field(default=_FOUR_GIB, gt=0)
    trt_fp16_enable: bool = True
    trt_engine_cache_enable: bool = True
    # Relative to EngineConfig.cache_dir unless absolute; with the default
    # cache_dir this is ".cache/trt_engines".
    trt_engine_cache_path: Path = Path("trt_engines")


class CUDAOptions(BaseModel):
    """Provider options for ``CUDAExecutionProvider``.

    ``gpu_mem_limit`` is not a field: it is derived per device from physical
    VRAM by :meth:`face_engine.core.execution.ExecutionEngine.cuda_mem_limit`
    as ``vram_fraction`` of TOTAL physical VRAM. Set
    ``gpu_mem_limit_override`` to pin it.

    ``free_vram_fraction`` (off by default) additionally caps it at a fraction
    of the VRAM free when the session is built. Do not turn it on casually:
    under Windows WDDM "free" reads ~0 on a busy card while allocations still
    succeed (the driver pages), and a cap derived from it gave a 0-byte arena;
    the CUDA session failed and, unless strict, fell back to CPU
    (full test suite with the app running, 2026-09-27).
    """

    model_config = ConfigDict(frozen=True)

    arena_extend_strategy: str = "kNextPowerOfTwo"
    # HEURISTIC, not DEFAULT: with the cuDNN 9 that torch cu128 loads, DEFAULT
    # sends every convolution down ORT's "Conv running in Fallback mode" path.
    # Measured 2026-09-28 (RTX 4070, batch 1, FP32), DEFAULT -> HEURISTIC with
    # TF32 off: SCRFD-10G 10.9 -> 4.7 ms, XSeg-3 21.2 -> 13.3, BiSeNet 17.1 ->
    # 10.9, HyperSwap-1a 44.3 -> 21.6, GPEN-512 152.5 -> 76.8, ArcFace 12.0 ->
    # 3.4; outputs equal to <= 1.6e-5 (HyperSwap 1.2e-2 either way).
    cudnn_conv_algo_search: str = "HEURISTIC"
    # TF32 off = exact FP32. On, HyperSwap runs 21.6 -> 9.8 ms but its output
    # moves by up to 1.5e-2 (~1.9/255); swap identity is precision-sensitive,
    # so it is opt-in.
    use_tf32: bool = False
    do_copy_in_default_stream: bool = True
    vram_fraction: float = Field(default=0.80, gt=0.0, le=1.0)
    free_vram_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    gpu_mem_limit_override: int | None = Field(default=None, gt=0)

    @field_validator("arena_extend_strategy")
    @classmethod
    def _check_arena(cls, value: str) -> str:
        if value not in {"kNextPowerOfTwo", "kSameAsRequested"}:
            raise ValueError(f"unknown arena_extend_strategy {value!r}")
        return value

    @field_validator("cudnn_conv_algo_search")
    @classmethod
    def _check_algo(cls, value: str) -> str:
        if value not in {"EXHAUSTIVE", "HEURISTIC", "DEFAULT"}:
            raise ValueError(f"unknown cudnn_conv_algo_search {value!r}")
        return value


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


class EngineConfig(BaseModel):
    """Top-level runtime configuration.

    Attributes:
        device_id: CUDA ordinal for both GPU providers.
        providers: Preference-ordered provider list. Providers the installed
            onnxruntime build does not offer are dropped before a session is
            created.
        strict: When True, a session whose granted primary provider differs
            from the first *available* requested provider raises
            :class:`~face_engine.core.execution.ProviderFallbackError`
            instead of logging a warning. ORT drops an EP that fails to load
            silently and returns a working CPU session; ``strict`` turns that
            into an error.
        register_gpu_dlls: On Windows, register TensorRT / torch / nvidia wheel
            library folders before the first session so ORT can load them.
        cache_dir: Root for engine caches and hash sidecars.
        models_dir: Where model files are stored.
        intra_op_num_threads: 0 lets ORT decide.
    """

    model_config = ConfigDict(frozen=True)

    device_id: int = Field(default_factory=lambda: _env_int("FACE_ENGINE_DEVICE_ID", 0), ge=0)
    providers: list[Provider] = Field(default_factory=lambda: list(DEFAULT_PROVIDER_ORDER))
    strict: bool = False
    register_gpu_dlls: bool = True
    cache_dir: Path = Field(default_factory=lambda: _env_path("FACE_ENGINE_CACHE_DIR", Path(".cache")))
    models_dir: Path = Field(
        default_factory=lambda: _env_path("FACE_ENGINE_MODELS_DIR", Path(".cache/models")))
    tensorrt: TensorRTOptions = Field(default_factory=TensorRTOptions)
    cuda: CUDAOptions = Field(default_factory=CUDAOptions)
    intra_op_num_threads: int = Field(default=0, ge=0)
    log_severity_level: int = Field(default=3, ge=0, le=4)

    @field_validator("providers")
    @classmethod
    def _non_empty(cls, value: list[Provider]) -> list[Provider]:
        if not value:
            raise ValueError("providers must name at least one execution provider")
        if len(set(value)) != len(value):
            raise ValueError("providers must not repeat")
        return value

    def resolved_cache_dir(self) -> Path:
        """Absolute cache root."""
        return self.cache_dir.expanduser().resolve()

    def resolved_models_dir(self) -> Path:
        """Absolute models directory."""
        return self.models_dir.expanduser().resolve()

    def resolved_trt_cache_dir(self) -> Path:
        """Absolute TensorRT engine cache directory (relative paths hang off the cache root)."""
        path = self.tensorrt.trt_engine_cache_path.expanduser()
        return path if path.is_absolute() else (self.resolved_cache_dir() / path).resolve()

    def tensorrt_provider_options(self) -> dict[str, Any]:
        """Options dict for ``TensorrtExecutionProvider`` (shape profiles excluded)."""
        return {
            "device_id": self.device_id,
            "trt_max_workspace_size": self.tensorrt.trt_max_workspace_size,
            "trt_fp16_enable": self.tensorrt.trt_fp16_enable,
            "trt_engine_cache_enable": self.tensorrt.trt_engine_cache_enable,
            "trt_engine_cache_path": str(self.resolved_trt_cache_dir()),
        }
