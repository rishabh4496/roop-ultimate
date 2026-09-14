"""Strict TensorRT Execution Provider session construction.

The normal roop runtime intentionally keeps CUDA and CPU fallbacks because it
supports a number of legacy, fixed-batch model exports.  The throughput path
in this module has a different contract: a session is created with TensorRT
only, its dynamic input profile is explicit, and construction fails if ORT
cannot keep the requested execution provider active.

The profile strings follow ONNX Runtime's documented format, for example
``target:1x3x256x256,source:1x512``.  Engine and timing caches are allowed
filesystem state; video frames never use those directories.
"""

from __future__ import annotations
from roop.degrade import swallowed as _swallowed

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


Shape = Tuple[int, ...]
ProfileShapes = Tuple[Shape, Shape, Shape]
_DLL_HANDLES: List[Any] = []


class TensorRTSessionError(RuntimeError):
    """Base error for a session that cannot satisfy the strict TRT contract."""


class StaticBatchError(TensorRTSessionError):
    """Raised when a model export cannot accept the requested batch profile."""


def _truthy(value: Any, default: bool = False) -> bool:
    """Parse an environment-style boolean without treating ``None`` as false."""

    if value is None:
        return bool(default)
    return str(value).strip().lower() not in ("", "0", "false", "no", "off")


def _positive_int(value: Any, default: int) -> int:
    """Return a positive integer, falling back for malformed environment input."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, parsed)


def _finite_float(value: Any, default: float) -> float:
    """Return a finite float for hardware probing values."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _safe_name(value: str) -> str:
    """Make a cache prefix legal and stable on Windows and POSIX."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned.strip("_.-") or "model"


def prepare_tensorrt_runtime() -> None:
    """Register packaged TensorRT/CUDA DLL directories before ORT import.

    Windows Python 3.8+ does not reliably resolve dependent DLLs from PATH
    alone.  The repository's environment carries ``tensorrt_libs`` beside
    the wheel and CUDA/cuDNN beside PyTorch; registering both locations is
    required before ONNX Runtime attempts to load its TensorRT provider.
    """

    search_paths: List[Path] = []
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir():
            search_paths.append(torch_lib)
    except (ImportError, OSError, RuntimeError):
        pass
    try:
        import tensorrt

        trt_libs = Path(tensorrt.__file__).resolve().parent.parent / "tensorrt_libs"
        if trt_libs.is_dir():
            search_paths.append(trt_libs)
    except (ImportError, OSError, RuntimeError):
        try:
            import tensorrt_libs

            trt_libs = Path(tensorrt_libs.__file__).resolve().parent
            if trt_libs.is_dir():
                search_paths.append(trt_libs)
        except (ImportError, OSError, RuntimeError):
            pass
    for directory in dict.fromkeys(search_paths):
        directory_text = str(directory)
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_HANDLES.append(os.add_dll_directory(directory_text))
            except OSError:
                pass
        current_path = os.environ.get("PATH", "")
        if directory_text not in current_path.split(os.pathsep):
            os.environ["PATH"] = directory_text + os.pathsep + current_path


def _model_cache_prefix(model_path: Path, device_id: int) -> str:
    """Generate a model/version-specific prefix without reading video data."""

    try:
        stat = model_path.stat()
        stamp = f"{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    except OSError:
        stamp = str(model_path).encode("utf-8")
    digest = hashlib.sha1(stamp + str(model_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return _safe_name(f"roop_{model_path.stem}_{digest}_gpu{int(device_id)}")


def _shape_text(shape: Sequence[int]) -> str:
    """Encode one shape using the TensorRT EP ``xd`` syntax."""

    if not shape or any(int(value) <= 0 for value in shape):
        raise ValueError(f"profile shapes must contain positive dimensions: {shape!r}")
    return "x".join(str(int(value)) for value in shape)


def format_profile_shapes(shapes: Mapping[str, Shape]) -> str:
    """Format input shapes in deterministic ONNX Runtime provider syntax."""

    if not shapes:
        raise ValueError("at least one TensorRT profile input is required")
    return ",".join(
        f"{name}:{_shape_text(shapes[name])}" for name in sorted(shapes)
    )


def _dimension_is_dynamic(dimension: Any) -> bool:
    """Return whether an ONNX dimension is symbolic or unknown."""

    if dimension is None:
        return True
    if isinstance(dimension, int):
        return dimension <= 0
    return not str(dimension).isdigit()


def _dimension_value(dimension: Any) -> Optional[int]:
    """Convert a fixed ONNX dimension to an integer when possible."""

    if isinstance(dimension, int) and dimension > 0:
        return int(dimension)
    try:
        parsed = int(str(dimension))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _read_model_input_shapes(model_path: Path) -> Dict[str, Tuple[Any, ...]]:
    """Read graph input dimensions without constructing a fallback inference session."""

    try:
        import onnx

        model = onnx.load(str(model_path), load_external_data=False)
        initializers = {initializer.name for initializer in model.graph.initializer}
        result: Dict[str, Tuple[Any, ...]] = {}
        for value_info in model.graph.input:
            if value_info.name in initializers:
                continue
            tensor_shape = value_info.type.tensor_type.shape.dim
            dimensions: List[Any] = []
            for dimension in tensor_shape:
                if dimension.dim_value > 0:
                    dimensions.append(int(dimension.dim_value))
                elif dimension.dim_param:
                    dimensions.append(str(dimension.dim_param))
                else:
                    dimensions.append(None)
            result[value_info.name] = tuple(dimensions)
        if result:
            return result
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        onnx_error = error
    else:
        onnx_error = RuntimeError("ONNX graph did not contain usable inputs")

    # The fallback is metadata-only.  It is never used to run inference and
    # exists for minimal installations that omit the optional ``onnx`` wheel.
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return {
            meta.name: tuple(meta.shape)
            for meta in session.get_inputs()
        }
    except Exception as error:
        raise TensorRTSessionError(
            f"cannot inspect ONNX inputs for {model_path}: {onnx_error}"
        ) from error


def _default_spatial_shape(rank: int, dimension_index: int) -> int:
    """Choose a conservative concrete value for common image input symbols."""

    if rank >= 4 and dimension_index >= rank - 2:
        return 640
    return 1


def derive_profile_shapes(
    model_path: str | os.PathLike[str],
    min_batch: int = 1,
    opt_batch: int = 8,
    max_batch: int = 16,
    shape_overrides: Optional[Mapping[str, Shape]] = None,
    explicit_profiles: Optional[Mapping[str, ProfileShapes]] = None,
    require_dynamic_batch: bool = True,
    allow_static_batch_one: bool = False,
    default_spatial_shape: Tuple[int, int] = (640, 640),
) -> Tuple[Dict[str, Shape], Dict[str, Shape], Dict[str, Shape]]:
    """Derive complete min/opt/max profiles for every model input.

    ``shape_overrides`` supplies the optimal concrete shape for dynamic
    non-batch dimensions.  ``explicit_profiles`` can supply all three shapes
    for models such as RetinaFace whose output geometry depends on a caller's
    chosen detector resolution.  A static batch-one face swap export is
    rejected by default because running it repeatedly is exactly the batch
    starvation this optimized path is intended to remove.
    """

    minimum = max(1, int(min_batch))
    optimal = max(minimum, int(opt_batch))
    maximum = max(optimal, int(max_batch))
    if maximum < 2 and require_dynamic_batch and not allow_static_batch_one:
        raise StaticBatchError(
            "strict TensorRT mode requires a batch profile larger than one; "
            "set max_batch >= 2 or explicitly use the legacy fixed-batch path"
        )
    overrides = dict(shape_overrides or {})
    explicit = dict(explicit_profiles or {})
    model_inputs = _read_model_input_shapes(Path(model_path))
    minimum_shapes: Dict[str, Shape] = {}
    optimal_shapes: Dict[str, Shape] = {}
    maximum_shapes: Dict[str, Shape] = {}

    for name, declared in model_inputs.items():
        if name in explicit:
            profile = explicit[name]
            if len(profile) != 3:
                raise ValueError(f"profile for {name!r} must contain min/opt/max shapes")
            min_shape, opt_shape, max_shape = tuple(
                tuple(int(item) for item in shape) for shape in profile
            )
            if not (len(min_shape) == len(opt_shape) == len(max_shape) == len(declared)):
                raise ValueError(f"profile rank mismatch for input {name!r}")
            minimum_shapes[name] = min_shape
            optimal_shapes[name] = opt_shape
            maximum_shapes[name] = max_shape
            continue

        override = overrides.get(name)
        if override is not None and len(override) != len(declared):
            raise ValueError(f"shape override rank mismatch for input {name!r}")
        min_values: List[int] = []
        opt_values: List[int] = []
        max_values: List[int] = []
        for index, dimension in enumerate(declared):
            fixed = _dimension_value(dimension)
            dynamic = _dimension_is_dynamic(dimension)
            if index == 0 and dynamic:
                min_values.append(minimum)
                opt_values.append(optimal)
                max_values.append(maximum)
                continue
            if index == 0 and not dynamic and fixed == 1 and require_dynamic_batch:
                if maximum > 1 and not allow_static_batch_one:
                    raise StaticBatchError(
                        f"{Path(model_path).name} input {name!r} is static batch-1; "
                        "export a dynamic-batch ONNX model before using the strict "
                        "TensorRT pipeline"
                    )
            if fixed is not None:
                min_values.append(fixed)
                opt_values.append(fixed)
                max_values.append(fixed)
                continue
            if override is not None:
                concrete = int(override[index])
            elif len(declared) >= 4 and index == len(declared) - 2:
                concrete = int(default_spatial_shape[0])
            elif len(declared) >= 4 and index == len(declared) - 1:
                concrete = int(default_spatial_shape[1])
            else:
                concrete = _default_spatial_shape(len(declared), index)
            if concrete <= 0:
                raise ValueError(f"dynamic dimension {index} of {name!r} is not positive")
            min_values.append(concrete)
            opt_values.append(concrete)
            max_values.append(concrete)
        minimum_shapes[name] = tuple(min_values)
        optimal_shapes[name] = tuple(opt_values)
        maximum_shapes[name] = tuple(max_values)

    if not model_inputs:
        raise TensorRTSessionError(f"ONNX model has no graph inputs: {model_path}")
    return minimum_shapes, optimal_shapes, maximum_shapes


@dataclass(frozen=True)
class TensorRTSessionConfig:
    """Hardware-aware, strict TensorRT EP settings.

    The defaults match the two supported machines: a 4 GiB TRT workspace and
    up to 16 items on the 12 GiB card, or a 1.5 GiB workspace and a smaller
    dynamic profile on the 6 GiB laptop.  The batch profile remains dynamic on
    both devices; the laptop governor is conservative rather than silently
    reverting to CPU execution.
    """

    cache_path: Path | str = Path("models/trt_cache")
    device_id: int = 0
    fp16: bool = True
    builder_optimization_level: int = 3
    workspace_size: Optional[int] = None
    min_batch: int = 1
    opt_batch: Optional[int] = None
    max_batch: Optional[int] = None
    default_spatial_shape: Tuple[int, int] = (640, 640)
    require_dynamic_batch: bool = True
    allow_static_batch_one: bool = False
    shape_overrides: Mapping[str, Shape] = field(default_factory=dict)
    explicit_profiles: Mapping[str, ProfileShapes] = field(default_factory=dict)
    engine_cache_prefix: Optional[str] = None
    extra_provider_options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_environment(
        cls,
        cache_path: Optional[str | os.PathLike[str]] = None,
        device_id: int = 0,
    ) -> "TensorRTSessionConfig":
        """Build settings from the repository's dual-device environment."""

        total_gb = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                total_gb = float(
                    torch.cuda.get_device_properties(int(device_id)).total_memory
                ) / float(1024**3)
        except Exception as _degrade_error:
            _swallowed("roop/trt_session_builder.py:365", _degrade_error, "fallback continued")
            total_gb = 0.0
        laptop = 0.0 < total_gb < 7.0
        default_opt = 2 if laptop else 8
        default_max = 4 if laptop else 16
        opt_batch = _positive_int(os.environ.get("ROOP_TRT_OPT_BATCH", default_opt), default_opt)
        max_batch = _positive_int(os.environ.get("ROOP_TRT_MAX_BATCH", default_max), default_max)
        max_batch = max(opt_batch, max_batch)
        default_cache = Path(cache_path or os.environ.get("ROOP_TRT_CACHE_DIR", "models/trt_cache"))
        configured_workspace = os.environ.get("ROOP_TRT_WORKSPACE_BYTES")
        if configured_workspace:
            workspace = _positive_int(configured_workspace, 1536 * 1024 * 1024)
        else:
            workspace = (1536 if laptop else 4096) * 1024 * 1024
        return cls(
            cache_path=default_cache,
            device_id=int(device_id),
            fp16=_truthy(os.environ.get("ROOP_TRT_FP16", "1"), True),
            builder_optimization_level=min(
                5,
                max(0, _positive_int(os.environ.get("ROOP_TRT_BUILDER_LEVEL", 3), 3)),
            ),
            workspace_size=workspace,
            min_batch=_positive_int(os.environ.get("ROOP_TRT_MIN_BATCH", 1), 1),
            opt_batch=opt_batch,
            max_batch=max_batch,
            default_spatial_shape=(
                _positive_int(os.environ.get("ROOP_TRT_PROFILE_HEIGHT", 640), 640),
                _positive_int(os.environ.get("ROOP_TRT_PROFILE_WIDTH", 640), 640),
            ),
            require_dynamic_batch=_truthy(
                os.environ.get("ROOP_TRT_REQUIRE_DYNAMIC_BATCH", "1"), True
            ),
            allow_static_batch_one=_truthy(
                os.environ.get("ROOP_TRT_ALLOW_STATIC_BATCH1", "0"), False
            ),
        )

    def resolved_batches(self) -> Tuple[int, int, int]:
        """Return validated min/optimal/max batch sizes."""

        minimum = max(1, int(self.min_batch))
        optimal = max(minimum, int(self.opt_batch or minimum))
        maximum = max(optimal, int(self.max_batch or optimal))
        return minimum, optimal, maximum

    def resolved_workspace(self) -> int:
        """Return the workspace cap while respecting the hardware tier."""

        default = 1536 * 1024 * 1024 if self.max_batch is not None and self.max_batch <= 4 else 4096 * 1024 * 1024
        return max(256 * 1024 * 1024, int(self.workspace_size or default))

    def provider_options(
        self,
        model_path: str | os.PathLike[str],
    ) -> Dict[str, Any]:
        """Create the complete TensorRT provider option dictionary."""

        minimum, optimal, maximum = self.resolved_batches()
        min_shapes, opt_shapes, max_shapes = derive_profile_shapes(
            model_path,
            min_batch=minimum,
            opt_batch=optimal,
            max_batch=maximum,
            shape_overrides=self.shape_overrides,
            explicit_profiles=self.explicit_profiles,
            require_dynamic_batch=self.require_dynamic_batch,
            allow_static_batch_one=self.allow_static_batch_one,
            default_spatial_shape=self.default_spatial_shape,
        )
        cache_dir = Path(self.cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.engine_cache_prefix or _model_cache_prefix(Path(model_path), self.device_id)
        profile_digest = hashlib.sha1(
            (
                format_profile_shapes(min_shapes)
                + "|"
                + format_profile_shapes(opt_shapes)
                + "|"
                + format_profile_shapes(max_shapes)
            ).encode("utf-8")
        ).hexdigest()[:10]
        options: Dict[str, Any] = dict(self.extra_provider_options)
        options.update(
            {
                "device_id": int(self.device_id),
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_dir),
                "trt_engine_cache_prefix": _safe_name(f"{prefix}_{profile_digest}"),
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": str(cache_dir),
                "trt_fp16_enable": bool(self.fp16),
                "trt_builder_optimization_level": int(self.builder_optimization_level),
                "trt_max_workspace_size": int(self.resolved_workspace()),
                "trt_max_partition_iterations": 2000,
                "trt_min_subgraph_size": 1,
                "trt_force_sequential_engine_build": True,
                "trt_build_heuristics_enable": True,
                "trt_context_memory_sharing_enable": True,
                "trt_layer_norm_fp32_fallback": True,
                "trt_auxiliary_streams": 0,
                "trt_cuda_graph_enable": _truthy(
                    os.environ.get("ROOP_TRT_CUDA_GRAPH", "0"), False
                ),
                "trt_profile_min_shapes": format_profile_shapes(min_shapes),
                "trt_profile_opt_shapes": format_profile_shapes(opt_shapes),
                "trt_profile_max_shapes": format_profile_shapes(max_shapes),
            }
        )
        return options


def assert_strict_tensorrt_session(session: Any, model_path: str | os.PathLike[str]) -> None:
    """Reject an ORT session whose active provider chain is not TRT-only."""

    try:
        active = [str(name) for name in session.get_providers()]
    except Exception as error:
        raise TensorRTSessionError(
            f"TensorRT session for {model_path} did not expose its provider chain"
        ) from error
    if active != ["TensorrtExecutionProvider"]:
        raise TensorRTSessionError(
            f"strict TensorRT session rejected for {model_path}: active providers {active!r}; "
            "CUDAExecutionProvider/CPUExecutionProvider fallbacks are forbidden"
        )


def build_tensorrt_session(
    model_path: str | os.PathLike[str],
    config: Optional[TensorRTSessionConfig] = None,
    session_options: Any = None,
) -> Any:
    """Create and validate an ORT session with TensorRT as the only provider.

    The function deliberately does not append CUDA or CPU providers.  ORT's
    documented default recommendation is to add CUDA for unsupported nodes,
    but this performance contract requires a loud failure instead of an
    unmeasured mixed-EP execution.  The caller can then choose a compatible
    model export or use the repository's explicitly separate legacy path.
    """

    prepare_tensorrt_runtime()
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise TensorRTSessionError("onnxruntime-gpu with TensorRT support is required") from error
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    available = {str(name) for name in ort.get_available_providers()}
    if "TensorrtExecutionProvider" not in available:
        raise TensorRTSessionError(
            f"TensorRTExecutionProvider is unavailable; ORT providers are {sorted(available)!r}"
        )
    settings = config or TensorRTSessionConfig.from_environment()
    options = session_options or ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [("TensorrtExecutionProvider", settings.provider_options(path))]
    try:
        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    except Exception as error:
        raise TensorRTSessionError(
            f"TensorRT session construction failed for {path}; no fallback was attempted: {error}"
        ) from error
    try:
        assert_strict_tensorrt_session(session, path)
    except Exception:
        del session
        raise
    return session


def build_trt_session(
    model_path: str | os.PathLike[str],
    config: Optional[TensorRTSessionConfig] = None,
    session_options: Any = None,
) -> Any:
    """Backward-compatible alias for :func:`build_tensorrt_session`."""

    return build_tensorrt_session(model_path, config=config, session_options=session_options)


__all__ = [
    "ProfileShapes",
    "Shape",
    "StaticBatchError",
    "TensorRTSessionConfig",
    "TensorRTSessionError",
    "assert_strict_tensorrt_session",
    "build_trt_session",
    "build_tensorrt_session",
    "derive_profile_shapes",
    "format_profile_shapes",
    "prepare_tensorrt_runtime",
]
