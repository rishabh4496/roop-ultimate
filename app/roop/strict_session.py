"""One way to build a GPU ONNX Runtime session that cannot quietly run on the CPU.

WHY. ``onnxruntime.InferenceSession(path, providers=[CUDA, CPU])`` never raises
when the CUDA provider fails to load (a missing cuDNN DLL is Win32 error 126, a
CPU-only wheel shadowing onnxruntime-gpu, a rejected provider option): it logs,
drops the provider and returns a perfectly working CPU session. The picture is
valid, the return code is 0, only the speed is wrong. ``roop.predictor`` already
knows how to ask a built session which providers it ended up with; what was
missing was a single construction path that always asks, so a model loader
cannot forget to.

WHAT IT DOES, in order:

1. Takes the provider chain the caller already resolved. In this code base that
   is ``precision_policy.providers_for(...)``, which carries the options the
   project has measured and tuned: the cuDNN conv-algo policy (HEURISTIC, with a
   device-probed lowering for the CodeFormer family), the per-model TensorRT
   precision, shape profiles, the GPU-scaled TensorRT workspace, the namespaced
   engine cache under ``models/trt_cache``, ``gpu_mem_limit``, and
   ``do_copy_in_default_stream``. They are NOT re-decided here -- see
   "Options" below.
2. Removes ``CPUExecutionProvider`` from a GPU chain. (ORT appends its CPU EP to
   every session anyway, so this alone prevents nothing; step 5 is the guard.)
3. Fails with a RuntimeError, and what to do about it, when ORT does not list
   ``CUDAExecutionProvider`` at all.
4. Builds ONCE. There is no retry ladder in here: a chain that cannot be built
   raises. Loaders with their own policy on that signal (the swapper's FP32
   rebuild, the canaries) rely on it, and a silent step down would hand them a
   session on a provider they did not ask for with their state flags stale.
   Callers that want a ladder use ``backend_manager.build_session_with_fallback``.
5. Asserts the BUILT session (``predictor.assert_session_providers``): CUDA or
   TensorRT requested and neither active is a hard error, not a warning. A
   TensorRT -> CUDA drop is still the GPU: printed and recorded, not fatal.
6. Prints one confirmation line per tag: the active provider, the device, and the
   memory limits the session was given.

A chain that holds no CUDA/TensorRT provider (``--execution-provider cpu``, DML,
ROCm, CoreML, or a model that is CPU-only by design) is a configuration, not a
fallback: it is built as given and the line says so.

Options. The brief for this module listed fixed values (``kNextPowerOfTwo``
arena, ``cudnn_conv_algo_search='DEFAULT'``, an 8 GiB cap, TensorRT FP16 for
everything). They are deliberately not applied: DEFAULT is 55-241% slower than
HEURISTIC on this app's models (cudnn_algo.py has the table), forcing TensorRT
FP16 undoes the per-model FP32 routing that exists because mixed precision
measured IoU 0.81 on XSeg, and the arena/cap/workspace are already resolved from
the card in ``core.decode_execution_providers``. ``ROOP_CUDNN_CONV_ALGO``,
``ROOP_CUDA_ARENA_STRATEGY``, ``ROOP_CUDA_MEM_LIMIT`` and
``ROOP_TRT_WORKSPACE_MB`` change them for an operator who wants to.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Iterable, List, Optional, Sequence

from roop.degrade import swallowed as _swallowed

CUDA_EP = "CUDAExecutionProvider"
TENSORRT_EP = "TensorrtExecutionProvider"
CPU_EP = "CPUExecutionProvider"

_GIB = 1024 ** 3
_lock = threading.Lock()
_announced: set = set()


class StrictSessionError(RuntimeError):
    """A GPU session was requested and cannot be built on a GPU provider."""


# --------------------------------------------------------------------------
# provider chains
# --------------------------------------------------------------------------

def _name(provider: Any) -> str:
    if isinstance(provider, (tuple, list)) and provider:
        provider = provider[0]
    return str(provider)


def _options(provider: Any) -> dict:
    if isinstance(provider, (tuple, list)) and len(provider) > 1 and isinstance(provider[1], dict):
        return provider[1]
    return {}


def _is_gpu(name: str) -> bool:
    lowered = name.lower()
    return "cuda" in lowered or "tensorrt" in lowered


def without_cpu(providers: Iterable[Any]) -> List[Any]:
    """The chain with every ``CPUExecutionProvider`` entry removed."""
    return [p for p in providers if _name(p) != CPU_EP]


def has_gpu_provider(providers: Iterable[Any]) -> bool:
    return any(_is_gpu(_name(p)) for p in providers)


def _resolve_chain(provider: str) -> List[Any]:
    """The project's tuned chain for ``'cuda'`` or ``'tensorrt'``."""
    key = str(provider or "").strip().lower()
    if key not in ("cuda", "tensorrt"):
        raise ValueError(
            "provider must be 'cuda' or 'tensorrt' (got %r); pass providers=[...] "
            "to build any other configuration" % (provider,))
    from roop import core
    return list(core.decode_execution_providers([key]))


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

def _missing_provider_message(missing: str, available: Sequence[str], tag: str) -> str:
    try:
        from roop import predictor
        environment = predictor.format_environment()
    except Exception as exc:  # the diagnostic must never hide the real error
        _swallowed("roop/strict_session.py:environment", exc, "fallback continued")
        environment = "  (environment report unavailable: %s)" % exc
    try:
        import onnxruntime
        ort_version = onnxruntime.__version__
    except Exception as exc:
        _swallowed("roop/strict_session.py:ort_version", exc, "fallback continued")
        ort_version = "unimportable (%s)" % exc
    return (
        "%s: %s is not available in this onnxruntime, so a GPU session cannot be "
        "built and the CPU is deliberately not used instead.\n"
        "  onnxruntime         : %s\n"
        "  available providers : %s\n"
        "Most likely causes, in order:\n"
        "  1. a CPU-only 'onnxruntime' (or onnxruntime-directml) wheel is installed "
        "next to onnxruntime-gpu and shadows it:\n"
        "       app\\env\\Scripts\\python.exe -m pip list | findstr /i onnxruntime\n"
        "     fix: stop the app, then run Install from Pinokio (it runs "
        "provision_runtime.py, which removes every onnxruntime variant before "
        "installing onnxruntime-gpu).\n"
        "  2. a CUDA 12 / cuDNN 9 DLL cannot be loaded (Win32 error 126): run "
        "tests/test_gpu_bindings.py and read the child stderr, and check that "
        "run.py registered the torch/lib and nvidia\\*\\bin directories.\n"
        "  3. the NVIDIA driver is missing or too old for CUDA 12: run nvidia-smi.\n"
        "%s"
        % (tag, missing, ort_version, ", ".join(available) or "(none)", environment))


def _device_summary(device_id: int) -> str:
    """'NVIDIA GeForce RTX 4070 (cc 8.9, 12282 MiB)' without importing torch."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_id)
            return "%s (cc %d.%d, %d MiB)" % (
                props.name, props.major, props.minor, props.total_memory // (1024 ** 2))
    except Exception as exc:
        _swallowed("roop/strict_session.py:device_torch", exc, "fallback continued")
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            total = pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024 ** 2)
            return "%s (%d MiB)" % (name, total)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        _swallowed("roop/strict_session.py:device_nvml", exc, "fallback continued")
    return "device %d (name unavailable)" % device_id


def _limits_summary(chain: Sequence[Any], active: Sequence[str]) -> str:
    """The memory limits the session was handed, as they appear in the chain."""
    parts: List[str] = []
    for entry in chain:
        opts = _options(entry)
        name = _name(entry)
        if name == CUDA_EP:
            limit = opts.get("gpu_mem_limit")
            parts.append("arena=%s, gpu_mem_limit=%s" % (
                opts.get("arena_extend_strategy", "ORT default"),
                ("%.1f GiB" % (int(limit) / _GIB)) if limit else "unlimited"))
        elif name == TENSORRT_EP:
            workspace = opts.get("trt_max_workspace_size")
            parts.append("trt_workspace=%s, trt_fp16=%s, engine_cache=%s" % (
                ("%.2f GiB" % (int(workspace) / _GIB)) if workspace else "TensorRT default",
                opts.get("trt_fp16_enable", "n/a"),
                os.path.basename(str(opts.get("trt_engine_cache_path", "")).rstrip("\\/")) or "off"))
    return "; ".join(parts) or "no GPU memory options set"


def _model_name(model: Any) -> str:
    if isinstance(model, (bytes, bytearray)):
        return "<in-memory model, %d bytes>" % len(model)
    return os.path.basename(str(model))


def _announce(tag: str, model: Any, requested: Sequence[Any],
              active: Sequence[str], chain: Sequence[Any], device_id: int) -> None:
    with _lock:
        if tag in _announced:
            return
        _announced.add(tag)
    print("[Session] %s: active=%s | requested=%s | %s | %s | model=%s" % (
        tag,
        active[0] if active else "?",
        " > ".join(_name(p) for p in requested),
        _device_summary(device_id),
        _limits_summary(chain, active),
        _model_name(model)), flush=True)


def reset_announcements() -> None:
    """Forget which tags were announced (tests, and model reloads)."""
    with _lock:
        _announced.clear()


# --------------------------------------------------------------------------
# the factory
# --------------------------------------------------------------------------

def create_strictly_accelerated_session(
    model_path: "str | bytes",
    provider: str = "cuda",
    *,
    providers: Optional[Sequence[Any]] = None,
    sess_options: Any = None,
    tag: Optional[str] = None,
    forbid_cpu_nodes: bool = False,
) -> "ort.InferenceSession":  # noqa: F821 - onnxruntime is imported lazily
    """Build an ONNX Runtime session that is on the GPU or does not exist.

    ``provider`` is ``'cuda'`` or ``'tensorrt'`` and selects the project's tuned
    chain from ``core.decode_execution_providers``. ``providers`` -- the chain a
    loader already resolved with ``providers_for`` -- takes precedence and is
    used as given (minus CPU). A ``providers`` chain with no CUDA/TensorRT entry
    is a deliberate CPU/DML/ROCm/CoreML configuration and is built unchanged.

    ``forbid_cpu_nodes=True`` additionally sets ORT's
    ``session.disable_cpu_ep_fallback``, which makes construction fail if ANY
    node would be assigned to the CPU EP. Off by default: most real graphs keep
    a few shape/index nodes on the CPU and are healthy; it is for models known
    to be fully supported. It is set on ``sess_options``, which is therefore
    modified when the caller passes one.

    Raises ``StrictSessionError`` (a ``RuntimeError``) when the GPU provider is
    missing or the session cannot be built on one, and
    ``predictor.ProviderAssertionError`` when ORT returned a session whose active
    providers hold no GPU (``ROOP_STRICT_PROVIDER=0`` downgrades the latter to a
    warning, as everywhere else).
    """
    import onnxruntime as ort
    from roop import predictor

    # A path, or the serialized model itself (the swapper hands ORT bytes after
    # it patches a graph). A bytes model has no file name, so it needs a tag.
    if isinstance(model_path, (bytes, bytearray)):
        if not tag:
            raise ValueError("tag is required when the model is passed as bytes")
        model = bytes(model_path)
        label = tag
    else:
        model = str(model_path)
        label = tag or os.path.splitext(os.path.basename(model))[0]

    explicit = providers is not None
    chain = list(providers) if explicit else _resolve_chain(provider)
    if not has_gpu_provider(chain):
        if not explicit:
            raise StrictSessionError(
                "%s: provider=%r resolved to %s, which holds no CUDA/TensorRT "
                "provider (the backend was not admitted); refusing to build a CPU "
                "session for a model that was asked to run accelerated."
                % (label, provider, [_name(p) for p in chain] or "[]"))
        if sess_options is None:
            from roop.utilities import get_onnx_session_options
            sess_options = get_onnx_session_options()
        session = ort.InferenceSession(model, sess_options, providers=chain)
        with _lock:
            first = label not in _announced
            _announced.add(label)
        if first:
            print("[Session] %s: %s by configuration (no GPU provider requested)"
                  % (label, ", ".join(session.get_providers()) or "?"), flush=True)
        return session

    chain = without_cpu(chain)
    available = list(ort.get_available_providers())
    if CUDA_EP not in available:
        raise StrictSessionError(_missing_provider_message(CUDA_EP, available, label))
    if TENSORRT_EP not in available and any(_name(p) == TENSORRT_EP for p in chain):
        # ORT would drop it with a warning anyway; say so instead of leaving it to stderr.
        print("[Session] %s: %s is not available in this onnxruntime; using CUDA"
              % (label, TENSORRT_EP), flush=True)
        chain = [p for p in chain if _name(p) != TENSORRT_EP]

    try:
        from roop.trt_native_runner import is_native_loaded
        if is_native_loaded(label) or (not isinstance(model, bytes) and is_native_loaded(model)):
            if any(_name(p) == TENSORRT_EP for p in chain):
                print("[Session] %s: already loaded in native TensorRT runner; excluding ORT %s"
                      % (label, TENSORRT_EP), flush=True)
                chain = [p for p in chain if _name(p) != TENSORRT_EP]
    except Exception as _exc:
        _swallowed("roop/strict_session.py:is_native_loaded", _exc, "native check fallback")

    if sess_options is None:
        from roop.utilities import get_onnx_session_options
        sess_options = get_onnx_session_options()
    if forbid_cpu_nodes:
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    # One attempt, no ladder: if the chain cannot be built the caller sees the
    # exception. Several loaders run their own policy on that signal (the swapper
    # rebuilds on TensorRT FP32, then on CUDA, and fixes its own state flags); a
    # silent step down in here would hand them a session on a provider they did
    # not ask for and leave that state wrong. Callers that DO want a ladder use
    # backend_manager.build_session_with_fallback.
    from contextlib import nullcontext
    from roop.trt_build_guard import around_trt_build
    _guard = around_trt_build("session:" + label) if any(_name(p) == TENSORRT_EP for p in chain) else nullcontext()
    with _guard:
        session = ort.InferenceSession(model, sess_options, providers=chain)

    # ORT also fails without raising: it drops the GPU provider and returns a
    # working CPU session. Ask the built session. TensorRT -> CUDA is the one
    # sanctioned drop (still the GPU): printed and recorded, not fatal.
    predictor.verify_built(session, chain, label)
    active = list(session.get_providers())
    device_id = 0
    for entry in chain:
        if "device_id" in _options(entry):
            device_id = int(_options(entry)["device_id"])
            break
    _announce(label, model, chain, active, chain, device_id)
    return session
