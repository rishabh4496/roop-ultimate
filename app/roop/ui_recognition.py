"""Logic behind Settings > Face recognition: hardware advice, model catalogue, apply.

Framework-neutral on purpose. The Gradio UI (app/ui/) is frozen and every new control is
React over FastAPI, so this module has no UI import: ``routes_recognition.py`` serves it
and ``RecognitionPanel.jsx`` draws it. A Gradio front end could call the same functions.

What a selection DOES (and does not). The chosen model + provider drive
``face_analyser.set_recognition_model`` / ``extract_face_embedding`` / ``IdentityBank``.
Live swap identity MATCHING is not rerouted through it: the w600k vector also feeds the
swapper, and every gate constant is calibrated per model (only w600k and AdaFace have
one). ``SCOPE_NOTE`` says so and the panel shows it; do not let a UI claim more.

The hardware advice is a RECOMMENDATION from measured behaviour, never an automatic
change: the provider follows the machine, but the recognition MODEL is only suggested
where a measured cost difference justifies it (CPU), because switching the model switches
the identity metric.
"""

import os
import platform
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from roop.degrade import swallowed as _swallowed
from roop.recognition_registry import RECOGNITION_REGISTRY, get_model_spec

TIER_ENTHUSIAST_ADA = "TIER_ENTHUSIAST_ADA"
TIER_MID_AMPERE = "TIER_MID_AMPERE"
TIER_CUDA_GENERIC = "TIER_CUDA_GENERIC"
TIER_DIRECTML = "TIER_DIRECTML"
TIER_APPLE_SILICON = "TIER_APPLE_SILICON"
TIER_CPU = "TIER_CPU"

DEFAULT_MODEL = "default"
FOLLOW_APP = "app"            # provider value meaning "whatever the rest of the app runs on"

SCOPE_NOTE = (
    "Used by the embedding API (face_analyser.extract_face_embedding) and IdentityBank. "
    "Live swap matching is NOT switched by this: it keeps using w600k (or AdaFace when "
    "ROOP_ADAFACE=1) until each model's match threshold has been calibrated."
)

_TRT_MIN_VRAM_GB = 7.0        # TensorRT is not admitted below this (the 6 GB laptop tier)

# value -> (label, ORT provider that must be present)
_PROVIDERS: Dict[str, Tuple[str, Optional[str]]] = {
    FOLLOW_APP: ("Same as the app's provider", None),
    "cuda": ("CUDA", "CUDAExecutionProvider"),
    "tensorrt": ("TensorRT", "TensorrtExecutionProvider"),
    "directml": ("DirectML", "DmlExecutionProvider"),
    "coreml": ("CoreML", "CoreMLExecutionProvider"),
    "cpu": ("CPU", "CPUExecutionProvider"),
}


@dataclass(frozen=True)
class HardwareAdvice:
    tier: str
    label: str
    device_name: str
    vram_gb: float
    compute_capability: str
    architecture: str
    provider: str                 # recommended provider key (a _PROVIDERS key)
    strategy: str                 # one-line description of the tuning that provider gets
    model_hint: Optional[str]     # a registry key, or None = keep the current model
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- hardware advice

def probe_hardware(device_id: int = 0) -> Dict[str, Any]:
    """Cheap capability probe: CUDA device name / compute capability / VRAM via torch.

    runtime_optimizer.HardwareProfiler.profile() reports more but takes ~15 s (ffmpeg,
    TensorRT builder, NVML), far too slow for a panel that loads on page open.
    """
    info: Dict[str, Any] = {"cuda": False, "name": "", "capability": None, "vram_gb": 0.0}
    try:
        import torch
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(device_id)
            info.update(cuda=True, name=props.name, capability=(props.major, props.minor),
                        vram_gb=round(props.total_memory / 1024 ** 3, 2))
    except Exception as exc:
        _swallowed("roop/ui_recognition.py:probe_hardware", exc, "no CUDA info; treated as non-CUDA")
    return info


def _architecture(capability: Optional[Tuple[int, int]]) -> str:
    if not capability:
        return ""
    try:
        from roop.runtime_optimizer import HardwareProfiler
        return HardwareProfiler._architecture(capability)
    except Exception as exc:
        _swallowed("roop/ui_recognition.py:_architecture", exc, "architecture family unknown")
        return "SM %d.%d" % capability


def classify_hardware(probe: Dict[str, Any], ort_providers: List[str],
                      system: str, machine: str) -> HardwareAdvice:
    """Pure function (testable without a GPU): hardware facts -> tier + recommendation.

    Tiers come from the compute capability and the providers ORT actually offers, never
    from the marketing name, so an unknown future card lands in the nearest tier by what
    it can do instead of being treated as a 4070.
    """
    cap = probe.get("capability")
    vram = float(probe.get("vram_gb") or 0.0)
    name = probe.get("name") or ""
    arch = _architecture(tuple(cap) if cap else None)
    cc = "%d.%d" % tuple(cap) if cap else ""
    common = dict(device_name=name, vram_gb=vram, compute_capability=cc, architecture=arch)
    have = set(ort_providers)

    if probe.get("cuda") and cap and "CUDAExecutionProvider" in have:
        if tuple(cap) >= (8, 9):
            if "TensorrtExecutionProvider" in have and vram >= _TRT_MIN_VRAM_GB:
                return HardwareAdvice(
                    TIER_ENTHUSIAST_ADA, "Ada-class GPU", provider="tensorrt",
                    strategy="TensorRT FP16, fixed 1x3x112x112 profile (CUDA with cuDNN EXHAUSTIVE is the fallback)",
                    model_hint=None,
                    reason="Measured on an RTX 4070: TensorRT 1.3-2.5 ms vs CUDA 1-6 ms per face; FP16 "
                           "embeddings stay within 0.99996 cosine of CPU.", **common)
            return HardwareAdvice(
                TIER_ENTHUSIAST_ADA, "Ada-class GPU", provider="cuda",
                strategy="CUDA, cuDNN EXHAUSTIVE, power-of-two arena",
                model_hint=None,
                reason=("TensorRT is not offered below %.0f GB of VRAM." % _TRT_MIN_VRAM_GB
                        if "TensorrtExecutionProvider" in have else
                        "TensorRT is not available in this onnxruntime build."), **common)
        if tuple(cap) >= (8, 0):
            return HardwareAdvice(
                TIER_MID_AMPERE, "Ampere-class GPU", provider="cuda",
                strategy="CUDA, cuDNN HEURISTIC, arena grown in requested-size chunks",
                model_hint=None,
                reason="Keeps the per-session arena small so it coexists with the swapper and "
                       "enhancers on a 6-12 GB card.", **common)
        return HardwareAdvice(
            TIER_CUDA_GENERIC, "Older CUDA GPU", provider="cuda",
            strategy="CUDA, cuDNN HEURISTIC",
            model_hint=None,
            reason="Pre-Ampere capability %s: CUDA without TensorRT FP16 tuning." % cc, **common)

    if "DmlExecutionProvider" in have and system == "Windows":
        return HardwareAdvice(
            TIER_DIRECTML, "DirectML GPU", provider="directml",
            strategy="DirectML (memory-pattern optimisation off, as DirectML requires)",
            model_hint=None,
            reason="No CUDA device; DirectML covers AMD / Intel GPUs on Windows.", **common)

    if system == "Darwin" and machine == "arm64" and "CoreMLExecutionProvider" in have:
        return HardwareAdvice(
            TIER_APPLE_SILICON, "Apple Silicon", provider="coreml",
            strategy="CoreML (default options); falls back to CPU if it cannot bind",
            model_hint=None,
            reason="Not measured on this hardware: the engine verifies the provider after a "
                   "real inference and says if CoreML did not take.", **common)

    return HardwareAdvice(
        TIER_CPU, "CPU only", provider="cpu",
        strategy="CPU, one thread per physical core",
        model_hint="mobilefacenet",
        reason="Measured on a 24-core desktop CPU: MobileFaceNet 3.9 ms vs the default 41 ms vs "
               "Glint-R100 310 ms per face. A different model is a different identity metric; "
               "this is a suggestion, not an automatic switch.", **common)


def hardware_advice(device_id: int = 0) -> HardwareAdvice:
    try:
        import onnxruntime as ort
        providers = list(ort.get_available_providers())
    except Exception as exc:
        _swallowed("roop/ui_recognition.py:hardware_advice", exc, "no onnxruntime providers listed")
        providers = []
    return classify_hardware(probe_hardware(device_id), providers, platform.system(), platform.machine())


# --------------------------------------------------------------- catalogue / options

def model_catalog(models_dir: str) -> List[Dict[str, Any]]:
    """One row per registered recogniser, with the specs worth showing and whether the
    file is already on disk (existence only; the hash is checked when it is loaded)."""
    by_file: Dict[str, List[str]] = {}
    for spec in RECOGNITION_REGISTRY.values():
        by_file.setdefault(spec.filename, []).append(spec.name)
    rows = []
    for spec in RECOGNITION_REGISTRY.values():
        twins = [n for n in by_file[spec.filename] if n != spec.name]
        rows.append({
            "name": spec.name,
            "display_name": spec.display_name,
            "input": "%dx%d" % tuple(spec.input_size),
            "color_space": spec.color_space,
            "output_dim": spec.output_dim,
            "quality_score": spec.extract_quality_score,
            "downloaded": os.path.isfile(os.path.join(models_dir, spec.filename)),
            "same_file_as": twins,
        })
    return rows


def provider_options() -> List[Dict[str, Any]]:
    """The provider selector: each strategy, and whether this onnxruntime can offer it."""
    try:
        import onnxruntime as ort
        have = set(ort.get_available_providers())
    except Exception as exc:
        _swallowed("roop/ui_recognition.py:provider_options", exc, "no onnxruntime providers listed")
        have = set()
    options = []
    for value, (label, ep) in _PROVIDERS.items():
        available = ep is None or ep in have
        options.append({"value": value, "label": label, "available": available,
                        "reason": "" if available else "%s is not in this onnxruntime build" % ep})
    return options


# ----------------------------------------------------------------- selection / apply

def normalise_selection(model: Any, provider: Any) -> Tuple[str, str]:
    """Validate a (model, provider) pair; ValueError names the valid choices."""
    model = str(model or DEFAULT_MODEL)
    provider = str(provider or FOLLOW_APP).lower()
    get_model_spec(model)                                    # ValueError listing valid models
    if provider not in _PROVIDERS:
        raise ValueError("Unsupported provider '%s'. Valid options: %s" % (provider, sorted(_PROVIDERS)))
    return model, provider


def saved_selection(cfg: Any) -> Tuple[str, str]:
    """The persisted choice, with anything invalid (hand-edited config.yaml) reset to defaults."""
    try:
        return normalise_selection(getattr(cfg, "recognition_model", DEFAULT_MODEL),
                                   getattr(cfg, "recognition_provider", FOLLOW_APP))
    except ValueError as exc:
        _swallowed("roop/ui_recognition.py:saved_selection", exc, "invalid saved choice; defaults used")
        return DEFAULT_MODEL, FOLLOW_APP


def active_engine_info() -> Optional[Dict[str, Any]]:
    """What is loaded right now (None if nothing has been built yet)."""
    from roop import face_analyser
    if face_analyser.recognition_model_name() is None:
        return None
    return face_analyser.get_recognition_engine().describe()


def apply_selection(model: Any, provider: Any, models_dir: str,
                    gpu_id: Optional[int] = None) -> Dict[str, Any]:
    """Validate, (re)build the engine, and return the signals the UI shows.

    May download the model on first use (up to ~260 MB) and may spend minutes building a
    TensorRT engine on a cold cache. A failed build raises and leaves the previous engine
    serving (face_analyser.set_recognition_model builds before it publishes).
    """
    from roop import face_analyser
    model, provider = normalise_selection(model, provider)
    previous = (face_analyser.recognition_model_name(),
                (active_engine_info() or {}).get("requested_device"))
    downloaded_before = {r["name"]: r["downloaded"] for r in model_catalog(models_dir)}[model]
    engine = face_analyser.set_recognition_model(
        model, None if provider == FOLLOW_APP else provider, gpu_id, models_dir)
    info = engine.describe()
    note = "Using %s on %s." % (get_model_spec(model).display_name,
                                (info["active_providers"] or ["CPUExecutionProvider"])[0].replace("ExecutionProvider", ""))
    if info["degraded"]:
        note += " WARNING: a GPU provider was requested but the session is CPU-only (%s)." % "; ".join(info["fallback_log"])
    return {
        "model": model,
        "provider": provider,
        "changed": previous != (model, info["requested_device"]),
        "reinitialized": True,
        "downloaded": not downloaded_before,
        "degraded": info["degraded"],
        "active": info,
        "message": note,
        "scope": SCOPE_NOTE,
    }


def status(cfg: Any, models_dir: str, device_id: int = 0) -> Dict[str, Any]:
    """Everything the panel renders, in one call."""
    model, provider = saved_selection(cfg)
    return {
        "selection": {"model": model, "provider": provider},
        "advice": hardware_advice(device_id).as_dict(),
        "models": model_catalog(models_dir),
        "providers": provider_options(),
        "active": active_engine_info(),
        "scope": SCOPE_NOTE,
    }
