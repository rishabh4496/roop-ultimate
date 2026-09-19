"""Transactional startup state machine for Roop Ultimate.

Enforces deterministic, structured startup across 10 sequential phases:
1. BOOT
2. DEPENDENCY_PREFLIGHT
3. DLL_RUNTIME_PREFLIGHT
4. ORT_PREFLIGHT
5. GPU_PREFLIGHT
6. PROVIDER_ADMISSION
7. CONFIG_LOAD
8. MODEL_RUNTIME_INIT
9. API_READY
10. UI_READY

Each phase must return SUCCESS, DEGRADED, or FATAL.
Any FATAL result immediately halts startup, prints structured diagnostic output,
and prevents subsequent phases from executing.
"""
from __future__ import annotations

import enum
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


class StartupPhase(str, enum.Enum):
    BOOT = "BOOT"
    DEPENDENCY_PREFLIGHT = "DEPENDENCY_PREFLIGHT"
    DLL_RUNTIME_PREFLIGHT = "DLL_RUNTIME_PREFLIGHT"
    ORT_PREFLIGHT = "ORT_PREFLIGHT"
    GPU_PREFLIGHT = "GPU_PREFLIGHT"
    PROVIDER_ADMISSION = "PROVIDER_ADMISSION"
    CONFIG_LOAD = "CONFIG_LOAD"
    MODEL_RUNTIME_INIT = "MODEL_RUNTIME_INIT"
    API_READY = "API_READY"
    UI_READY = "UI_READY"


class PhaseStatus(str, enum.Enum):
    SUCCESS = "SUCCESS"
    DEGRADED = "DEGRADED"
    FATAL = "FATAL"


PHASE_SEQUENCE: Tuple[StartupPhase, ...] = (
    StartupPhase.BOOT,
    StartupPhase.DEPENDENCY_PREFLIGHT,
    StartupPhase.DLL_RUNTIME_PREFLIGHT,
    StartupPhase.ORT_PREFLIGHT,
    StartupPhase.GPU_PREFLIGHT,
    StartupPhase.PROVIDER_ADMISSION,
    StartupPhase.CONFIG_LOAD,
    StartupPhase.MODEL_RUNTIME_INIT,
    StartupPhase.API_READY,
    StartupPhase.UI_READY,
)


@dataclass(frozen=True)
class PhaseResult:
    phase: StartupPhase
    status: PhaseStatus
    component: str
    reason: Optional[str] = None
    detected_version: Optional[str] = None
    expected_version: Optional[str] = None
    next_action: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "status": self.status.value,
            "component": self.component,
            "reason": self.reason,
            "detected_version": self.detected_version,
            "expected_version": self.expected_version,
            "next_action": self.next_action,
            "details": self.details,
        }


def format_fatal_diagnostic(
    stage: str | StartupPhase,
    component: str,
    reason: str,
    detected_version: str | None = None,
    expected_version: str | None = None,
    next_action: str | None = None,
) -> str:
    """Format structured fatal diagnostic output per prompt specification."""
    phase_str = stage.value if isinstance(stage, StartupPhase) else str(stage)
    det_ver = str(detected_version or "none")
    exp_ver = str(expected_version or "none")
    nxt_act = str(next_action or "Check installation logs or run reset.js")
    return (
        f"[Startup:FATAL]\n"
        f"stage={phase_str}\n"
        f"component={component}\n"
        f"reason={reason}\n"
        f"detected_version={det_ver}\n"
        f"expected_version={exp_ver}\n"
        f"next_action={nxt_act}"
    )


def has_nvidia_hardware() -> bool:
    """Return whether this machine has physical NVIDIA GPU hardware."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            name = torch.cuda.get_device_name(0).lower()
            if any(k in name for k in ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla")):
                return True
    except Exception:
        pass

    # Check via nvidia-smi command if available
    import shutil
    import subprocess
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            res = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0 and "GPU" in res.stdout:
                return True
        except Exception:
            pass

    # Check Windows driver DLL
    if sys.platform == "win32":
        system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
        if os.path.exists(os.path.join(system32, "nvcuda.dll")):
            return True

    return False


class StartupStateMachine:
    """State machine governing the 10 transactional startup phases."""

    def __init__(self, halt_on_fatal: bool = True):
        self._halt_on_fatal = halt_on_fatal
        self._results: Dict[StartupPhase, PhaseResult] = {}
        self._current_phase: Optional[StartupPhase] = None
        self._fatal_encountered: bool = False
        self._start_time: float = time.monotonic()

    @property
    def current_phase(self) -> Optional[StartupPhase]:
        return self._current_phase

    @property
    def results(self) -> Dict[StartupPhase, PhaseResult]:
        return dict(self._results)

    @property
    def is_failed(self) -> bool:
        return self._fatal_encountered

    def get_result(self, phase: StartupPhase) -> Optional[PhaseResult]:
        return self._results.get(phase)

    def reset(self) -> None:
        """Reset state machine to initial unstarted state."""
        self._results.clear()
        self._current_phase = None
        self._fatal_encountered = False
        self._start_time = time.monotonic()

    def can_transition_to(self, target: StartupPhase) -> bool:
        """A phase can only execute if no FATAL has occurred and prior phases succeeded or degraded."""
        idx = PHASE_SEQUENCE.index(target)
        if idx == 0:
            self.reset()
            return True
        if self._fatal_encountered:
            return False
        for prev in PHASE_SEQUENCE[:idx]:
            res = self._results.get(prev)
            if res is None or res.status == PhaseStatus.FATAL:
                return False
        return True

    def record(self, result: PhaseResult) -> PhaseResult:
        """Record phase outcome and enforce transactional halt on FATAL."""
        self._results[result.phase] = result
        self._current_phase = result.phase

        if result.status == PhaseStatus.FATAL:
            self._fatal_encountered = True
            msg = format_fatal_diagnostic(
                stage=result.phase,
                component=result.component,
                reason=result.reason or "Fatal startup failure",
                detected_version=result.detected_version,
                expected_version=result.expected_version,
                next_action=result.next_action,
            )
            print(msg, file=sys.stderr, flush=True)
            if self._halt_on_fatal:
                sys.exit(f"{result.reason or 'Fatal startup failure'}")
        elif result.status == PhaseStatus.DEGRADED:
            print(
                f"[Startup:DEGRADED] stage={result.phase.value} "
                f"component={result.component} "
                f"reason={result.reason or 'Operational with degraded capabilities'}",
                flush=True,
            )
        else:
            print(
                f"[Startup:OK] stage={result.phase.value} component={result.component}",
                flush=True,
            )

        return result

    def record_fatal(
        self,
        phase: StartupPhase,
        component: str,
        reason: str,
        detected_version: str | None = None,
        expected_version: str | None = None,
        next_action: str | None = None,
        details: Dict[str, Any] | None = None,
    ) -> PhaseResult:
        res = PhaseResult(
            phase=phase,
            status=PhaseStatus.FATAL,
            component=component,
            reason=reason,
            detected_version=detected_version,
            expected_version=expected_version,
            next_action=next_action,
            details=details or {},
        )
        return self.record(res)

    def record_degraded(
        self,
        phase: StartupPhase,
        component: str,
        reason: str,
        details: Dict[str, Any] | None = None,
    ) -> PhaseResult:
        res = PhaseResult(
            phase=phase,
            status=PhaseStatus.DEGRADED,
            component=component,
            reason=reason,
            details=details or {},
        )
        return self.record(res)

    def record_success(
        self,
        phase: StartupPhase,
        component: str,
        details: Dict[str, Any] | None = None,
    ) -> PhaseResult:
        res = PhaseResult(
            phase=phase,
            status=PhaseStatus.SUCCESS,
            component=component,
            details=details or {},
        )
        return self.record(res)

    def execute_phase(self, phase: StartupPhase, fn: Callable[..., PhaseResult], *args, **kwargs) -> PhaseResult:
        """Run a phase function if permissible, halting on fatal."""
        if not self.can_transition_to(phase):
            return self.record_fatal(
                phase=phase,
                component=phase.value.lower(),
                reason=f"Cannot transition to {phase.value}; previous phase failed or incomplete",
                next_action="Resolve fatal failures in earlier startup phases",
            )
        try:
            res = fn(*args, **kwargs)
            return self.record(res)
        except SystemExit:
            raise
        except Exception as exc:
            return self.record_fatal(
                phase=phase,
                component=phase.value.lower(),
                reason=f"Unhandled exception during {phase.value}: {exc}",
                next_action="Check full stack trace and repair environment",
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "current_phase": self._current_phase.value if self._current_phase else None,
            "fatal": self._fatal_encountered,
            "elapsed_seconds": round(time.monotonic() - self._start_time, 3),
            "phases": {p.value: r.as_dict() for p, r in self._results.items()},
        }


# Global singleton instance for runtime access
_GLOBAL_STATE_MACHINE = StartupStateMachine()


def get_startup_state_machine() -> StartupStateMachine:
    return _GLOBAL_STATE_MACHINE


# ── Concrete Phase Execution Implementations ───────────────────────────────

def execute_boot() -> PhaseResult:
    """Phase 1: BOOT - Initialize process environment, UTF-8 streams, and priority."""
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    try:
        from roop import keep_awake
        keep_awake.boost_process_priority()
    except Exception:
        pass

    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    os.environ["AV_LOG_LEVEL"] = "error"

    return PhaseResult(
        phase=StartupPhase.BOOT,
        status=PhaseStatus.SUCCESS,
        component="process_bootstrap",
        details={"platform": sys.platform, "python_version": sys.version},
    )


def execute_dependency_preflight() -> PhaseResult:
    """Phase 2: DEPENDENCY_PREFLIGHT - Verify core Python libraries and ABI contracts."""
    import numpy as np

    np_version = getattr(np, "__version__", "unknown")
    try:
        np_major = int(np_version.split(".")[0])
        if np_major >= 2:
            return PhaseResult(
                phase=StartupPhase.DEPENDENCY_PREFLIGHT,
                status=PhaseStatus.FATAL,
                component="numpy",
                reason="InsightFace C-bindings require numpy<2.0.0",
                detected_version=np_version,
                expected_version="<2.0.0 (e.g. 1.26.4)",
                next_action='Run: uv pip install "numpy<2.0.0" or reinstall via reset.js',
            )
    except Exception as exc:
        return PhaseResult(
            phase=StartupPhase.DEPENDENCY_PREFLIGHT,
            status=PhaseStatus.FATAL,
            component="numpy",
            reason=f"Failed to inspect numpy version: {exc}",
            detected_version=np_version,
            expected_version="<2.0.0",
            next_action='Run: uv pip install "numpy<2.0.0"',
        )

    for pkg_name in ("yaml", "cv2", "PIL"):
        try:
            __import__(pkg_name)
        except ImportError as exc:
            return PhaseResult(
                phase=StartupPhase.DEPENDENCY_PREFLIGHT,
                status=PhaseStatus.FATAL,
                component=pkg_name,
                reason=f"Required package {pkg_name} is missing: {exc}",
                detected_version="missing",
                expected_version="installed",
                next_action="Run: uv pip install -r requirements.txt",
            )

    nvidia_present = has_nvidia_hardware()
    try:
        import torch
        torch_ver = getattr(torch, "__version__", "unknown")
        if nvidia_present and not torch.cuda.is_available():
            return PhaseResult(
                phase=StartupPhase.DEPENDENCY_PREFLIGHT,
                status=PhaseStatus.FATAL,
                component="torch",
                reason="NVIDIA hardware detected but PyTorch has no CUDA support",
                detected_version=torch_ver,
                expected_version="torch>=2.4.0+cu124",
                next_action="Run: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
            )
    except ImportError as exc:
        if nvidia_present:
            return PhaseResult(
                phase=StartupPhase.DEPENDENCY_PREFLIGHT,
                status=PhaseStatus.FATAL,
                component="torch",
                reason=f"PyTorch is missing on this NVIDIA installation: {exc}",
                detected_version="missing",
                expected_version="torch>=2.4.0+cu124",
                next_action="Run: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
            )

    return PhaseResult(
        phase=StartupPhase.DEPENDENCY_PREFLIGHT,
        status=PhaseStatus.SUCCESS,
        component="dependencies",
        detected_version=np_version,
        expected_version="<2.0.0",
    )


def execute_dll_runtime_preflight() -> PhaseResult:
    """Phase 3: DLL_RUNTIME_PREFLIGHT - Register and verify Windows native runtime libraries."""
    if sys.platform == "win32":
        try:
            import roop.gpu_preflight as gp
            reg_fn = getattr(gp, "register_gpu_runtime_dirs", None)
            dirs = reg_fn() if reg_fn else []
            check_fn = getattr(gp, "_check_tensorrt_dlls", None)
            trt_ok = check_fn(dirs) if check_fn else True
            if has_nvidia_hardware() and not trt_ok:
                return PhaseResult(
                    phase=StartupPhase.DLL_RUNTIME_PREFLIGHT,
                    status=PhaseStatus.DEGRADED,
                    component="tensorrt_dlls",
                    reason="TensorRT runtime DLLs (nvinfer.dll) were not found in registered directories or PATH; CUDA fallback will be used if needed",
                    detected_version="none",
                    expected_version="TensorRT 10.x runtime libraries",
                    next_action="Install TensorRT runtime libraries via Pinokio or pip install tensorrt",
                    details={"registered_dirs": dirs},
                )
        except Exception:
            pass

    return PhaseResult(
        phase=StartupPhase.DLL_RUNTIME_PREFLIGHT,
        status=PhaseStatus.SUCCESS,
        component="runtime_dlls",
    )


def execute_ort_preflight() -> PhaseResult:
    """Phase 4: ORT_PREFLIGHT - Authoritative ONNX Runtime probe and session validation."""
    try:
        from roop.gpu_preflight import get_preflight_result
        result = get_preflight_result()
    except Exception as exc:
        return PhaseResult(
            phase=StartupPhase.ORT_PREFLIGHT,
            status=PhaseStatus.FATAL,
            component="onnxruntime",
            reason=f"Authoritative ONNX Runtime preflight probe failed: {type(exc).__name__}: {exc}",
            next_action="Reinstall onnxruntime-gpu via Pinokio or uv pip",
        )

    if not result.get("onnxruntime_importable"):
        stage = result.get("failure_stage") or "package_missing"
        reason = result.get("failure_reason") or "ONNX Runtime import failed"
        return PhaseResult(
            phase=StartupPhase.ORT_PREFLIGHT,
            status=PhaseStatus.FATAL,
            component="onnxruntime",
            reason=f"{stage}: {reason}",
            detected_version=result.get("onnxruntime_version") or "none",
            expected_version="onnxruntime-gpu>=1.18.0",
            next_action="Repair the Pinokio environment before starting the app",
        )

    providers = list(result.get("available_providers") or [])
    if not providers:
        stage = result.get("failure_stage") or "unknown"
        reason = result.get("failure_reason") or "provider registry is empty"
        return PhaseResult(
            phase=StartupPhase.ORT_PREFLIGHT,
            status=PhaseStatus.FATAL,
            component="onnxruntime",
            reason=f"[FATAL] ONNX Runtime exposes no execution providers. stage={stage}; reason={reason}.",
            detected_version=result.get("onnxruntime_version") or "none",
            expected_version="CUDAExecutionProvider, TensorrtExecutionProvider",
            next_action="Reinstall onnxruntime-gpu with matching CUDA runtime",
        )

    if has_nvidia_hardware() and "CUDAExecutionProvider" not in providers:
        return PhaseResult(
            phase=StartupPhase.ORT_PREFLIGHT,
            status=PhaseStatus.FATAL,
            component="onnxruntime-gpu",
            reason="CUDAExecutionProvider is not compiled into installed ONNX Runtime on this NVIDIA machine",
            detected_version=result.get("onnxruntime_version") or "none",
            expected_version="onnxruntime-gpu with CUDAExecutionProvider",
            next_action="Install onnxruntime-gpu instead of onnxruntime CPU build",
        )

    active = result.get("active_provider") or "none"
    stage = result.get("failure_stage")
    reason = result.get("failure_reason")

    if active == "TensorrtExecutionProvider" and result.get("tensorrt_session_usable"):
        print("[OK] TensorRT minimal session verified as active.", flush=True)
        print(f"[Runtime] provider preflight: requested=auto active={active} available={providers}", flush=True)
        return PhaseResult(
            phase=StartupPhase.ORT_PREFLIGHT,
            status=PhaseStatus.SUCCESS,
            component="onnxruntime",
            details={"active": active, "providers": providers},
        )
    elif "CUDAExecutionProvider" in providers and active == "CUDAExecutionProvider":
        print(
            "[WARNING] TensorRT is not active; CUDA is the validated fallback. "
            f"stage={stage or 'none'}; reason={reason or 'TensorRT was not selected'}",
            flush=True,
        )
        print(f"[Runtime] provider preflight: requested=auto active={active} available={providers}", flush=True)
        return PhaseResult(
            phase=StartupPhase.ORT_PREFLIGHT,
            status=PhaseStatus.DEGRADED,
            component="onnxruntime",
            reason=reason or "TensorRT is not active; CUDA is the validated fallback",
            details={"active": active, "providers": providers},
        )
    elif active == "CPUExecutionProvider":
        print(
            "[WARNING] GPU providers are not active; CPU fallback is validated. "
            f"stage={stage or 'none'}; reason={reason or 'GPU provider unavailable'}",
            flush=True,
        )
        print(f"[Runtime] provider preflight: requested=auto active={active} available={providers}", flush=True)
        return PhaseResult(
            phase=StartupPhase.ORT_PREFLIGHT,
            status=PhaseStatus.DEGRADED,
            component="onnxruntime",
            reason=reason or "GPU providers are not active; CPU fallback is validated",
            details={"active": active, "providers": providers},
        )
    else:
        print(
            f"[WARNING] no recognized accelerated provider is active. stage={stage or 'none'}; reason={reason or 'provider chain unavailable'}",
            flush=True,
        )
        print(f"[Runtime] provider preflight: requested=auto active={active} available={providers}", flush=True)
        return PhaseResult(
            phase=StartupPhase.ORT_PREFLIGHT,
            status=PhaseStatus.FATAL,
            component="onnxruntime",
            reason=reason or f"No recognized accelerated provider is active: {active}",
            next_action="Run env/Scripts/python.exe tests/diag_device.py to diagnose hardware",
        )


def execute_gpu_preflight(device_id: int = 0) -> PhaseResult:
    """Phase 5: GPU_PREFLIGHT - Probe physical GPU VRAM, architecture, and safety tier."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > int(device_id):
            vram_gb = torch.cuda.get_device_properties(int(device_id)).total_memory / (1024 ** 3)
            gpu_name = torch.cuda.get_device_name(int(device_id))
            from roop.backend_manager import is_sub_7gb_gpu
            sub_7gb = is_sub_7gb_gpu(int(device_id))
            if sub_7gb:
                return PhaseResult(
                    phase=StartupPhase.GPU_PREFLIGHT,
                    status=PhaseStatus.DEGRADED,
                    component="gpu",
                    reason=f"Sub-7GB GPU tier detected ({gpu_name}: {vram_gb:.1f}GB); 0/0 pools enforced",
                    details={"gpu_name": gpu_name, "vram_gb": vram_gb, "sub_7gb": True},
                )
            return PhaseResult(
                phase=StartupPhase.GPU_PREFLIGHT,
                status=PhaseStatus.SUCCESS,
                component="gpu",
                details={"gpu_name": gpu_name, "vram_gb": vram_gb, "sub_7gb": False},
            )
        elif has_nvidia_hardware():
            return PhaseResult(
                phase=StartupPhase.GPU_PREFLIGHT,
                status=PhaseStatus.FATAL,
                component="torch_cuda",
                reason="NVIDIA hardware present but PyTorch cannot initialize CUDA",
                next_action="Verify NVIDIA GPU drivers and PyTorch CUDA compatibility",
            )
        else:
            return PhaseResult(
                phase=StartupPhase.GPU_PREFLIGHT,
                status=PhaseStatus.DEGRADED,
                component="gpu",
                reason="No CUDA GPU available; running CPU pipeline",
                details={"sub_7gb": False, "cpu_only": True},
            )
    except Exception as exc:
        return PhaseResult(
            phase=StartupPhase.GPU_PREFLIGHT,
            status=PhaseStatus.FATAL,
            component="gpu",
            reason=f"GPU preflight failed with exception: {exc}",
            next_action="Verify display drivers and hardware stability",
        )


def execute_provider_admission(requested: str | None = None, device_id: int = 0) -> PhaseResult:
    """Phase 6: PROVIDER_ADMISSION - Validate requested backend against hardware and policy."""
    from roop.backend_manager import canonical_provider_decision
    decision = canonical_provider_decision(requested, device_id)
    req = decision.requested
    admitted = decision.admitted
    active = decision.active.replace("ExecutionProvider", "").lower()

    if req == "auto":
        if decision.degraded and active == "cuda":
            return PhaseResult(
                phase=StartupPhase.PROVIDER_ADMISSION,
                status=PhaseStatus.DEGRADED,
                component="provider",
                reason=decision.degradation_reason or "TensorRT unavailable; AUTO provider selected CUDA fallback",
                details={"requested": req, "admitted": admitted, "active": active},
            )
        return PhaseResult(
            phase=StartupPhase.PROVIDER_ADMISSION,
            status=PhaseStatus.SUCCESS,
            component="provider",
            details={"requested": req, "admitted": admitted, "active": active},
        )

    if req == "tensorrt":
        if active != "tensorrt":
            return PhaseResult(
                phase=StartupPhase.PROVIDER_ADMISSION,
                status=PhaseStatus.DEGRADED,
                component="tensorrt",
                reason=decision.degradation_reason or f"TensorRT requested but active provider is {active}",
                details={"requested": req, "admitted": admitted, "active": active},
            )
        return PhaseResult(
            phase=StartupPhase.PROVIDER_ADMISSION,
            status=PhaseStatus.SUCCESS,
            component="provider",
            details={"requested": req, "admitted": admitted, "active": active},
        )

    if active == "cpu":
        import torch
        if req == "cpu":
            return PhaseResult(
                phase=StartupPhase.PROVIDER_ADMISSION,
                status=PhaseStatus.SUCCESS,
                component="provider",
                details={"requested": req, "admitted": admitted, "active": active},
            )
        if os.environ.get("ROOP_DISALLOW_CPU_FALLBACK") == "1":
            return PhaseResult(
                phase=StartupPhase.PROVIDER_ADMISSION,
                status=PhaseStatus.FATAL,
                component="provider",
                reason="CPU fallback is forbidden by policy for requested GPU accelerator",
                next_action="Fix GPU acceleration or select cpu explicitly",
            )
        if torch.cuda.is_available() and has_nvidia_hardware() and req in ("cuda", "tensorrt"):
            return PhaseResult(
                phase=StartupPhase.PROVIDER_ADMISSION,
                status=PhaseStatus.FATAL,
                component="provider",
                reason=f"GPU provider '{req}' requested on NVIDIA hardware fell back to CPU",
                next_action="Verify CUDA drivers and ORT GPU runtime",
            )
        return PhaseResult(
            phase=StartupPhase.PROVIDER_ADMISSION,
            status=PhaseStatus.DEGRADED,
            component="provider",
            reason=decision.degradation_reason or "CPU fallback permitted for non-accelerated environment",
            details={"requested": req, "admitted": admitted, "active": active},
        )

    if decision.degraded:
        return PhaseResult(
            phase=StartupPhase.PROVIDER_ADMISSION,
            status=PhaseStatus.DEGRADED,
            component="provider",
            reason=decision.degradation_reason or f"Provider degraded: requested {req}, active {active}",
            details={"requested": req, "admitted": admitted, "active": active},
        )

    return PhaseResult(
        phase=StartupPhase.PROVIDER_ADMISSION,
        status=PhaseStatus.SUCCESS,
        component="provider",
        details={"requested": req, "admitted": admitted, "active": active},
    )


def execute_config_load(config_path: str = "config.yaml") -> PhaseResult:
    """Phase 7: CONFIG_LOAD - Load Settings, applying defaults and provider overrides."""
    from settings import Settings
    try:
        cfg = Settings(config_path)
        return PhaseResult(
            phase=StartupPhase.CONFIG_LOAD,
            status=PhaseStatus.SUCCESS,
            component="settings",
            details={"provider": cfg.provider, "max_threads": cfg.max_threads},
        )
    except Exception as exc:
        return PhaseResult(
            phase=StartupPhase.CONFIG_LOAD,
            status=PhaseStatus.FATAL,
            component="settings",
            reason=f"Failed to load application configuration: {exc}",
            next_action="Check config.yaml syntax or restore default_config.yaml",
        )


def execute_model_runtime_init(cfg=None) -> PhaseResult:
    """Phase 8: MODEL_RUNTIME_INIT - Decode providers and initialize execution threads."""
    from roop.core import decode_execution_providers
    import roop.globals as roop_globals

    target_cfg = cfg or getattr(roop_globals, "CFG", None)
    provider_name = getattr(target_cfg, "provider", "auto") if target_cfg else "auto"

    providers = [provider_name]
    if provider_name == "tensorrt":
        providers.append("cuda")

    decoded = decode_execution_providers(providers)
    if not decoded:
        return PhaseResult(
            phase=StartupPhase.MODEL_RUNTIME_INIT,
            status=PhaseStatus.FATAL,
            component="execution_providers",
            reason="decode_execution_providers resolved an empty provider list",
            next_action="Check ONNX Runtime installation and provider registration",
        )

    roop_globals.execution_providers = decoded
    if target_cfg:
        roop_globals.execution_threads = getattr(target_cfg, "max_threads", 3)

    first = decoded[0]
    first_name = first[0] if isinstance(first, (tuple, list)) else first
    active_short = str(first_name).replace("ExecutionProvider", "").lower()

    if provider_name == "tensorrt" and active_short != "tensorrt":
        return PhaseResult(
            phase=StartupPhase.MODEL_RUNTIME_INIT,
            status=PhaseStatus.DEGRADED,
            component="model_runtime",
            reason=f"Requested TensorRT but model runtime decoded to {active_short}",
            details={"execution_providers": [str(p) for p in decoded]},
        )

    return PhaseResult(
        phase=StartupPhase.MODEL_RUNTIME_INIT,
        status=PhaseStatus.SUCCESS,
        component="model_runtime",
        details={"execution_providers": [str(p) for p in decoded]},
    )


def execute_api_ready(api_thread, api_port: int, timeout: float = 180.0) -> PhaseResult:
    """Phase 9: API_READY - Verify FastAPI loopback connectivity and readiness."""
    import socket
    import roop.globals as roop_globals

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not api_thread.is_alive():
            return PhaseResult(
                phase=StartupPhase.API_READY,
                status=PhaseStatus.FATAL,
                component="api_server",
                reason="FastAPI server thread terminated prematurely before readiness",
                next_action="Check console logs for uvicorn import or binding errors",
            )
        if getattr(roop_globals, "CFG", None) is not None:
            try:
                with socket.create_connection(("127.0.0.1", int(api_port)), timeout=0.25):
                    return PhaseResult(
                        phase=StartupPhase.API_READY,
                        status=PhaseStatus.SUCCESS,
                        component="api_server",
                        details={"port": api_port},
                    )
            except OSError:
                pass
        time.sleep(0.1)

    return PhaseResult(
        phase=StartupPhase.API_READY,
        status=PhaseStatus.FATAL,
        component="api_server",
        reason=f"API server failed to respond on port {api_port} within {timeout:.0f}s",
        next_action=f"Verify port {api_port} is not in use and firewall allows loopback connections",
    )


def execute_ui_ready(api_port: int, is_react: bool = True) -> PhaseResult:
    """Phase 10: UI_READY - Verify UI build and emit Pinokio capture URL."""
    if is_react:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent.parent
        dist_index = repo_root / "react-ui" / "dist" / "index.html"
        if not dist_index.is_file():
            return PhaseResult(
                phase=StartupPhase.UI_READY,
                status=PhaseStatus.FATAL,
                component="react-ui",
                reason=f"React production build not found at {dist_index}",
                expected_version="Built bundle in react-ui/dist",
                next_action="Run: cd react-ui && npm run build",
            )

        ready_url = f"http://127.0.0.1:{api_port}"
        print(f"[Backend] listening on {ready_url}", flush=True)
        return PhaseResult(
            phase=StartupPhase.UI_READY,
            status=PhaseStatus.SUCCESS,
            component="ui",
            details={"url": ready_url},
        )

    return PhaseResult(
        phase=StartupPhase.UI_READY,
        status=PhaseStatus.SUCCESS,
        component="ui",
    )
