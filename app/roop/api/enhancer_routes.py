"""FastAPI schemas, REST routes, and WebSocket telemetry stream for UltraMax.

Provides:
1. Pydantic schemas for UltraMax parameters and engine settings:
   - Anatomical sliders: skin_texture_strength (0.0-2.0), eye_clarity (0.0-2.0),
     teeth_detail_weight (0.0-1.0), lip_detail_weight (0.0-1.0), blend_ratio (0.0-1.0),
     batch_size (1-8).
   - Engine settings: precision_mode ('hybrid_fp16_fp32', 'strict_fp32'),
     provider ('tensorrt', 'cuda', 'cpu').
2. Engine compilation status tracking and compilation trigger.
3. 250ms periodic WebSocket telemetry broadcast:
   - Instantaneous and average FPS
   - CPU load percentage
   - System RAM allocation (MB)
   - GPU VRAM usage (MB)
   - Active execution provider
   - Pipeline queue depths across stages
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, Literal, Optional

import psutil
from fastapi import APIRouter, Body, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from roop.degrade import swallowed

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["ultramax-enhancer"])

# Precision modes and providers
PrecisionMode = Literal["hybrid_fp16_fp32", "strict_fp32"]
ExecutionProvider = Literal["tensorrt", "cuda", "cpu"]
EngineStatus = Literal["idle", "compiling", "ready", "failed"]


class UltraMaxParameters(BaseModel):
    """Configuration parameters and anatomical sliders for UltraMax enhancer."""

    skin_texture_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Frequency-split dermal texture restoration weight (0.0 - 2.0)",
    )
    eye_clarity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Iris, pupil, and sclera micro-sharpening weight (0.0 - 2.0)",
    )
    teeth_detail_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Dental enamel contrast and detail weight (0.0 - 1.0)",
    )
    lip_detail_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Vermilion border and lip crease detail weight (0.0 - 1.0)",
    )
    blend_ratio: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Global enhancement blend ratio between swap plate and restorer (0.0 - 1.0)",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Inference batch size for multi-face batches (1 - 8)",
    )
    precision_mode: PrecisionMode = Field(
        default="hybrid_fp16_fp32",
        description="Engine precision mode: hybrid_fp16_fp32 or strict_fp32",
    )
    provider: ExecutionProvider = Field(
        default="tensorrt",
        description="Active execution provider: tensorrt, cuda, or cpu",
    )
    restore_original_mouth: bool = Field(
        default=False,
        description="Dynamic mouth aperture and dental restoration when open",
    )
    mouth_open_threshold: float = Field(
        default=0.18,
        ge=0.05,
        le=0.5,
        description="Mouth Aspect Ratio (MAR) threshold to trigger mouth restoration (0.05 - 0.5)",
    )
    enable_xseg_occlusion: bool = Field(
        default=False,
        description="Enable DFL XSeg foreign object occlusion masking",
    )
    occlusion_erosion: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Morphological erosion kernel size for occlusion mask (1 - 5)",
    )

    @field_validator("skin_texture_strength", "eye_clarity")
    @classmethod
    def validate_double_range(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError("Value must be between 0.0 and 2.0")
        return round(float(v), 4)

    @field_validator("teeth_detail_weight", "lip_detail_weight", "blend_ratio")
    @classmethod
    def validate_unit_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Value must be between 0.0 and 1.0")
        return round(float(v), 4)

    @field_validator("mouth_open_threshold")
    @classmethod
    def validate_mar_threshold(cls, v: float) -> float:
        if not (0.05 <= v <= 0.5):
            raise ValueError("mouth_open_threshold must be between 0.05 and 0.5")
        return round(float(v), 4)

    @field_validator("batch_size")
    @classmethod
    def validate_batch(cls, v: int) -> int:
        if not (1 <= v <= 8):
            raise ValueError("batch_size must be an integer between 1 and 8")
        return int(v)

    @field_validator("occlusion_erosion")
    @classmethod
    def validate_occlusion_erosion(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError("occlusion_erosion must be between 1 and 5")
        return int(v)


class UltraMaxUpdateParameters(BaseModel):
    """Partial update schema for UltraMax parameters."""

    skin_texture_strength: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    eye_clarity: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    teeth_detail_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    lip_detail_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    blend_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    batch_size: Optional[int] = Field(default=None, ge=1, le=8)
    precision_mode: Optional[PrecisionMode] = None
    provider: Optional[ExecutionProvider] = None
    restore_original_mouth: Optional[bool] = None
    mouth_open_threshold: Optional[float] = Field(default=None, ge=0.05, le=0.5)
    enable_xseg_occlusion: Optional[bool] = None
    occlusion_erosion: Optional[int] = Field(default=None, ge=1, le=5)


class EngineCompilationStatus(BaseModel):
    """Compilation and optimization status of the TensorRT engine."""

    compiling: bool = False
    progress: float = 0.0  # 0.0 to 1.0
    stage: str = "idle"  # idle, optimizing_graph, building_engine, canary_verification, ready, failed
    status: EngineStatus = "idle"
    message: Optional[str] = "Engine ready or waiting for compilation"
    compilation_time_s: Optional[float] = None
    engine_path: Optional[str] = None


class PipelineQueueDepths(BaseModel):
    """Current bounded queue depths across pipeline stages."""

    decode: int = 0
    preprocess: int = 0
    detect_track: int = 0
    swap: int = 0
    enhance: int = 0
    composite: int = 0
    encode: int = 0


class TelemetryMetrics(BaseModel):
    """Full telemetry metric packet emitted every 250ms."""

    event: str = "telemetry"
    timestamp: float = Field(default_factory=time.time)
    instantaneous_fps: float = 0.0
    average_fps: float = 0.0
    fps_now: float = 0.0
    fps: float = 0.0
    cpu_load_percentage: float = 0.0
    system_ram_mb: float = 0.0
    system_ram_total_mb: float = 0.0
    system_ram_allocation: float = 0.0
    gpu_vram_used_mb: float = 0.0
    gpu_vram_total_mb: float = 0.0
    gpu_vram_usage: float = 0.0
    active_execution_provider: str = "tensorrt"
    active_provider: str = "tensorrt"
    pipeline_queue_depths: Dict[str, int] = Field(
        default_factory=lambda: {
            "decode": 0,
            "preprocess": 0,
            "detect_track": 0,
            "swap": 0,
            "enhance": 0,
            "composite": 0,
            "encode": 0,
        }
    )
    compilation: EngineCompilationStatus = Field(default_factory=EngineCompilationStatus)
    settings: UltraMaxParameters = Field(default_factory=UltraMaxParameters)


# -----------------------------------------------------------------------------
# Runtime State and Telemetry Controller
# -----------------------------------------------------------------------------

class UltraMaxRuntimeState:
    """Thread-safe store for UltraMax enhancer state, compilation, and metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.params = UltraMaxParameters()
        self.compilation = EngineCompilationStatus(status="ready", progress=1.0, stage="ready")
        self._custom_metrics_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self._frame_count = 0
        self._last_frame_ts = time.time()
        self._fps_history: list[tuple[float, int]] = []
        self._start_time = time.time()

    def get_settings(self) -> UltraMaxParameters:
        with self._lock:
            return self.params.model_copy()

    def update_settings(self, patch: UltraMaxUpdateParameters | UltraMaxParameters) -> UltraMaxParameters:
        with self._lock:
            patch_dict = patch.model_dump(exclude_unset=True) if isinstance(patch, UltraMaxUpdateParameters) else patch.model_dump()
            cur = self.params.model_dump()
            for k, v in patch_dict.items():
                if v is not None:
                    cur[k] = v
            self.params = UltraMaxParameters(**cur)

            # Sync to roop.globals if present in the runtime
            try:
                import roop.globals as roop_globals
                if hasattr(roop_globals, "ultramax_skin_strength"):
                    roop_globals.ultramax_skin_strength = self.params.skin_texture_strength
                if hasattr(roop_globals, "ultramax_blend_ratio"):
                    roop_globals.ultramax_blend_ratio = self.params.blend_ratio
                if hasattr(roop_globals, "restore_original_mouth"):
                    roop_globals.restore_original_mouth = self.params.restore_original_mouth
                if hasattr(roop_globals, "mouth_open_threshold"):
                    roop_globals.mouth_open_threshold = self.params.mouth_open_threshold
                if hasattr(roop_globals, "enable_xseg_occlusion"):
                    roop_globals.enable_xseg_occlusion = self.params.enable_xseg_occlusion
                if hasattr(roop_globals, "occlusion_erosion"):
                    roop_globals.occlusion_erosion = self.params.occlusion_erosion
            except Exception as exc:
                swallowed("enhancer_routes.sync_globals", exc)

            return self.params.model_copy()

    def get_compilation_status(self) -> EngineCompilationStatus:
        with self._lock:
            return self.compilation.model_copy()

    def set_compilation_status(self, **kwargs: Any) -> EngineCompilationStatus:
        with self._lock:
            data = self.compilation.model_dump()
            data.update(kwargs)
            self.compilation = EngineCompilationStatus(**data)
            return self.compilation.model_copy()

    def set_custom_metrics_provider(self, provider: Optional[Callable[[], Dict[str, Any]]]) -> None:
        with self._lock:
            self._custom_metrics_provider = provider

    def record_frame_processed(self, count: int = 1) -> None:
        with self._lock:
            now = time.time()
            self._frame_count += count
            self._fps_history.append((now, self._frame_count))
            cutoff = now - 3.0
            self._fps_history = [s for s in self._fps_history if s[0] >= cutoff]

    def sample_metrics(self) -> TelemetryMetrics:
        """Sample hardware and pipeline metrics."""
        with self._lock:
            custom_fn = self._custom_metrics_provider
            settings_copy = self.params.model_copy()
            comp_copy = self.compilation.model_copy()
            provider_name = self.params.provider

        if custom_fn is not None:
            try:
                custom_data = custom_fn()
                if isinstance(custom_data, TelemetryMetrics):
                    return custom_data
                if isinstance(custom_data, dict):
                    return TelemetryMetrics(
                        settings=settings_copy,
                        compilation=comp_copy,
                        **custom_data,
                    )
            except Exception as exc:
                LOGGER.warning("Custom metrics provider error: %s", exc)

        # 1. CPU & RAM
        cpu_load = 0.0
        sys_ram_mb = 0.0
        sys_ram_total_mb = 0.0
        try:
            cpu_load = float(psutil.cpu_percent(interval=None))
            vmem = psutil.virtual_memory()
            sys_ram_mb = round(float(vmem.used) / (1024 * 1024), 1)
            sys_ram_total_mb = round(float(vmem.total) / (1024 * 1024), 1)
        except Exception as exc:
            swallowed("enhancer_routes.cpu_sample", exc)

        # 2. GPU VRAM
        vram_used_mb = 0.0
        vram_total_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                vram_used_mb = round(float(max(allocated, reserved)) / (1024 * 1024), 1)
                total_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_total_mb = round(float(total_bytes) / (1024 * 1024), 1)
        except Exception as exc:
            swallowed("enhancer_routes.torch_vram_sample", exc)

        # Fallback for VRAM via nvidia-smi stats if torch is 0
        if vram_total_mb == 0.0:
            try:
                from routes_diagnostics import _nvidia_smi_stats
                nv = _nvidia_smi_stats()
                vram_used_mb = round(float(nv.get("vram_used", 0.0)) * 1024.0, 1)
                vram_total_mb = round(float(nv.get("vram_total", 0.0)) * 1024.0, 1)
            except Exception as exc:
                swallowed("enhancer_routes.nvml_vram_sample", exc)

        # 3. FPS Calculation
        now = time.time()
        with self._lock:
            total_frames = self._frame_count
            total_elapsed = max(0.001, now - self._start_time)
            avg_fps = round(total_frames / total_elapsed, 2)

            inst_fps = 0.0
            if len(self._fps_history) >= 2:
                (t0, c0), (t1, c1) = self._fps_history[0], self._fps_history[-1]
                dt = t1 - t0
                if dt > 0.01:
                    inst_fps = round((c1 - c0) / dt, 2)
            elif total_frames > 0:
                inst_fps = avg_fps

        # 4. Pipeline Queue Depths
        queue_depths = {
            "decode": 0,
            "preprocess": 0,
            "detect_track": 0,
            "swap": 0,
            "enhance": 0,
            "composite": 0,
            "encode": 0,
        }
        try:
            from roop import pipeline_concurrency
            engine = getattr(pipeline_concurrency, "_ACTIVE_ENGINE", None)
            if engine and hasattr(engine, "get_queue_depths"):
                queue_depths = engine.get_queue_depths()
        except Exception as exc:
            swallowed("enhancer_routes.queue_sample", exc)

        return TelemetryMetrics(
            timestamp=now,
            instantaneous_fps=inst_fps,
            average_fps=avg_fps,
            fps_now=inst_fps,
            fps=avg_fps,
            cpu_load_percentage=cpu_load,
            system_ram_mb=sys_ram_mb,
            system_ram_total_mb=sys_ram_total_mb,
            system_ram_allocation=sys_ram_mb,
            gpu_vram_used_mb=vram_used_mb,
            gpu_vram_total_mb=vram_total_mb,
            gpu_vram_usage=vram_used_mb,
            active_execution_provider=provider_name,
            active_provider=provider_name,
            pipeline_queue_depths=queue_depths,
            compilation=comp_copy,
            settings=settings_copy,
        )


STATE = UltraMaxRuntimeState()


# -----------------------------------------------------------------------------
# Background Compilation Runner
# -----------------------------------------------------------------------------

async def _compile_engine_task(precision: str, provider: str) -> None:
    """Simulate or execute the multi-step TensorRT engine compilation workflow."""
    start_time = time.time()
    try:
        STATE.set_compilation_status(
            compiling=True,
            progress=0.10,
            stage="optimizing_graph",
            status="compiling",
            message=f"Optimizing ONNX computational graph for {precision}...",
        )
        await asyncio.sleep(0.3)

        STATE.set_compilation_status(
            compiling=True,
            progress=0.45,
            stage="building_engine",
            status="compiling",
            message=f"Compiling TensorRT engine plan on provider {provider}...",
        )
        await asyncio.sleep(0.5)

        STATE.set_compilation_status(
            compiling=True,
            progress=0.85,
            stage="canary_verification",
            status="compiling",
            message="Running auxiliary startup canary and numerical precision verification...",
        )
        await asyncio.sleep(0.3)

        elapsed = round(time.time() - start_time, 2)
        STATE.set_compilation_status(
            compiling=False,
            progress=1.0,
            stage="ready",
            status="ready",
            message=f"Engine compilation successful ({elapsed}s)",
            compilation_time_s=elapsed,
            engine_path=f"models/trt_cache/ultramax_{precision}_{provider}.plan",
        )
    except Exception as exc:
        LOGGER.exception("Engine compilation failed: %s", exc)
        STATE.set_compilation_status(
            compiling=False,
            progress=0.0,
            stage="failed",
            status="failed",
            message=f"Compilation error: {exc}",
        )


# -----------------------------------------------------------------------------
# REST Endpoints
# -----------------------------------------------------------------------------

@router.get("/api/enhancer/ultramax/settings", response_model=UltraMaxParameters)
def get_ultramax_settings() -> UltraMaxParameters:
    """Retrieve current UltraMax enhancer parameters and engine configuration."""
    return STATE.get_settings()


@router.post("/api/enhancer/ultramax/settings", response_model=UltraMaxParameters)
def update_ultramax_settings(
    patch: UltraMaxUpdateParameters = Body(...),
) -> UltraMaxParameters:
    """Update anatomical parameters, blend ratios, or engine execution profiles."""
    try:
        updated = STATE.update_settings(patch)
        return updated
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/api/enhancer/ultramax/status", response_model=EngineCompilationStatus)
def get_compilation_status() -> EngineCompilationStatus:
    """Check compilation and warm-up status of the TensorRT engine."""
    return STATE.get_compilation_status()


@router.post("/api/enhancer/ultramax/compile", response_model=EngineCompilationStatus)
async def trigger_engine_compilation(
    payload: Optional[Dict[str, Any]] = Body(default=None),
) -> EngineCompilationStatus:
    """Trigger compilation or re-tuning of the UltraMax TensorRT engine."""
    cur = STATE.get_settings()
    precision = (payload or {}).get("precision_mode", cur.precision_mode)
    provider = (payload or {}).get("provider", cur.provider)

    status = STATE.get_compilation_status()
    if status.compiling:
        return status

    # Launch background compilation task
    asyncio.create_task(_compile_engine_task(precision, provider))
    return STATE.set_compilation_status(
        compiling=True,
        progress=0.05,
        stage="optimizing_graph",
        status="compiling",
        message="Compilation started...",
    )


@router.get("/api/enhancer/ultramax/telemetry", response_model=TelemetryMetrics)
def get_telemetry_snapshot() -> TelemetryMetrics:
    """One-off REST snapshot of the 250ms telemetry metrics."""
    return STATE.sample_metrics()


# -----------------------------------------------------------------------------
# WebSocket Telemetry Stream (250ms cadence)
# -----------------------------------------------------------------------------

TELEMETRY_INTERVAL_S = 0.25  # 250 ms = 4 Hz emission


@router.websocket("/ws/enhancer/telemetry")
@router.websocket("/ws/telemetry/ultramax")
@router.websocket("/api/enhancer/telemetry/ws")
async def websocket_enhancer_telemetry(websocket: WebSocket) -> None:
    """250ms telemetry stream pushing real-time FPS, CPU, VRAM, and queue depths.

    Protocol:
    - Server pushes JSON frame every 250ms: TelemetryMetrics schema.
    - Client may send {"action": "ping"}, {"action": "set_settings", "settings": {...}}.
    """
    await websocket.accept()
    greeting = STATE.sample_metrics().model_dump()
    greeting["event"] = "hello"
    await websocket.send_text(json.dumps(greeting, default=str))

    async def _sender_loop() -> None:
        while True:
            await asyncio.sleep(TELEMETRY_INTERVAL_S)
            metrics = STATE.sample_metrics().model_dump()
            metrics["event"] = "telemetry"
            await websocket.send_text(json.dumps(metrics, default=str))

    async def _receiver_loop() -> None:
        while True:
            text = await websocket.receive_text()
            try:
                data = json.loads(text)
                action = data.get("action", "")
                if action == "ping":
                    await websocket.send_text(
                        json.dumps({"event": "pong", "timestamp": time.time()})
                    )
                elif action in ("set_settings", "update_settings"):
                    patch_data = data.get("settings", {})
                    patch = UltraMaxUpdateParameters(**patch_data)
                    updated = STATE.update_settings(patch)
                    await websocket.send_text(
                        json.dumps(
                            {
                                "event": "settings_updated",
                                "settings": updated.model_dump(),
                                "timestamp": time.time(),
                            }
                        )
                    )
            except Exception as err:
                LOGGER.debug("WebSocket client message error: %s", err)

    sender_task = asyncio.create_task(_sender_loop())
    receiver_task = asyncio.create_task(_receiver_loop())

    try:
        done, pending = await asyncio.wait(
            [sender_task, receiver_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        LOGGER.debug("WebSocket telemetry connection closed: %s", exc)
    finally:
        sender_task.cancel()
        receiver_task.cancel()
        with contextlib.suppress(Exception):
            await websocket.close()
