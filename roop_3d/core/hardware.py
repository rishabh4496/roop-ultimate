"""Read-only hardware probes: provider advertisement is not execution proof."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
import platform
import shutil
import subprocess
from typing import Literal

from pydantic import BaseModel, Field


class GPU(BaseModel):
    index: int
    name: str
    total_vram_mib: int
    free_vram_mib: int
    driver_version: str
    state: Literal["VERIFIED_LOCAL"] = "VERIFIED_LOCAL"


class Hardware(BaseModel):
    timestamp: str
    os: dict[str, str]
    logical_cpu_count: int | None
    physical_cpu_count: int | None = None
    physical_cpu_count_state: Literal["UNKNOWN"] = "UNKNOWN"
    gpus: list[GPU] = Field(default_factory=list)
    gpu_probe_state: Literal["VERIFIED_LOCAL", "UNKNOWN"] = "UNKNOWN"
    advertised_providers: list[str] | None = None
    provider_probe_state: Literal["VERIFIED_LOCAL", "UNKNOWN"] = "UNKNOWN"
    provider_execution_verified: bool = False
    ort_version: str | None = None
    cuda_runtime_version: str | None = None
    cuda_runtime_state: Literal["UNKNOWN"] = "UNKNOWN"
    probe_notes: list[str] = Field(default_factory=list)


def probe_hardware() -> Hardware:
    result = Hardware(timestamp=datetime.now(timezone.utc).isoformat(),
                      os={"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
                      logical_cpu_count=os.cpu_count())
    command = shutil.which("nvidia-smi")
    if command:
        try:
            completed = subprocess.run([command, "--query-gpu=index,name,memory.total,memory.free,driver_version",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                       timeout=5, check=True)
            for row in csv.reader(completed.stdout.splitlines()):
                index, name, total, free, driver = [item.strip() for item in row]
                result.gpus.append(GPU(index=int(index), name=name, total_vram_mib=int(total),
                                       free_vram_mib=int(free), driver_version=driver))
            result.gpu_probe_state = "VERIFIED_LOCAL" if result.gpus else "UNKNOWN"
        except (OSError, subprocess.SubprocessError, ValueError):
            result.probe_notes.append("NVIDIA hardware probe failed; no GPU profile inferred")
    else:
        result.probe_notes.append("nvidia-smi unavailable; GPU fields are unknown")
    try:
        import onnxruntime as ort
        result.advertised_providers = ort.get_available_providers()
        result.ort_version = ort.__version__
        result.provider_probe_state = "VERIFIED_LOCAL"
    except Exception:
        result.probe_notes.append("ONNX Runtime import/probe failed; provider fields are unknown")
    result.probe_notes.append("No inference sessions created; GPU execution and CUDA runtime are unverified")
    return result
