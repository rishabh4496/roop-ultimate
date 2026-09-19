#!/usr/bin/env python3
"""Authoritative, fail-closed provisioning for the Python runtime.

This module deliberately owns the platform-dependent part of installation.
Pinokio only starts this script unconditionally. That prevents a thrown or
mis-scoped ``when`` expression from skipping every GPU branch and leaving a
successful-looking but unusable environment behind.

The script is used only during install/update or an explicit TensorRT repair.
It is not part of normal application startup, so it never performs an
uninstall/install loop on every launch.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence


ORT_DISTRIBUTIONS = (
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-rocm",
    "onnxruntime-silicon",
)
TRT_PACKAGES = (
    "tensorrt-cu12",
    "tensorrt-cu12-libs",
    "tensorrt-cu12-bindings",
)


class ProvisioningError(RuntimeError):
    """A required provisioning step failed."""


@dataclass(frozen=True)
class Hardware:
    system: str
    architecture: str
    vendor: str
    gpu_names: tuple[str, ...] = ()
    vram_mb: tuple[int, ...] = ()
    nvidia_smi: str | None = None


def _print(message: str) -> None:
    print(f"[Provision] {message}", flush=True)


def _run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    rendered = " ".join(_quote(part) for part in command)
    _print(rendered)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr, flush=True)
    if check and result.returncode != 0:
        raise ProvisioningError(
            f"command failed with exit code {result.returncode}: {rendered}"
        )
    return result


def _quote(value: str) -> str:
    if re.search(r"\s", value):
        return repr(value)
    return value


def _uv() -> str:
    executable = shutil.which("uv")
    if not executable:
        raise ProvisioningError(
            "uv is not available on PATH; Pinokio's Python package manager is required"
        )
    return executable


def _uv_pip(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    # --python binds uv to the interpreter Pinokio activated for this step. It
    # prevents a globally installed uv from modifying a different environment.
    return _run([_uv(), "pip", *arguments, "--python", sys.executable])


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_distributions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if name:
            result[_normalise(name)] = version
    return result


def _remove_installed(names: Iterable[str]) -> None:
    installed = _installed_distributions()
    for name in names:
        key = _normalise(name)
        if key in installed:
            _uv_pip(["uninstall", name])


def _install(
    packages: Sequence[str],
    *,
    index_url: str | None = None,
    extra_index_url: str | None = None,
    force_reinstall: bool = False,
    no_deps: bool = False,
) -> None:
    arguments = ["install", *packages]
    if index_url:
        arguments.extend(["--index-url", index_url])
    if extra_index_url:
        arguments.extend(["--extra-index-url", extra_index_url])
    if force_reinstall:
        arguments.append("--force-reinstall")
    if no_deps:
        arguments.append("--no-deps")
    _uv_pip(arguments)


def _nvidia_smi_executable() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if os.name == "nt":
        candidates = (
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"),
                         "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
            os.path.join(os.environ.get("ProgramW6432", "C:\\Program Files"),
                         "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
        )
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
    return None


def _query_nvidia() -> tuple[str | None, list[str], list[int]]:
    executable = _nvidia_smi_executable()
    if not executable:
        return None, [], []
    result = _run(
        [executable, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        check=False,
    )
    if result.returncode != 0:
        return executable, [], []
    names: list[str] = []
    memory: list[int] = []
    for line in result.stdout.splitlines():
        columns = [column.strip() for column in line.split(",")]
        if not columns:
            continue
        names.append(columns[0])
        if len(columns) > 1:
            try:
                memory.append(int(float(columns[1])))
            except ValueError:
                memory.append(0)
    return executable, names, memory


def _display_adapter_names() -> list[str]:
    if os.name == "nt":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell:
            result = _run(
                [powershell, "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                check=False,
            )
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    lspci = shutil.which("lspci")
    if lspci:
        result = _run([lspci], check=False)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return []


def detect_hardware() -> Hardware:
    system = platform.system()
    architecture = platform.machine()
    nvidia_smi, nvidia_names, nvidia_memory = _query_nvidia()
    if nvidia_names:
        return Hardware(system, architecture, "nvidia", tuple(nvidia_names),
                        tuple(nvidia_memory), nvidia_smi)

    adapters = _display_adapter_names()
    lowered = " ".join(adapters).lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered:
        return Hardware(system, architecture, "nvidia", tuple(adapters))
    if any(token in lowered for token in ("amd", "radeon", "advanced micro devices")):
        return Hardware(system, architecture, "amd", tuple(adapters))
    return Hardware(system, architecture, "cpu", tuple(adapters))


def _install_common_torch_dependencies() -> None:
    _install(["filelock", "fsspec", "jinja2", "networkx", "typing-extensions", "sympy"])


def _install_onnxruntime(package: str) -> None:
    _remove_installed(ORT_DISTRIBUTIONS)
    _install([package])


def _install_nvidia_runtime(*, xformers: bool = False) -> None:
    torch_packages = ["torch==2.7.0", "torchvision==0.22.0"]
    if xformers:
        torch_packages.append("xformers")
    _install(
        torch_packages,
        index_url="https://download.pytorch.org/whl/cu128",
        force_reinstall=True,
        no_deps=True,
    )
    _install_common_torch_dependencies()
    _install_onnxruntime("onnxruntime-gpu==1.23.2")
    _install(
        [f"{package}==10.9.0.34" for package in TRT_PACKAGES],
        extra_index_url="https://pypi.nvidia.com/",
    )


def _install_amd_runtime() -> None:
    if os.name == "nt":
        _install(["torch", "torch-directml", "torchvision", "torchaudio", "numpy==1.26.4"],
                 force_reinstall=True)
        _install_onnxruntime("onnxruntime-directml")
    else:
        _install(
            ["torch==2.7.0", "torchvision==0.22.0"],
            index_url="https://download.pytorch.org/whl/rocm6.3",
            force_reinstall=True,
            no_deps=True,
        )
        _install_common_torch_dependencies()
        _remove_installed(ORT_DISTRIBUTIONS)
        _install(["https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/"
                  "onnxruntime_rocm-1.19.0-cp310-cp310-linux_x86_64.whl"])


def _install_cpu_runtime() -> None:
    _install(
        ["torch==2.7.0", "torchvision==0.22.0"],
        index_url="https://download.pytorch.org/whl/cpu",
        force_reinstall=True,
        no_deps=True,
    )
    _install_common_torch_dependencies()
    _install_onnxruntime("onnxruntime==1.17.1")


def _install_darwin_runtime(hardware: Hardware) -> None:
    _install(
        ["torch==2.7.0", "torchvision==0.22.0"],
        index_url="https://download.pytorch.org/whl/cpu",
        force_reinstall=True,
        no_deps=True,
    )
    _install_common_torch_dependencies()
    _install_onnxruntime(
        "onnxruntime-silicon==1.16.3" if hardware.architecture == "arm64"
        else "onnxruntime==1.17.1"
    )


def provision(hardware: Hardware, *, tensorrt_only: bool = False, xformers: bool = False) -> None:
    _print("hardware=" + json.dumps(asdict(hardware), sort_keys=True))
    if hardware.vendor == "nvidia" and not hardware.nvidia_smi:
        raise ProvisioningError(
            "NVIDIA hardware was detected, but nvidia-smi is unavailable; "
            "refusing to silently provision a CPU runtime"
        )
    if tensorrt_only:
        if hardware.vendor != "nvidia":
            raise ProvisioningError(
                "TensorRT repair requested, but nvidia-smi did not identify an NVIDIA GPU"
            )
        _install_onnxruntime("onnxruntime-gpu==1.23.2")
        _install([f"{package}==10.9.0.34" for package in TRT_PACKAGES],
                 extra_index_url="https://pypi.nvidia.com/")
        return

    if hardware.vendor == "nvidia":
        _install_nvidia_runtime(xformers=xformers)
    elif hardware.vendor == "amd":
        _install_amd_runtime()
    elif hardware.system == "Darwin":
        _install_darwin_runtime(hardware)
    else:
        _install_cpu_runtime()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorrt-only", action="store_true")
    parser.add_argument("--xformers", action="store_true")
    args = parser.parse_args(argv)
    try:
        provision(detect_hardware(), tensorrt_only=args.tensorrt_only, xformers=args.xformers)
    except (ProvisioningError, OSError) as exc:
        print(f"[FATAL] runtime provisioning failed: {exc}", file=sys.stderr, flush=True)
        return 1
    _print("runtime package provisioning completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
