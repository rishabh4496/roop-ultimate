"""Read-only runtime health checks used by the compatibility-gated updater.

The checks intentionally run in a separate Python process.  A provider/model
session created for validation is therefore released when this process exits,
which bounds TensorRT cache and GPU-memory side effects.  No check downloads a
model, writes application configuration, or writes output media.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import importlib.metadata
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _result(name: str, ok: bool, detail: str, **extra: Any) -> dict[str, Any]:
    value = {"name": name, "ok": bool(ok), "detail": detail}
    value.update(extra)
    return value


def _requirements_check(source_root: Path) -> dict[str, Any]:
    path = source_root / "app" / "requirements.txt"
    if not path.is_file():
        return _result("dependencies", False, f"requirements file is missing: {path}")
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except Exception as exc:
        return _result("dependencies", False,
                       f"dependency specifier parser is unavailable: {exc}",
                       classification="UNVERIFIED")

    missing: list[str] = []
    incompatible: list[str] = []
    malformed: list[str] = []
    checked: list[dict[str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return _result("dependencies", False, f"requirements file could not be read: {exc}")
    for number, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            requirement = Requirement(line)
        except Exception as exc:
            malformed.append(f"line {number}: {line} ({exc})")
            continue
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(requirement.name)
            continue
        checked.append({"name": requirement.name, "version": installed})
        try:
            matches = not requirement.specifier or Version(installed) in requirement.specifier
        except Exception as exc:
            malformed.append(f"line {number}: {line} ({exc})")
            continue
        if not matches:
            incompatible.append(f"{requirement.name}=={installed} does not satisfy {requirement.specifier}")
    if malformed:
        return _result("dependencies", False,
                       "dependency requirements could not be verified", checked=checked,
                       malformed=malformed, missing=missing, incompatible=incompatible,
                       classification="UNVERIFIED")
    if missing or incompatible:
        return _result("dependencies", False,
                       "installed dependencies do not satisfy the application requirements",
                       checked=checked, missing=missing, incompatible=incompatible)
    return _result("dependencies", True, f"verified {len(checked)} direct requirements", checked=checked)


def _resolve_npm() -> str | None:
    """Find npm without depending on the caller's PATH.

    WHY THIS IS NOT `shutil.which` ALONE.  Pinokio ships node/npm inside its own
    toolchain (``<PINOKIO_HOME>/bin/miniforge``) and puts them on PATH only
    inside a Pinokio-managed shell.  This health worker is spawned as a plain
    child process by `update_manager`, so on the physical RTX 3060 host the bare
    lookup MISSED an npm that is installed and working, the check returned
    `ok: False`, and the whole health report -- and therefore the updater's
    activation gate -- was reported unhealthy on a healthy machine.

    Same defect class as the bare `shutil.which('ffmpeg')` that made the
    hardware profile record a machine with no NVDEC and no NVENC; the resolution
    order here mirrors `HardwareProfiler._resolve_ffmpeg`.  Resolution is
    deliberately runtime-only: no absolute path is written into any script.
    """
    found = shutil.which("npm")
    if found:
        return found
    home = os.environ.get("PINOKIO_HOME")
    if not home or not os.path.isdir(home):
        try:
            cfg = os.path.join(os.path.expanduser("~"), ".pinokio", "config.json")
            with open(cfg, "r", encoding="utf-8") as handle:
                home = json.load(handle).get("home")
        except Exception:
            home = None
    # <PINOKIO_HOME>/api/<launcher>/app/this_file.py
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if not home or not os.path.isdir(home):
        home = os.path.dirname(os.path.dirname(os.path.dirname(app_dir)))
    roots = (
        os.path.join(home, "bin", "miniforge"),
        os.path.join(home, "bin", "miniconda"),
        os.path.join(home, "bin", "miniforge", "Library", "bin"),
        os.path.join(app_dir, "env"),
    )
    for root in roots:
        for name in ("npm.cmd", "npm.exe", "npm"):
            candidate = os.path.join(root, name)
            if os.path.isfile(candidate):
                return candidate
    return None


def _node_dependencies_check(source_root: Path, data_root: Path) -> dict[str, Any]:
    """Validate both shipped React generations without installing anything."""
    npm = _resolve_npm()
    if not npm:
        return _result("node-dependencies", False, "npm is not available for dependency validation",
                       classification="UNVERIFIED")
    checked: list[str] = []
    failures: list[str] = []
    for relative in ("react-ui",):
        package = source_root / relative / "package.json"
        modules = data_root / relative / "node_modules"
        if not package.is_file():
            failures.append(f"{relative}: candidate package.json is missing")
            continue
        if not modules.is_dir():
            failures.append(f"{relative}: installed node_modules is missing")
            continue
        try:
            result = subprocess.run(
                [npm, "ls", "--depth=0", "--json", "--prefix", str(data_root / relative)],
                cwd=str(data_root), capture_output=True, text=True, check=False, timeout=60,
            )
            parsed = json.loads(result.stdout or "{}")
            if result.returncode or parsed.get("error"):
                detail = result.stderr.strip() or parsed.get("error", {}).get("summary", "npm ls failed")
                failures.append(f"{relative}: {detail}")
            else:
                checked.append(relative)
        except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{relative}: npm dependency tree could not be verified: {exc}")
    return _result("node-dependencies", not failures,
                   f"verified installed dependency trees: {', '.join(checked)}" if not failures
                   else "; ".join(failures), checked=checked, failures=failures)


def _read_config(data_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = data_root / "app" / "config.yaml"
    if not path.is_file():
        return None, f"application configuration is missing: {path}"
    try:
        import yaml
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return None, f"application configuration could not be read: {exc}"
    if not isinstance(value, dict):
        return None, "application configuration is not a mapping"
    return value, None


def _runtime_and_provider(source_root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        return (_result("provider", False, f"ONNX Runtime could not be imported: {exc}"),
                _result("gpu", False, "GPU validation could not run because ONNX Runtime is unavailable"))
    requested = os.environ.get("ROOP_EXECUTION_PROVIDER") or config.get("provider") or "cpu"
    requested = str(requested).strip().lower()
    try:
        from roop import backend_manager
        resolved = backend_manager.resolve_provider_names([requested], device_id=0)
    except Exception as exc:
        return (_result("provider", False, f"provider resolution failed: {exc}", requested=requested),
                _result("gpu", False, "GPU validation could not run because provider resolution failed"))
    available = [str(item) for item in ort.get_available_providers()]
    if not resolved:
        return (_result("provider", False,
                        "no usable ONNX Runtime execution provider was resolved",
                        requested=requested, available=available),
                _result("gpu", False, "GPU validation could not run because no provider was resolved"))
    first = str(resolved[0])
    provider_ok = any(first.lower() == item.lower() for item in available)
    provider = _result("provider", provider_ok,
                       f"requested '{requested}' resolved to {resolved}",
                       requested=requested, resolved=resolved, available=available)

    gpu_names = ("CUDAExecutionProvider", "TensorrtExecutionProvider", "ROCMExecutionProvider")
    needs_gpu = any(any(gpu_name.lower() == item.lower() for gpu_name in gpu_names)
                    for item in resolved)
    if not needs_gpu:
        gpu = _result("gpu", True, "resolved provider chain does not require a GPU", required=False)
    else:
        try:
            import torch
            visible = bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count()) if visible else 0
            if not visible or count < 1:
                gpu = _result("gpu", False,
                              "a CUDA-capable provider resolved but no CUDA device is usable",
                              cuda_visible=visible, device_count=count)
            else:
                props = torch.cuda.get_device_properties(0)
                torch.cuda.synchronize(0)
                gpu = _result("gpu", True,
                              f"CUDA device is usable: {props.name}, {props.total_memory // (1024 * 1024)} MiB",
                              cuda_visible=True, device_count=count, gpu_name=str(props.name),
                              vram_mb=int(props.total_memory // (1024 * 1024)),
                              compute_capability="%d.%d" % torch.cuda.get_device_capability(0))
        except Exception as exc:
            gpu = _result("gpu", False, f"CUDA device validation failed: {exc}")
    return provider, gpu


def _feed_shape(shape: list[Any], output_size: int) -> list[int]:
    values: list[int] = []
    for index, dimension in enumerate(shape):
        try:
            value = int(dimension)
        except (TypeError, ValueError):
            value = 1 if index == 0 else (output_size if len(shape) == 4 and index in (2, 3) else 512)
        values.append(max(1, value))
    return values


def _smoke_feeds(session: Any, output_size: int) -> dict[str, Any]:
    import numpy as np

    feeds: dict[str, Any] = {}
    for input_meta in session.get_inputs():
        shape = _feed_shape(list(input_meta.shape), output_size)
        input_type = str(input_meta.type).lower()
        if "float16" in input_type:
            dtype = np.float16
        elif "float" in input_type:
            dtype = np.float32
        elif "int64" in input_type:
            dtype = np.int64
        elif "int32" in input_type:
            dtype = np.int32
        else:
            raise RuntimeError(f"unsupported smoke-test input type {input_meta.type} for {input_meta.name}")
        if np.issubdtype(dtype, np.floating):
            value = np.full(shape, 0.125, dtype=dtype)
        else:
            value = np.ones(shape, dtype=dtype)
        feeds[input_meta.name] = value
    return feeds


def _model_check(source_root: Path, data_root: Path, config: dict[str, Any], resolved: list[str]) -> list[dict[str, Any]]:
    try:
        import numpy as np
        import onnxruntime as ort
        from roop.processors.FaceSwapInsightFace import SWAP_MODELS
    except Exception as exc:
        failed = _result("models", False, f"model validation imports failed: {exc}")
        return [failed, _result("inference", False, "inference smoke test was not run")]

    selected = str(config.get("swap_model") or "realswap").strip()
    if selected not in SWAP_MODELS:
        return [_result("models", False, f"configured swap model is not supported by this source: {selected}"),
                _result("inference", False, "inference smoke test was not run")]
    keys = [selected]
    secondary = SWAP_MODELS[selected].get("secondary")
    if secondary and secondary in SWAP_MODELS:
        keys.append(secondary)
    sessions: list[tuple[str, Any, int]] = []
    model_errors: list[str] = []
    for key in keys:
        spec = SWAP_MODELS[key]
        path = data_root / "app" / "models" / str(spec["file"])
        if not path.is_file() or path.stat().st_size <= 0:
            model_errors.append(f"{key}: required local model is missing or empty: {path}")
            continue
        try:
            session = ort.InferenceSession(str(path), providers=resolved)
            actual = [str(item) for item in session.get_providers()]
            if not actual or actual[0].lower() != str(resolved[0]).lower():
                model_errors.append(f"{key}: session provider chain {actual} does not start with {resolved[0]}")
            sessions.append((key, session, int(spec.get("output_size", 128))))
        except Exception as exc:
            model_errors.append(f"{key}: model session initialization failed: {exc}")
    model_result = _result("models", not model_errors,
                           "all configured local model sessions initialized" if not model_errors
                           else "; ".join(model_errors), selected=selected, models=keys)
    if model_errors or not sessions:
        return [model_result, _result("inference", False, "inference smoke test was not run")]

    inference_errors: list[str] = []
    for key, session, output_size in sessions:
        try:
            outputs = session.run(None, _smoke_feeds(session, output_size))
            if not outputs:
                raise RuntimeError("session returned no outputs")
            for output in outputs:
                array = np.asarray(output)
                if array.size == 0:
                    raise RuntimeError("session returned an empty output")
                if array.dtype.kind in "fc" and not np.isfinite(array).all():
                    raise RuntimeError("session returned non-finite values")
        except Exception as exc:
            inference_errors.append(f"{key}: {exc}")
    sessions.clear()
    gc.collect()
    return [
        model_result,
        _result("inference", not inference_errors,
                "one finite inference completed for every configured model session"
                if not inference_errors else "; ".join(inference_errors), models=keys),
    ]


def _launch_check(source_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    app_root = source_root / "app"
    run_path = app_root / "run.py"
    if not run_path.is_file():
        return _result("launch", False, f"application launch entry point is missing: {run_path}")
    # The shipped run.py path reads the provider from config.yaml.  Passing a
    # provider flag here would be unsafe: run.py and core.parse_args each own
    # part of argument parsing and the latter rejects this extra argument.
    requested = str(os.environ.get("ROOP_EXECUTION_PROVIDER") or config.get("provider") or "cpu").strip()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    finally:
        sock.close()
    env = os.environ.copy()
    env.update({"ROOP_API_PORT": str(port), "ROOP_UPDATE_HEALTH": "1"})
    try:
        process = subprocess.Popen(
            [sys.executable, str(run_path)],
            cwd=str(app_root), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )
    except OSError as exc:
        return _result("launch", False, f"application process could not start: {exc}")

    # DRAIN THE CHILD CONCURRENTLY.  The pipe was previously read only AFTER the
    # probe finished, so the child's startup output accumulated in an OS pipe
    # buffer of a few kilobytes.  This application prints far more than that
    # before it serves: every ONNX Runtime session dumps a ~1 KB provider-option
    # block and the model loader prints one per model.  Once the buffer fills,
    # the child BLOCKS ON WRITE and can never reach the point where it binds the
    # port -- the probe then times out against a process that is alive, healthy
    # and simply gagged.  A reader thread removes that failure mode entirely.
    collected: list[str] = []

    def _drain(stream):
        try:
            for line in iter(stream.readline, ""):
                collected.append(line)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    reader = None
    if process.stdout is not None:
        reader = threading.Thread(target=_drain, args=(process.stdout,), daemon=True)
        reader.start()

    # A COLD START IS NOT A 90-SECOND OPERATION ON EVERY TARGET.  Where the
    # provider policy admits TensorRT, `run.py` may build engines on first use,
    # which runs into minutes; where it resolves to CUDA/CPU (the sub-7GB
    # laptop tier) startup is fast.  A single 90 s budget therefore passed on
    # one validation GPU and timed out on the other while both were healthy.
    # The budget is generous and bounded, and overridable for slow hosts.
    try:
        budget = float(os.environ.get("ROOP_HEALTH_LAUNCH_TIMEOUT", "300"))
    except ValueError:
        budget = 300.0
    deadline = time.monotonic() + max(30.0, budget)
    response_status: int | None = None
    launch_error = ""
    url = f"http://127.0.0.1:{port}/api/meta"
    while time.monotonic() < deadline and process.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                response_status = int(response.status)
                if response_status == 200:
                    break
        except (OSError, urllib.error.URLError) as exc:
            launch_error = str(exc)
        time.sleep(0.25)
    alive_after_probe = process.poll() is None
    try:
        if alive_after_probe:
            process.terminate()
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)
    if reader is not None:
        reader.join(timeout=10)
    output = "".join(collected)[-12000:]
    ok = response_status == 200 and alive_after_probe
    detail = (f"/api/meta returned HTTP 200 on loopback port {port}"
              if ok else f"application did not pass launch probe: {launch_error or output[-1000:]}")
    return _result("launch", ok, detail, port=port, response_status=response_status,
                   process_output=output)


def run_health(source_root: Path, data_root: Path, skip_launch: bool = False) -> dict[str, Any]:
    source_root = source_root.resolve()
    data_root = data_root.resolve()
    app_path = source_root / "app"
    if str(app_path) not in sys.path:
        sys.path.insert(0, str(app_path))
    results: list[dict[str, Any]] = []
    dependency = _requirements_check(source_root)
    results.extend([dependency, _node_dependencies_check(source_root, data_root)])
    config, config_error = _read_config(data_root)
    if config_error:
        results.append(_result("configuration", False, config_error))
        results.append(_result("provider", False, "provider validation was not run"))
        results.append(_result("gpu", False, "GPU validation was not run"))
    else:
        results.append(_result("configuration", True, "application configuration loaded without mutation"))
        provider, gpu = _runtime_and_provider(source_root, config or {})
        results.extend([provider, gpu])
        if provider.get("ok"):
            # Launch first, before this health process creates TensorRT/ORT
            # sessions.  The launcher owns its own provider process and must
            # not compete with validation sessions for GPU memory.
            if not skip_launch:
                results.append(_launch_check(source_root, config or {}))
            results.extend(_model_check(source_root, data_root, config or {}, provider["resolved"]))
        else:
            results.append(_result("models", False, "model loading was not run because provider validation failed"))
            results.append(_result("inference", False, "inference smoke test was not run because provider validation failed"))
            if not skip_launch:
                results.append(_launch_check(source_root, config or {}))
    if skip_launch:
        results.append(_result("launch", True, "launch probe deferred until post-activation validation"))
    healthy = all(item.get("ok") for item in results)
    return {"healthy": healthy, "source_root": str(source_root), "data_root": str(data_root),
            "checks": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--skip-launch", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    diagnostic_output = io.StringIO()
    try:
        # Third-party imports and the application provider probes may print
        # warnings. Keep the machine-readable stdout channel unambiguous for
        # update_manager; preserve those diagnostics on stderr instead.
        with contextlib.redirect_stdout(diagnostic_output):
            report = run_health(args.source_root, args.data_root, args.skip_launch)
    except Exception as exc:
        report = {"healthy": False, "checks": [_result("health-worker", False,
                                                          f"unexpected health-check failure: {exc}")]}
    if diagnostic_output.getvalue():
        print(diagnostic_output.getvalue(), file=sys.stderr, end="")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("UPDATE RUNTIME HEALTH: " + ("HEALTHY" if report["healthy"] else "FAILED"))
        for item in report.get("checks", []):
            print(f"- {item['name']}: {'PASS' if item['ok'] else 'FAIL'} — {item['detail']}")
    return 0 if report.get("healthy") else 2


if __name__ == "__main__":
    raise SystemExit(main())
