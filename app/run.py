#!/usr/bin/env python3

import os
import socket
import sys
import time

import numpy as np


_GPU_DLL_HANDLES = []


def _register_gpu_runtime_dirs():
    """Register process-local TensorRT/CUDA DLL directories without PATH edits."""
    dll_dirs = []

    def _add(directory):
        if directory and os.path.isdir(directory) and directory not in dll_dirs:
            dll_dirs.append(directory)

    for module_name, resolver in (
        ("tensorrt", lambda m: os.path.join(
            os.path.dirname(os.path.dirname(m.__file__)), "tensorrt_libs")),
        ("tensorrt_libs", lambda m: os.path.dirname(m.__file__)),
        ("torch", lambda m: os.path.join(os.path.dirname(m.__file__), "lib")),
    ):
        try:
            module = __import__(module_name)
            if getattr(module, "__file__", None):
                _add(resolver(module))
        except Exception:
            continue

    try:
        import nvidia
        roots = ([os.path.dirname(nvidia.__file__)]
                 if getattr(nvidia, "__file__", None)
                 else list(getattr(nvidia, "__path__", [])))
        for root in roots:
            for current, _dirs, _files in os.walk(root):
                if os.path.basename(current).lower() == "bin":
                    _add(current)
    except Exception:
        pass

    for directory in dll_dirs:
        try:
            if hasattr(os, "add_dll_directory"):
                _GPU_DLL_HANDLES.append(os.add_dll_directory(directory))
        except Exception:
            pass


_register_gpu_runtime_dirs()

from roop.degrade import swallowed as _swallowed


def run_preflight_checks():
    from roop.gpu_preflight import get_preflight_result
    from roop.startup_state_machine import (
        StartupPhase,
        get_startup_state_machine,
        execute_boot,
        execute_dependency_preflight,
        execute_dll_runtime_preflight,
        execute_ort_preflight,
        execute_gpu_preflight,
        execute_provider_admission,
    )
    sm = get_startup_state_machine()
    sm.execute_phase(StartupPhase.BOOT, execute_boot)
    sm.execute_phase(StartupPhase.DEPENDENCY_PREFLIGHT, execute_dependency_preflight)
    sm.execute_phase(StartupPhase.DLL_RUNTIME_PREFLIGHT, execute_dll_runtime_preflight)
    sm.execute_phase(StartupPhase.ORT_PREFLIGHT, execute_ort_preflight)
    sm.execute_phase(StartupPhase.GPU_PREFLIGHT, execute_gpu_preflight)
    sm.execute_phase(StartupPhase.PROVIDER_ADMISSION, execute_provider_admission)


if __name__ == "__main__":
    run_preflight_checks()

# Force UTF-8 encoding for standard streams to avoid UnicodeEncodeError on Windows terminals with non-UTF-8 locale
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception as _degrade_error:
        _swallowed("run.py:13", _degrade_error, "fallback continued")
        pass

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["AV_LOG_LEVEL"] = "error"

# Apply the advanced perf knobs from config.yaml to os.environ BEFORE any roop
# module is imported (ProcessMgr reads ROOP_PROFILE/ROOP_BATCH_SWAP at import
# time). 'auto'/blank means "leave it alone" so the launcher's env and the
# VRAM auto-tuner keep working; an explicit value overrides them.
def _apply_perf_env():
    try:
        with open('config.yaml', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        # Check if default_config.yaml is available to seed the initial environment
        try:
            with open('default_config.yaml', 'r') as f:
                content = f.read()
        except FileNotFoundError:
            return
    except Exception as _degrade_error:
        _swallowed("run.py:28", _degrade_error, "fallback continued")
        return

    try:
        import yaml
        cfg = yaml.safe_load(content) or {}
    except Exception as _degrade_error:
        _swallowed("run.py:28", _degrade_error, "fallback continued")
        return

    # Build a cheap hardware-only profile before importing the model pipeline.
    # This does not create sessions or TensorRT engines.  It only publishes a
    # bounded, provenance-tagged runtime hint for stages that opt into the
    # rollout; explicit config values and explicit environment values win.
    try:
        from roop.runtime_optimizer import RuntimeOptimizer
        _runtime_optimizer = RuntimeOptimizer(settings=cfg)
        _startup_profile = _runtime_optimizer.startup_profile()
        _applied = _runtime_optimizer.apply_environment(_startup_profile, cfg)
        print("[RuntimeOptimizer] startup profile: "
              f"GPU={_startup_profile.hardware.gpu_name or 'none'} "
              f"VRAM={_startup_profile.hardware.vram_total_gb:.1f}GB "
              f"workers={_startup_profile.tuning.worker_count} "
              f"contexts={_startup_profile.tuning.trt_context_count} "
              f"queue={_startup_profile.tuning.queue_depth} "
              f"applied={sorted(_applied)}", flush=True)
    except Exception as exc:
        # The optimizer is advisory at startup.  A missing optional probe must
        # never prevent the established provider/fallback path from launching.
        print(f"[RuntimeOptimizer] startup profile unavailable: {exc}", flush=True)

    # The setting -> ROOP_* mapping is settings.ENV_SETTINGS, applied by
    # settings.apply_env: one implementation shared with the comparison benches,
    # and the same 'an explicit environment value wins' contract as before.
    from settings import apply_env as _apply_env
    _apply_env(cfg, os.environ)


_apply_perf_env()

# Windows asyncio fixes (Python 3.10 ProactorEventLoop): a peer aborting an
# accept must not close the LISTENING socket, and a pipe closing under a
# transport must not raise from the loop's own callback. Both live in
# roop.win_asyncio_compat so api.run_api applies them on every entry path too.
from roop.win_asyncio_compat import install as _install_win_asyncio_compat
_install_win_asyncio_compat()

from roop import core
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda_device_id', type=int, default=0,
                    help='CUDA device index within CUDA_VISIBLE_DEVICES (distributed workers use 0)')
parser.add_argument('--execution-provider', default=None, help='Execution provider override: auto, cpu, cuda, tensorrt, rocm, or dml')
parser.add_argument('--source', '--source-path', dest='source_reference_path', default=None,
                    help='source image or folder of same-identity reference images')
parser.add_argument('--project', dest='project', default=None,
                    help='portable .roop project session to load')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the --project headlessly and exit')
parser.add_argument('--output', dest='output', default=None,
                    help='override the output directory or exact filename for a headless project render')
parser.add_argument('--export-fcpxml', dest='export_fcpxml', default=None,
                    help='export the --project timeline as Final Cut Pro XML')
parser.add_argument('--export-edl', dest='export_edl', default=None,
                    help='export the --project timeline as a Resolve CMX3600 EDL')
parser.add_argument('--scene-detect', action='store_true', default=False,
                    help='run scene-cut detection before NLE export')
# Headless benchmark. Runs the same engine, scoring and recommendation the
# React panel drives -- the CLI renders the dashboard as text rather than
# computing anything of its own.
parser.add_argument('--benchmark', action='store_true',
                    help='run the hardware benchmark and print the results '
                         'dashboard, then exit without starting the UI')
parser.add_argument('--benchmark-faces', dest='benchmark_faces',
                    choices=['1', '2', 'all'], default='1',
                    help='target face complexity: 1 face, 2 faces, or a crowd')
parser.add_argument('--benchmark-mode', dest='benchmark_mode',
                    choices=['quick', 'full', 'regression'], default='quick',
                    help='quick profile (~30s), full stress and thermal test (~90s), '
                         'or regression: a real 300-frame 1080p render checked '
                         'for fps, VRAM, frame latency and SSIM/PSNR against a baseline')
parser.add_argument('--benchmark-frames', dest='benchmark_frames', type=int, default=300,
                    help='regression: frames in the timed render')
parser.add_argument('--benchmark-clip', dest='benchmark_clip', default=None,
                    help='regression: clip to render (default: a generated 1080p clip)')
parser.add_argument('--benchmark-source', dest='benchmark_source', default=None,
                    help='regression: source face image')
parser.add_argument('--benchmark-threads', dest='benchmark_threads', type=int, default=None,
                    help='regression: worker threads (default: config max_threads)')
parser.add_argument('--benchmark-update-baseline', dest='benchmark_update_baseline',
                    action='store_true',
                    help='regression: record this run as the new baseline')
parser.add_argument('--benchmark-apply', dest='benchmark_apply',
                    action='store_true',
                    help='apply the recommended settings when the run finishes '
                         '(without this the run is saved but nothing changes)')
parser.add_argument('--ui', choices=['react', 'gradio', 'legacy'], default=None,
                    help='UI to launch: react (default for React launcher) or legacy/gradio')
parser.add_argument('--react', action='store_true', default=False,
                    help='force React client mode')
parser.add_argument('--diagnose-runtime', action='store_true', default=False,
                    help='run standalone diagnostic probe and print runtime environment report without launching servers or models')
args = parser.parse_args()
_project_flags = ('render', 'output', 'export_fcpxml', 'export_edl', 'scene_detect')
if not getattr(args, 'project', None) and any(getattr(args, flag, None) for flag in _project_flags):
    parser.error('--render, --output, --export-fcpxml, --export-edl, and --scene-detect require --project')
if getattr(args, 'diagnose_runtime', False):
    from roop.runtime_diagnostics import run_diagnose_runtime
    sys.exit(run_diagnose_runtime())
if getattr(args, 'react', False) or getattr(args, 'ui', None) == 'react':
    os.environ['ROOP_REACT_CLIENT'] = '1'
elif getattr(args, 'ui', None) in ('gradio', 'legacy'):
    os.environ.pop('ROOP_REACT_CLIENT', None)
from roop import globals
# Normalize to onnxruntime's exact provider names — naive concatenation makes
# 'cudaExecutionProvider' (wrong case), which get_device() and the GPU guard
# would not recognize during the window before ui.main overwrites this.
_PROVIDER_NAMES = {
    'cpu': 'CPUExecutionProvider',
    'cuda': 'CUDAExecutionProvider',
    'tensorrt': 'TensorrtExecutionProvider',
    'rocm': 'ROCMExecutionProvider',
    'dml': 'DmlExecutionProvider',
}
_requested_provider = str(args.execution_provider or '').strip().lower()
if _requested_provider:
    globals.execution_providers = [_PROVIDER_NAMES.get(
        _requested_provider, _requested_provider + 'ExecutionProvider')]

def _run_cli_benchmark(faces: str, mode: str, apply_result: bool) -> int:
    """Delegate to the one shared implementation.

    The renderer lives in roop.benchmark.ui_dashboard so this entry point and
    roop/core.py's `--benchmark` cannot drift into printing different numbers
    for the same run.
    """
    from roop.benchmark.ui_dashboard import run_cli_benchmark
    return run_cli_benchmark(faces=faces, mode=mode, apply_result=apply_result)


def _announce_react_backend_when_ready(api_thread, api_port):
    """Publish the React URL only after both halves of startup are usable."""
    from roop.startup_state_machine import (
        StartupPhase,
        PhaseStatus,
        get_startup_state_machine,
        execute_api_ready,
        execute_ui_ready,
    )
    sm = get_startup_state_machine()
    api_res = sm.execute_phase(StartupPhase.API_READY, execute_api_ready, api_thread, api_port)
    if api_res.status == PhaseStatus.FATAL:
        return False
    ui_res = sm.execute_phase(StartupPhase.UI_READY, execute_ui_ready, api_port, True)
    return ui_res.status != PhaseStatus.FATAL


if __name__ == '__main__':
    if getattr(args, 'benchmark', False) and args.benchmark_mode == 'regression':
        from roop.benchmark.regression import run_regression_from_args
        sys.exit(run_regression_from_args(args))
    if getattr(args, 'benchmark', False):
        import time
        sys.exit(_run_cli_benchmark(args.benchmark_faces, args.benchmark_mode,
                                    args.benchmark_apply))

    if getattr(args, 'project', None) and (
            getattr(args, 'render', False) or getattr(args, 'export_fcpxml', None)
            or getattr(args, 'export_edl', None) or getattr(args, 'scene_detect', False)):
        # Project execution is dispatched by core after provider admission so
        # the headless path uses the same model/runtime initialization as the UI.
        os.environ['ROOP_HEADLESS_PROJECT'] = '1'

    if getattr(args, 'project', None) and (
            getattr(args, 'render', False) or getattr(args, 'export_fcpxml', None)
            or getattr(args, 'export_edl', None) or getattr(args, 'scene_detect', False)):
        # The core project dispatcher owns the headless lifecycle. Do not start
        # the HTTP daemon or a UI thread for a non-interactive render/export.
        core.run()
        raise SystemExit(0)

    # Opt out of Windows background throttling (EcoQoS) and raise process
    # priority so analysis/processing speed is identical whether the app
    # window is foreground or covered by other windows.
    from roop import keep_awake
    keep_awake.boost_process_priority()

    import threading
    from api import run_api
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    # Pinokio's launcher waits for a concrete loopback URL before advancing
    # to the React shell.  API_READY is deliberately announced after
    # core.run() returns from CONFIG_LOAD and MODEL_RUNTIME_INIT.  Publishing
    # it from a concurrent thread here races the transactional state machine
    # and attempts API_READY while those required phases are still pending.
    api_port = int(os.environ.get("ROOP_API_PORT", "8001"))
    if os.environ.get("ROOP_REACT_CLIENT") != "1":
        # Preserve the legacy launcher's existing URL capture behavior. Its
        # actual Gradio URL is emitted later by ui/main.py, so do not print
        # a web address here.
        print(f"[Backend] API daemon running on port {api_port}", flush=True)

    core.run()

    # core.run() launches the legacy Gradio UI and blocks in ITS OWN loop.  The
    # API above is a DAEMON thread, so it dies the instant this process exits --
    # which is the moment core.run() returns.
    #
    # For a React client that coupling is a live outage, and it was observed as
    # one: a second launcher instance collided on the Gradio port (both React
    # launchers derive it as ROOP_API_PORT + 2), ui/main.py CAUGHT the
    # "When localhost is not accessible, a shareable link must be created"
    # error, set run_server = False, closed the UI and RETURNED NORMALLY.  The
    # backend had already logged "[Backend] listening on 127.0.0.1:42003"
    # successfully; run.py then fell off the end of __main__ and took it down,
    # and the React UI showed ECONNREFUSED on every poll.
    #
    # Note the failure returns rather than raising, so wrapping core.run() in
    # try/except does NOT catch this -- the return itself has to be handled.
    #
    # Gradio is incidental to the React clients; they speak only to the API.  So
    # when a React launcher started us, outlive Gradio and keep serving.  The
    # legacy launcher is unchanged: there Gradio IS the application, and its
    # shutdown should still end the process.
    if os.environ.get("ROOP_REACT_CLIENT") == "1" and api_thread.is_alive():
        if not _announce_react_backend_when_ready(api_thread, api_port):
            sys.exit(1)
        print(f"[Backend] API daemon running on port {api_port} "
              f"- stop this script in Pinokio to shut it down.", flush=True)
        # Repeat the share banner here, after the model-loading output, so the
        # token is the last thing on the Pinokio terminal, not buried above it.
        import api_access as _api_access
        if _api_access.get_policy().share:
            print(_api_access.get_policy().banner(api_port), flush=True)
        try:
            while api_thread.is_alive():
                api_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            print("[Backend] interrupted; shutting down.", flush=True)
