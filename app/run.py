#!/usr/bin/env python3

import os
import sys

# Force UTF-8 encoding for standard streams to avoid UnicodeEncodeError on Windows terminals with non-UTF-8 locale
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["AV_LOG_LEVEL"] = "error"

# Apply the advanced perf knobs from config.yaml to os.environ BEFORE any roop
# module is imported (ProcessMgr reads ROOP_PROFILE/ROOP_BATCH_SWAP at import
# time). 'auto'/blank means "leave it alone" so the launcher's env and the
# VRAM auto-tuner keep working; an explicit value overrides them.
def _apply_perf_env():
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
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

    def _set(var, val):
        # A caller (including a controlled benchmark) owns an explicit
        # process environment value.  Config is only the fallback; otherwise
        # an A/B arm can be silently replaced before modules import it.
        if var in os.environ:
            return
        if val is None:
            return
        s = str(val).strip()
        if s and s.lower() != 'auto':
            os.environ[var] = s

    _set('ROOP_TRT_POOL', cfg.get('perf_trt_pool'))
    _set('ROOP_TRT_BUILDER_OPT_LEVEL', cfg.get('trt_builder_optimization_level'))
    _set('ROOP_TRT_AUX_STREAMS', cfg.get('trt_auxiliary_streams'))
    if cfg.get('trt_cuda_graph') is not None and 'ROOP_TRT_CUDA_GRAPH' not in os.environ:
        graph = cfg.get('trt_cuda_graph')
        graph_on = graph is True or str(graph).strip().lower() in ('1', 'true', 'yes', 'on')
        os.environ['ROOP_TRT_CUDA_GRAPH'] = '1' if graph_on else '0'
    _set('ROOP_CV_THREADS', cfg.get('cpu_opencv_threads'))
    _set('ROOP_ORT_INTRA_THREADS', cfg.get('cpu_ort_intra_threads'))
    _set('ROOP_ORT_INTER_THREADS', cfg.get('cpu_ort_inter_threads'))
    _set('ROOP_FFMPEG_THREADS', cfg.get('cpu_ffmpeg_threads'))
    _set('ROOP_DETMASK_POOL', cfg.get('perf_detmask_pool'))
    _set('ROOP_DETECTOR_POOL', cfg.get('perf_detector_pool'))
    _set('ROOP_EXPR_POOL', cfg.get('perf_expr_pool'))
    _set('ROOP_ENCODER_PRESET', cfg.get('perf_encoder_preset'))
    for var, key in (('ROOP_PROFILE', 'perf_profile'), ('ROOP_BATCH_SWAP', 'perf_batch_swap'),
                     ('ROOP_NVDEC', 'perf_nvdec'),
                     # Identity/tracking features that used to be reachable only
                     # by editing a launcher's environment. Same 'auto' contract:
                     # leave the env alone and let each module keep its own
                     # default, so exposing them changed no shipped behaviour.
                     ('ROOP_FACE_DEMARCATE', 'face_demarcate'),
                     ('ROOP_TRACK_STITCH', 'track_stitch'),
                     ('ROOP_VERIFY_SWAP', 'verify_swap'),
                     ('ROOP_UPRIGHT_REMEASURE', 'upright_remeasure')):
        if var in os.environ:
            continue
        v = str(cfg.get(key, 'auto')).strip().lower()
        if v == 'on' or (v == 'auto' and var == 'ROOP_BATCH_SWAP'):
            os.environ[var] = '1'
            if var == 'ROOP_BATCH_SWAP' and 'ROOP_BATCH_SWAP_XFRAME' not in os.environ:
                os.environ['ROOP_BATCH_SWAP_XFRAME'] = '1'
        elif v == 'off':
            os.environ[var] = '0'
            if var == 'ROOP_BATCH_SWAP' and 'ROOP_BATCH_SWAP_XFRAME' not in os.environ:
                os.environ['ROOP_BATCH_SWAP_XFRAME'] = '0'

    # Not tri-state: a model choice and a priority class.
    _rec = str(cfg.get('recognizer', 'default')).strip().lower()
    if _rec == 'adaface' and 'ROOP_ADAFACE' not in os.environ:
        os.environ['ROOP_ADAFACE'] = '1'
    elif _rec == 'default' and 'ROOP_ADAFACE' not in os.environ:
        os.environ['ROOP_ADAFACE'] = '0'
    # Only the names keep_awake._PRIORITY_CLASSES accepts; it falls back to
    # 'high' for anything else, so passing a value it does not know through
    # would present as a working setting that does nothing.
    _pri = str(cfg.get('process_priority', 'auto')).strip().lower()
    if _pri in ('high', 'above_normal', 'normal') and 'ROOP_PRIORITY' not in os.environ:
        os.environ['ROOP_PRIORITY'] = _pri

_apply_perf_env()

# Windows asyncio fix: Python 3.10 on Windows raises ConnectionResetError
# (WinError 10054) in asyncio ProactorEventLoop when a subprocess pipe closes.
# This is a known CPython bug fixed in 3.11. Patch swallows the spurious error.
import sys as _sys
if _sys.platform == 'win32':
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport as _T
        _orig_ccl = _T._call_connection_lost
        def _patched_ccl(self, exc):
            try:
                _orig_ccl(self, exc)
            except ConnectionResetError:
                pass
        _T._call_connection_lost = _patched_ccl
    except Exception:
        pass

from roop import core
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--execution-provider', default='cuda', help='Execution provider: cpu or cuda')
args = parser.parse_args()
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
globals.execution_providers = [_PROVIDER_NAMES.get(
    args.execution_provider.lower(), args.execution_provider + 'ExecutionProvider')]

if __name__ == '__main__':
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
    # to the React shell.  The API owns the port, so publish the detected
    # address here instead of making the launcher guess or hard-code it.
    api_port = int(os.environ.get("ROOP_API_PORT", "8001"))
    print(f"[Backend] listening on http://127.0.0.1:{api_port}", flush=True)

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
        print("[Backend] the legacy Gradio UI has stopped. This does NOT affect "
              "the React client.", flush=True)
        print(f"[Backend] still serving the API on http://127.0.0.1:{api_port} "
              f"- stop this script in Pinokio to shut it down.", flush=True)
        try:
            while api_thread.is_alive():
                api_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            print("[Backend] interrupted; shutting down.", flush=True)
