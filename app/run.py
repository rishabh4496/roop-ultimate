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

    def _set(var, val):
        if val is None:
            return
        s = str(val).strip()
        if s and s.lower() != 'auto':
            os.environ[var] = s

    _set('ROOP_TRT_POOL', cfg.get('perf_trt_pool'))
    _set('ROOP_TRT_BUILDER_OPT_LEVEL', cfg.get('trt_builder_optimization_level'))
    _set('ROOP_TRT_AUX_STREAMS', cfg.get('trt_auxiliary_streams'))
    if cfg.get('trt_cuda_graph') is not None:
        graph = cfg.get('trt_cuda_graph')
        graph_on = graph is True or str(graph).strip().lower() in ('1', 'true', 'yes', 'on')
        os.environ['ROOP_TRT_CUDA_GRAPH'] = '1' if graph_on else '0'
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
        v = str(cfg.get(key, 'auto')).strip().lower()
        if v == 'on' or (v == 'auto' and var == 'ROOP_BATCH_SWAP'):
            os.environ[var] = '1'
            if var == 'ROOP_BATCH_SWAP':
                os.environ['ROOP_BATCH_SWAP_XFRAME'] = '1'
        elif v == 'off':
            os.environ[var] = '0'
            if var == 'ROOP_BATCH_SWAP':
                os.environ['ROOP_BATCH_SWAP_XFRAME'] = '0'

    # Not tri-state: a model choice and a priority class.
    _rec = str(cfg.get('recognizer', 'default')).strip().lower()
    if _rec == 'adaface':
        os.environ['ROOP_ADAFACE'] = '1'
    elif _rec == 'default':
        os.environ['ROOP_ADAFACE'] = '0'
    # Only the names keep_awake._PRIORITY_CLASSES accepts; it falls back to
    # 'high' for anything else, so passing a value it does not know through
    # would present as a working setting that does nothing.
    _pri = str(cfg.get('process_priority', 'auto')).strip().lower()
    if _pri in ('high', 'above_normal', 'normal'):
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
    threading.Thread(target=run_api, daemon=True).start()
    core.run()
