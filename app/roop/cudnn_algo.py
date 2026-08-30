"""Per-model cuDNN convolution-algorithm policy, verified on the live device.

WHY THIS EXISTS. ORT's CUDA EP picks conv algorithms three ways:
`cudnn_conv_algo_search` = EXHAUSTIVE, HEURISTIC or DEFAULT. The first two go
through cuDNN's *frontend graph API*; DEFAULT uses the legacy path. This app
sets HEURISTIC globally in core.py because it is much faster -- measured on an
RTX 3060 Laptop (Ampere, sm86, ORT 1.23.2, driver 616.56):

    hyperswap_1a_256   24.1 ms HEURISTIC ->  82.2 ms DEFAULT   (+241%)
    GFPGANv1.4         83.6 ms           -> 175.1 ms           (+110%)
    GPEN-BFR-512      185.5 ms           -> 387.4 ms           (+109%)
    gpen_bfr_256       25.3 ms           ->  49.4 ms           ( +96%)
    retinaface_r50      7.5 ms           ->  12.1 ms           ( +61%)
    xseg               23.6 ms           ->  36.7 ms           ( +55%)

So HEURISTIC must stay the default. But the CodeFormer family cannot use it on
that device at all:

    Conv '/blocks.3/conv/Conv'
      CUDNN_FE failure 8: HEURISTIC_QUERY_FAILED  (conv.cc:225, create_execution_plans)
      CUDNN_FE failure 7: GRAPH_EXECUTION_FAILED  (conv.cc:485, graph->execute)

HEURISTIC_QUERY_FAILED is the primary error and GRAPH_EXECUTION_FAILED is its
consequence; only the second reaches the render log, which is why this looked
like an execution fault rather than a planning one. EXHAUSTIVE fails the same
way (also a frontend path). `cudnn_conv_use_max_workspace=0` does NOT help, so
it is not a workspace-size problem. DEFAULT works:

    codeformer.fp16   RUN-FAIL HEURISTIC ->  316.6 ms DEFAULT  (only working mode)
    restoreformer++     1815.3 ms        ->  346.0 ms          (-81%, also FASTER)

MEASURED CONSEQUENCE IN THE PRODUCT. With HEURISTIC, ProcessMgr catches the
per-frame GPU error and writes the ORIGINAL frame, so Codeformer, Codeformer
(fp16), UltraMax (which runs codeformer.fp16 internally) and Restoreformer++
produced 60/60 unswapped frames while the swap audit still reported
"swapped (every face) 100.0%". Only the identity check caught it (0.961 against
a good swap's 0.41-0.45).

WHY THIS IS PROBED AND NOT HARDCODED. DEFAULT may well be slower than HEURISTIC
wherever HEURISTIC works, and the RTX 4070 runs UltraMax end to end today. Per
the dual-GPU validation rules a fix must not be applied globally on one card's
evidence, and capability must be detected rather than inferred from a GPU name.
So the suspect models are *probed once on the live device* and the verdict is
cached under the same GPU/driver/CUDA/ORT identity used for engine caches. A
device where HEURISTIC works keeps HEURISTIC and is completely unaffected.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading

# Models whose conv pattern is known to be capable of tripping the cuDNN
# frontend. Membership only means "probe this one"; it never by itself changes
# a setting. Anything not listed keeps the global HEURISTIC with no probe cost.
#
# These are the exact keys the processors pass to `providers_for`, not guesses:
#   Enhance_CodeFormer        'codeformer' / 'codeformer_fp16'
#   Enhance_RestoreFormerPPlus 'restoreformer_pp'
#   Enhance_UltraMax          'ultramax'
# A key that does not match a real call site would silently never probe, which
# is the failure mode this comment exists to prevent; tests/test_cudnn_algo.py
# asserts every name here is actually reachable from a processor.
SUSPECT_MODEL_KEYS = frozenset({
    'codeformer',
    'codeformer_fp16',
    'restoreformer_pp',
    'ultramax',
})

# Substrings of the cuDNN frontend failures above, matched case-insensitively.
_CUDNN_FE_MARKERS = (
    'cudnn_fe failure',
    'heuristic_query_failed',
    'graph_execution_failed',
    'failed to initialize cudnn frontend',
)

_LOCK = threading.Lock()
_MEMO = {}


def is_cudnn_frontend_error(exc):
    """Is this the cuDNN frontend planning/execution failure described above?"""
    text = str(exc).lower()
    return any(marker in text for marker in _CUDNN_FE_MARKERS)


def _cache_dir():
    return os.path.join(os.path.dirname(__file__), '..', 'models', 'runtime_profiles')


def _device_key(device_id=0):
    """GPU + driver + CUDA + ORT identity, so a verdict cannot cross devices."""
    try:
        from roop import backend_manager
        # Reuse the engine-cache identity: it already carries GPU name, sm,
        # CUDA, driver and ORT version, and is exactly what must invalidate
        # this verdict too.
        return backend_manager.cache_namespace('cudnnalgo', device_id)
    except Exception:
        return 'unknown-device'


def _cache_path(device_id=0):
    key = _device_key(device_id)
    safe = ''.join(c if (c.isalnum() or c in '._-') else '_' for c in key)
    return os.path.join(_cache_dir(), 'cudnn_algo_%s.json' % safe[:180])


def _load(device_id=0):
    try:
        with open(_cache_path(device_id), 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _store(model_key, algo, device_id=0):
    path = _cache_path(device_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = _load(device_id)
        data[model_key] = algo
        fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '.',
                                   dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
                fh.write('\n')
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except Exception:
        pass  # a cache we cannot write is a slow probe, never a failed render


def record(model_key, algo, device_id=0):
    """Persist a verdict for this model on this device."""
    with _LOCK:
        _MEMO[model_key] = algo
        _store(model_key, algo, device_id)


def known_algo(model_key, device_id=0):
    """The cached verdict for this model on this device, if any."""
    with _LOCK:
        if model_key in _MEMO:
            return _MEMO[model_key]
        algo = _load(device_id).get(model_key)
        if algo:
            _MEMO[model_key] = algo
        return algo


def apply_algo(providers, algo):
    """Return `providers` with the CUDA EP's conv algo search set to `algo`."""
    if not algo:
        return list(providers or ())
    out = []
    for entry in (providers or ()):
        if (isinstance(entry, (tuple, list)) and len(entry) == 2
                and str(entry[0]) == 'CUDAExecutionProvider'):
            opts = dict(entry[1] or {})
            opts['cudnn_conv_algo_search'] = algo
            out.append((entry[0], opts))
        else:
            out.append(entry)
    return out


def has_cuda_ep(providers):
    for entry in (providers or ()):
        name = entry[0] if isinstance(entry, (tuple, list)) else entry
        if str(name) == 'CUDAExecutionProvider':
            return True
    return False


def probe(model_key, model_path, providers, session_options=None, device_id=0):
    """Decide this model's conv algo on the live device, once, and cache it.

    Builds a throwaway session under HEURISTIC and runs one inference at the
    model's own declared input shape. If that raises the cuDNN frontend
    failure the verdict is DEFAULT; otherwise HEURISTIC is kept and this device
    is unaffected. The result is cached, so the probe is paid once per
    (model, GPU, driver, CUDA, ORT).
    """
    if not model_key or not has_cuda_ep(providers):
        return None
    cached = known_algo(model_key, device_id)
    if cached:
        return cached
    if model_key not in SUSPECT_MODEL_KEYS:
        return None
    if not model_path or not os.path.exists(model_path):
        return None

    verdict = 'HEURISTIC'
    sess = None
    try:
        import numpy as np
        import onnxruntime

        sess = onnxruntime.InferenceSession(
            model_path, session_options,
            providers=apply_algo(providers, "HEURISTIC"))
        # Without this, a failing CUDA EP silently re-initialises on the CPU EP
        # and the exception that surfaces is whatever the CPU path then hits --
        # for codeformer.fp16 a SimplifiedLayerNormFusion graph error -- so the
        # cuDNN signature never reaches is_cudnn_frontend_error() and the probe
        # records nothing. Keep the original EP error.
        try:
            sess.disable_fallback()
        except Exception:
            pass
        feed = {}
        for spec in sess.get_inputs():
            shape = [1 if (d is None or isinstance(d, str)) else int(d)
                     for d in (spec.shape or [])]
            t = str(spec.type)
            if 'float16' in t:
                dtype = np.float16
            elif 'double' in t:
                dtype = np.float64
            elif 'int64' in t:
                dtype = np.int64
            else:
                dtype = np.float32
            feed[spec.name] = np.zeros(shape, dtype=dtype)
        try:
            sess.run(None, feed)
        except Exception as exc:
            if is_cudnn_frontend_error(exc):
                verdict = 'DEFAULT'
                print('[cuDNN] %s: cuDNN frontend conv planning failed on this '
                      'device; using cudnn_conv_algo_search=DEFAULT for this '
                      'model. Other models keep HEURISTIC.' % model_key)
            else:
                # An unrelated failure says nothing about the algo search, so
                # do not cache a verdict off it.
                return None
    except Exception:
        return None
    finally:
        del sess

    record(model_key, verdict, device_id)
    return verdict
