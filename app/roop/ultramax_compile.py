"""Build/warm the actual local UltraMax graph and report real cache files."""
from pathlib import Path


def compile_session(precision, provider, batch_size=1):
    import onnxruntime as ort
    from roop import globals as runtime, aux_canary
    from roop.model_integrity import require_local_model
    from roop.precision_policy import providers_for
    from roop.engine_cache import qualify_providers
    from roop.predictor import warmup_session
    from roop.utilities import get_onnx_session_options

    model = require_local_model('CodeFormer/codeformer.fp16.onnx')
    expected = {'tensorrt': 'TensorrtExecutionProvider', 'cuda': 'CUDAExecutionProvider', 'cpu': 'CPUExecutionProvider'}[provider]
    if expected not in ort.get_available_providers():
        raise RuntimeError(f'{expected} is unavailable; no engine was compiled')
    configured = list(runtime.execution_providers or [])
    if provider == 'cpu':
        configured = ['CPUExecutionProvider']
    elif provider == 'cuda':
        configured = [p for p in configured if 'Tensorrt' not in str(p)]
    if not any(expected in str(p) for p in configured):
        raise RuntimeError(f'{expected} has no admitted runtime configuration')
    chain, _decision = providers_for('ultramax', configured, model,
                                    requested='fp32' if precision == 'strict_fp32' else 'mixed')
    chain = qualify_providers(chain, model, batch_size=batch_size)
    def build(providers):
        return ort.InferenceSession(model, get_onnx_session_options(
            optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED), providers=providers)
    session = build(chain)
    if not warmup_session(session, 'ultramax:compile', once=False):
        raise RuntimeError('UltraMax warm-up did not execute')
    import numpy as np
    from roop.predictor import _concrete_shape, _numpy_dtype, DEFAULT_WARMUP_HW
    feed = {meta.name: np.zeros(_concrete_shape(meta, DEFAULT_WARMUP_HW), dtype=_numpy_dtype(meta))
            for meta in session.get_inputs()}
    for output in session.run(None, feed):
        if not np.isfinite(output).all():
            raise RuntimeError('UltraMax compilation produced non-finite output')
    session, chain, _verdict = aux_canary.guard('codeformer.fp16', session, model, chain, build)
    actual = session.get_providers()[0]
    if actual != expected:
        raise RuntimeError(f'Requested {expected}, but verification selected {actual}; no requested-engine success')
    engine = None
    if provider == 'tensorrt':
        options = session.get_provider_options().get(expected, {})
        cache = Path(options.get('trt_engine_cache_path') or '.')
        prefix = options.get('trt_engine_cache_prefix', '')
        engines = list(cache.glob(f'{prefix}*.engine')) if prefix else []
        if not engines:
            raise RuntimeError('TensorRT warm-up produced no matching cached .engine file')
        engine = str(max(engines, key=lambda p: p.stat().st_mtime_ns).resolve())
    return {'engine_path': engine, 'provider': actual}
