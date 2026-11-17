"""Content and shape-qualified ORT TensorRT engine filenames.

The parent namespace still supplies the GPU, driver and runtime identity.
ORT's supported trt_engine_cache_prefix applies to .engine/.profile files;
renaming the provider-owned files afterwards would invalidate its references.
"""
from __future__ import annotations

import hashlib
import json
import re


def engine_cache_prefix(model_path, precision, input_shape, batch_profile, pixel_boost=None):
    from roop.precision_policy import _model_digest
    digest = _model_digest(str(model_path))
    shape = re.sub(r'[^A-Za-z0-9]+', 'x', str(input_shape)).strip('x')
    if len(shape) > 48:
        shape = hashlib.sha256(shape.encode()).hexdigest()[:16]
    profile = hashlib.sha256(json.dumps(batch_profile, sort_keys=True,
                                        default=str).encode()).hexdigest()[:16]
    return f'm{digest}_{precision}_s{shape}_b{profile}_pb{pixel_boost or 0}'


def qualify_providers(providers, model_path, pixel_boost=None, batch_size=None):
    if not model_path:
        return list(providers or ())
    from roop.trt_shape_profile import graph_inputs
    from roop import globals as runtime
    inputs = [(spec.name, spec.dims) for spec in graph_inputs(model_path)]
    result = []
    for provider in providers or ():
        if isinstance(provider, (list, tuple)) and 'tensorrt' in provider[0].lower():
            name, options = provider[0], dict(provider[1])
            enabled = lambda key: str(options.get(key, False)).lower() in ('1', 'true', 'on')
            precision = ('int8' if enabled('trt_int8_enable') else
                         'bf16' if enabled('trt_bf16_enable') else
                         'fp16' if enabled('trt_fp16_enable') else 'fp32')
            profile = {key: options.get(key, '') for key in
                       ('trt_profile_min_shapes', 'trt_profile_opt_shapes', 'trt_profile_max_shapes')}
            profile['requested_batch'] = int(batch_size if batch_size is not None else getattr(runtime, 'ultramax_batch_size', 1) or 1)
            # Include effective build constraints, including mixed precision's
            # FP32 islands. Two builds with the same FP16 flag can differ.
            profile['constraints'] = {k: v for k, v in options.items()
                                      if k.startswith('trt_') and 'cache' not in k}
            options['trt_engine_cache_prefix'] = engine_cache_prefix(
                model_path, precision, inputs, profile, pixel_boost if pixel_boost is not None else getattr(runtime, 'subsample_size', None))
            result.append((name, options))
        else:
            result.append(provider)
    return result
