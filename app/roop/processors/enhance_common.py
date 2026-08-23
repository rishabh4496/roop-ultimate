"""Shared post-processing contract for the face restorers.

Every enhancer here ends the same three lines — clip to [-1, 1], rescale to
[0, 255], cast to uint8 — and hands back `(frame, scale_factor)`. Two things
about that ending are traps, and both were found the expensive way in GPEN
before being written down here.
"""

import cv2
import numpy as np


def is_usable(result):
    """False when the model returned anything non-finite.

    `np.clip` does NOT remove NaN — it propagates it — and `uint8(NaN)` is 0.
    So a single overflowed value becomes a black pixel and a saturated graph
    becomes a completely black face, with no exception, no warning, and a
    perfectly normal-looking `(512, 512, 3) uint8` on the way out. Verified:

        np.clip(nan, -1, 1)                  -> nan   (inf clips fine, nan does not)
        np.full(..., nan) -> post -> uint8   -> every value 0

    This is not hypothetical here. GPEN's 1024/2048 weights overflow in FP16
    under TensorRT and painted exactly that, which is why GPEN grew a guard;
    the frame upscaler hit the same thing (ESRGAN x4 goes black under TRT
    FP16). Any enhancer running FP16 on a graph nobody has stress-tested is one
    overflow away from it, and a black face reads as "the app is broken"
    rather than "this model overflowed".
    """
    return bool(np.isfinite(result).all())


def sized(result, input_size):
    """`(frame, scale_factor)` in the form paste_upscale expects.

    scale_factor is `result_width / input_size` as an INTEGER, because
    paste_upscale multiplies the paste matrix by it. That is fine while the
    model output is the same size as the crop or larger (512→1, 1024→2,
    2048→4), but a model SMALLER than the crop gives int(256/512) = 0, which
    collapses the paste matrix to zero and blanks the face.

    So a downscaling model is resized back to the crop size here and reports 1.
    The saving that tier exists for is in the network, not in carrying a
    smaller buffer through the paste — and an INTER_CUBIC upsample of a 256px
    crop costs a fraction of what the 512px net would have.
    """
    if result.shape[1] < input_size:
        result = cv2.resize(result, (input_size, input_size),
                            interpolation=cv2.INTER_CUBIC)
        return result, 1
    return result, max(1, int(result.shape[1] / input_size))


def fp32_trt_providers(providers, tag):
    """`providers` with TensorRT forced to FP32, on its own engine cache.

    Some restorers do not survive TensorRT's FP16 kernels. Two distinct
    failures have been seen, and the second is the dangerous one:

      OVERFLOW -> NaN. GPEN at 1024/2048. `np.clip` does not strip NaN and
        uint8(NaN) is 0, so the face comes out solid black. Loud, and
        `is_usable` catches it.

      COLLAPSE -> a flat image. GFPGAN v1.4. Its output range shrinks from
        [-1.00, 1.00] to [-0.47, -0.14] — every value finite, nothing to catch,
        and the result is a uniform grey face that still looks like "an image".
        Measured 2026-08-24: FP16 gave pixel std 16.0 and detail 0.08 against
        FP32's 65.2 and 4.35, a mean absolute difference of 59/255 from the CUDA
        reference, while FP32 matched CUDA to 0.03.

    `tag` keeps each model's FP32 engine in its own cache directory so it can
    never collide with the FP16 engines built for detection and the other
    stages. ROOP_<TAG>_FP16=1 opts back in, for re-measuring only.
    """
    import os
    if os.environ.get(f'ROOP_{tag.upper()}_FP16', '0') == '1':
        return providers
    patched = []
    for p in providers:
        if isinstance(p, (tuple, list)) and len(p) == 2 and 'tensorrt' in str(p[0]).lower():
            name, opts = p[0], dict(p[1])
            opts['trt_fp16_enable'] = False
            cache = opts.get('trt_engine_cache_path')
            if cache:
                fp32_cache = f'{cache}_{tag}_fp32'
                os.makedirs(fp32_cache, exist_ok=True)
                opts['trt_engine_cache_path'] = fp32_cache
            patched.append((name, opts))
        else:
            patched.append(p)
    return patched


def looks_collapsed(result, source):
    """True when a restorer returned a degenerate, near-uniform image.

    `is_usable` only rejects non-finite output, which misses a precision
    COLLAPSE: every value finite, but the dynamic range gone. That is exactly
    how GFPGAN's FP16 engine failed, and it shipped undetected because a flat
    grey face is still a valid-looking array.

    Deliberately conservative — it must never fire on a face that is
    legitimately low-contrast. It asks for a near-total loss of variation
    relative to the input the restorer was handed, which a real restoration
    never produces: the FP16 case measured 16.0 against the input's own spread,
    a quarter of what FP32 returned.
    """
    try:
        import numpy as np
        s_std = float(np.asarray(source, dtype=np.float32).std())
        r_std = float(np.asarray(result, dtype=np.float32).std())
        return s_std > 8.0 and r_std < s_std * 0.35
    except Exception:
        return False
