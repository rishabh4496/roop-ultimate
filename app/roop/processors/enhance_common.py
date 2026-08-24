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

    CALL THIS ON THE FLOAT OUTPUT, BEFORE THE uint8 CAST. On an integer array
    `np.isfinite` is always True, so a call placed after `convertScaleAbs`
    cannot ever fire — it reads as a safety net and is not one. All three of
    GPEN 256 Pro, GPEN Realistic and UltraMax had it on the wrong side (fixed
    2026-08-24); they now use the cheap `np.isfinite(sum)` before the cast and
    `looks_collapsed` after it, which is the check that CAN see a uint8 result.

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


def luma_only_recolour(restored, source, chroma=0.0, lab_exact=False):
    """The restorer's LUMINANCE carried on the SOURCE's chrominance.

    THE FAILURE THIS FIXES. Every GAN restorer here drifts in colour, and each
    drifts its own way: GPEN pushes the whole face pink and paints magenta onto
    the eyelids, while CodeFormer does the opposite — it desaturates and lifts.
    Measured against the crop the restorer was handed, over 5 real frames of
    s1.mp4 (`tests/diag_ultramax_cost_and_colour.py`):

        enhancer          chroma drift   dLAB-a   dLAB-b     dL   saturation
        UltraMax                  2.51    -0.96    +0.35   +2.22       x0.958
        GPEN Realistic            0.31    -0.11    -0.08   +4.06       x0.966
        GPEN 256 Pro              0.38    -0.07    -0.00   +1.54       x1.011

    Less red, brighter, less saturated is exactly what "pale" looks like, and
    the two GPEN processors do not have it for one reason only: they already
    run this function. UltraMax did not, and the user reported the difference
    before anything here was measured.

    THE MECHANISM. A luminance-only edit is the same signed offset on all three
    BGR channels, so adding `grey(restored) - grey(source)` to the SOURCE moves
    brightness and leaves hue and saturation exactly where the swapper put
    them. Two C++ passes, 0.27 ms at 512, and it cannot touch detail: the
    restored image's every high-frequency variation survives in the grey delta.

    `lab_exact` swaps in a true LAB L-channel replacement. It is more precise
    (0.11 residual drift against the grey delta's 0.30, on a ~2.9 problem) and
    more than twice the cost; the two are indistinguishable on footage.

    `chroma` lerps back toward the restorer's own colour, for re-measuring only.
    0 is the default and the entire point.

    Raises cv2.error rather than swallowing it. A colour fix that silently
    no-ops leaves a plausible image carrying the exact cast it exists to
    remove, and nothing anywhere would show it — so each caller catches this
    and says so once.
    """
    import numpy as np
    if lab_exact:
        lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)[:, :, 0]
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        d = cv2.subtract(cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY),
                         cv2.cvtColor(source, cv2.COLOR_BGR2GRAY),
                         dtype=cv2.CV_16S)
        out = cv2.add(source, cv2.merge((d, d, d)), dtype=cv2.CV_8U)
    if chroma > 0.0:
        out = cv2.addWeighted(out, 1.0 - chroma, restored, chroma, 0.0)
    return out


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

    COST, because this runs on EVERY face of EVERY restorer that has a pool.
    The obvious spelling — `np.asarray(x, np.float32).std()` — allocates a
    786k-element float copy of each 512 image and then makes a second pass over
    it, and measured **3.28 ms per face** on an RTX 4070: 11.3% of UltraMax's
    entire 33.5 ms Run(), for a guard that fires approximately never.
    `cv2.meanStdDev` reads the uint8 directly in C++ and costs **0.318 ms**,
    10.3x less. It is not an approximation — see `_global_std`.
    """
    try:
        s_std = _global_std(source)
        return s_std > 8.0 and _global_std(result) < s_std * 0.35
    except Exception:
        return False


def _global_std(img):
    """The population std over ALL channels, via one C++ pass.

    `cv2.meanStdDev` returns PER-CHANNEL mean and std, which is not what the
    caller wants — the std of the flattened image also carries the spread
    BETWEEN the channel means. Recombining them exactly is the parallel-axis
    theorem: pooling equal-sized groups,

        var_total = mean_c(var_c + mean_c^2) - (mean_c mean_c)^2

    This is exact, not an estimate. Verified against `np.float32(img).std()` on
    a random 512x512x3: 73.9585919644853 vs 73.95858764648438 — the whole
    difference is float32 vs float64 accumulation inside numpy's own reduction,
    and it is 6 orders of magnitude below the 0.35 ratio being tested.

    A decimated view (`img[::4, ::4]`) is another 2.4x cheaper and was measured
    too, but it is an APPROXIMATION (74.10 against 73.96 on the same array) and
    there is no reason to accept one for 0.19 ms.
    """
    import numpy as np
    m, sd = cv2.meanStdDev(img)
    m, sd = m.ravel(), sd.ravel()
    return float(np.sqrt(max(0.0, float(np.mean(sd * sd + m * m) - np.mean(m) ** 2))))
