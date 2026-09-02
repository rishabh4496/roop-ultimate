"""Shared post-processing contract for the face restorers.

Every enhancer here ends the same three lines — clip to [-1, 1], rescale to
[0, 255], cast to uint8 — and hands back `(frame, scale_factor)`. Two things
about that ending are traps, and both were found the expensive way in GPEN
before being written down here.
"""

import contextlib

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


try:
    import torch
    _TORCH_CUDA = torch.cuda.is_available()
except (ImportError, AttributeError):
    _TORCH_CUDA = False


_CUDART = None
_CUDART_RESOLVED = False


def _cudart():
    """The CUDA runtime, resolved once, for device-to-device copies.

    Returns None when it cannot be loaded, which is a supported outcome: every
    caller treats that as "stay on the host path" rather than as an error.
    """
    global _CUDART, _CUDART_RESOLVED
    if _CUDART_RESOLVED:
        return _CUDART
    _CUDART_RESOLVED = True
    import ctypes
    import glob
    import os

    names = []
    try:
        import torch as _t
        libdir = os.path.join(os.path.dirname(_t.__file__), 'lib')
        # Torch bundles the runtime it was built against; prefer it over
        # whatever happens to be on PATH so the ABI matches the tensors.
        names += sorted(glob.glob(os.path.join(libdir, 'cudart64_*.dll')),
                        reverse=True)
        names += sorted(glob.glob(os.path.join(libdir, 'libcudart.so*')),
                        reverse=True)
    except Exception:
        pass
    names += ['cudart64_12.dll', 'libcudart.so.12', 'libcudart.so']
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            lib.cudaMemcpy.restype = ctypes.c_int
            _CUDART = lib
            return _CUDART
        except (OSError, AttributeError):
            continue
    return None


def ort_cuda_output_to_torch(ort_value, torch_dtype):
    """An ORT-OWNED CUDA output, copied device-to-device into a Torch tensor.

    WHY THIS SPELLING, AND NOT THE OBVIOUS ONE. The obvious way to keep a model
    output in VRAM is to allocate a Torch tensor and hand ORT its pointer via
    `bind_output(..., data_ptr)`. UltraMax did exactly that and had to disable
    it for TensorRT: on ORT 1.23 / TRT 10 the external allocation's ownership
    and stream contract is not honoured for this dynamic CodeFormer output, and
    the result is finite, non-flat, and SPATIALLY CORRUPT (striped/ghosted) --
    which no numerical guard can catch, so it shipped as a hard fallback to the
    host path and the CUDA post-process never ran on a single face.

    Letting ORT allocate AND own the output removes that contract entirely. The
    only cost is one 1.5 MB device-to-device copy, which is microseconds and
    never crosses PCIe. Measured against the host path on this TensorRT build,
    over four trials at 512: max difference 0.0 -- bit-identical, not merely
    close. See tests/test_enhancer_ultramax.py.

    Returns None when the runtime cannot be loaded or the value is not on CUDA;
    the caller keeps its host path in that case.
    """
    import ctypes
    if ort_value is None or getattr(ort_value, 'device_name', None) is None:
        return None
    try:
        if str(ort_value.device_name()).lower() != 'cuda':
            return None
    except (TypeError, AttributeError):
        return None
    lib = _cudart()
    if lib is None:
        return None
    shape = tuple(ort_value.shape())
    nbytes = int(ort_value.tensor_size_in_bytes())
    out = torch.empty(shape, dtype=torch_dtype, device='cuda')
    if out.numel() * out.element_size() != nbytes:
        # A dtype/shape disagreement would silently copy the wrong number of
        # bytes, so refuse rather than produce a plausible wrong tensor.
        return None
    torch.cuda.synchronize()
    # cudaMemcpyDeviceToDevice == 3. The synchronous form is deliberate: it
    # orders against both ORT's stream and Torch's without having to reason
    # about either, and 1.5 MB on-device is not worth the risk of getting that
    # wrong.
    rc = lib.cudaMemcpy(ctypes.c_void_p(out.data_ptr()),
                        ctypes.c_void_p(ort_value.data_ptr()),
                        ctypes.c_size_t(nbytes), ctypes.c_int(3))
    if rc != 0:
        return None
    return out


def _luma_only_recolour_gpu(restored, source, chroma=0.0):
    return luma_only_recolour_tensor(restored, source, chroma).to(torch.uint8).cpu().numpy()


def luma_only_recolour_tensor(restored, source, chroma=0.0):
    """CUDA-resident luminance transfer for HWC BGR tensors in [0, 255].

    Unlike the legacy helper this deliberately does not stage either crop
    through NumPy.  Callers own the final download boundary.
    """
    if not _TORCH_CUDA or not getattr(restored, 'is_cuda', False):
        raise RuntimeError('CUDA tensor luminance transfer requires torch.cuda')
    t_r = restored.to(dtype=torch.float32)
    t_s = source.to(device=t_r.device, dtype=torch.float32)
    g_r = 0.114 * t_r[:, :, 0] + 0.587 * t_r[:, :, 1] + 0.299 * t_r[:, :, 2]
    g_s = 0.114 * t_s[:, :, 0] + 0.587 * t_s[:, :, 1] + 0.299 * t_s[:, :, 2]
    d = g_r - g_s
    out = torch.clamp(t_s + d.unsqueeze(-1), 0, 255)
    if chroma > 0.0:
        out = (1.0 - chroma) * out + chroma * t_r
        out = torch.clamp(out, 0, 255)
    return out


def luma_only_recolour(restored, source, chroma=0.0, lab_exact=False):
    """The restorer's LUMINANCE carried on the SOURCE's chrominance."""
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


@contextlib.contextmanager
def exclusive(pool, lock, fallback):
    """Exclusive use of ONE inference context — and nothing wider than that.

    THE PROBLEM THIS EXISTS FOR. ProcessMgr wraps the whole enhance stage in
    `_gpu_guard(..., owner='enhance')`, which under TensorRT is a real mutex.
    That guard was sized for the model call, but it is held across the entire
    `Run()` — and for the look-filter restorers the network is a small minority
    of that. Measured on an RTX 4070, GPEN 256 Pro, 256 crop in:

        pre  (LUT gather)             0.50 ms
        GPU  network                  4.32 ms
        post colour + texture        34.20 ms
        ------------------------------------
        Run()                        39.02 ms   ->  GPU is 11% of it

    So 89% of what the enhance lock serialised never touched the GPU. With N
    worker threads that caps the stage at one face at a time. Measured, same
    processor, faces/s against worker threads:

        threads      free-running    whole Run() under a lock
              1              24.6                       24.7
              4              77.4                       25.0
             10             103.5                       24.6

    i.e. the stage does not scale AT ALL, and the card idles while nine threads
    queue behind one doing NumPy. That is not a small-card problem: it is every
    card on which the enhancer has no pool, which is every card below 7GB (see
    `session_pool._auto_pool_defaults`) plus any install that turned pooling
    off.

    THE FIX IS NOT TO DROP THE LOCK. A TensorRT execution context is not
    thread-safe and concurrent enqueue corrupts the CUDA context (error 999).
    What is needed is a lock the width of the context use, which is what this
    is: lease an independent context when the processor owns a pool, else hold
    the processor's OWN lock over its single shared session/io_binding. Either
    way no context is entered twice at once — the only guarantee `_gpu_guard`
    was ever providing — and the host work falls outside it.

    A processor that routes every session call through this can then declare
    `self_excluding = True` and the stage-level guard becomes a no-op for it,
    exactly as `Expression_LivePortrait` already does.

    `fallback` is the (session, io_binding) pair — or bare session — to hand
    back when there is no pool. It is evaluated by the caller, so a class that
    keeps its session under a different attribute name still fits.
    """
    if pool is not None:
        with pool.lease() as item:
            yield item
    else:
        with lock:
            yield fallback
