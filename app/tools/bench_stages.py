"""Isolated GPU cost of every swappable model, per pipeline stage.

The swap phase runs the GPU at ~98%, so wall time is set by total GPU work per
frame and nothing else. That makes "which model" the main remaining lever, and
the only honest way to compare is one model at a time with nothing else on the
card. Run it, then put the numbers where the user makes the choice.

Three traps this file exists to avoid, each of which silently yields CPU numbers
that look like an ordinary result (GFPGAN measured 483 ms/call on CPU against 12
ms on TensorRT — a 43x error that arrives looking plausible):

  1. `import torch` BEFORE `onnxruntime` makes ORT reject the CUDA EP.
  2. `tensorrt_libs` must be on the DLL search path.
  3. That is still not enough: onnxruntime_providers_tensorrt.dll also needs
     cublas64_12.dll, which in this environment ships inside torch/lib. Adding
     the DIRECTORY does not import torch, so it does not trip trap 1.

The resolved provider is printed per model rather than assumed.

Usage:  python tools/bench_stages.py [enhancer|mask|swap|all]
"""
import os, sys, time

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../app
os.chdir(APP)

for _d in ('env/Lib/site-packages/tensorrt_libs',
           'env/Lib/site-packages/torch/lib',
           'env/Lib/site-packages/nvidia/cuda_runtime/bin'):
    _p = os.path.abspath(os.path.join(APP, _d))
    if os.path.isdir(_p):
        os.add_dll_directory(_p)
        os.environ['PATH'] = _p + os.pathsep + os.environ['PATH']
    else:
        print(f'WARN missing DLL dir {_p}')

import onnxruntime           # noqa: E402  MUST precede torch
import numpy as np           # noqa: E402

CACHE = os.path.join(APP, 'models', 'trt_cache', 'mixed')
TRT_OPTS = {
    'device_id': 0,
    'trt_fp16_enable': True,
    'trt_engine_cache_enable': True,
    'trt_engine_cache_path': CACHE,
    'trt_max_partition_iterations': 2000,
}
PROVIDERS = [('TensorrtExecutionProvider', TRT_OPTS),
             'CUDAExecutionProvider', 'CPUExecutionProvider']

# `note` carries what the raw ms/call does NOT tell you on its own.
STAGES = {
    # GPEN 1024/2048 are deliberately absent. The app forces them onto a FP32
    # TensorRT engine (they overflow in FP16 and return a black face, see
    # Enhance_GPEN._fp32_trt_providers), so timing them under the FP16 options
    # below would report a configuration that never runs — flatteringly. They
    # need their own FP32 cache to be measured honestly. DMDNet is absent too:
    # it is a .pth torch model, not single-file ONNX.
    'enhancer': [
        ('RestoreFormer++', 'models/restoreformer_plus_plus.onnx', 'app default'),
        ('CodeFormer',      'models/CodeFormer/CodeFormerv0.1.onnx', 'has a fidelity input'),
        ('GPEN-BFR-512',    'models/GPEN-BFR-512.onnx', ''),
        ('GPEN-BFR-256',    'models/gpen_bfr_256.onnx', 'output resized back to crop size'),
        ('GFPGAN v1.4',     'models/GFPGANv1.4.onnx', ''),
    ],
    'mask': [
        ('DFL XSeg',        'models/xseg.onnx', ''),
        ('XSeg 3',          'models/xseg_3.onnx', ''),
        ('Occluder',        'models/face_occluder.onnx', ''),
        ('FaceParser',      'models/resnet18.onnx', 'BiSeNet'),
        ('FastSAM',         'models/fastsam_s.onnx', ''),
        ('MobileSAM enc',   'models/mobilesam_encoder.onnx', 'enc+dec BOTH run per call'),
        ('MobileSAM dec',   'models/mobilesam_decoder.onnx', 'enc+dec BOTH run per call'),
    ],
    # output_size drives pixel-boost tiling: tiles = (subsample/output_size)**2,
    # so a 128 model at the 256px default runs FOUR times per face. Per-face cost
    # is tiles x ms/call, not ms/call.
    'swap': [
        ('inswapper',    'models/inswapper_128.onnx', '128 -> 4 tiles @256'),
        ('reswapper',    'models/reswapper_256.onnx', '256 -> 1 tile'),
        ('hyperswap_1a', 'models/hyperswap_1a_256.onnx', '256 -> 1 tile'),
        ('hyperswap_1b', 'models/hyperswap_1b_256.onnx', '256 -> 1 tile'),
        ('hyperswap_1c', 'models/hyperswap_1c_256.onnx', '256 -> 1 tile'),
        ('ghost_1',      'models/ghost_1_256.onnx', '256 -> 1 tile'),
        ('ghost_3',      'models/ghost_3_256.onnx', '256 -> 1 tile'),
        ('simswap',      'models/simswap_256.onnx', '256 -> 1 tile'),
        ('simswap_512',  'models/simswap_unofficial_512.onnx', '512 native'),
        ('blendswap',    'models/blendswap_256.onnx', '256 -> 1 tile'),
        ('uniface',      'models/uniface_256.onnx', '256 -> 1 tile'),
        ('hififace',     'models/hififace_unofficial_256.onnx', '256 -> 1 tile'),
    ],
}

_NP = {
    'tensor(float)': np.float32, 'tensor(float16)': np.float16,
    'tensor(double)': np.float64, 'tensor(int64)': np.int64,
    'tensor(int32)': np.int32, 'tensor(bool)': np.bool_,
}
# These were 5 and 25, and at those counts this tool invented differences that
# are not there. Each model is loaded, warmed, timed and freed in turn, so five
# warm-up calls leave the measurement sitting on whatever the GPU clocks and the
# TensorRT context are doing immediately after that model's own session was
# built — and that is not the same for the first row as for the sixth.
#
# It showed up on a pair that cannot differ: xseg.onnx and xseg_3.onnx are the
# SAME network (430 nodes, 233 initializers, identical op set, identical opset
# version — only the weights and the tensor names differ). At 5/25 they measured
# 3.36 and 5.83 ms, a 1.74x gap. At 60/150 they measure 2.54 and 2.75, and
# interleaved in one warmed process, alternating which goes first over 8 rounds,
# 2.564 +/- 0.032 against 2.570 +/- 0.054 — the same number.
#
# Every row was inflated and the RATIOS were wrong, which is worse, because the
# ratios are the part that gets quoted and acted on. Occluder was reported as the
# most expensive mask engine (5.02, 1.71x) when it is in fact the cheapest
# (2.31, 0.91x).
#
# For two models within a few percent of each other, this still is not enough:
# build both sessions, warm both, then alternate. Between-process variance is
# larger than within-process variance, so a single sequential pass can never
# resolve a 5% gap however long each row runs.
N_WARM, N_ITER = 60, 150
rng = np.random.default_rng(0)


def make_feeds(sess):
    """Plausible inputs from the declared signature. Dtype is taken FROM the
    model — feeding float64 where float32 is declared fails inside ORT, which
    then reads as 'this model is broken' rather than 'the harness guessed'."""
    feeds, img_shape = {}, None
    for inp in sess.get_inputs():
        shape = [1 if (isinstance(d, str) or d is None or d < 1) else d
                 for d in inp.shape]
        dt = _NP.get(inp.type, np.float32)
        if len(shape) == 4:                       # image tensor
            arr = (rng.random(shape) * 2 - 1).astype(dt)
            img_shape = shape
        elif np.issubdtype(dt, np.floating):      # embedding / fidelity scalar
            arr = rng.random(shape).astype(dt)
        else:
            arr = np.ones(shape, dtype=dt)
        feeds[inp.name] = arr
    return feeds, img_shape


def bench(stage, rows):
    print(f'\n=== {stage} ===')
    print(f'{"model":16s} {"provider":10s} {"input":18s} {"init":>7s} '
          f'{"ms/call":>9s} {"rel":>6s}  note')
    base = None
    for name, rel, note in rows:
        path = os.path.join(APP, rel)
        if not os.path.exists(path):
            print(f'{name:16s} MISSING {rel}')
            continue
        t0 = time.perf_counter()
        try:
            sess = onnxruntime.InferenceSession(path, None, providers=PROVIDERS)
        except Exception as e:
            print(f'{name:16s} LOAD FAILED {type(e).__name__}: {str(e)[:60]}')
            continue
        init = time.perf_counter() - t0
        prov = sess.get_providers()[0].replace('ExecutionProvider', '')
        feeds, img_shape = make_feeds(sess)
        shp = 'x'.join(map(str, img_shape)) if img_shape else '?'
        out0 = sess.get_outputs()[0].name
        try:
            for _ in range(N_WARM):
                sess.run([out0], feeds)
            t = time.perf_counter()
            for _ in range(N_ITER):
                sess.run([out0], feeds)
            ms = (time.perf_counter() - t) / N_ITER * 1000
        except Exception as e:
            print(f'{name:16s} {prov:10s} {shp:18s} RUN FAILED '
                  f'{type(e).__name__}: {str(e)[:55]}')
            del sess
            continue
        if base is None:
            base = ms
        print(f'{name:16s} {prov:10s} {shp:18s} {init:6.1f}s {ms:8.2f} '
              f'{ms / base:5.2f}x  {note}')
        del sess


want = (sys.argv[1] if len(sys.argv) > 1 else 'all').lower()
for _stage, _rows in STAGES.items():
    if want in ('all', _stage):
        bench(_stage, _rows)

print('\nIsolated cost. In-app the same call reports several times this because '
      'N workers contend for one GPU — the RATIO is what transfers.')
print('Clip2Seg (torch CLIP) and SAM2 are not single-file ONNX and are not '
      'covered here.')
