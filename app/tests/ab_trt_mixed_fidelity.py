"""XSeg and RestoreFormer++ on the production TensorRT session vs FP32, on real aligned crops.

    env/Scripts/python.exe tests/ab_trt_mixed_fidelity.py [--faces 120]

The precision policy ships both on TensorRT 'mixed' (FP16 kernels, layer-norm FP32 fallback). This asks
what that costs in fidelity, against the gate the project states for it: mask IoU >= 0.995 and enhancer
PSNR >= 45 dB versus an FP32 baseline.

Arms, all fed the same crops through the processors' own pre/post-processing:
  ref       CUDA EP, FP32, TF32 off  -- the reference (also checked against the CPU EP)
  tf32      CUDA EP, FP32, TF32 on   -- a second reduced-precision control: if it reproduces `ref`, the
                                        metric is not simply hypersensitive
  trt_mix   the production session   (providers_for(..., 'mixed'))
  trt_fp32  TensorRT with FP16 off   (providers_for(..., requested='fp32'))
and steady wall ms per call at batch 1, which is what a render pays.

It builds the production sessions exactly the way the processors do (same providers_for, same session
options) and it also prints, per model, which execution provider ran each node under ORT profiling:
a session whose first provider is TensorRT can still have partitions on CUDA/CPU.
"""
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2                                                 # noqa: E402
import numpy as np                                         # noqa: E402
import onnxruntime                                         # noqa: E402
import angle_bench as ab                                   # noqa: E402
import fixtures                                            # noqa: E402

IOU_GATE, PSNR_GATE = 0.995, 45.0
CLIPS = (('double/d4.mp4', 0, 11), ('Love.mp4', 1100, 13), ('double/d1.mp4', 0, 9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--faces', type=int, default=120)
    ap.add_argument('--date', default='2026-10-05')
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, 'config.yaml'))
    g = ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), 'None', 'None', sync_config=True)
    from roop import face_util as fu
    from roop.precision_policy import providers_for
    from roop.utilities import (get_onnx_session_options, get_small_card_safe_providers,
                                resolve_relative_path)
    from roop.restore_ultra_optimizer import BUFFER_POOL
    from insightface.utils import face_align

    crops256, crops512 = [], []
    for rel, lo, step in CLIPS:
        cap = cv2.VideoCapture(fixtures.clip(rel, required=True))
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        got, i = 0, 0
        while got < args.faces and i < 400:
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            if i % step:
                continue
            for f in fu._detect_faces_raw(fr, aux=False, unclamped=True) or []:
                crops256.append(face_align.norm_crop(fr, f.kps, 256))
                crops512.append(face_align.norm_crop(fr, f.kps, 512))
                got += 1
        cap.release()
    crops256, crops512 = crops256[:args.faces], crops512[:args.faces]
    print('real aligned crops: %d' % len(crops256), flush=True)

    def build(path, providers, profiling=False):
        so = get_onnx_session_options()
        if profiling:
            so.enable_profiling = True
        return onnxruntime.InferenceSession(path, so, providers=providers)

    def cuda32(tf32):
        return [('CUDAExecutionProvider', {'use_tf32': '1' if tf32 else '0', 'cudnn_conv_algo_search': 'DEFAULT'}),
                'CPUExecutionProvider']

    def timed(fn, items, warm=8):
        for c in items[:warm]:
            fn(c)
        t = time.perf_counter()
        for c in items:
            fn(c)
        return round((time.perf_counter() - t) / len(items) * 1000, 2)

    def node_providers(path, providers, feed):
        sess = build(path, providers, profiling=True)
        for _ in range(10):
            sess.run(None, feed)
        prof = sess.end_profiling()
        by = collections.Counter()
        for e in json.load(open(prof)):
            if e.get('cat') == 'Node' and e.get('name', '').endswith('_kernel_time'):
                by[e.get('args', {}).get('provider')] += 1
        os.remove(prof)
        return {k: round(v / 10.0, 1) for k, v in by.items()}

    out = {'faces': len(crops256), 'gates': {'mask_iou': IOU_GATE, 'enhancer_psnr_db': PSNR_GATE}}

    # ---- XSeg -------------------------------------------------------------------------------
    xp = resolve_relative_path('../models/xseg.onnx')
    base = get_small_card_safe_providers(g.execution_providers, model_path=xp, stage='mask:xseg')
    mix, _ = providers_for('masking:xseg', base, xp)
    t32, _ = providers_for('masking:xseg', base, xp, requested='fp32')
    sess = {'ref': build(xp, cuda32(False)), 'tf32': build(xp, cuda32(True)),
            'trt_mix': build(xp, mix), 'trt_fp32': build(xp, t32)}
    name = sess['ref'].get_inputs()[0].name

    def xseg(s, crop):
        x = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC).astype('float32') / 255.0
        o = np.clip(s.run(None, {name: x[None, ...]})[0][0], 0, 1.0)
        o[o < 0.1] = 0
        return (1.0 - o)[..., 0] if o.ndim == 3 else 1.0 - o

    def iou(a, b):
        ma, mb = a > 0.5, b > 0.5
        u = float((ma | mb).sum())
        return 1.0 if u == 0 else float((ma & mb).sum()) / u

    refs = [xseg(sess['ref'], c) for c in crops256]
    cpu = build(xp, ['CPUExecutionProvider'])
    ref_vs_cpu = min(iou(xseg(cpu, c), r) for c, r in zip(crops256[:15], refs[:15]))
    res = {'reference_vs_cpu_iou_min': ref_vs_cpu,
           'nodes_per_run_by_provider': node_providers(xp, mix, {name: np.zeros((1, 256, 256, 3), np.float32)})}
    for arm in ('tf32', 'trt_mix', 'trt_fp32'):
        v = np.array([iou(xseg(sess[arm], c), r) for c, r in zip(crops256, refs)])
        res[arm] = {'iou_min': float(v.min()), 'iou_mean': float(v.mean()), 'iou_p01': float(np.percentile(v, 1)),
                    'faces_below_gate': int((v < IOU_GATE).sum()),
                    'ms_per_call': timed(lambda c, a=arm: xseg(sess[a], c), crops256)}
    out['xseg'] = res
    print('XSEG', json.dumps(res), flush=True)
    del sess, cpu

    # ---- RestoreFormer++ -----------------------------------------------------------------------
    rp = resolve_relative_path('../models/restoreformer_plus_plus.onnx')
    mix, _ = providers_for('restoreformer_pp', g.execution_providers, rp)
    t32, _ = providers_for('restoreformer_pp', g.execution_providers, rp, requested='fp32')
    sess = {'ref': build(rp, cuda32(False)), 'tf32': build(rp, cuda32(True)),
            'trt_mix': build(rp, mix), 'trt_fp32': build(rp, t32)}
    iname = sess['ref'].get_inputs()[0].name

    def rf(s, crop):
        return BUFFER_POOL.postprocess_model_output(
            s.run(None, {iname: BUFFER_POOL.prepare_model_input(crop)})[0][0])

    def psnr(a, b):
        mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
        return 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)

    refs = [rf(sess['ref'], c) for c in crops512]
    res = {'nodes_per_run_by_provider': node_providers(rp, mix, {iname: np.zeros((1, 3, 512, 512), np.float32)})}
    for arm in ('tf32', 'trt_mix', 'trt_fp32'):
        ps = np.array([psnr(rf(sess[arm], c), r) for c, r in zip(crops512, refs)])
        res[arm] = {'psnr_min_db': float(ps.min()), 'psnr_mean_db': float(ps.mean()),
                    'psnr_p01_db': float(np.percentile(ps, 1)), 'faces_below_gate': int((ps < PSNR_GATE).sum()),
                    'ms_per_call': timed(lambda c, a=arm: rf(sess[a], c), crops512[:40])}
    out['restoreformer++'] = res
    print('RF++', json.dumps(res), flush=True)

    path = os.path.join(REPO, 'docs', 'perf', 'trt_mixed_fidelity_%s.json' % args.date)
    json.dump(out, open(path, 'w'), indent=1)
    print('wrote', path)


if __name__ == '__main__':
    main()
