"""ROOP_TRT_BUILD_HEURISTICS: engines built with heuristics ON (shipped, mixed) vs OFF.

    env/Scripts/python.exe tests/ab_trt_build_heuristics.py [--calls 300] [--faces 40]

Prints the TensorRT / ORT versions, then for the detector (SCRFD det_10g at the live det size and at 640), the
swapper, XSeg and RestoreFormer++ builds each model's production session TWICE in one process -- once per flag
value, each in its own engine-cache namespace (the flag is part of `core.decode_execution_providers`'s
`builder_config` digest) -- plus a CUDA FP32 reference, and reports:

  * latency, batch 1: the two arms alternate call by call (ABAB...), so clock/thermal drift and the first
    arm's engine build land on both; median / p10 / p90 wall ms per `session.run`;
  * fidelity against the FP32 reference with the project's own metrics: detector box IoU / keypoint px,
    swapper SSIM (swap_canary's), XSeg mask IoU, RestoreFormer++ PSNR.

Latency alone would be a trap: swap_canary.py records the same options minus build heuristics producing a swapper
engine that runs at full speed and paints the wrong face (SSIM 0.75). The decision this feeds: if TensorRT >= 10
shows no latency difference, the item stops there.
"""
import argparse
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

FLAGS = ('1', '0')           # '1' = shipped (heuristics on under mixed); '0' = off


def pct(a, q):
    return float(np.percentile(a, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calls', type=int, default=300)
    ap.add_argument('--faces', type=int, default=40)
    ap.add_argument('--date', default='2026-10-06')
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, 'config.yaml'))
    g = ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), 'None', 'None', sync_config=True)
    import tensorrt
    print('TensorRT %s | onnxruntime %s | trt_precision=%s | det_size=%s | swap_model=%s'
          % (tensorrt.__version__, onnxruntime.__version__, g.CFG.trt_precision, g.face_detector_size,
             g.CFG.swap_model), flush=True)
    from roop import face_util as fu, swap_canary
    from roop.core import decode_execution_providers
    from roop.precision_policy import providers_for
    from roop.processors.FaceSwapInsightFace import _swap_providers
    from roop.utilities import (get_onnx_session_options, get_small_card_safe_providers,
                                resolve_relative_path)
    from roop.restore_ultra_optimizer import BUFFER_POOL
    from insightface.model_zoo.scrfd import SCRFD
    from insightface.utils import face_align

    # ---- real material -----------------------------------------------------------------------
    frames, crops256, crops512, embeds = [], [], [], []
    cap = cv2.VideoCapture(fixtures.clip('double/d4.mp4', required=True))
    i = 0
    while len(frames) < args.faces and i < 600:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i % 9:
            continue
        faces = fu.get_all_faces(fr) or []
        if not faces:
            continue
        frames.append(fr)
        f = faces[0]
        crops256.append(face_align.norm_crop(fr, f.kps, 256))
        crops512.append(face_align.norm_crop(fr, f.kps, 512))
        e = np.asarray(f.embedding, np.float32)
        embeds.append(e / np.linalg.norm(e))
    cap.release()
    print('real frames/crops: %d' % len(frames), flush=True)

    def sess_opts():
        return get_onnx_session_options()

    def build(path, providers):
        return onnxruntime.InferenceSession(path, sess_opts(), providers=providers)

    def cuda32():
        return [('CUDAExecutionProvider', {'use_tf32': '0', 'cudnn_conv_algo_search': 'DEFAULT'}),
                'CPUExecutionProvider']

    def providers_for_flag(flag):
        os.environ['ROOP_TRT_BUILD_HEURISTICS'] = flag
        return decode_execution_providers(['tensorrt'])

    def interleaved(fns, n, warm=15):
        """fns: {label: callable()}. Alternate calls round-robin; returns ms arrays."""
        labels = list(fns)
        for _ in range(warm):
            for lb in labels:
                fns[lb]()
        out = {lb: [] for lb in labels}
        for _ in range(n):
            for lb in labels:
                t = time.perf_counter()
                fns[lb]()
                out[lb].append((time.perf_counter() - t) * 1000.0)
        return out

    results = {}

    def report(name, lat, fidelity):
        row = {'latency_ms': {lb: {'median': round(pct(v, 50), 3), 'p10': round(pct(v, 10), 3),
                                   'p90': round(pct(v, 90), 3)} for lb, v in lat.items()},
               'fidelity': fidelity}
        a, b = lat['heur_on'], lat['heur_off']
        row['off_over_on_median'] = round(pct(b, 50) / pct(a, 50), 4)
        results[name] = row
        print('%-22s on %.3f ms | off %.3f ms | off/on %.3f | %s'
              % (name, pct(a, 50), pct(b, 50), row['off_over_on_median'], json.dumps(fidelity)), flush=True)

    # ---- detector (SCRFD) at the live size and at 640 ---------------------------------------------
    det_path = os.path.join(resolve_relative_path('..'), 'models', 'buffalo_l', 'det_10g.onnx')
    for size in sorted({int(str(g.face_detector_size).split('x')[0]), 640}):
        t0 = time.time()
        sess = {}
        for flag, label in zip(FLAGS, ('heur_on', 'heur_off')):
            g.execution_providers = providers_for_flag(flag)
            prov = fu._face_analysis_providers()
            prov, _ = providers_for('recognition:buffalo_l', prov)
            sess[label] = build(det_path, prov)
        ref = build(det_path, cuda32())
        dets = {}
        for label, s in list(sess.items()) + [('ref', ref)]:
            d = SCRFD(model_file=det_path, session=s)
            d.prepare(0, input_size=(size, size), det_thresh=float(g.face_detector_threshold))
            dets[label] = d
        blob = cv2.dnn.blobFromImage(cv2.resize(frames[0], (size, size)), 1.0 / 128, (size, size),
                                     (127.5, 127.5, 127.5), swapRB=True)
        name = sess['heur_on'].get_inputs()[0].name
        lat = interleaved({lb: (lambda s=s: s.run(None, {name: blob})) for lb, s in sess.items()}, args.calls)

        def iou(a, b):
            x0, y0 = max(a[0], b[0]), max(a[1], b[1])
            x1, y1 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, x1 - x0) * max(0, y1 - y0)
            u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / u if u > 0 else 0.0
        fid = {}
        for label in ('heur_on', 'heur_off'):
            ious, kpx, miss = [], [], 0
            for fr in frames:
                bb, kk = dets['ref'].detect(fr, max_num=0, metric='default')
                b2, k2 = dets[label].detect(fr, max_num=0, metric='default')
                if len(bb) != len(b2):
                    miss += 1
                    continue
                for x, y, kx, ky in zip(bb, b2, kk, k2):
                    ious.append(iou(x, y))
                    kpx.append(float(np.abs(kx - ky).max()))
            fid[label] = {'frames_with_different_count': int(miss), 'box_iou_min': round(float(min(ious or [0])), 4),
                          'kps_max_px': round(float(max(kpx or [0])), 2)}
        report('detector@%d' % size, lat, fid)
        print('   (built+measured in %.0fs)' % (time.time() - t0), flush=True)
        del sess, ref, dets

    # ---- swapper ------------------------------------------------------------------------------------
    sw_path = resolve_relative_path('../models/hyperswap_1a_256.onnx')
    t0 = time.time()
    sess = {}
    for flag, label in zip(FLAGS, ('heur_on', 'heur_off')):
        g.execution_providers = providers_for_flag(flag)
        sess[label] = build(sw_path, _swap_providers(g.execution_providers, 'swapper:hyperswap', sw_path))
    ref = build(sw_path, cuda32())
    inputs = {i.name: i for i in ref.get_inputs()}
    tname = [n for n, i in inputs.items() if len(i.shape) == 4][0]
    sname = [n for n, i in inputs.items() if len(i.shape) == 2][0]

    def swap_feed(k):
        c = cv2.cvtColor(crops256[k % len(crops256)], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return {tname: ((c - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32),
                sname: embeds[(k + 3) % len(embeds)][None].astype(np.float32)}
    feeds = [swap_feed(k) for k in range(len(crops256))]
    cnt = [0]

    def run_swap(s):
        def f():
            s.run(None, feeds[cnt[0] % len(feeds)])
            cnt[0] += 1
        return f
    lat = interleaved({lb: run_swap(s) for lb, s in sess.items()}, args.calls)
    fid = {}
    refs = [ref.run(None, f)[0] for f in feeds]
    for label, s in sess.items():
        ss = [swap_canary._compare(s.run(None, f)[0], r) for f, r in zip(feeds, refs)]
        fid[label] = {'ssim_min': round(float(min(ss)), 4), 'ssim_mean': round(float(np.mean(ss)), 4)}
    # the canary's own synthetic cases, as the app would gate it
    try:
        canary = {lb: swap_canary.check_engine(s, lambda: ref, tag='ab:' + lb).min_ssim
                  for lb, s in sess.items()}
        fid['canary_min_ssim'] = canary
    except Exception as exc:                                  # pragma: no cover - harness only
        fid['canary_error'] = '%s: %s' % (type(exc).__name__, str(exc)[:120])
    report('swapper(hyperswap)', lat, fid)
    print('   (built+measured in %.0fs)' % (time.time() - t0), flush=True)
    del sess, ref

    # ---- XSeg ---------------------------------------------------------------------------------------
    xp = resolve_relative_path('../models/xseg.onnx')
    t0 = time.time()
    sess = {}
    for flag, label in zip(FLAGS, ('heur_on', 'heur_off')):
        g.execution_providers = providers_for_flag(flag)
        base = get_small_card_safe_providers(g.execution_providers, model_path=xp, stage='mask:xseg')
        prov, _ = providers_for('masking:xseg', base, xp)
        sess[label] = build(xp, prov)
    ref = build(xp, cuda32())
    xn = ref.get_inputs()[0].name

    def xseg_in(c):
        return (cv2.resize(c, (256, 256), interpolation=cv2.INTER_CUBIC).astype('float32') / 255.0)[None]
    xfeeds = [{xn: xseg_in(c)} for c in crops256]
    cnt = [0]

    def run_x(s):
        def f():
            s.run(None, xfeeds[cnt[0] % len(xfeeds)])
            cnt[0] += 1
        return f
    lat = interleaved({lb: run_x(s) for lb, s in sess.items()}, args.calls)

    def mask(out):
        o = np.clip(out[0][0], 0, 1.0)
        o[o < 0.1] = 0
        return (1.0 - o) > 0.5
    fid = {}
    rm = [mask(ref.run(None, f)) for f in xfeeds]
    for label, s in sess.items():
        v = []
        for f, r in zip(xfeeds, rm):
            m = mask(s.run(None, f))
            u = float((m | r).sum())
            v.append(1.0 if u == 0 else float((m & r).sum()) / u)
        fid[label] = {'iou_min': round(min(v), 4), 'iou_mean': round(float(np.mean(v)), 5)}
    report('xseg', lat, fid)
    print('   (built+measured in %.0fs)' % (time.time() - t0), flush=True)
    del sess, ref

    # ---- RestoreFormer++ --------------------------------------------------------------------------
    rp = resolve_relative_path('../models/restoreformer_plus_plus.onnx')
    t0 = time.time()
    sess = {}
    for flag, label in zip(FLAGS, ('heur_on', 'heur_off')):
        g.execution_providers = providers_for_flag(flag)
        prov, _ = providers_for('restoreformer_pp', g.execution_providers, rp)
        sess[label] = build(rp, prov)
    ref = build(rp, cuda32())
    rn = ref.get_inputs()[0].name
    # prepare_model_input hands back a REUSED buffer: copy, or every feed aliases the last crop
    rfeeds = [{rn: BUFFER_POOL.prepare_model_input(c).copy()} for c in crops512]
    cnt = [0]

    def run_r(s):
        def f():
            s.run(None, rfeeds[cnt[0] % len(rfeeds)])
            cnt[0] += 1
        return f
    lat = interleaved({lb: run_r(s) for lb, s in sess.items()}, max(60, args.calls // 4), warm=8)

    def psnr(a, b):
        mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
        return 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
    fid = {}
    rr = [BUFFER_POOL.postprocess_model_output(ref.run(None, f)[0][0]) for f in rfeeds]
    for label, s in sess.items():
        v = [psnr(BUFFER_POOL.postprocess_model_output(s.run(None, f)[0][0]), r) for f, r in zip(rfeeds, rr)]
        fid[label] = {'psnr_min_db': round(float(min(v)), 2), 'psnr_mean_db': round(float(np.mean(v)), 2)}
    report('restoreformer++', lat, fid)
    print('   (built+measured in %.0fs)' % (time.time() - t0), flush=True)

    os.environ.pop('ROOP_TRT_BUILD_HEURISTICS', None)
    path = os.path.join(REPO, 'docs', 'perf', 'trt_build_heuristics_%s.json' % args.date)
    json.dump({'tensorrt': tensorrt.__version__, 'onnxruntime': onnxruntime.__version__, 'results': results},
              open(path, 'w'), indent=1)
    print('wrote', path)


if __name__ == '__main__':
    main()
