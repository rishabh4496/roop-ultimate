"""Static-shape TensorRT sessions: production call path vs persistent device I/O + user stream vs + CUDA graph.

    env/Scripts/python.exe tests/ab_trt_bound_sessions.py [--calls 300] [--stability 10000] [--models det,xseg,rfpp]

Arms, per model, all on the PRODUCTION provider options (same engine cache, no rebuild):
  prod   the call path the app makes today: SCRFD `session.run(numpy)`; XSeg and RestoreFormer++
         `bind_cpu_input` + an ORT-allocated device output + `copy_outputs_to_cpu`
  bound  roop.trt_bound_session.BoundStaticSession: persistent device buffers bound once, pinned staging,
         its own torch stream passed as user_compute_stream
  graph  the same plus `trt_cuda_graph_enable` (ORT captures the launch sequence after warm-up and replays it)

Reported: single-thread latency with the arms interleaved call by call; whether every output is
BIT-IDENTICAL to `prod` over real inputs; two-thread pooled throughput with one session (hence one stream)
per thread, as the pools run; and a long stability run per bound arm (every 100th call re-checked against its
first answer, latency drift first vs last 1000 calls, device memory before/after, any exception).
Keep an arm only where outputs are bit-identical and the long run is stable.
"""
import argparse
import json
import os
import sys
import threading
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


def pct(a, q):
    return round(float(np.percentile(a, q)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calls', type=int, default=300)
    ap.add_argument('--stability', type=int, default=10000)
    ap.add_argument('--models', default='det,xseg,rfpp')
    ap.add_argument('--det-sizes', default='', help='comma list; default live size and 640')
    ap.add_argument('--date', default='2026-10-06')
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, 'config.yaml'))
    g = ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), 'None', 'None', sync_config=True)
    import torch
    from roop import face_util as fu
    from roop.precision_policy import providers_for
    from roop.trt_bound_session import BoundStaticSession
    from roop.utilities import (get_onnx_session_options, get_small_card_safe_providers,
                                resolve_relative_path)
    from roop.restore_ultra_optimizer import BUFFER_POOL
    from insightface.utils import face_align

    frames, crops256, crops512 = [], [], []
    cap = cv2.VideoCapture(fixtures.clip('double/d4.mp4', required=True))
    i = 0
    while len(frames) < 40 and i < 600:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i % 9:
            continue
        fs = fu._detect_faces_raw(fr, aux=False, unclamped=True) or []
        if not fs:
            continue
        frames.append(fr)
        crops256.append(face_align.norm_crop(fr, fs[0].kps, 256))
        crops512.append(face_align.norm_crop(fr, fs[0].kps, 512))
    cap.release()
    print('real inputs: %d' % len(frames), flush=True)

    def so():
        return get_onnx_session_options()

    # ---- the production call paths -----------------------------------------------------------------
    class ProdPlain:                     # SCRFD: session.run(numpy)
        def __init__(self, path, providers, **_):
            self.s = onnxruntime.InferenceSession(path, so(), providers=providers)
            self.names = [o.name for o in self.s.get_outputs()]
            self.inn = self.s.get_inputs()[0].name

        def run(self, feed):
            return self.s.run(self.names, {self.inn: feed[self.inn]})

    class ProdBinding:                   # XSeg / RF++ processors: bind_cpu_input + ORT-allocated output
        def __init__(self, path, providers, **_):
            self.s = onnxruntime.InferenceSession(path, so(), providers=providers)
            self.inn = self.s.get_inputs()[0].name
            self.out = self.s.get_outputs()[0].name
            self.iob = self.s.io_binding()
            self.iob.bind_output(self.out, 'cuda')
            self.lock = threading.Lock()

        def run(self, feed):
            with self.lock:
                self.iob.bind_cpu_input(self.inn, feed[self.inn])
                self.s.run_with_iobinding(self.iob)
                return self.iob.copy_outputs_to_cpu()

    class Bound:
        def __init__(self, path, providers, cuda_graph, input_shapes, sample):
            self.b = BoundStaticSession(path, providers, input_shapes=input_shapes, cuda_graph=cuda_graph,
                                        session_options=so(), sample_feed=sample, outputs=ONLY.get(path))

        def run(self, feed):
            return self.b.run(feed)

    results = {}
    ONLY = {}      # model path -> the one output the app binds (XSeg/RF++ ProdBinding bind outputs[0])

    def measure(name, path, providers, prod_cls, feeds, input_shapes):
        print('\n== %s' % name, flush=True)
        sample = feeds[0]
        arms = {}
        t0 = time.time()
        arms['prod'] = prod_cls(path, providers)
        arms['bound'] = Bound(path, providers, False, input_shapes, sample)
        try:
            arms['graph'] = Bound(path, providers, True, input_shapes, sample)
        except Exception as exc:
            print('   graph arm failed to build: %s: %s' % (type(exc).__name__, str(exc)[:200]), flush=True)
        for lb in ('bound', 'graph'):
            if lb in arms:
                print('   %-5s providers %s' % (lb, arms[lb].b.active_providers), flush=True)
        print('   built in %.0fs' % (time.time() - t0), flush=True)
        row = {}
        # bit-identity vs prod over real inputs
        ref = [arms['prod'].run(f) for f in feeds]
        for lb in ('bound', 'graph'):
            if lb not in arms:
                continue
            same, worst = 0, 0.0
            try:
                for f, r in zip(feeds, ref):
                    out = arms[lb].run(f)
                    eq = all(np.array_equal(a, b) for a, b in zip(r, out))
                    same += int(eq)
                    if not eq:
                        worst = max(worst, max(float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
                                               for a, b in zip(r, out)))
                row[lb] = {'bit_identical_inputs': '%d/%d' % (same, len(feeds)), 'max_abs_diff_when_not': worst}
            except Exception as exc:
                row[lb] = {'error': '%s: %s' % (type(exc).__name__, str(exc)[:200])}
                arms.pop(lb)
        # latency, interleaved
        labels = list(arms)
        cnt = [0]
        for _ in range(15):
            for lb in labels:
                arms[lb].run(feeds[cnt[0] % len(feeds)])
        lat = {lb: [] for lb in labels}
        for _ in range(args.calls):
            f = feeds[cnt[0] % len(feeds)]
            cnt[0] += 1
            for lb in labels:
                t = time.perf_counter()
                arms[lb].run(f)
                lat[lb].append((time.perf_counter() - t) * 1000.0)
        for lb in labels:
            row.setdefault(lb, {})['latency_ms'] = {'median': pct(lat[lb], 50), 'p10': pct(lat[lb], 10), 'p90': pct(lat[lb], 90)}
        base = pct(lat['prod'], 50)
        for lb in labels:
            row[lb]['vs_prod'] = round(pct(lat[lb], 50) / base, 4)
        print('   latency median ms: ' + ' | '.join('%s %.3f (x%.3f)' % (lb, pct(lat[lb], 50), pct(lat[lb], 50) / base)
                                                    for lb in labels), flush=True)
        print('   identity: ' + ' | '.join('%s %s' % (lb, row[lb].get('bit_identical_inputs', row[lb].get('error')))
                                           for lb in labels if lb != 'prod'), flush=True)
        results[name] = row
        return arms

    def pooled(name, path, providers, prod_cls, feeds, input_shapes, n_calls):
        """Two pooled sessions per arm, one thread each: aggregate calls/s."""
        out = {}
        for lb in ('prod', 'bound', 'graph'):
            try:
                if lb == 'prod':
                    pool = [prod_cls(path, providers) for _ in range(2)]
                else:
                    pool = [Bound(path, providers, lb == 'graph', input_shapes, feeds[0]) for _ in range(2)]
            except Exception as exc:
                out[lb] = {'error': '%s: %s' % (type(exc).__name__, str(exc)[:160])}
                continue
            for s in pool:
                for f in feeds[:5]:
                    s.run(f)
            barrier = threading.Barrier(3)
            err = []

            def work(s, k):
                try:
                    barrier.wait()
                    for j in range(n_calls):
                        s.run(feeds[(j + k) % len(feeds)])
                except Exception as exc:                   # noqa: BLE001
                    err.append(exc)
            ts = [threading.Thread(target=work, args=(s, k)) for k, s in enumerate(pool)]
            for t in ts:
                t.start()
            barrier.wait()
            t0 = time.perf_counter()
            for t in ts:
                t.join()
            dt = time.perf_counter() - t0
            out[lb] = {'calls_per_s': round(2 * n_calls / dt, 1)} if not err else {'error': str(err[0])[:160]}
            if lb != 'prod':
                for s in pool:
                    s.b.close()
        base = out.get('prod', {}).get('calls_per_s')
        for lb, v in out.items():
            if base and 'calls_per_s' in v:
                v['vs_prod'] = round(v['calls_per_s'] / base, 3)
        print('   pooled x2 threads calls/s: ' + ' | '.join('%s %s' % (lb, v.get('calls_per_s', v.get('error')))
                                                             for lb, v in out.items()), flush=True)
        results[name]['pooled_2x'] = out

    def stability(name, path, providers, feeds, input_shapes, n):
        out = {}
        for lb, graph in (('bound', False), ('graph', True)):
            try:
                s = Bound(path, providers, graph, input_shapes, feeds[0])
            except Exception as exc:
                out[lb] = {'error': '%s: %s' % (type(exc).__name__, str(exc)[:160])}
                continue
            first = [s.run(f) for f in feeds]
            free0 = torch.cuda.mem_get_info()[0] / 2 ** 20
            lat, bad, err = [], 0, None
            try:
                for j in range(n):
                    f = j % len(feeds)
                    t = time.perf_counter()
                    o = s.run(feeds[f])
                    lat.append((time.perf_counter() - t) * 1000.0)
                    if j % 100 == 0 and not all(np.array_equal(a, b) for a, b in zip(first[f], o)):
                        bad += 1
            except Exception as exc:                       # noqa: BLE001
                err = '%s: %s' % (type(exc).__name__, str(exc)[:160])
            free1 = torch.cuda.mem_get_info()[0] / 2 ** 20
            out[lb] = {'calls': len(lat), 'rechecks_not_identical': bad, 'error': err,
                       'median_first1000_ms': pct(lat[:1000], 50) if lat else None,
                       'median_last1000_ms': pct(lat[-1000:], 50) if lat else None,
                       'device_free_mib_before': round(free0), 'device_free_mib_after': round(free1)}
            s.b.close()
            print('   stability %-5s %s' % (lb, json.dumps(out[lb])), flush=True)
        results[name]['stability'] = out

    which = set(args.models.split(','))

    # ---- detector ---------------------------------------------------------------------------------
    if 'det' in which:
        det_path = os.path.join(resolve_relative_path('..'), 'models', 'buffalo_l', 'det_10g.onnx')
        sizes = ([int(x) for x in args.det_sizes.split(',') if x] or
                 sorted({int(str(g.face_detector_size).split('x')[0]), 640}))
        for size in sizes:
            prov = fu._face_analysis_providers()
            prov, _ = providers_for('recognition:buffalo_l', prov)
            blobs = [cv2.dnn.blobFromImage(cv2.resize(fr, (size, size)), 1.0 / 128, (size, size),
                                           (127.5, 127.5, 127.5), swapRB=True) for fr in frames]
            tmp = onnxruntime.InferenceSession(det_path, so(), providers=prov)
            nm = tmp.get_inputs()[0].name
            del tmp
            feeds = [{nm: b} for b in blobs]
            name = 'detector@%d' % size
            try:
                arms = measure(name, det_path, prov, ProdPlain, feeds, {nm: (1, 3, size, size)})
            except Exception as exc:                       # noqa: BLE001
                # det_10g DECLARES static outputs from its 640 export (12800 anchors). ORT checks a pre-bound
                # output against that declared shape, so at any other size the run is refused ("Got: 8192
                # Expected: 12800"): no preallocated-output binding, hence no CUDA graph, off the 640 size.
                results[name] = {'unsupported': '%s: %s' % (type(exc).__name__, str(exc).replace(chr(10), ' ')[:240])}
                print('   %s: bound/graph arms UNSUPPORTED: %s' % (name, results[name]['unsupported']), flush=True)
                continue
            for a in arms.values():
                if hasattr(a, 'b'):
                    a.b.close()
            pooled(name, det_path, prov, ProdPlain, feeds, {nm: (1, 3, size, size)}, args.calls)
            if args.stability:
                stability(name, det_path, prov, feeds, {nm: (1, 3, size, size)}, args.stability)

    # ---- XSeg --------------------------------------------------------------------------------------
    if 'xseg' in which:
        xp = resolve_relative_path('../models/xseg.onnx')
        base = get_small_card_safe_providers(g.execution_providers, model_path=xp, stage='mask:xseg')
        prov, _ = providers_for('masking:xseg', base, xp)
        nm = onnxruntime.InferenceSession(xp, so(), providers=prov).get_inputs()[0].name
        feeds = [{nm: (cv2.resize(c, (256, 256), interpolation=cv2.INTER_CUBIC).astype('float32') / 255.0)[None]}
                 for c in crops256]
        shapes = {nm: (1, 256, 256, 3)}
        ONLY[xp] = [onnxruntime.InferenceSession(xp, so(), providers=prov).get_outputs()[0].name]
        arms = measure('xseg', xp, prov, ProdBinding, feeds, shapes)
        for a in arms.values():
            if hasattr(a, 'b'):
                a.b.close()
        pooled('xseg', xp, prov, ProdBinding, feeds, shapes, args.calls)
        if args.stability:
            stability('xseg', xp, prov, feeds, shapes, args.stability)

    # ---- RestoreFormer++ ------------------------------------------------------------------------------
    if 'rfpp' in which:
        rp = resolve_relative_path('../models/restoreformer_plus_plus.onnx')
        prov, _ = providers_for('restoreformer_pp', g.execution_providers, rp)
        nm = onnxruntime.InferenceSession(rp, so(), providers=prov).get_inputs()[0].name
        # prepare_model_input hands back a REUSED buffer: copy, or every feed aliases the last crop
        feeds = [{nm: BUFFER_POOL.prepare_model_input(c).copy()} for c in crops512]
        shapes = {nm: (1, 3, 512, 512)}
        ONLY[rp] = [onnxruntime.InferenceSession(rp, so(), providers=prov).get_outputs()[0].name]
        arms = measure('restoreformer++', rp, prov, ProdBinding, feeds, shapes)
        for a in arms.values():
            if hasattr(a, 'b'):
                a.b.close()
        pooled('restoreformer++', rp, prov, ProdBinding, feeds, shapes, max(60, args.calls // 4))
        if args.stability:
            stability('restoreformer++', rp, prov, feeds, shapes, min(args.stability, 3000))

    path = os.path.join(REPO, 'docs', 'perf', 'trt_bound_sessions_%s.json' % args.date)
    json.dump(results, open(path, 'w'), indent=1, default=float)
    print('\nwrote', path)


if __name__ == '__main__':
    main()
