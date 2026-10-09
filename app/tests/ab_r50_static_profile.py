"""retinaface_r50 TensorRT profile: band (320-1280, opt 512, batch 1-8) vs static 1x3x640x640.

    # one arm per process (the provider list, the engine and the pool are rebuilt per arm):
    ROOP_TRT_STATIC_PROFILE=0 env/Scripts/python.exe tests/ab_r50_static_profile.py run --arm band_a --out output/r50_static/band_a.json
    ROOP_TRT_STATIC_PROFILE=1 env/Scripts/python.exe tests/ab_r50_static_profile.py run --arm static_a --out output/r50_static/static_a.json
    env/Scripts/python.exe tests/ab_r50_static_profile.py compare --a band_a.json band_b.json --b static_a.json static_b.json

`run` renders nothing: for every frame of each clip window it calls the pipeline's own
`face_util._detect_faces_raw(frame, aux=False)` (the production detector call: context padding, the
adaptive pyramid, the DIoU merge; `_detect_faces_raw` pins r50 to 640 itself), timing that call and
recording every face's box, score and 5 keypoints. Device-wide VRAM (`torch.cuda.mem_get_info`) is
sampled after the pool is built, after the first call and every 25 frames.

`compare` pairs faces between two arms per frame (greedy by IoU, >= 0.5 to count as the same face) and
reports the acceptance numbers: box IoU of matched pairs (>= 0.99), keypoint distance (<= 0.5 px),
faces present in only one arm, detect ms (median / mean / p90; ABBA means over the arms given) and VRAM.
An A-vs-A pair (same flag twice) is the null control: numerics that differ there are noise, not the profile.
"""
import argparse
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('ROOP_PROFILE', '1')

CLIPS = [('d1', 'double/d1.mp4', 0, 418), ('d4', 'double/d4.mp4', 0, 600), ('d6', 'double/d6.mp4', 0, 488)]


def _used_mib():
    import torch
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1048576.0


def run(args):
    import cv2
    import fixtures
    from settings import Settings
    import angle_bench as ab
    cfg = Settings(os.path.join(APP, 'config.yaml'))
    g = ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), 'None', 'None', sync_config=True)
    g.detector_engine = 'retinaface_r50'
    g.g_desired_face_analysis = ['landmark_3d_68', 'landmark_2d_106', 'detection', 'recognition']
    from roop import face_util as fu
    from roop import baseline_probe as bp
    from roop import trt_shape_profile as tsp

    out = {'arm': args.arm, 'static_flag': os.environ.get('ROOP_TRT_STATIC_PROFILE'),
           'static_enabled': tsp.static_enabled(), 'clips': {}, 'vram': []}
    out['vram'].append(['after init_pipeline', round(_used_mib(), 1)])
    only = [c for c in args.clips.split(',') if c]
    first = True
    for clip, rel, lo, hi in CLIPS:
        if only and clip not in only:
            continue
        hi = min(hi, lo + args.limit) if args.limit else hi
        cap = cv2.VideoCapture(fixtures.clip(rel, required=True))
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        before = bp.snapshot()
        frames, ms = {}, []
        for idx in range(lo, hi):
            ok, fr = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            faces = fu._detect_faces_raw(fr, aux=False)
            dt = (time.perf_counter() - t0) * 1000.0
            if first:
                first = False
                out['vram'].append(['after the very first detect call (pool built, engine loaded)', round(_used_mib(), 1)])
                out['first_call_ms'] = round(dt, 1)
                continue                                     # not part of the steady-state sample
            ms.append(dt)
            rows = sorted(([float(v) for v in f.bbox] + [float(f.det_score)] + [float(v) for v in f.kps.reshape(-1)]
                           for f in faces), key=lambda r: -r[4])
            frames[idx] = [[round(v, 3) for v in r] for r in rows]
            if len(ms) % 25 == 0:
                out['vram'].append(['%s frame %d' % (clip, idx), round(_used_mib(), 1)])
        cap.release()
        after = bp.snapshot()
        counters = {k: sum(after[k].values()) - sum(before.get(k, {}).values()) for k in after
                    if k.startswith(('rescue.', 'detect.', 'raw.')) and
                    sum(after[k].values()) != sum(before.get(k, {}).values())}
        out['clips'][clip] = {'frames': frames, 'ms': [round(v, 3) for v in ms], 'counters': counters,
                              'shape': [int(cv2.VideoCapture(fixtures.clip(rel, required=True)).get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        int(cv2.VideoCapture(fixtures.clip(rel, required=True)).get(cv2.CAP_PROP_FRAME_HEIGHT))]}
        print('[ab] %s %s: %d frames, median %.2f ms, mean %.2f ms, %d faces' % (
            args.arm, clip, len(ms), st.median(ms), st.mean(ms), sum(len(v) for v in frames.values())), flush=True)
    out['vram'].append(['end of run', round(_used_mib(), 1)])
    pools = {}
    try:
        from roop import retinaface
        p = retinaface._POOLS.get('r50')
        if p:
            pools['width'] = len(p['items'])
            pools['pinned_input_size'] = [getattr(d, 'pinned_input_size', None) for d in p['items']]
            sess = p['items'][0].session
            opts = (sess.get_provider_options() or {}).get('TensorrtExecutionProvider') or {}
            pools['provider'] = sess.get_providers()[0]
            pools['profile'] = {k: v for k, v in opts.items() if k.startswith('trt_profile') or k == 'trt_engine_cache_path'}
    except Exception as exc:                                  # a probe must not kill the arm
        pools['error'] = repr(exc)
    out['pool'] = pools
    print('[ab] pool:', json.dumps(pools), flush=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh)
    print('[ab] wrote', args.out)
    return 0


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pair_faces(fa, fb):
    """Greedy one-to-one pairing by IoU >= 0.5. Returns (pairs, only_a, only_b)."""
    cand = sorted(((iou(a, b), i, j) for i, a in enumerate(fa) for j, b in enumerate(fb)), reverse=True)
    used_a, used_b, pairs = set(), set(), []
    for v, i, j in cand:
        if v < 0.5:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((fa[i], fb[j], v))
    return (pairs, [fa[i] for i in range(len(fa)) if i not in used_a],
            [fb[j] for j in range(len(fb)) if j not in used_b])


def load(path):
    with open(path, encoding='utf-8') as fh:
        d = json.load(fh)
    for c in d['clips'].values():
        c['frames'] = {int(k): v for k, v in c['frames'].items()}
    return d


def compare_pair(a, b):
    rep = {}
    for clip in a['clips']:
        if clip not in b['clips']:
            continue
        fa, fb = a['clips'][clip]['frames'], b['clips'][clip]['frames']
        ious, kdist, only_a, only_b, dup_a, dup_b, n_a, n_b = [], [], 0, 0, 0, 0, 0, 0
        worst = []
        for idx in sorted(set(fa) | set(fb)):
            la, lb = fa.get(idx, []), fb.get(idx, [])
            n_a += len(la)
            n_b += len(lb)
            pairs, oa, ob = pair_faces(la, lb)
            only_a += len(oa)
            only_b += len(ob)
            for x, y, v in pairs:
                ious.append(v)
                d = max(((x[5 + 2 * k] - y[5 + 2 * k]) ** 2 + (x[6 + 2 * k] - y[6 + 2 * k]) ** 2) ** 0.5 for k in range(5))
                kdist.append(d)
                if v < 0.99 or d > 0.5:
                    worst.append((idx, round(v, 4), round(d, 3)))
            for lst, which in ((la, 'a'), (lb, 'b')):          # duplicate boxes inside ONE arm's frame
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        if iou(lst[i], lst[j]) > 0.5:
                            if which == 'a':
                                dup_a += 1
                            else:
                                dup_b += 1
        rep[clip] = {
            'faces_a': n_a, 'faces_b': n_b, 'matched': len(ious), 'only_in_a': only_a, 'only_in_b': only_b,
            'iou_min': round(min(ious), 5) if ious else None, 'iou_median': round(st.median(ious), 5) if ious else None,
            'pairs_iou_lt_0.99': sum(1 for v in ious if v < 0.99),
            'kps_max_px_median': round(st.median(kdist), 4) if kdist else None,
            'kps_max_px_max': round(max(kdist), 3) if kdist else None,
            'pairs_kps_gt_0.5px': sum(1 for d in kdist if d > 0.5),
            'dup_boxes_a': dup_a, 'dup_boxes_b': dup_b, 'worst_pairs(frame,iou,kps_px)': worst[:8]}
    return rep


def ms_stats(arms, clip):
    vals = [v for a in arms for v in a['clips'][clip]['ms']]
    vals.sort()
    return {'median': round(st.median(vals), 3), 'mean': round(st.mean(vals), 3),
            'p90': round(vals[(len(vals) * 9) // 10], 3), 'n': len(vals)}


def compare(args):
    a_arms, b_arms = [load(p) for p in args.a], [load(p) for p in args.b]
    arms = a_arms + b_arms
    print('A = %s   B = %s' % ([x['arm'] for x in a_arms], [x['arm'] for x in b_arms]))
    for x in arms:
        print('  arm %-12s static_enabled=%s pool=%s' % (x['arm'], x['static_enabled'], json.dumps(x.get('pool'))))
    print('\n== same-face agreement, first A arm vs first B arm')
    for clip, r in compare_pair(a_arms[0], b_arms[0]).items():
        print(clip, json.dumps(r))
    if len(a_arms) > 1:
        print('\n== NULL control: first A arm vs second A arm (same flag, separate process)')
        for clip, r in compare_pair(a_arms[0], a_arms[1]).items():
            print(clip, json.dumps(r))
    print('\n== detect ms/frame (production call, all A arms pooled vs all B arms pooled)')
    for clip in a_arms[0]['clips']:
        sa, sb = ms_stats(a_arms, clip), ms_stats(b_arms, clip)
        print('%-3s A %s | B %s | B/A median %.3f mean %.3f' % (
            clip, sa, sb, sb['median'] / sa['median'], sb['mean'] / sa['mean']))
        for x in arms:
            print('      %-12s median %.3f mean %.3f' % (x['arm'], st.median(x['clips'][clip]['ms']), st.mean(x['clips'][clip]['ms'])))
    print('\n== VRAM, device-wide MiB')
    for x in arms:
        samples = [v for k, v in x['vram'] if k.startswith(('d', 'e')) and k != 'after init_pipeline'] or [0]
        first = dict(x['vram'])
        print('  %-12s init %s | after first call %s | peak sample %.1f | end %s | first call %s ms' % (
            x['arm'], first.get('after init_pipeline'),
            first.get('after the very first detect call (pool built, engine loaded)'), max(samples),
            first.get('end of run'), x.get('first_call_ms')))
    print('\n== detector/rescue counters (A first arm vs B first arm)')
    for clip in a_arms[0]['clips']:
        ca, cb = a_arms[0]['clips'][clip]['counters'], b_arms[0]['clips'][clip]['counters']
        print(clip, 'A', json.dumps(ca), '\n   B', json.dumps(cb))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--arm', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--clips', default='d1,d4,d6')
    r.add_argument('--limit', type=int, default=0)
    c = sub.add_parser('compare')
    c.add_argument('--a', nargs='+', required=True, help='arm JSONs of the baseline (band) flag')
    c.add_argument('--b', nargs='+', required=True, help='arm JSONs of the candidate (static) flag')
    args = ap.parse_args()
    return run(args) if args.cmd == 'run' else compare(args)


if __name__ == '__main__':
    sys.exit(main())
