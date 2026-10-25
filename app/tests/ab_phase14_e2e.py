"""Phase 14 end to end: the pre-pass filter and the detection interval on REAL renders, with the swap rate as the guard.

`probe_phase14_prepass.py` measures the pre-pass alone. This renders the same window through the whole pipeline
(pre-pass, swap, enhance, encode) under each arm and reports what a user would see: wall time, swap rate, faces
seen and how far each arm's PIXELS are from the baseline arm's - against the null spread between two renders of the
SAME arm, which is the floor below which a difference is not evidence of anything (AGENTS.md: 0.71/255 mean).

Arms are env-driven (`ROOP_PREPASS_FILTER`, `ROOP_PREPASS_DETECT_INTERVAL`): both are read at use time, once per
pre-pass, so a change between two renders in one process takes effect. The schedule is counterbalanced and starts with
the baseline twice (the null control); the first render of a process also pays the TensorRT engine build.

    app\\env\\Scripts\\python.exe app\\tests\\ab_phase14_e2e.py --clip <clip.mp4> --frames 600 \
        --arms base,base,euro,k4,k4,euro,base --out docs/perf/phase14/e2e_<clip>.json
"""
import argparse
import io
import json
import os
import re
import statistics as st
import sys
import time
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2                                                  # noqa: E402
import numpy as np                                          # noqa: E402

ARMS = {
    'base': {'ROOP_PREPASS_FILTER': 'adaptive', 'ROOP_PREPASS_DETECT_INTERVAL': '1'},
    'euro': {'ROOP_PREPASS_FILTER': 'one_euro', 'ROOP_PREPASS_DETECT_INTERVAL': '1'},
    'zp': {'ROOP_PREPASS_FILTER': 'zero_phase', 'ROOP_PREPASS_DETECT_INTERVAL': '1'},
    'k2': {'ROOP_PREPASS_FILTER': 'adaptive', 'ROOP_PREPASS_DETECT_INTERVAL': '2'},
    'k4': {'ROOP_PREPASS_FILTER': 'adaptive', 'ROOP_PREPASS_DETECT_INTERVAL': '4'},
    'k8': {'ROOP_PREPASS_FILTER': 'adaptive', 'ROOP_PREPASS_DETECT_INTERVAL': '8'},
    # K=4 with the landmark net re-measuring every flow-placed face (roop/landmark_refresh.py)
    'k4r': {'ROOP_PREPASS_FILTER': 'adaptive', 'ROOP_PREPASS_DETECT_INTERVAL': '4',
            'ROOP_PREPASS_LANDMARK_REFRESH': '1'},
}

SWAPPED_RE = re.compile(r'swapped \([^)]*\)\s+(\d+)\s+([\d.]+)%')
SEEN_RE = re.compile(r'faces seen\s+(\d+)')
NOFACE_RE = re.compile(r'frames with no face detected at all\s+(\d+)\s+([\d.]+)%')
PLAN_RE = re.compile(r'\[Stage2\] plan: (\d+) frame\(s\) hold a face matched')
CALLS_RE = re.compile(r'prepass_detect_interval (\d+): (\d+) detector call')


def run_arm(name, args, swapper, mask, threads, tag):
    import compare_enhancers_video as C
    import sample_bench as sb
    for k in ('ROOP_PREPASS_FILTER', 'ROOP_PREPASS_DETECT_INTERVAL', 'ROOP_PREPASS_LANDMARK_REFRESH'):
        os.environ[k] = ARMS[name].get(k, '0')
    out_dir = os.path.join(APP, 'output', 'ab_phase14', tag)
    buf = io.StringIO()
    orig = sb.run_swap
    sb.run_swap = lambda *a, **kw: orig(*a, frames=int(args.frames), **kw)
    t0 = time.time()
    try:
        with redirect_stdout(buf):
            path, elapsed = C.render(args.clip, args.source, args.enhancer, out_dir, swapper, mask, threads,
                                     overrides={'sparse_auto_range': False})
    finally:
        sb.run_swap = orig
    wall = time.time() - t0
    text = re.sub(r'\x1b\[[0-9;]*m', '', buf.getvalue())
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'arm.log'), 'w', encoding='utf-8', errors='ignore') as fh:
        fh.write(text)

    def first(rx, g=1, cast=int):
        m = rx.search(text)
        return cast(m.group(g)) if m else None
    return dict(arm=name, tag=tag, path=path, wall=wall, elapsed=elapsed, swapped=first(SWAPPED_RE),
                swap_pct=first(SWAPPED_RE, 2, float), seen=first(SEEN_RE), noface_pct=first(NOFACE_RE, 2, float),
                plan_frames=first(PLAN_RE), detector_calls=first(CALLS_RE, 2))


def mean_abs_diff(a, b, step=5):
    """Mean absolute pixel difference (0-255) over every `step`-th frame of two outputs; None if unreadable."""
    ca, cb = cv2.VideoCapture(a), cv2.VideoCapture(b)
    if not ca.isOpened() or not cb.isOpened():
        return None
    diffs, i = [], 0
    while True:
        ra, fa = ca.read()
        rb, fb = cb.read()
        if not ra or not rb:
            break
        if i % step == 0 and fa.shape == fb.shape:
            diffs.append(float(np.mean(np.abs(fa.astype(np.int16) - fb.astype(np.int16)))))
        i += 1
    ca.release()
    cb.release()
    return float(np.mean(diffs)) if diffs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', required=True)
    ap.add_argument('--source', default='akansha')
    ap.add_argument('--enhancer', default='GPEN Realistic')
    ap.add_argument('--frames', type=int, default=600)
    ap.add_argument('--arms', default='base,base,euro,k4,k4,euro,base')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    args.clip = os.path.abspath(args.clip)
    if args.out:
        args.out = os.path.abspath(args.out)

    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'), encoding='utf-8')) or {}
    swapper = cfg.get('swap_model', 'realswap')
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg', 'None': 'None'}.get(
        cfg.get('mask_engine', 'RealityUX'), cfg.get('mask_engine'))
    threads = int(cfg.get('max_threads', 16))
    names = [a.strip() for a in args.arms.split(',')]
    print('=' * 78)
    print('[ab] %s  first %d frames  %s / %s / threads=%d  arms %s' % (
        os.path.basename(args.clip), args.frames, args.enhancer, swapper, threads, names))
    print('=' * 78, flush=True)
    results = []
    for n, name in enumerate(names):
        tag = '%s_%d' % (name, n)
        print('\n[ab] running %s ...' % tag, flush=True)
        r = run_arm(name, args, swapper, mask, threads, tag)
        results.append(r)
        print('[ab] %s: %.1fs -> %.2f fps | swapped %s (%s%%) of %s seen | no-face %s%% | plan %s | detector calls %s' % (
            tag, r['elapsed'], args.frames / r['elapsed'] if r['elapsed'] else 0.0, r['swapped'], r['swap_pct'],
            r['seen'], r['noface_pct'], r['plan_frames'], r['detector_calls']), flush=True)

    base = next((r for r in results if r['arm'] == 'base' and r['path']), None)
    print('\n' + '=' * 78)
    print('  %-8s %8s %8s %9s %8s %10s %7s %12s' % ('arm', 'seconds', 'fps', 'swapped', 'swap %', 'no-face %', 'plan', 'px diff vs 1st base'))
    for r in results:
        r['px_diff'] = mean_abs_diff(base['path'], r['path']) if base and r['path'] and r is not base else None
        print('  %-8s %8.1f %8.2f %9s %8s %10s %7s %12s' % (
            r['tag'], r['elapsed'], args.frames / r['elapsed'] if r['elapsed'] else 0.0, r['swapped'], r['swap_pct'],
            r['noface_pct'], r['plan_frames'], '-' if r['px_diff'] is None else '%.3f' % r['px_diff']))
    for arm in sorted({r['arm'] for r in results}):
        rows = [r for r in results if r['arm'] == arm and r['elapsed']]
        if rows:
            print('  mean %-5s %.1f s (%.2f fps), swap %% %s, plan %s' % (
                arm, st.mean(r['elapsed'] for r in rows), args.frames / st.mean(r['elapsed'] for r in rows),
                [r['swap_pct'] for r in rows], [r['plan_frames'] for r in rows]))
    print('=' * 78)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(dict(clip=args.clip, frames=args.frames, arms=names, results=results), fh, indent=1, default=str)
        print('[ab] wrote', args.out)


if __name__ == '__main__':
    main()
