"""End-to-end A/B of one environment switch on REAL renders, counterbalanced, with a null control and the swap rate as
the guard.

    app\\env\\Scripts\\python.exe app\\tests\\ab_env_e2e.py --clip <clip.mp4> --frames 150 \
        --env ROOP_BLEND_ROI_MATTE --a 0 --b 1 --schedule a,a,b,b,a,b,b,a --out docs/perf/<dir>/<name>.json

Same machinery as ``ab_phase14_e2e.py`` (the live config.yaml: enhancer, swapper, mask, threads, via
``compare_enhancers_video.render``); only the arms differ. The schedule starts with ``a,a`` (the NULL control: the
spread between two renders of one config is the floor below which a difference is not evidence of anything) and is
counterbalanced because the first render of a process also pays any engine build.

Reported per arm: render seconds (the harness's own ``elapsed``), processing fps, swapped faces and the swap rate,
faces seen, the mean absolute pixel difference of each output against the FIRST ``a`` output (the null's own spread
being the noise floor, AGENTS.md: 0.71/255), and every ``--grep`` line the render printed (a stage that ran says so).
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

import cv2          # noqa: E402
import numpy as np  # noqa: E402

SWAPPED_RE = re.compile(r'swapped \([^)]*\)\s+(\d+)\s+([\d.]+)%')
SEEN_RE = re.compile(r'faces seen\s+(\d+)')


def run_arm(label, env, args, swapper, mask, threads, tag):
    import compare_enhancers_video as C
    import sample_bench as sb
    for k, v in env.items():
        os.environ[k] = v
    out_dir = os.path.join(APP, 'output', 'ab_env', tag)
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
    lines = [l.strip() for l in text.splitlines() if args.grep and re.search(args.grep, l)]
    return dict(arm=label, tag=tag, path=path, wall=wall, elapsed=elapsed, fps=args.frames / elapsed if elapsed else None,
                swapped=first(SWAPPED_RE), swap_pct=first(SWAPPED_RE, 2, float), seen=first(SEEN_RE),
                size=os.path.getsize(path) if path and os.path.exists(path) else None, grep=lines[-4:])


def mean_abs_diff(a, b, step=5):
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
    ap.add_argument('--enhancer', default=None, help='default: selected_enhancer from config.yaml')
    ap.add_argument('--frames', type=int, default=150)
    ap.add_argument('--env', required=True, help='the environment variable under test')
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--also', action='append', default=[], help='KEY=VALUE held for BOTH arms')
    ap.add_argument('--schedule', default='a,a,b,b,a,b,b,a')
    ap.add_argument('--grep', default=r'\[Grain\]|grain match|\[Pipeline\]')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    args.clip = os.path.abspath(args.clip)
    if args.out:
        args.out = os.path.abspath(args.out)         # the render harness changes directory
    held = dict(kv.split('=', 1) for kv in args.also)

    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'), encoding='utf-8')) or {}
    swapper = cfg.get('swap_model', 'realswap')
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg', 'None': 'None'}.get(
        cfg.get('mask_engine', 'RealityUX'), cfg.get('mask_engine'))
    threads = int(cfg.get('max_threads', 16))
    args.enhancer = args.enhancer or cfg.get('selected_enhancer', 'UltraMax')
    names = [a.strip() for a in args.schedule.split(',')]
    print('=' * 78)
    print('[ab] %s first %d frames  %s / %s / %s / threads=%d  %s: a=%r b=%r  schedule %s' % (
        os.path.basename(args.clip), args.frames, args.enhancer, swapper, mask, threads, args.env, args.a, args.b, names),
        flush=True)
    results = []
    for n, name in enumerate(names):
        env = dict(held)
        env[args.env] = args.a if name == 'a' else args.b
        r = run_arm(name, env, args, swapper, mask, threads, '%02d_%s' % (n, name))
        results.append(r)
        print('[ab] #%d %s  render %.1fs  %.2f fps  swapped %s (%s%%)  seen %s  wall %.0fs' % (
            n, name, r['elapsed'] or 0, r['fps'] or 0, r['swapped'], r['swap_pct'], r['seen'], r['wall']), flush=True)
        for l in r['grep']:
            print('       ' + l[:200], flush=True)
    ref = next(r for r in results if r['arm'] == 'a')
    for r in results:
        r['diff_vs_first_a'] = None if r is ref else mean_abs_diff(ref['path'], r['path'])
    print('\n' + '-' * 78)
    print('%-3s %-5s %9s %8s %8s %8s %10s %12s' % ('#', 'arm', 'render s', 'fps', 'swapped', 'swap %', 'seen', 'diff vs a0'))
    for n, r in enumerate(results):
        print('%-3d %-5s %9.1f %8.2f %8s %8s %10s %12s' % (
            n, r['arm'], r['elapsed'] or 0, r['fps'] or 0, r['swapped'], r['swap_pct'], r['seen'],
            '-' if r['diff_vs_first_a'] is None else '%.4f' % r['diff_vs_first_a']))
    for arm in ('a', 'b'):
        v = [r['fps'] for r in results if r['arm'] == arm and r['fps']]
        if len(v) > 1:
            print('arm %s fps: mean %.2f  min %.2f  max %.2f  spread %.1f%%' % (
                arm, st.mean(v), min(v), max(v), 100.0 * (max(v) - min(v)) / st.mean(v)))
    null = results[:2]
    if len(null) == 2 and null[0]['arm'] == null[1]['arm'] == 'a' and null[0]['fps'] and null[1]['fps']:
        print('NULL control (a vs a): %.2f%% apart in fps' % (100.0 * abs(null[0]['fps'] - null[1]['fps']) / st.mean([null[0]['fps'], null[1]['fps']])))
    fa = [r['fps'] for r in results[2:] if r['arm'] == 'a' and r['fps']]
    fb = [r['fps'] for r in results[2:] if r['arm'] == 'b' and r['fps']]
    if fa and fb:
        print('b vs a (counterbalanced runs only): %+.1f%% fps' % (100.0 * (st.mean(fb) / st.mean(fa) - 1.0)))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=1, default=float)


if __name__ == '__main__':
    main()
