"""A/B the stabilized render's chunk size: ROOP_STAB_CHUNK_MB.

    env/Scripts/python.exe tests/ab_stab_chunk_mb.py \
        --clip "G:/pinokio/roop-keep/inverted/Cervical....mp4" --source harjot

WHAT IS BEING TESTED. `_run_stab_parallel` hands blocks to workers through a
shared queue. The chunk holds `width * (fits // width)` blocks, and `fits` comes
from the decoded-frame memory budget — so the budget decides how many WHOLE
ROUNDS of work a chunk contains. One round means every worker gets exactly one
block and the chunk's wall time is gated by its unluckiest one; two rounds let a
worker that drew an easy block come back for another.

At 1280x720 with the default 1536 MB that is 1 round (10 blocks for 10 workers).
3072 MB buys 2 rounds (20 blocks). Simulation put the gain at about +5%; this is
the measurement.

HOW IT IS MEASURED, and why not simply total wall time. Wall time includes the
detection pre-pass, model/engine warm-up and the final encode, none of which this
knob touches, so it dilutes the effect being measured. The number that isolates
it is the per-chunk `proc` rate the stabilized loop already prints, taken over
the steady state — the first chunks of an arm pay engine build and cache warming
and are dropped. Both are reported.

Arms run in ONE process, A/B/A/B, so TensorRT engines and model pools are shared
and the ordering is counterbalanced: a difference that only appears in the first
pair is warm-up, not the knob.
"""

import argparse
import io
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

# BEFORE importing anything from roop: `_PROFILE` is bound at import time in
# procmgr_runtime, so setting it later leaves the per-chunk lines unprinted and
# this harness silently measures nothing. config.yaml's perf_profile is 'auto',
# which the tri-state loop leaves alone, so this survives _apply_perf_env().
os.environ['ROOP_PROFILE'] = '1'

import compare_enhancers_video as C     # applies the perf env at import
import cv2

CHUNK_RE = re.compile(
    r'STAB CHUNK +(\d+)\].*?frames= *(\d+).*?proc= *([\d.]+)s \( *([\d.]+) FPS\)'
    r'.*?imbalance= *([\d.]+)s \(slow= *([\d.]+)s / fast= *([\d.]+)s\)')
GEOM_RE = re.compile(r'\[Stabilize\] parallel: (\d+) workers, (\d+) blocks x (\d+)f')


def parse(text):
    rows = []
    for ln in text.replace('\r', '\n').split('\n'):
        ln = re.sub(r'\x1b\[[0-9;]*m', '', ln)
        m = CHUNK_RE.search(ln)
        if m:
            c, f, proc, fps, imb, slow, fast = m.groups()
            rows.append(dict(chunk=int(c), frames=int(f), proc=float(proc),
                             fps=float(fps), imb=float(imb),
                             slow=float(slow), fast=float(fast)))
    g = GEOM_RE.search(re.sub(r'\x1b\[[0-9;]*m', '', text))
    geom = tuple(int(x) for x in g.groups()) if g else None
    return rows, geom


def run_arm(args, spec, tag, swapper, mask, threads):
    """`spec` = (chunk_mb, rounds_cap). rounds_cap=1 reproduces the OLD
    behaviour exactly — one block per worker, nothing for the queue to hand
    out — which is the arm the whole question is about."""
    chunk_mb, rounds_cap = spec
    for var, val in (('ROOP_STAB_CHUNK_MB', chunk_mb),
                     ('ROOP_STAB_BLOCKS_PER_WORKER', rounds_cap)):
        if val:
            os.environ[var] = str(val)
        else:
            os.environ.pop(var, None)
    out_dir = os.path.join(APP, 'output', 'ab_stab', tag)
    buf = io.StringIO()
    t0 = time.time()
    with redirect_stdout(buf):
        path, elapsed = C.render(args.clip, args.source, args.enhancer, out_dir,
                                 swapper, mask, threads)
    wall = time.time() - t0
    text = buf.getvalue()
    rows, geom = parse(text)
    log = os.path.join(out_dir, 'arm.log')
    os.makedirs(out_dir, exist_ok=True)
    with open(log, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(text)
    return dict(tag=tag, chunk_mb=chunk_mb, wall=wall, elapsed=elapsed,
                rows=rows, geom=geom, path=path, log=log)


def summarize(r, drop=1):
    # drop=1, not 2: the arms share one process, so only the FIRST arm of the run
    # pays engine build, and the counterbalanced ordering already moves that
    # penalty between arms across reps. Dropping 2 threw away every sample of an
    # arm whose chunks are big enough that a short clip only holds two.
    rows = r['rows'][drop:]
    if not rows:
        return None
    fps = [x['fps'] for x in rows]
    frames = sum(x['frames'] for x in rows)
    proc = sum(x['proc'] for x in rows)
    idle = [x['imb'] / x['slow'] * 100 for x in rows if x['slow'] > 0]
    return dict(n=len(rows), median_fps=st.median(fps),
                agg_fps=frames / proc if proc else 0.0,
                median_idle=st.median(idle) if idle else float('nan'),
                imb=st.median([x['imb'] for x in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', required=True)
    ap.add_argument('--source', default='harjot')
    ap.add_argument('--enhancer', default='UltraMax')
    ap.add_argument('--reps', type=int, default=1, help='passes over the arm set')
    ap.add_argument('--only', default='', help='comma-separated arm labels to run')
    ap.add_argument('--reverse', action='store_true',
                    help='run the arm set in reverse order (to counterbalance a '
                         'previous forward pass)')
    args = ap.parse_args()

    import yaml
    with open(os.path.join(APP, 'config.yaml')) as f:
        cfg = yaml.safe_load(f) or {}
    swapper = cfg.get('swap_model', 'realswap')
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg',
            'None': 'None'}.get(cfg.get('mask_engine', 'RealityUX'),
                                cfg.get('mask_engine'))
    threads = int(cfg.get('max_threads', 16))

    cap = cv2.VideoCapture(args.clip)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    print('=' * 78)
    print(f"[ab] {os.path.basename(args.clip)}  {n} frames  {w}x{h}")
    print(f"[ab] {args.enhancer} / {swapper} / {mask} / threads={threads}")
    print(f"[ab] arms: 1 round (old) | 2 rounds (new default) | "
          f"4 rounds (CHUNK_MB=3072), {args.reps} pass(es)")
    print('=' * 78, flush=True)

    # (chunk_mb, rounds_cap, label)
    ARMS = [((None, 1), '1round_OLD'),
            ((None, None), '2round_NEW'),
            ((3072, None), '4round_3072')]
    if args.only:
        keep = {x.strip() for x in args.only.split(',') if x.strip()}
        ARMS = [a for a in ARMS if a[1] in keep]
        if not ARMS:
            raise SystemExit(f"--only matched no arm; labels are "
                             f"1round_OLD, 2round_NEW, 4round_3072")
    results = []
    for rep in range(args.reps):
        order = list(ARMS)
        if bool(args.reverse) ^ bool(rep % 2):
            order.reverse()                      # counterbalance
        for spec, name in order:
            tag = f"{name}_rep{rep}"
            print(f"\n[ab] running {tag} ...", flush=True)
            r = run_arm(args, spec, tag, swapper, mask, threads)
            s = summarize(r)
            results.append((name, r, s))
            g = r['geom']
            print(f"[ab] {tag}: geometry {g[0]} workers x {g[1]} blocks x {g[2]}f"
                  if g else f"[ab] {tag}: geometry not reported")
            if s:
                print(f"[ab] {tag}: {s['n']} chunks | median {s['median_fps']:.2f} fps "
                      f"| aggregate {s['agg_fps']:.2f} fps | idle {s['median_idle']:.1f}% "
                      f"| wall {r['wall']:.0f}s", flush=True)
            else:
                print(f"[ab] {tag}: NO chunk lines — the stabilized path did not "
                      f"run; this A/B measures nothing.", flush=True)

    print('\n' + '=' * 78)
    print(f"  {'arm':>12} {'chunks':>7} {'median fps':>11} {'agg fps':>9} "
          f"{'idle %':>8} {'wall s':>8}")
    for name, r, s in results:
        if s:
            print(f"  {name:>12} {s['n']:7d} {s['median_fps']:11.2f} "
                  f"{s['agg_fps']:9.2f} {s['median_idle']:8.1f} {r['wall']:8.0f}")
    for key in ('agg_fps', 'median_fps'):
        a = [s[key] for nm, _r, s in results if s and nm == 'A_default']
        b = [s[key] for nm, _r, s in results if s and nm == 'B_3072']
        if a and b:
            ma, mb_ = st.mean(a), st.mean(b)
            print(f"\n  {key}: A {ma:.2f} -> B {mb_:.2f} = {mb_/ma:.3f}x "
                  f"({(mb_/ma - 1) * 100:+.1f}%)   per-rep A={['%.2f'%x for x in a]} "
                  f"B={['%.2f'%x for x in b]}")
    print('=' * 78)


if __name__ == '__main__':
    main()
