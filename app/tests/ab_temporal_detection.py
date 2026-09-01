"""A/B `temporal_detection` end to end, with the swap rate as a guard.

WHY IT MIGHT HELP. In the swap loop the detect stage runs inside `_gpu_guard`,
competing with swap, mask and enhance for the GPU. `temporal_detection` moves it
into a dedicated pre-pass that runs alone, and the loop then reads a precomputed
cache instead of detecting per frame — the detect block in ProcessMgr is skipped
entirely when that cache exists. detect was 42.4% of one profiled render, so the
ceiling here is large.

WHY IT MIGHT NOT. The pre-pass still detects every frame, so the total detection
WORK is unchanged; only the contention and the scheduling change. Whether that
converts into wall clock is exactly the question, and reasoning about it has been
wrong here before.

WHAT IS GUARDED. fps alone would endorse a setting that goes faster by finding
fewer faces, so the swap rate is measured alongside it: a run that detects and
swaps fewer faces has not got faster, it has done less. Both arms also fix
`face_detector_size` at 512, so this measures the temporal change ON TOP of that
one rather than the two mixed together.

Arms are counterbalanced (A/B then B/A) because end-to-end render time on this
machine moves by ~18% with position alone — larger than the effect being looked
for, and it has already produced one false +10% this session.

    env/Scripts/python.exe tests/ab_temporal_detection.py
    env/Scripts/python.exe tests/ab_temporal_detection.py --vary face_detector_size --a 640 --b 512

WHAT IT HAS FOUND SO FAR — both NEUTRAL, and both looked like wins first:

    temporal_detection   off 10.88 -> on 10.84 fps   (+0%)
    face_detector_size   640 12.56 -> 512 12.69 fps  (+1%)

In each case the FIRST arm of the process is several fps slower than every
later one, because it pays the TensorRT engine build for whatever geometry it is
the first to use. Read uncounterbalanced, those runs say +21.8% and +9.8%. Both
are the warm-up.

The det_size result is the one worth understanding: 512 IS 1.30x faster than 640
at the detector (14.27 -> 10.95 ms/frame, measured directly, with slightly BETTER
recall). It does not show up end to end because "detect = 42.4% of the profile"
is 42.4% of wall clock SUMMED ACROSS WORKER THREADS, not of the render. With ten
threads overlapping on one saturated GPU, giving one stage back 3.3 ms of thread
time does not shorten the render unless that stage is what the GPU is waiting on.
Stage share is not a speedup budget.
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

import fixtures
import compare_enhancers_video as C
import cv2

def VARY_OVERRIDES(flag):
    """What the two arms differ by. Set by --vary so this harness can A/B any
    single setting end to end with the swap rate as a guard."""
    key, a, b = _VARY
    return {key: (b if flag else a)}


_VARY = ('temporal_detection', False, True)

SWAPPED_RE = re.compile(r'swapped \([^)]*\)\s+(\d+)\s+([\d.]+)%')
SEEN_RE = re.compile(r'faces seen\s+(\d+)')
NOFACE_RE = re.compile(r'frames with no face detected at all\s+(\d+)\s+([\d.]+)%')


def run_arm(args, temporal, tag, swapper, mask, threads):
    out_dir = os.path.join(APP, 'output', 'ab_temporal', tag)
    buf = io.StringIO()
    t0 = time.time()
    with redirect_stdout(buf):
        path, elapsed = C.render(
            args.clip, args.source, args.enhancer, out_dir, swapper, mask, threads,
            overrides=dict(VARY_OVERRIDES(temporal)))
    wall = time.time() - t0
    text = re.sub(r'\x1b\[[0-9;]*m', '', buf.getvalue())
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'arm.log'), 'w', encoding='utf-8',
              errors='ignore') as f:
        f.write(text)

    seen = SWAPPED_RE.search(text)
    tot = SEEN_RE.search(text)
    nof = NOFACE_RE.search(text)
    return dict(tag=tag, temporal=bool(temporal), wall=wall, elapsed=elapsed,
                swapped=int(seen.group(1)) if seen else None,
                swap_pct=float(seen.group(2)) if seen else None,
                seen=int(tot.group(1)) if tot else None,
                noface_pct=float(nof.group(2)) if nof else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', default=fixtures.clip('inverted/s1.mp4'))
    ap.add_argument('--source', default='harjot')
    ap.add_argument('--enhancer', default='GPEN Realistic')
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--vary', default='temporal_detection',
                    help='globals key to A/B')
    ap.add_argument('--a', default='False', help='arm A value')
    ap.add_argument('--b', default='True', help='arm B value')
    args = ap.parse_args()

    import yaml
    with open(os.path.join(APP, 'config.yaml')) as f:
        cfg = yaml.safe_load(f) or {}
    swapper = cfg.get('swap_model', 'realswap')
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg',
            'None': 'None'}.get(cfg.get('mask_engine', 'RealityUX'),
                                cfg.get('mask_engine'))
    threads = int(cfg.get('max_threads', 16))

    def _coerce(v):
        if v in ('True', 'False'):
            return v == 'True'
        return v
    global _VARY
    _VARY = (args.vary, _coerce(args.a), _coerce(args.b))

    cap = cv2.VideoCapture(args.clip)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print('=' * 78)
    print(f"[ab] {os.path.basename(args.clip)}  {n} frames  {args.enhancer} / "
          f"{swapper} / threads={threads} / det 512")
    print(f"[ab] {_VARY[0]}: {_VARY[1]!r} vs {_VARY[2]!r}, {args.reps} pass(es), "
          f"counterbalanced")
    print('=' * 78, flush=True)

    results = []
    for rep in range(args.reps):
        order = [(False, f'{_VARY[0]}={_VARY[1]}'), (True, f'{_VARY[0]}={_VARY[2]}')]
        if rep % 2:
            order.reverse()
        for temporal, name in order:
            tag = f'{name}_rep{rep}'.replace('/', '_')
            print(f"\n[ab] running {tag} ...", flush=True)
            r = run_arm(args, temporal, tag, swapper, mask, threads)
            results.append(r)
            fps = n / r['elapsed'] if r['elapsed'] else 0.0
            print(f"[ab] {tag}: {r['elapsed']:.1f}s -> {fps:.2f} fps | "
                  f"swapped {r['swapped']} ({r['swap_pct']}%) of {r['seen']} seen | "
                  f"no-face {r['noface_pct']}%", flush=True)

    print('\n' + '=' * 78)
    print(f"  {_VARY[0][:6]:>6} {'fps':>8} {'swapped':>9} {'swap %':>8} {'no-face %':>10}")
    for r in results:
        fps = n / r['elapsed'] if r['elapsed'] else 0.0
        print(f"  {str(_VARY[2] if r['temporal'] else _VARY[1]):>6} {fps:8.2f} "
              f"{str(r['swapped']):>9} {str(r['swap_pct']):>8} {str(r['noface_pct']):>10}")

    def mean_fps(flag):
        v = [n / r['elapsed'] for r in results if r['temporal'] is flag and r['elapsed']]
        return st.mean(v) if v else None

    off, on = mean_fps(False), mean_fps(True)
    if off and on:
        print(f"\n  temporal OFF {off:.2f} fps -> ON {on:.2f} fps = {on/off:.3f}x "
              f"({(on/off - 1) * 100:+.1f}%)")
        sw = {False: [r['swap_pct'] for r in results if r['temporal'] is False
                      and r['swap_pct'] is not None],
              True: [r['swap_pct'] for r in results if r['temporal'] is True
                     and r['swap_pct'] is not None]}
        if sw[False] and sw[True]:
            print(f"  swap rate    {st.mean(sw[False]):.1f}% -> {st.mean(sw[True]):.1f}%"
                  f"   <- a speedup that moves this DOWN is not a speedup")
    print('=' * 78)


if __name__ == '__main__':
    main()
