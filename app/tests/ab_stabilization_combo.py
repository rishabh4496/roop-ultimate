"""Counterbalanced A/B of the three stabilization switches, end to end.

WHY. `stabilize_face` / `stabilize_mask` / `stabilize_enhancer` do not compose
into three independent costs -- they select a whole EXECUTION PATH:

    kps only, no mask, no enhancer  ->  two-pass   (smooth sequentially, then
                                                    swap N-wide; buffers nothing)
    anything else                   ->  parallel-chunk (holds 3 + 2*queue_capacity
                                                    chunk-sized frame buffers)

The live config here is `face=false, mask=true, enhancer=true`, which is the
worst cell in that table: it takes the buffering path AND forgoes the landmark
smoothing, while the shipped fresh-install default is all three ON. This
measures the shipped kps-only cell against what is actually running.

ARMS ARE COUNTERBALANCED (A B B A) because end-to-end time on this machine
moves several per cent with position alone -- the first arm of a process also
pays any TensorRT engine build. Reading a forward-only pair has produced false
double-digit results here more than once.

SWAP RATE IS REPORTED BESIDE fps. A configuration that goes faster by finding
or swapping fewer faces has not got faster, and stabilization feeds the
temporal gap-filling that finds them, so this is exactly the setting where that
guard matters.

    env/Scripts/python.exe tests/ab_stabilization_combo.py --video <clip> --end 600

RESULT 2026-09-06 -- NEUTRAL, the hypothesis is REJECTED. RTX 4070, b1.mp4
frames 0-600, realswap / UltraMax / RealityUX / tensorrt / threads 8:

    arm        pos   wall s     fps   swap %
    current      1    365.5    1.64     57.5
    kps_only     2    363.4    1.65     57.3
    kps_only     3    374.4    1.60     56.8
    current      4    343.7    1.75     57.1

    current 1.69 -> kps_only 1.63 = 0.961x (-3.9%), swap rate identical

Switching to the two-pass path does NOT speed this render up; it is marginally
slower. Do not re-propose it as a performance fix. Note also that the SAME
configuration read 1.64 at position 1 and 1.75 at position 4 -- a forward-only
pair would have reported +6.7%.

TWO THINGS THIS RUN ALSO ESTABLISHED, both worth not re-deriving:

* `detect` is ~41% of summed thread time at ~99 ms/call against the ~11
  ms/frame retinaface@512 measures alone. That gap is `_prof('detect')`
  wrapping `lease_face_analyser()`'s blocking `Queue.get()`: 8 workers on a
  pool of 2. It is queue wait, not detector cost, and stage share is not a
  speedup budget.

* DO NOT compare `ms/call` across arms with different stabilization WIDTHS.
  The profile sums wall clock across worker threads, so per-call time inflates
  with concurrency by construction. Only wall clock compares. (Recorded because
  this analysis made exactly that error once.)

CAVEAT ON ABSOLUTE VALUES. The swap audit for these arms reads 102 of 2199
faces swapped (4.6%) -- `harjot,gargee` are not the people in this clip, so the
identity gate refused 95.4%. The A/B RATIO is valid (both arms did identical
work), but the absolute fps and the stage mix are not representative of a run
whose faces actually swap. Pass `--sources` that match the clip before quoting
any absolute number from here.
"""

import argparse
import os
import re
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import fixtures  # noqa: E402

PY = os.path.join(APP, 'env', 'Scripts', 'python.exe')
SWAPPED_RE = re.compile(r'swapped\s+(\d+)\s+\(([\d.]+)%\)')
FRAMES_RE = re.compile(r'(\d+)\s+frames, swapped\s+(\d+)\s+\(([\d.]+)%\)')
STAGE_RE = re.compile(r'^\s*(\w+)\s+([\d.]+)s\s+([\d.]+)%\s+(\d+)\s+calls', re.M)

# face, mask, enhancer
ARMS = {
    'current':  ('0', '1', '1'),
    'kps_only': ('1', '0', '0'),
}


def run_arm(args, arm, tag):
    face, mask, enh = ARMS[arm]
    cmd = [PY, os.path.join(HERE, 'two_face_video.py'),
           '--tag', tag, '--video', args.video, '--sources', args.sources,
           '--start', str(args.start), '--end', str(args.end),
           '--provider', args.provider, '--swap-model', args.swap_model,
           '--enhancer', args.enhancer, '--mask-engine', args.mask_engine,
           '--codec', args.codec, '--threads', str(args.threads),
           '--tracking', args.tracking,
           '--stabilize-face', face, '--stabilize-mask', mask,
           '--stabilize-enhancer', enh]
    env = dict(os.environ, ROOP_PROFILE='1')
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True,
                         errors='ignore', env=env, cwd=APP)
    wall = time.time() - t0
    text = (out.stdout or '') + (out.stderr or '')
    log = os.path.join(APP, 'output', 'bench_two_face', tag, 'arm.log')
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(text)
    m = FRAMES_RE.search(text)
    stages = {k: (float(s), float(p), int(c))
              for k, s, p, c in STAGE_RE.findall(text)}
    return dict(arm=arm, tag=tag, wall=wall, rc=out.returncode,
                frames=int(m.group(1)) if m else None,
                swapped=int(m.group(2)) if m else None,
                swap_pct=float(m.group(3)) if m else None,
                stages=stages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default=fixtures.clip('b1.mp4'))
    ap.add_argument('--sources', default='harjot,gargee')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=600)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--provider', default='tensorrt')
    ap.add_argument('--swap-model', default='realswap')
    ap.add_argument('--enhancer', default='UltraMax')
    ap.add_argument('--mask-engine', default='RealityUX')
    ap.add_argument('--codec', default='hevc_nvenc')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--tracking', default='0')
    args = ap.parse_args()

    n = args.end - args.start
    print('=' * 76)
    print(f"[ab-stab] {os.path.basename(args.video)} frames {args.start}-{args.end} "
          f"({n})")
    print(f"[ab-stab] {args.swap_model} / {args.enhancer} / {args.mask_engine} / "
          f"{args.provider} / threads={args.threads} / tracking={args.tracking}")
    print(f"[ab-stab] current(face0 mask1 enh1) vs kps_only(face1 mask0 enh0), "
          f"{args.reps} pass(es), counterbalanced")
    print('=' * 76, flush=True)

    results = []
    for rep in range(args.reps):
        order = ['current', 'kps_only']
        if rep % 2:
            order.reverse()
        for arm in order:
            tag = f'stab_{arm}_rep{rep}'
            print(f"\n[ab-stab] running {tag} ...", flush=True)
            r = run_arm(args, arm, tag)
            results.append(r)
            fps = n / r['wall'] if r['wall'] else 0.0
            print(f"[ab-stab] {tag}: rc={r['rc']} {r['wall']:.1f}s -> {fps:.2f} fps "
                  f"| frames {r['frames']} swapped {r['swapped']} "
                  f"({r['swap_pct']}%)", flush=True)

    print('\n' + '=' * 76)
    print(f"  {'arm':>10} {'pos':>4} {'wall s':>9} {'fps':>8} {'swapped':>9} {'swap %':>8}")
    for i, r in enumerate(results):
        fps = n / r['wall'] if r['wall'] else 0.0
        print(f"  {r['arm']:>10} {i + 1:>4} {r['wall']:9.1f} {fps:8.2f} "
              f"{str(r['swapped']):>9} {str(r['swap_pct']):>8}")

    def mean_fps(arm):
        v = [n / r['wall'] for r in results if r['arm'] == arm and r['wall']]
        return st.mean(v) if v else None

    a, b = mean_fps('current'), mean_fps('kps_only')
    if a and b:
        print(f"\n  current {a:.2f} fps -> kps_only {b:.2f} fps = {b / a:.3f}x "
              f"({(b / a - 1) * 100:+.1f}%)")
        for arm in ('current', 'kps_only'):
            sw = [r['swap_pct'] for r in results
                  if r['arm'] == arm and r['swap_pct'] is not None]
            if sw:
                print(f"  {arm:>10} swap rate {st.mean(sw):.1f}%")
        print("\n  A faster arm that swapped fewer faces has not got faster.")

    for arm in ('current', 'kps_only'):
        st_rows = [r['stages'] for r in results if r['arm'] == arm and r['stages']]
        if st_rows:
            print(f"\n  {arm} stage seconds (first arm of its kind):")
            for k, (s, p, c) in sorted(st_rows[0].items(),
                                       key=lambda kv: -kv[1][0])[:8]:
                print(f"    {k:<16} {s:8.1f}s {p:6.1f}%  {c:>7} calls")


if __name__ == '__main__':
    main()
