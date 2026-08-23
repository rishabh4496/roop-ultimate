"""Does dropping det_size and stepping the temporal scan cost angle recall?

The two proposed savings on the detect stage (42.4% of a render) are
`face_detector_size` 640 -> 512 and `ROOP_TEMPORAL_STEP` 1 -> 2. Both trade
against the hard cases, so both are measured here on the hard cases: the
`inverted/` clips are profile, upside-down and yoga footage, which is where a
detector fails if it is going to.

TWO SEPARATE QUESTIONS, measured separately.

1. det_size. Straightforward: detect the same frames at each size and compare
   recall and landmark agreement, bucketed by how difficult the pose is.

2. ROOP_TEMPORAL_STEP. This one is NOT a detector question at all, and reading
   it as one would give the wrong answer. Stepping the scan does not make
   detection worse — the frames it scans are detected exactly as before. It
   replaces the SKIPPED frames with a LINEAR INTERPOLATION between neighbours.
   So the thing to measure is interpolation error against the real landmarks,
   and specifically whether that error concentrates on the frames where the
   head is moving or turned — which is exactly where a straight line between
   two poses stops being a good model.

Pose difficulty comes from `solve_pose_5pt`, so "extreme angle" is the
pipeline's own definition rather than a guess.

    env/Scripts/python.exe tests/measure_detect_tradeoffs.py
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('ROOP_TRT_POOL', '4')

import cv2
import numpy as np

import angle_bench as ab

CLIPS = [
    'Cervical Spondylosis Stretches _ Exercises - Ask Doctor Jo.mp4',
    '10 Minute Daily Stretching Routine For Women Over 50!.mp4',
    'Daily Face Yoga _ Face Sculpting Massage for every day _ 8 Min. to Radiant Skin _ Natural Glow.mp4',
]
ROOT = r'G:/pinokio/roop-keep/inverted'


def pose_of(kps):
    from roop.face_util import solve_pose_5pt
    try:
        p = solve_pose_5pt(np.asarray(kps, np.float32))
        return tuple(float(x) for x in p[:3])
    except Exception:
        return (0.0, 0.0, 0.0)


def difficulty(kps):
    """One number for 'how hard is this pose', from the pipeline's own solver."""
    pitch, yaw, roll = pose_of(kps)
    return max(abs(yaw), abs(roll) * 0.7, abs(pitch) * 0.7), (pitch, yaw, roll)


def scan(clip, det_size, frames, engine=None):
    """Detect at a given det_size (and engine). Returns {frame_idx: (bbox, kps)}."""
    import roop.globals as g
    import roop.face_util as fu
    g.face_detector_size = str(det_size)
    if engine:
        g.detector_engine = engine
    out, t0 = {}, time.perf_counter()
    cap = cv2.VideoCapture(clip)
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        faces = fu.get_all_faces(fr) or []
        if faces:
            f = max(faces, key=lambda x: x.bbox[2] - x.bbox[0])
            out[fi] = (np.asarray(f.bbox, np.float32), np.asarray(f.kps, np.float32))
    cap.release()
    return out, time.perf_counter() - t0


def bucket(d):
    if d < 15:
        return 'frontal (<15 deg)'
    if d < 35:
        return 'moderate (15-35)'
    if d < 60:
        return 'hard (35-60)'
    return 'extreme (>60 deg)'


ORDER = ['frontal (<15 deg)', 'moderate (15-35)', 'hard (35-60)', 'extreme (>60 deg)']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=400, help='frames sampled per clip')
    ap.add_argument('--sizes', default='640,512,416')
    args = ap.parse_args()

    g = ab.init_pipeline('tensorrt', 'realswap', 'GPEN Realistic', 'mask_realityux', 0.0)
    g.detector_engine = g.CFG.detector_engine
    g.face_detector_threshold = float(g.CFG.face_detector_threshold)
    print(f"[detect] engine={g.detector_engine} thr={g.face_detector_threshold}")
    sizes = [int(s) for s in args.sizes.split(',')]

    ref_size = sizes[0]
    agg = {s: {'found': 0, 'total': 0, 'by': {}} for s in sizes}
    interp_rows = []

    for name in CLIPS:
        clip = os.path.join(ROOT, name)
        if not os.path.exists(clip):
            continue
        cap = cv2.VideoCapture(clip)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # A CONTIGUOUS run, because the temporal question is about neighbours.
        start = max(0, n // 3)
        frames = list(range(start, min(n, start + args.frames)))
        print(f"\n[detect] {name[:44]}  frames {frames[0]}-{frames[-1]}", flush=True)

        scans = {}
        for s in sizes:
            res, secs = scan(clip, s, frames)
            scans[s] = res
            print(f"   det_size {s}: {len(res)}/{len(frames)} found "
                  f"({len(res)/len(frames):.1%})  {secs:.1f}s "
                  f"({len(frames)/secs:.1f} fps)", flush=True)

        ref = scans[ref_size]
        for s in sizes:
            for fi in frames:
                d = None
                if fi in ref:
                    d, _ = difficulty(ref[fi][1])
                b = bucket(d) if d is not None else None
                if b is None:
                    continue
                a = agg[s]['by'].setdefault(b, {'found': 0, 'total': 0, 'kps_err': []})
                a['total'] += 1
                if fi in scans[s]:
                    a['found'] += 1
                    if s != ref_size:
                        a['kps_err'].append(
                            float(np.linalg.norm(scans[s][fi][1] - ref[fi][1], axis=1).mean()))
            agg[s]['found'] += len(scans[s])
            agg[s]['total'] += len(frames)

        # ---- temporal step 2: interpolate the odd frames, compare to truth ----
        for fi in frames:
            a, b = fi - 1, fi + 1
            if fi in ref and a in ref and b in ref and (fi - frames[0]) % 2 == 1:
                lerp = (ref[a][1] + ref[b][1]) / 2.0
                err = float(np.linalg.norm(lerp - ref[fi][1], axis=1).mean())
                eye = float(np.linalg.norm(ref[fi][1][1] - ref[fi][1][0])) or 1.0
                d, _ = difficulty(ref[fi][1])
                interp_rows.append((bucket(d), err, err / eye * 100.0))

    print("\n" + "=" * 78)
    print("  1. det_size — recall by pose difficulty (difficulty judged at "
          f"{ref_size})")
    print(f"  {'bucket':22s} {'n':>6} " + " ".join(f"{s:>13}" for s in sizes))
    for b in ORDER:
        if b not in agg[ref_size]['by']:
            continue
        n = agg[ref_size]['by'][b]['total']
        cells = []
        for s in sizes:
            a = agg[s]['by'].get(b, {'found': 0, 'total': 1})
            cells.append(f"{a['found']/max(1,a['total']):12.1%} ")
        print(f"  {b:22s} {n:6d} " + " ".join(cells))
    print(f"  {'ALL':22s} {agg[ref_size]['total']:6d} " +
          " ".join(f"{agg[s]['found']/max(1,agg[s]['total']):12.1%} " for s in sizes))

    print(f"\n  landmark shift vs det_size {ref_size} (px, where both found a face)")
    for s in sizes[1:]:
        for b in ORDER:
            e = agg[s]['by'].get(b, {}).get('kps_err', [])
            if e:
                print(f"    {s} @ {b:22s} mean {np.mean(e):5.2f} px  p95 {np.percentile(e,95):5.2f} px")

    print("\n" + "-" * 78)
    print("  2. ROOP_TEMPORAL_STEP=2 — error of the interpolated frames")
    print(f"  {'bucket':22s} {'n':>6} {'mean px':>9} {'p95 px':>8} "
          f"{'mean %eye':>10} {'p95 %eye':>9}")
    for b in ORDER:
        rows = [r for r in interp_rows if r[0] == b]
        if not rows:
            continue
        px = [r[1] for r in rows]
        pe = [r[2] for r in rows]
        print(f"  {b:22s} {len(rows):6d} {np.mean(px):9.2f} {np.percentile(px,95):8.2f} "
              f"{np.mean(pe):9.1f}% {np.percentile(pe,95):8.1f}%")
    if interp_rows:
        allpe = [r[2] for r in interp_rows]
        print(f"  {'ALL':22s} {len(interp_rows):6d} "
              f"{np.mean([r[1] for r in interp_rows]):9.2f} "
              f"{np.percentile([r[1] for r in interp_rows],95):8.2f} "
              f"{np.mean(allpe):9.1f}% {np.percentile(allpe,95):8.1f}%")
    print("\n  %eye = landmark error as a share of interocular distance. The swap")
    print("  is aligned from these 5 points, so this is the misalignment the")
    print("  interpolated frames would be swapped with.")
    print("=" * 78)


if __name__ == '__main__':
    main()
