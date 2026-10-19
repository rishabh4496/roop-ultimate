"""Stage 1 sparse pre-scan on real footage: what does it MISS, and what does it cost?

Ground truth is the production detector (`get_all_faces`, full rescue ladder, aux models)
run on EVERY frame of the window. A frame that has a face (universal) or a face of the
target (targeted: cosine distance <= the swap threshold) but lies outside the scan's mask
would be passed through UNSWAPPED, so "missed frames" is the number that matters; "skipped"
is what the scan buys. The old implementation (git HEAD at the time of the rewrite, loaded
from `--old`) is run on the same window for the same two numbers plus wall clock.

Also measures the decode cost of retrieving the per-interval midpoint (the deviation from
"retrieve only on sampled indices"): grab-only vs grab+retrieve(grid) vs grab+retrieve(grid+mid).

Usage (cwd = app):
    env/Scripts/python.exe tests/probe_sparse_scan_stage1.py --video <clip> --frames 1200 \
        [--old <path to the old sparse_range_scan.py>] [--out docs/perf/sparse_scan_stage1.json]
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def decode_cost(video, start, frames, stride):
    """Seconds to decode the window three ways (same frames, fresh capture each)."""
    out = {}
    for name in ('grab_only', 'grid_retrieve', 'grid_plus_mid_retrieve'):
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        t0 = time.perf_counter()
        for off in range(frames):
            if not cap.grab():
                break
            on_grid = off % stride == 0
            on_mid = name == 'grid_plus_mid_retrieve' and off % stride == stride - stride // 2
            if name != 'grab_only' and (on_grid or on_mid):
                cap.retrieve()
        out[name] = round(time.perf_counter() - t0, 3)
        cap.release()
    return out


def load_old(path):
    spec = importlib.util.spec_from_file_location('old_sparse_range_scan', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def missed(truth, mask):
    idx = np.flatnonzero(truth & ~mask)
    return [int(i) for i in idx]


def summarise(name, ranges, mask, stats, truth, seconds):
    miss = missed(truth, mask)
    row = dict(name=name, seconds=round(seconds, 2), ranges=len(ranges),
               active=int(mask.sum()), total=int(mask.size),
               skipped_pct=round(100.0 * (1 - mask.mean()), 1),
               truth_frames=int(truth.sum()), missed=len(miss),
               missed_first=miss[:12], mode=stats.get('mode'), stride=stats.get('stride'),
               cuts=stats.get('cuts'), mids=stats.get('mids_inspected'),
               workers=stats.get('workers'), detector_errors=stats.get('detector_errors'))
    print(f"  {name:<22} {seconds:7.2f}s  ranges={len(ranges):3d}  active={row['active']}/{row['total']}"
          f"  skipped={row['skipped_pct']}%  truth={row['truth_frames']}  MISSED={len(miss)}"
          f"  {miss[:8] if miss else ''}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=1200)
    ap.add_argument('--old', default=None, help='path to the pre-rewrite sparse_range_scan.py')
    ap.add_argument('--target-frame', type=int, default=None,
                    help='window-relative frame whose largest face is the target (default: first face)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import yaml
    import angle_bench as ab
    with open(os.path.join(APP, 'config.yaml')) as fh:
        cfg = yaml.safe_load(fh) or {}
    g = ab.init_pipeline(cfg.get('provider') or 'cuda', cfg.get('swap_model') or 'inswapper',
                         'None', 'None', sync_config=True)
    # what ProcessMgr.initialize sets for a normal swap (no 68-pt model needed here)
    g.g_desired_face_analysis = ["landmark_2d_106", "detection", "recognition"]
    swap_distance = float(cfg.get('max_face_distance', 0.65))

    from roop.face_util import get_all_faces
    from roop import sparse_range_scan as new

    cap = cv2.VideoCapture(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = min(args.frames, n_video - args.start)
    end = args.start + frames
    cap.release()
    print(f"{os.path.basename(args.video)}: window {args.start}..{end} ({frames} frames, {fps:.3f} fps), "
          f"detector={g.detector_engine} thr={g.face_detector_threshold} swap_distance={swap_distance}")

    # ---- ground truth: production detection on every frame ---------------------------
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    per_frame = []
    for _ in range(frames):
        ok, fr = cap.read()
        if not ok:
            break
        faces = get_all_faces(fr)
        per_frame.append([(float(getattr(f, 'det_score', 0.0)),
                           None if getattr(f, 'embedding', None) is None else
                           np.asarray(f.embedding, np.float32) / (np.linalg.norm(f.embedding) + 1e-9),
                           float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
                          for f in faces])
    cap.release()
    t_dense = time.perf_counter() - t0
    frames = len(per_frame)
    universal = np.array([len(f) > 0 for f in per_frame], dtype=bool)
    print(f"dense production detection: {t_dense:.1f}s ({frames / t_dense:.1f} fps), "
          f"{int(universal.sum())}/{frames} frames have a face", flush=True)

    # ---- target: largest face of the first frame that has one ------------------------
    tf = args.target_frame
    if tf is None:
        tf = next((i for i, f in enumerate(per_frame) if any(e is not None for _, e, _ in f)), None)
    target_emb = None
    if tf is not None:
        cands = [(a, e) for _, e, a in per_frame[tf] if e is not None]
        if cands:
            target_emb = max(cands, key=lambda c: c[0])[1]
    targeted_truth = np.zeros(frames, dtype=bool)
    if target_emb is not None:
        for i, f in enumerate(per_frame):
            targeted_truth[i] = any(e is not None and (1.0 - float(e @ target_emb)) <= swap_distance
                                    for _, e, _ in f)
    print(f"target from window frame {tf}; {int(targeted_truth.sum())}/{frames} frames contain that "
          f"identity at distance <= {swap_distance}", flush=True)

    class T:                              # a target-face stand-in carrying only the embedding
        embedding = target_emb

    rows = []
    stride = new.sparse_stride(fps)
    print(f"decode cost over the window (S={stride}): {decode_cost(args.video, args.start, frames, stride)}",
          flush=True)

    def run(name, mod, truth, targeted):
        kw = dict(frame_start=args.start, frame_end=end, frame_count=frames, fps=fps)
        if mod is new:
            kw.update(swap_distance_threshold=swap_distance, det_thresh=float(g.face_detector_threshold))
        else:
            kw.update(threshold=swap_distance)
        faces = [T()] if targeted else None
        t = time.perf_counter()
        ranges, mask, stats = mod.scan_active_ranges(args.video, target_faces=faces, **kw)
        rows.append(summarise(name, ranges, mask, stats, truth, time.perf_counter() - t))

    print("\nuniversal mode (any face):")
    run('new', new, universal, False)
    run('new (2nd, warm)', new, universal, False)
    if args.old:
        old = load_old(args.old)
        run('old', old, universal, False)
    if target_emb is not None:
        print("\ntargeted mode (the swap's target identity only):")
        run('new', new, targeted_truth, True)
        if args.old:
            run('old', old, targeted_truth, True)

    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(dict(video=os.path.basename(args.video), window=[args.start, end], fps=fps,
                           dense_seconds=round(t_dense, 2), rows=rows), fh, indent=1)
        print('wrote', args.out)


if __name__ == '__main__':
    main()
