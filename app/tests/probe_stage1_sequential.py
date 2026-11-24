"""Stage 1 pre-scan vs dense ground truth: what does it MISS, and what does it cost?

Two steps, so the slow one is paid once per clip window:

  truth  Production detection (`get_all_faces`, full rescue ladder) on EVERY frame of the window, plus the hard cuts
         (64x36 luma delta between consecutive frames > CUT_DELTA_Y). Saved as <out>/<name>.npz.
  scan   The real `scan_active_ranges` on the same window, universal and targeted, scored against that file:
         `missed_raw` = truth frames outside the mask (they go out UNSWAPPED); `missed_persistent` ignores truth
         runs shorter than 3 frames (single-frame detector flicker the tracker would never swap).

    env/Scripts/python.exe tests/probe_stage1_sequential.py truth --video <clip> --start 3000 --frames 3000 --out <dir> --name b1a
    env/Scripts/python.exe tests/probe_stage1_sequential.py scan  --video <clip> --start 3000 --out <dir> --name b1a
    ROOP_HW_IO=0 ... scan     # the cv2 fallback source, for the A/B

cwd = app. The truth run takes ~1 min per 1000 frames; the scan a few seconds. Read `[SparseScan]`'s `samples`,
`skipped` (samples a positive shot made redundant) and `workers`/`util` beside the seconds: a scan that is faster
because it looked at fewer frames has not got faster. One render at a time: this loads the detector pool.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MIN_RUN = 3                    # a truth run shorter than this is detector flicker, not a face to swap


def persistent(truth, min_run=MIN_RUN):
    out = truth.copy()
    i, n = 0, len(truth)
    while i < n:
        if truth[i]:
            j = i
            while j + 1 < n and truth[j + 1]:
                j += 1
            if j - i + 1 < min_run:
                out[i:j + 1] = False
            i = j + 1
        else:
            i += 1
    return out


def _pipeline():
    import yaml
    import angle_bench as ab
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'), encoding='utf-8')) or {}
    g = ab.init_pipeline(cfg.get('provider') or 'cuda', cfg.get('swap_model') or 'inswapper', 'None', 'None',
                         sync_config=True)
    g.g_desired_face_analysis = ['landmark_2d_106', 'detection', 'recognition']
    return g, float(cfg.get('max_face_distance', 0.65))


def _target_embedding(video, start, frames, get_all_faces):
    """The first face (with an embedding) of the first window frame that has one - the 'person to swap'."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    emb = None
    for _ in range(frames):
        ok, fr = cap.read()
        if not ok:
            break
        faces = [f for f in get_all_faces(fr) if getattr(f, 'embedding', None) is not None]
        if faces:
            emb = np.asarray(faces[0].embedding, np.float32)
            break
    cap.release()
    return emb


def cmd_truth(a):
    from roop import sparse_range_scan as s
    from roop.face_util import get_all_faces
    _, swap_distance = _pipeline()
    cap = cv2.VideoCapture(a.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    per, thumbs, t0 = [], [], time.perf_counter()
    for _ in range(a.frames):
        ok, fr = cap.read()
        if not ok:
            break
        thumbs.append(s.luma_thumb(fr))
        per.append([None if getattr(f, 'embedding', None) is None else
                    np.asarray(f.embedding, np.float32) / (np.linalg.norm(f.embedding) + 1e-9)
                    for f in get_all_faces(fr)])
    cap.release()
    n = len(per)
    universal = np.array([len(f) > 0 for f in per])
    tf = next((i for i, f in enumerate(per) if any(e is not None for e in f)), None)
    target = np.zeros(n, bool)
    if tf is not None:
        emb = [e for e in per[tf] if e is not None][0]
        for i, f in enumerate(per):
            target[i] = any(e is not None and (1.0 - float(e @ emb)) <= swap_distance for e in f)
    cuts = [i for i in range(1, n) if s.luma_delta(thumbs[i - 1], thumbs[i]) > s.CUT_DELTA_Y]
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, a.name + '.npz'), universal=universal, target=target, cuts=np.array(cuts),
             fps=fps, start=a.start)
    print(f'{a.name}: {n} frames, universal {int(universal.sum())}, target {int(target.sum())}, cuts {len(cuts)}, '
          f'{time.perf_counter() - t0:.0f}s', flush=True)


def cmd_scan(a):
    from roop import sparse_range_scan as s
    from roop.face_util import get_all_faces
    g, swap_distance = _pipeline()
    d = np.load(os.path.join(a.out, a.name + '.npz'))
    uni, tar, fps = d['universal'].astype(bool), d['target'].astype(bool), float(d['fps'])
    n = len(uni)
    emb = _target_embedding(a.video, a.start, n, get_all_faces)

    class T:
        embedding = emb

    rows = []
    for mode, truth, faces in (('universal', uni, None), ('target', tar, [T()] if emb is not None else None)):
        if mode == 'target' and faces is None:
            continue
        p = persistent(truth)
        t = time.perf_counter()
        _, mask, st = s.scan_active_ranges(a.video, target_faces=faces, frame_start=a.start, frame_end=a.start + n,
                                           frame_count=n, fps=fps, swap_distance_threshold=swap_distance,
                                           det_thresh=float(g.face_detector_threshold))
        row = dict(clip=a.name, mode=mode, seconds=round(time.perf_counter() - t, 2),
                   active_pct=round(100 * float(mask.mean()), 1),
                   missed_raw=int((truth & ~mask).sum()), truth_raw=int(truth.sum()),
                   missed_persistent=int((p & ~mask).sum()), truth_persistent=int(p.sum()),
                   samples=st.get('samples_done'), skipped=st.get('skipped_samples'), cuts=st.get('cuts'),
                   backend=st.get('decode_backend'), workers=st.get('workers'), scan_mode=st.get('mode'),
                   util_pct=st.get('worker_utilization_pct'))
        rows.append(row)
        print('RESULT', json.dumps(row), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('truth', 'scan'))
    ap.add_argument('--video', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=3000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', required=True)
    a = ap.parse_args()
    (cmd_truth if a.cmd == 'truth' else cmd_scan)(a)


if __name__ == '__main__':
    main()
