"""Does Stage 3 follow the frames Stage 1 and Stage 2 decided? A real 3-stage render, scored against dense truth.

One real render of a clip window with Stage 1 ON (the regression harness leaves it off), recording per frame:

  Stage 1  the active-range mask                              (`_active_ranges_mask`)
  Stage 2  frames whose cache holds a face / matched plan     (`_temporal_faces`, `stage2_frame_plan`)
  Stage 3  frames that enter the GPU path, frames swapped (`swap_faces`), and detector calls Stage 3 makes
           ITSELF (`get_all_faces` after the pre-pass: a frame the cache did not cover)

and scores them against the dense truth written by `probe_stage1_sequential.py truth` (<dir>/<name>.npz).
`target_persistent` is the number that matters: truth runs of >= 3 frames (shorter ones are detector flicker)
and where each was lost: `lost_by_mask` (Stage 1), `lost_by_s2` (Stage 2 cache empty), `planned_not_swapped`
(Stage 3 refused), `never_swapped` (all of them). Other people's faces are not swaps, so `universal_*` is
context only.

    env/Scripts/python.exe tests/probe_stage3_follows_stages.py --video <clip> --start 600 --frames 900 \
        --truth <dir>/love.npz --out <dir>/love_follow.json

cwd = app. One render at a time (it loads the whole pipeline); the card must be idle.
"""
import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MIN_RUN = 3


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


def runs(indices):
    out, start, prev = [], None, None
    for i in indices:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append((start, prev))
            start = prev = i
    if start is not None:
        out.append((start, prev))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--start', type=int, required=True, help='absolute first frame of the window')
    ap.add_argument('--frames', type=int, default=900)
    ap.add_argument('--truth', required=True, help='the .npz from probe_stage1_sequential.py truth')
    ap.add_argument('--out', required=True)
    ap.add_argument('--out-dir', default=None, help='where the rendered file goes (default: next to --out)')
    args = ap.parse_args()
    os.chdir(APP)

    from roop.benchmark import regression as R
    log = lambda s: print(s, flush=True)                      # noqa: E731
    setup = R.prepare_pipeline(None, R.default_source(), log)
    g = setup.g
    g.sparse_auto_range = True

    # the target the truth used: the first face found in the window
    from roop.face_util import get_all_faces
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    target = None
    for _ in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            break
        faces = [f for f in (get_all_faces(frame) or []) if getattr(f, 'embedding', None) is not None]
        if faces:
            target = faces[0]
            break
    cap.release()
    if target is None:
        raise SystemExit('no face in the window')

    import roop.ProcessMgr as pmmod
    from roop.ProcessMgr import ProcessMgr
    rec = dict(mask=None, s2_faces=set(), plan=set(), entered=set(), swapped=set(), s3_detect=0, phase3=False,
               stage1=None, s2_scanned=None)
    lock = threading.Lock()

    orig_pre = ProcessMgr._precompute_temporal

    def pre(mgr, *a, **k):
        r = orig_pre(mgr, *a, **k)
        mask = getattr(mgr, '_active_ranges_mask', None)
        rec['mask'] = None if mask is None else np.asarray(mask, bool).copy()
        cache = getattr(mgr, '_temporal_faces', None) or {}
        rec['s2_faces'] = {int(i) for i, v in cache.items() if v}
        rec['plan'] = {int(i) for i in (getattr(mgr, 'stage2_frame_plan', None) or {})}
        rec['s2_scanned'] = int(getattr(mgr, '_temporal_covered', 0) or 0)
        stats = dict(getattr(mgr, '_active_ranges_stats', None) or {})
        rec['stage1'] = {k: stats[k] for k in ('mode', 'decode_backend', 'active_frames', 'skipped_frames',
                                                'samples', 'skipped_samples', 'elapsed_s', 'shots') if k in stats}
        rec['phase3'] = True
        return r

    ProcessMgr._precompute_temporal = pre

    orig_swap = ProcessMgr.swap_faces

    def swap(mgr, frame, *a, **k):
        idx = k.get('frame_idx')
        n, img = orig_swap(mgr, frame, *a, **k)
        if idx is not None:
            with lock:
                rec['entered'].add(int(idx))
                if n and n > 0:
                    rec['swapped'].add(int(idx))
        return n, img

    ProcessMgr.swap_faces = swap

    orig_detect = getattr(pmmod, 'get_all_faces', None)
    if orig_detect is not None:
        def counted(frame, *a, **k):
            if rec['phase3']:
                with lock:
                    rec['s3_detect'] += 1
            return orig_detect(frame, *a, **k)
        pmmod.get_all_faces = counted

    from roop.core import batch_process_with_options
    from roop.ProcessEntry import ProcessEntry
    from roop.target_selection import normalize_target_selection
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)), 'follow_render')
    os.makedirs(out_dir, exist_ok=True)
    g.INPUT_FACESETS = [setup.faceset]
    g.TARGET_FACES = [target]
    g.TARGET_FACE_GROUP = [0]
    g.output_path = out_dir
    setup.options.selection_state = normalize_target_selection(
        {"selection_mode": "multi_person", "person_ids": [0]}, person_count=1)
    pmmod._SWAP_LOG = {}
    entry = ProcessEntry(args.video, args.start, args.start + args.frames, R.probe(args.video)[2])
    t0 = time.perf_counter()
    batch_process_with_options([entry], setup.options, None)
    wall = time.perf_counter() - t0
    pmmod._SWAP_LOG = None

    n = args.frames
    truth = np.load(args.truth)
    off = args.start - int(truth['start'])
    universal, target_truth = truth['universal'][off:off + n], truth['target'][off:off + n]

    def mark(indices):
        a = np.zeros(n, bool)
        a[[i for i in indices if 0 <= i < n]] = True
        return a

    mask = rec['mask'] if rec['mask'] is not None else np.ones(n, bool)
    mask = np.pad(mask[:n], (0, max(0, n - len(mask))))
    s2, plan, entered, swapped = (mark(rec[k]) for k in ('s2_faces', 'plan', 'entered', 'swapped'))
    c = lambda a: int(np.count_nonzero(a))                    # noqa: E731
    res = dict(window=[args.start, args.start + n], wall_s=round(wall, 1), stage1=rec['stage1'],
               s2_scanned=rec['s2_scanned'], mask=c(mask), s2_face_frames=c(s2), s2_matched_plan=c(plan),
               s3_entered_gpu=c(entered), s3_swapped=c(swapped), s3_detector_calls=rec['s3_detect'],
               s2_faces_outside_mask=c(s2 & ~mask), s3_entered_outside_mask=c(entered & ~mask),
               plan_not_swapped=c(plan & ~swapped), swapped_without_plan=c(swapped & ~plan),
               swapped_without_s2_face=c(swapped & ~s2))
    for label, t in (('target_persistent', persistent(target_truth)), ('target_raw', target_truth),
                     ('universal_persistent', persistent(universal))):
        res[label] = dict(n=c(t), lost_by_mask=c(t & ~mask), lost_by_s2=c(t & mask & ~s2),
                          planned_not_swapped=c(t & s2 & ~swapped), swapped=c(t & swapped),
                          never_swapped=c(t & ~swapped))
    res['runs'] = dict(target_persistent_never_swapped=runs(np.nonzero(persistent(target_truth) & ~swapped)[0].tolist()),
                       plan_not_swapped=runs(np.nonzero(plan & ~swapped)[0].tolist()),
                       swapped_without_plan=runs(np.nonzero(swapped & ~plan)[0].tolist()))
    json.dump(res, open(args.out, 'w'), indent=1)
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
