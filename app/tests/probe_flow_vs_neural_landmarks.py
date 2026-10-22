"""Between detector key frames, are flow-placed landmarks worse than re-running the landmark net on the flow box?

    app\\env\\Scripts\\python.exe app\\tests\\probe_flow_vs_neural_landmarks.py --clip Weeds.mp4 --frames 300 --k 4

``FlowCoaster.step`` (the pre-pass's gap bridge) fits a SIMILARITY transform to the tracked point cloud and applies it
to the KEY FRAME's landmarks, so a flow-placed face carries the key frame's expression: a blink or a jaw opening between
two key frames is not followed. Phase 14 measured the damage at the median (0.8-1.4% of an eye distance) and at the p95
(9-18%). This asks whether the 106-point net, run on the box the flow produced, closes that tail:

    truth        the production detector + landmark_2d_106 on EVERY frame (what K = 1 gives)
    flow         FlowCoaster from the last key frame (anchored on the truth face there), K = --k
    neural       landmark_2d_106 run on the FLOW box (bbox + kps from the flow result)
    neural@truth the same net on the TRUE box: must reproduce truth, or the harness is not measuring what it says

Error = mean landmark distance / eye distance of the truth kps, in percent. Only single-face frames the detector saw at
>= 0.6 are scored; a flow step that was refused (not consistent) is counted and not scored (the pre-pass drops those chains).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures  # noqa: E402


def nme(a, b, kps):
    iod = float(np.linalg.norm(np.asarray(kps)[1] - np.asarray(kps)[0]))
    return 100.0 * float(np.mean(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1))) / max(iod, 1e-6)


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if not x.size:
        return None
    return dict(n=int(x.size), median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)), max=float(x.max()), mean=float(x.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', default='Weeds.mp4')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import angle_bench as ab
    from insightface.app.common import Face
    from roop import face_util
    from roop.flow_coast import FlowCoaster
    ab.init_pipeline('cuda', 'realswap', 'None', 'None')
    fa = face_util.get_face_analyser()
    lm_model = fa.models['landmark_2d_106']

    path = fixtures.clip(args.clip, required=True)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    coaster, since_key = FlowCoaster(), None
    rows, refused, no_truth, total = [], 0, 0, 0
    for n in range(args.frames):
        ok, fr = cap.read()
        if not ok:
            break
        total += 1
        faces = [f for f in (face_util.get_all_faces(fr) or []) if getattr(f, 'det_score', 0) >= 0.6]
        if len(faces) != 1 or getattr(faces[0], 'landmark_2d_106', None) is None:
            coaster, since_key = FlowCoaster(), None
            no_truth += 1
            continue
        truth = faces[0]
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if n % args.k == 0 or since_key is None:
            coaster.anchor(gray, truth.bbox, truth.landmark_2d_106, truth.kps)
            since_key = 0
            continue
        since_key += 1
        r = coaster.step(gray)
        if r is None or not r.consistent or r.landmarks is None or r.bbox is None:
            refused += 1
            continue
        probe = Face(bbox=np.asarray(r.bbox, np.float32), kps=np.asarray(r.kps, np.float32), det_score=1.0)
        lm_model.get(fr, probe)
        same = Face(bbox=np.asarray(truth.bbox, np.float32), kps=np.asarray(truth.kps, np.float32), det_score=1.0)
        lm_model.get(fr, same)
        rows.append(dict(frame=args.start + n, gap=since_key,
                         flow=nme(r.landmarks, truth.landmark_2d_106, truth.kps),
                         neural=nme(probe['landmark_2d_106'], truth.landmark_2d_106, truth.kps),
                         neural_true=nme(same['landmark_2d_106'], truth.landmark_2d_106, truth.kps)))

    out = dict(clip=args.clip, k=args.k, frames_read=total, no_single_face=no_truth, flow_refused=refused, scored=len(rows),
               flow=stats([r['flow'] for r in rows]), neural=stats([r['neural'] for r in rows]),
               neural_at_true_box=stats([r['neural_true'] for r in rows]),
               neural_better_share=float(np.mean([r['neural'] < r['flow'] for r in rows])) if rows else None,
               by_gap={g: dict(flow=stats([r['flow'] for r in rows if r['gap'] == g]),
                               neural=stats([r['neural'] for r in rows if r['gap'] == g]))
                       for g in sorted({r['gap'] for r in rows})})
    print(json.dumps(out, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(dict(summary=out, rows=rows), fh, indent=1)


if __name__ == '__main__':
    main()
