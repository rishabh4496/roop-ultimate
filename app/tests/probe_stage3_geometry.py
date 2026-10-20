"""Stage 3 probe: what precision and coordinates does the cached Stage 2 face data carry?

Runs the real pre-pass (`_precompute_temporal`) on a clip, then for the cached faces reports

  * the dtype of kps / landmark_2d_106 / landmark_3d_68 / pose / bbox / embedding, split into
    detected and gap-filled (interpolated / coasted) faces;
  * the distance, in pixels, between each DETECTED cached face and a fresh `get_all_faces` on
    the same decoded frame (the cache is smoothed, so a few px is expected; an axis swap or a
    scaling error on a vertical clip would be hundreds), and whether every cached point lies
    inside the frame.

    env/Scripts/python.exe tests/probe_stage3_geometry.py --clip <mp4> --source akansha [--rotate 90] [--out f.json]
"""
import argparse
import collections
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

import compare_enhancers_video as C            # noqa: E402
from roop.ProcessMgr import ProcessMgr         # noqa: E402

FIELDS = ('bbox', 'kps', 'landmark_2d_106', 'landmark_3d_68', 'pose', 'embedding')
SNAP = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', required=True)
    ap.add_argument('--source', default='akansha')
    ap.add_argument('--samples', type=int, default=25)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    orig = ProcessMgr._precompute_temporal

    def pre(self, *a, **k):
        r = orig(self, *a, **k)
        SNAP.update(self._temporal_faces or {})
        SNAP['_shape'] = getattr(self, '_temporal_frame_shape', None)
        return r

    ProcessMgr._precompute_temporal = pre
    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'))) or {}
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg', 'None': 'None'}.get(
        cfg.get('mask_engine', 'RealityUX'), cfg.get('mask_engine'))
    C.render(args.clip, args.source, args.enhancer if hasattr(args, 'enhancer') else 'GPEN Realistic',
             os.path.join(APP, 'output', 'probe_stage3_geom'), cfg.get('swap_model', 'realswap'),
             mask, int(cfg.get('max_threads', 16)), overrides={})

    shape = SNAP.pop('_shape', None)
    dtypes = {'detected': collections.defaultdict(collections.Counter),
              'gap_filled': collections.defaultdict(collections.Counter)}
    for idx, faces in SNAP.items():
        for f in faces:
            kind = 'gap_filled' if (f.get('_interpolated') or f.get('_coasted')) else 'detected'
            for name in FIELDS:
                v = f.get(name)
                dtypes[kind][name][('none' if v is None else str(np.asarray(v).dtype))] += 1

    cap = cv2.VideoCapture(args.clip)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    from roop.face_util import get_all_faces
    cand = [i for i in sorted(SNAP) if SNAP[i] and not any(
        x.get('_interpolated') or x.get('_coasted') for x in SNAP[i])]
    step = max(1, len(cand) // args.samples)
    errs = {'kps': [], 'landmark_2d_106': [], 'landmark_3d_68': [], 'bbox': []}
    out_of_frame = 0
    for i in cand[::step][:args.samples]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        fresh = get_all_faces(fr)
        for c in SNAP[i]:
            cc = np.asarray(c['bbox'], np.float64)[:2]
            best = min(fresh, key=lambda f: np.linalg.norm(np.asarray(f.bbox, np.float64)[:2] - cc), default=None)
            if best is None:
                continue
            for name in errs:
                a, b = c.get(name), best.get(name)
                if a is None or b is None:
                    continue
                a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
                if name == 'landmark_3d_68':
                    a, b = a[:, :2], b[:, :2]
                if a.shape == b.shape:
                    errs[name].append(float(np.linalg.norm(a.reshape(-1, 2 if name != 'bbox' else 2) - b.reshape(-1, 2 if name != 'bbox' else 2), axis=1).mean()))
            for name in ('kps', 'landmark_2d_106'):
                pts = np.asarray(c.get(name), np.float64).reshape(-1, 2)
                out_of_frame += int(((pts[:, 0] < -w * .1) | (pts[:, 0] > w * 1.1) | (pts[:, 1] < -h * .1) | (pts[:, 1] > h * 1.1)).any())
    res = dict(frame_wh=[w, h], frames=n, cached_frames=len(SNAP), temporal_frame_shape=shape,
               dtypes={k: {f: dict(c) for f, c in v.items()} for k, v in dtypes.items()},
               mean_px_error_vs_fresh_detection={k: (round(float(np.mean(v)), 3) if v else None,
                                                    round(float(np.max(v)), 3) if v else None, len(v))
                                                 for k, v in errs.items()},
               points_far_outside_frame=out_of_frame)
    print('GEOM', json.dumps(res))
    if args.out:
        json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
