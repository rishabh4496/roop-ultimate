#!/usr/bin/env python3
"""Capture, from a REAL temporal pre-pass, every track's landmarks twice: RAW (what the detector said, taken at
the entry of `AdaptiveLandmarkSmoother.smooth`) and as CACHED after the shipped smoothing, plus the never-smoothed
`landmark_3d_68`. Saved as an .npz so candidate filters can be compared offline on identical data.

    app/env/Scripts/python.exe app/tests/probe_stage25_capture.py <clip.mp4> <out.npz> [--source akansha]

The pre-pass is stopped as soon as it returns (a BaseException the pipeline's own `except Exception` cannot
swallow), so no swap pass is rendered.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                   # noqa: E402

import compare_enhancers_video as C                  # noqa: E402
from roop.ProcessMgr import ProcessMgr               # noqa: E402
from roop.temporal_smoother import AdaptiveLandmarkSmoother  # noqa: E402


class Done(BaseException):
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('clip')
    ap.add_argument('out')
    ap.add_argument('--source', default='akansha')
    args = ap.parse_args()

    raw = {}                                         # (track, frame) -> (kps, lm106)
    orig_smooth = AdaptiveLandmarkSmoother.smooth

    def smooth(self, kps, lm, track_id=None, frame_index=None, *a, **k):
        raw[(int(track_id), int(frame_index))] = (np.asarray(kps, np.float32).copy(),
                                                  None if lm is None else np.asarray(lm, np.float32).copy())
        return orig_smooth(self, kps, lm, *a, track_id=track_id, frame_index=frame_index, **k)

    AdaptiveLandmarkSmoother.smooth = smooth
    orig_pre = ProcessMgr._precompute_temporal

    def pre(self, *a, **k):
        # production runs stabilize_face true (config.yaml); the bench harness's options default it OFF, which
        # skips the whole smoothing block - set it as the user's config does
        self.options.stabilize_face = True
        orig_pre(self, *a, **k)
        rows = []
        for f_idx, faces in self._temporal_faces.items():
            for face in faces:
                tid = face.get('_track_id')
                if tid is None or face.get('kps') is None:
                    continue
                key = (int(tid), int(f_idx))
                r = raw.get(key)
                lm68 = face.get('landmark_3d_68')
                rows.append((int(tid), int(f_idx), r is not None,
                             np.asarray(face['kps'], np.float32),
                             np.asarray(face['landmark_2d_106'], np.float32) if face.get('landmark_2d_106') is not None else None,
                             None if lm68 is None else np.asarray(lm68, np.float32),
                             r[0] if r else None, r[1] if r else None,
                             bool(face.get('_interpolated')), bool(face.get('_coasted'))))
        rows.sort(key=lambda x: (x[0], x[1]))
        n = len(rows)
        out = dict(
            track=np.array([r[0] for r in rows], np.int32), frame=np.array([r[1] for r in rows], np.int32),
            has_raw=np.array([r[2] for r in rows], bool),
            kps_cur=np.stack([r[3] for r in rows]),
            lm106_cur=np.stack([r[4] if r[4] is not None else np.full((106, 2), np.nan, np.float32) for r in rows]),
            lm68=np.stack([r[5] if r[5] is not None else np.full((68, 3), np.nan, np.float32) for r in rows]),
            kps_raw=np.stack([r[6] if r[6] is not None else np.full((5, 2), np.nan, np.float32) for r in rows]),
            lm106_raw=np.stack([r[7] if r[7] is not None else np.full((106, 2), np.nan, np.float32) for r in rows]),
            interp=np.array([r[8] for r in rows], bool), coasted=np.array([r[9] for r in rows], bool),
            cuts=np.array(sorted(int(c) for c in (getattr(self, '_shot_boundaries', None) or [])), np.int32),
            n_frames=np.int32(getattr(self, '_temporal_covered', 0)))
        np.savez(args.out, **out)
        print('CAPTURED %d face rows, %d with a raw record, %d tracks, cuts %s -> %s' % (
            n, int(out['has_raw'].sum()), len(set(out['track'].tolist())), out['cuts'].tolist()[:8], args.out), flush=True)
        raise Done()

    ProcessMgr._precompute_temporal = pre
    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'), encoding='utf-8')) or {}
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg', 'None': 'None'}.get(
        cfg.get('mask_engine', 'RealityUX'), cfg.get('mask_engine'))
    try:
        C.render(args.clip, args.source, 'None', os.path.join(os.path.dirname(os.path.abspath(args.out)), 'cap_out'),
                 cfg.get('swap_model', 'realswap'), mask, int(cfg.get('max_threads', 16)),
                 overrides={'sparse_auto_range': False})
    except Done:
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
