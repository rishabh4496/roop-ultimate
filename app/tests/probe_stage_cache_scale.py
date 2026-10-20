#!/usr/bin/env python3
"""Cost of the Stage 2 cache at scale: write time, restore time, peak RSS and file size for N frames x 2 faces.

The faces carry the REAL field mix of a temporal pre-pass (float32 bbox / kps / both landmark sets / the
512-d embedding, mixed python-float and numpy-float32 `det_score`, the nested `_pose_v5` block that
`insightface.Face` turns into Face-in-Face), so the per-face size and the JSON-block path are the ones a
production clip hits. No GPU, no models.

    app/env/Scripts/python.exe app/tests/probe_stage_cache_scale.py 10000 40000

The restored faces ARE the set the pre-pass would have left resident (the swap pass consumes and evicts them
frame by frame), so the read peak is not extra memory the resume adds on top of a fresh run, except for the
transient JSON block being decoded. Measured on the 4070 host 2026-10-20: see docs/perf/stage6/.
"""
import os
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)

import psutil                                   # noqa: E402
from insightface.app.common import Face         # noqa: E402

from roop import stage_cache as sc              # noqa: E402
from roop.stage2_plan import build_stage2_plan  # noqa: E402


def make(n_frames, rng):
    emb = {t: rng.standard_normal(512).astype(np.float32) for t in (1, 2)}

    def face(track):
        f = Face(bbox=(rng.random(4) * 500).astype(np.float32), kps=(rng.random((5, 2)) * 500).astype(np.float32),
                 landmark_2d_106=(rng.random((106, 2)) * 500).astype(np.float32),
                 landmark_3d_68=(rng.random((68, 3)) * 500).astype(np.float32),
                 embedding=rng.standard_normal(512).astype(np.float32),
                 det_score=np.float32(0.9) if rng.random() < 0.1 else 0.9, pose=(rng.random(3) * 30).astype(np.float32))
        f['_track_id'] = track
        f['_interpolated'] = False
        f['roll_deg'] = 1.5
        f['_pose_v5'] = {'yaw': 1.0, 'pitch': 2.0, 'roll': 3.0, 'proportions': {'a': 1.0}, 'expression': {'c': 0.1},
                         'confidence': 0.9, 'inverted': False, 'available': True}
        return f

    faces = {i: [face(1), face(2)] for i in range(n_frames)}
    assign = {i: [((1.0, 2.0), 0, emb[1]), ((3.0, 4.0), None, emb[2])] for i in range(n_frames)}
    smap = {1: (0, emb[1]), 2: (None, emb[2])}
    pm = SimpleNamespace(_temporal_faces=faces, _track_assignments=assign, _track_source_map=smap,
                         _shot_boundaries={5}, _temporal_covered=n_frames, _active_ranges=None)
    pm.stage2_frame_plan, _ = build_stage2_plan(faces, smap)
    return pm


def run(n_frames):
    rng = np.random.default_rng(0)
    pm = make(n_frames, rng)
    n = sum(len(v) for v in pm._temporal_faces.values())
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.01)

    threading.Thread(target=sample, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="stagecache-scale-") as d:
        path = os.path.join(d, "scale.stage2.npz")
        t0 = time.perf_counter()
        sc.write_stage2(path, pm, frame_start=0, frame_count=n_frames, ranges_abs=None, plan=pm.stage2_frame_plan)
        t_w = time.perf_counter() - t0
        w_peak = peak[0] - base
        peak[0] = proc.memory_info().rss
        t0 = time.perf_counter()
        sc.restore_stage2(path, SimpleNamespace(), new_start=0, new_count=n_frames, current_ranges_abs=None,
                          build_plan=build_stage2_plan)
        t_r = time.perf_counter() - t0
        r_peak = peak[0] - base
        size = os.path.getsize(path)
    stop.set()
    print("faces %6d (%6d frames) | write %5.1f s +%4.0f MB peak RSS | read %5.1f s +%4.0f MB peak RSS | file %6.1f MB (%.2f KB/face)"
          % (n, n_frames, t_w, w_peak / 1048576, t_r, r_peak / 1048576, size / 1048576, size / 1024 / n), flush=True)


if __name__ == "__main__":
    for arg in (sys.argv[1:] or ["10000"]):
        run(int(arg))
