"""Where does the PRODUCTION pre-pass spend its time, and which Stage 11 asks can move it?

    ROOP_PROFILE=1 env/Scripts/python.exe tests/probe_prepass_stage11.py --section baseline --clip double/d4.mp4

Measurement only. It drives `ProcessMgr._precompute_temporal` -- the live "Analyzing faces" pass
(`procmgr_tracking._precompute_tracks` + `temporal_tracker`), NOT `optimized_prepass.FacePrepass`, which
only `optimized_processor` / `vectorized_pipeline` import -- through `angle_bench.init_pipeline`
(sync_config=True) so TensorRT's DLLs are on PATH and the models are the ones config.yaml names.

  baseline  N repeats of the same slice in one process: wall clock, fps, the `[Temporal] detector
            schedule` line, faces found, and the STAGE TIMING table. Repeat 1 pays the engine build and
            the first-arm penalty; compare repeats 2..N (that is the null control).
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

KEEP = os.environ.get("ROOP_MEDIA_DIR") or os.path.normpath(os.path.join(APP, "..", "..", "..", "roop-keep"))


def make_mgr():
    from roop.ProcessMgr import ProcessMgr
    from roop.ProcessOptions import ProcessOptions
    mgr = ProcessMgr(None)
    opts = ProcessOptions.__new__(ProcessOptions)
    for name, value in (
        ("stabilize_face", True), ("stabilize_method", "one_euro"),
        ("stabilize_min_cutoff", 0.1), ("stabilize_beta", 0.1),
        ("stabilize_landmarks", True), ("swap_mode", "selected"),
        ("selected_index", 0), ("face_distance_threshold", 0.75),
    ):
        setattr(opts, name, value)
    mgr.options = opts
    mgr.target_face_datas = []
    mgr.target_face_groups = []
    mgr.progress_gradio = None
    return mgr


def section_baseline(path, frames, repeats):
    import roop.globals as G
    from roop import procmgr_runtime as rt
    G.processing = True
    rows = []
    for r in range(repeats):
        rt._prof_reset()
        mgr = make_mgr()
        t0 = time.perf_counter()
        mgr._precompute_temporal(path, None, 0, frames, frames)
        dt = time.perf_counter() - t0
        tf = mgr._temporal_faces or {}
        n_faces = sum(len(v) for v in tf.values())
        real = sum(1 for v in tf.values() for f in v if not f.get("_interpolated") and not f.get("_coasted"))
        rows.append((r + 1, dt, frames / dt, len(tf), n_faces, real))
        print("[probe] repeat %d: %.1fs  %.1f fps  frames-with-face %d  faces %d  detected %d"
              % rows[-1], flush=True)
        rt._prof_report()
    print("\nrepeat | s | fps | frames-with-face | faces | detected")
    for row in rows:
        print("%d | %.1f | %.1f | %d | %d | %d" % row)


LADDER_CLIPS = ["double/d4.mp4", "Love.mp4", "Weeds.mp4", "b1.mp4", "single/s3.mp4", "single/s1.mp4"]


def section_ladder(frames):
    import json
    import cv2
    from roop import face_util as fu
    out = {}
    for rel in LADDER_CLIPS:
        path = os.path.join(KEEP, rel)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("[probe] cannot open", path)
            continue
        rows = []
        for _ in range(frames):
            ok, fr = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            a = fu.get_all_faces(fr, rescue=False) or []
            t1 = time.perf_counter()
            b = fu.get_all_faces(fr) or []
            t2 = time.perf_counter()
            rows.append((len(a), len(b), t1 - t0, t2 - t1))
        cap.release()
        out[rel] = rows
        n = max(len(rows), 1)
        print("[probe] %-16s %4d frames  first-pass faces on %3d  ladder faces on %3d  ladder-only frames %3d  "
              "first-pass %.1f ms  full call %.1f ms" % (
                  rel, len(rows), sum(1 for r in rows if r[0]), sum(1 for r in rows if r[1]),
                  sum(1 for r in rows if r[1] > r[0]),
                  1000 * sum(r[2] for r in rows) / n, 1000 * sum(r[3] for r in rows) / n), flush=True)
    dump = os.path.join(os.environ.get("ROOP_PROBE_OUT", "."), "stage11_ladder_rows.json")
    with open(dump, "w") as fh:
        json.dump(out, fh)
    print("[probe] wrote", dump)
    simulate(out)


def simulate_one(rows, k, n, rewind):
    """Replay a ladder-stride policy over recorded (first_n, full_n, t_first, t_full) rows.

    The ladder is skipped on a frame while the last `k` looked-at frames were faceless, except every `n`th
    frame. A skipped frame whose plain first pass finds a face ends the gap; with `rewind` the frames
    skipped before it get their ladder then. Lost = frames where the ladder found a face the plain pass did
    not and the policy never ran the ladder on. Cost uses the full call for a laddered frame, else the plain pass."""
    cost = 0.0
    streak = 0
    last = None
    skipped = []
    lost = 0
    laddered = 0
    for i, (a, b, t_first, t_full) in enumerate(rows):
        eligible = streak < k or last is None or (i - last) >= n
        if eligible:
            cost += t_full
            laddered += 1
            last = i
            streak = 0 if max(a, b) > 0 else streak + 1
            found = max(a, b) > 0
        elif a > 0:
            cost += t_first
            streak = 0
            found = True
        else:
            cost += t_first
            skipped.append(i)
            continue
        if found and skipped:
            if rewind:
                for j in skipped:
                    cost += rows[j][3] - rows[j][2]
                    laddered += 1
            else:
                lost += sum(1 for j in skipped if rows[j][1] > rows[j][0])
            skipped = []
    lost += sum(1 for j in skipped if rows[j][1] > rows[j][0])
    return cost, laddered, lost


def simulate(data):
    print("\npolicy                | clip             | laddered/frames | lost of ladder-only | cost vs exhaustive")
    for k, n, rewind in ((2, 3, True), (2, 3, False), (2, 8, True), (4, 8, True), (2, 8, False)):
        for rel, rows in data.items():
            total = sum(r[3] for r in rows) or 1e-9
            cost, laddered, lost = simulate_one(rows, k, n, rewind)
            only = sum(1 for r in rows if r[1] > r[0])
            print("k=%d N=%d %-8s | %-16s | %4d/%4d | %3d of %3d | %.2fx" % (
                k, n, "rewind" if rewind else "no-rewind", rel, laddered, len(rows), lost, only, cost / total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=("baseline", "ladder"), required=True)
    ap.add_argument("--clip", default="double/d4.mp4")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    import angle_bench as ab
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)
    path = args.clip if os.path.isabs(args.clip) else os.path.join(KEEP, args.clip)
    print("[probe] clip=%s frames=%d" % (path, args.frames))
    {"baseline": lambda: section_baseline(path, args.frames, args.repeats),
     "ladder": lambda: section_ladder(args.frames)}[args.section]()


if __name__ == "__main__":
    main()
