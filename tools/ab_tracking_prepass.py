"""Run the REAL tracking pre-pass over a clip, with shot awareness on and off,
and report what changed.

This drives roop's own TrackingMixin._precompute_temporal -- the same code the
render path runs -- so the numbers are the pipeline's, not a model of it.

What to look at:

  gap-filled faces
      Faces nobody detected. Their geometry is interpolated between two
      observations and their embedding is set to the track mean, so they pass
      every downstream identity gate by construction and are swapped
      unconditionally. Across a shot boundary that paints a swap onto whatever
      the next shot contains. A drop here is the bug being fixed.

  tracks
      With cuts respected a person appearing in several shots is several
      tracks unless Re-ID reconnects them on appearance. More tracks is
      expected and is the point: it is the refusal to inherit a track by
      POSITION across a cut.

`--cuts off` sets ROOP_CUT_FLOOR high enough that the detector never fires,
which reproduces the old behaviour exactly for an A/B on identical footage.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def resolve(pattern):
    path = pattern
    if any(c in pattern for c in "*?[") or not os.path.exists(pattern):
        m = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if m:
            path = m[0]
    if os.name == "nt":
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024) and buf.value:
                return buf.value
        except Exception:
            pass
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--cuts", choices=("on", "off"), default="on")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.cuts == "off":
        # Above the maximum a luma-thumbnail difference can reach, so the
        # detector cannot fire: the pre-change behaviour.
        os.environ["ROOP_CUT_FLOOR"] = "9.0"

    import cv2
    import roop.globals as G

    G.face_detector_threshold = 0.5
    G.face_detector_size = "512"
    G.default_det_size = True
    G.processing = True
    G.autorotate_faces = True

    from roop.ProcessMgr import ProcessMgr
    from roop.ProcessOptions import ProcessOptions

    path = resolve(args.video)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    frames = min(total, args.frames) if args.frames else total
    print(f"clip: {path}\nframes: {frames} of {total}   cuts: {args.cuts}\n")

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

    t0 = time.time()
    mgr._precompute_temporal(path, None, 0, frames, frames)
    elapsed = time.time() - t0

    faces = mgr._temporal_faces or {}
    n_frames = len(faces)
    n_faces = sum(len(v) for v in faces.values())
    n_interp = sum(1 for v in faces.values() for f in v if f.get("_interpolated"))
    n_coast = sum(1 for v in faces.values() for f in v if f.get("_coasted"))
    cuts = len(getattr(mgr, "_shot_boundaries", None) or ())

    print("\n" + "=" * 62)
    print(f"cuts detected          : {cuts}")
    print(f"frames with a face     : {n_frames}")
    print(f"faces total            : {n_faces}")
    print(f"  gap-filled (invented): {n_interp}  "
          f"({100.0 * n_interp / max(n_faces, 1):.1f}% of all faces)")
    print(f"  of those, coasted    : {n_coast}")
    print(f"refused as unbridgeable: {getattr(mgr, '_interp_refused', 0)}")
    print(f"  of those, for a cut  : {getattr(mgr, '_interp_refused_cut', 0)}")
    print(f"pre-pass wall clock    : {elapsed:.1f}s "
          f"({frames / max(elapsed, 1e-6):.1f} fps)")
    print("=" * 62)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"cuts_mode": args.cuts, "cuts": cuts, "frames": frames,
                       "face_frames": n_frames, "faces": n_faces,
                       "interpolated": n_interp, "coasted": n_coast,
                       "refused": int(getattr(mgr, "_interp_refused", 0)),
                       "refused_cut": int(getattr(mgr, "_interp_refused_cut", 0)),
                       "seconds": elapsed}, fh, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
