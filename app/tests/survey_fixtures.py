"""What is actually IN each clip, so a phase gets the right fixture.

WHY. Fixtures here have been chosen by name and by memory, and it has gone
wrong twice in ways that invalidated whole results: `duo/d4.mp4` and
`double/d4.mp4` are different clips sharing a filename (854x480 against
1280x720), and a "gradeability" survey found that `d1` and `d6` are 0%
gradeable for identity because the two people overlap in every frame -- after
they had already been used as identity fixtures.

The phases still open on this target each need particular material: occlusion
needs something crossing a face, night needs a dark scene, expression needs a
face doing something, pose needs real yaw. Guessing which clip is which, from
its filename, is how the two mistakes above happened.

So this measures, per clip, on a sparse sample:

    dims/frames/fps     identity of the file, and the duo/double discriminator
    luma p05/p50        NIGHT candidate: how dark, and how much of it is dark
    faces per frame      1 = single-face harnesses, 2+ = two_face_video
    face px              small faces stress detection and enhancer selection
    yaw span             POSE candidate: does the head actually turn
    face-box coverage    OCCLUSION candidate: a face that is repeatedly lost
                         while a large box remains nearby is usually occluded
    detection misses     frames where a face was expected and not found

Detection runs through `angle_bench.init_pipeline`, never a bare process:
without it TensorRT's DLLs are off PATH, ORT falls back to CPU silently, and
`face_util.get_all_faces` swallows detector exceptions and returns an empty
list -- which reads as "this clip has no faces" rather than "the detector was
never started".

    env/Scripts/python.exe tests/survey_fixtures.py
    env/Scripts/python.exe tests/survey_fixtures.py --dirs final "3d model" \
        --samples 40 --json survey.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import fixtures                # noqa: E402
import angle_bench as ab       # noqa: E402


def clip_files(subdir):
    try:
        root = fixtures.clip_dir(subdir)
    except Exception:
        return []
    if not root or not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if name.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
            out.append(os.path.join(root, name))
    return out


def survey_clip(path, samples, pose_fn):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"clip": path, "error": "could not open"}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if n <= 0:
        cap.release()
        return {"clip": path, "error": "no frames"}

    idxs = np.linspace(0, max(n - 2, 0), num=min(samples, n), dtype=int)
    lumas, counts, sizes, yaws, misses = [], [], [], [], 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lumas.append(float(gray.mean()))
        faces = _faces(frame)
        counts.append(len(faces))
        if not faces:
            misses += 1
            continue
        for f in faces:
            x0, y0, x1, y1 = f.bbox
            sizes.append(float(min(x1 - x0, y1 - y0)))
        yaw = pose_fn(max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])))
        if yaw is not None:
            yaws.append(float(yaw))
    cap.release()

    def q(v, p):
        return float(np.percentile(v, p)) if v else None

    return {
        "clip": os.path.relpath(path, os.path.dirname(os.path.dirname(path))),
        "path": path,
        "w": w, "h": h, "frames": n, "fps": round(fps, 2),
        "sampled": len(lumas),
        "luma_p05": q(lumas, 5), "luma_p50": q(lumas, 50),
        "faces_median": (float(np.median(counts)) if counts else 0.0),
        "faces_max": (int(max(counts)) if counts else 0),
        "face_px_p50": q(sizes, 50),
        "yaw_p05": q(yaws, 5), "yaw_p95": q(yaws, 95),
        "yaw_span": (q(yaws, 95) - q(yaws, 5)) if len(yaws) > 4 else None,
        "detect_miss_pct": round(100.0 * misses / max(len(lumas), 1), 1),
    }


def _faces(frame):
    from roop.face_util import get_all_faces
    return get_all_faces(frame) or []


def classify(row):
    """Which open phase this clip is plausible material for.

    Deliberately "candidate", not "fixture": a clip that LOOKS dark or turned on
    a sparse sample still has to be watched before a phase rests on it.
    """
    tags = []
    if row.get("error"):
        return ["unreadable"]
    if (row.get("luma_p50") or 255) < 60:
        tags.append("night")
    elif (row.get("luma_p05") or 255) < 40:
        tags.append("dark-passages")
    if (row.get("faces_median") or 0) >= 2:
        tags.append("two-face")
    elif (row.get("faces_median") or 0) >= 1:
        tags.append("single-face")
    if (row.get("yaw_span") or 0) >= 45:
        tags.append("pose")
    if (row.get("face_px_p50") or 999) < 110:
        tags.append("small-faces")
    # A clip that finds a face on most samples but loses it on a meaningful
    # minority is the shape of something crossing the face. It is also the
    # shape of a face leaving frame, which is why this is a lead, not a label.
    miss = row.get("detect_miss_pct") or 0
    if 5.0 <= miss <= 60.0:
        tags.append("occlusion-lead")
    if miss > 60.0:
        tags.append("mostly-faceless")
    return tags or ["plain"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*",
                    default=["double", "single", "expression", "final",
                             "3d model"])
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    provider = args.provider or str(cfg.provider)

    # Bring roop up properly before any detection. See the module docstring.
    ab.init_pipeline(provider, str(cfg.swap_model), "None",
                     str(cfg.mask_engine), sync_config=True)

    # The project's own pose solver, and nothing else. An ad-hoc
    # "nose offset over interocular distance" proxy was tried here first and
    # published yaw spans of 431 and 666 degrees -- physically impossible, and
    # the sort of number that discredits a whole table. `solve_pose_5pt`
    # separates yaw from pitch by weak perspective; the scalar ratio proxies it
    # replaced are each contaminated by the other angle, which is exactly how a
    # profile head that is also tilted reads as mid-angle.
    from roop.face_util import solve_pose_5pt

    def pose_fn(face):
        try:
            solved = solve_pose_5pt(np.asarray(face.kps, dtype=np.float32))
        except Exception:
            return None
        if solved is None:
            return None
        yaw = float(solved[0])
        # A yaw outside the physically reachable range is a solver failure, not
        # a turned head. Dropped rather than averaged in.
        return yaw if -100.0 <= yaw <= 100.0 else None

    rows = []
    for d in args.dirs:
        files = clip_files(d)
        if not files:
            print("[survey] %-12s no clips found" % d, flush=True)
            continue
        for f in files:
            row = survey_clip(f, args.samples, pose_fn)
            row["dir"] = d
            row["tags"] = classify(row)
            rows.append(row)
            if row.get("error"):
                print("[survey] %-12s %-46s ERROR %s"
                      % (d, os.path.basename(f)[:46], row["error"]), flush=True)
                continue
            print("[survey] %-10s %-40s %5dx%-5d %6df  luma %5.1f/%5.1f  "
                  "faces %3.1f/%d  px %5.0f  yaw %5s  miss %4.1f%%  %s"
                  % (d, os.path.basename(f)[:40], row["w"], row["h"],
                     row["frames"], row["luma_p05"] or -1, row["luma_p50"] or -1,
                     row["faces_median"], row["faces_max"],
                     row["face_px_p50"] or -1,
                     ("%.0f" % row["yaw_span"]) if row["yaw_span"] else "-",
                     row["detect_miss_pct"], ",".join(row["tags"])),
                  flush=True)

    print("\n[survey] candidates by phase")
    for tag, phase in (("night", "Phase 14 lighting/night"),
                       ("pose", "Phase 7 pose / hard angles"),
                       ("occlusion-lead", "Phase 10 foreign-object occlusion"),
                       ("small-faces", "Phase 3 detection stress"),
                       ("two-face", "Phase 11 interacting faces"),
                       ("single-face", "Phase 12 expression / eyes")):
        hits = [os.path.basename(r["path"]) for r in rows if tag in r["tags"]]
        print("[survey]   %-34s %s" % (phase, ", ".join(hits[:6]) or "NONE"))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print("[survey] wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
