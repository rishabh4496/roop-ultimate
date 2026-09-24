"""How much does the swapped face wobble against the head underneath it?

Non-circular by construction: it never reads the pipeline's own landmarks. An
INDEPENDENT detector (insightface SCRFD det_10g, not the render's retinaface)
is run on every frame of the source clip and of the rendered clip, faces are
paired by IoU, and three per-keypoint accelerations are measured, each as a
share of the interocular distance:

    orig  |k[t+1] - 2k[t] + k[t-1]| on the SOURCE    real motion + detector noise
    out   the same on the RENDER
    rel   the same on (out - orig), the offset of the pasted face from the
          head it sits on. Real head motion cancels; what is left is the swap
          moving relative to the head -- the "swimming" boundary -- plus the
          detector's own noise on two different images.

`rel` is the number that matters, and it needs a floor: run the tool with
`--floor`, which compares the source against itself re-encoded with the
render's x264 settings (same pixels, compression noise only). A render whose
`rel` sits at that floor adds no wobble a detector can see.

Everything is bucketed by |yaw| from `solve_pose_5pt` on the source, because
every earlier tracking/interpolation error here concentrated on turned heads.

    env/Scripts/python.exe tests/diag_landmark_jitter.py --run-dir output/bench_two_face/<tag>
    env/Scripts/python.exe tests/diag_landmark_jitter.py --run-dir ... --floor
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

YAW_BUCKETS = ((0.0, 20.0, "frontal <20"), (20.0, 45.0, "turned 20-45"),
               (45.0, 181.0, "profile >45"))


def load_detector():
    from insightface.model_zoo import get_model
    path = os.path.join(APP, "models", "buffalo_l", "det_10g.onnx")
    det = get_model(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    det.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.5)
    return det


def detect(det, frame):
    boxes, kpss = det.detect(frame, max_num=0, metric="default")
    if boxes is None or len(boxes) == 0:
        return []
    return [(b[:4].astype(np.float64), k.astype(np.float64)) for b, k in zip(boxes, kpss)]


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pair(src_faces, out_faces, thresh=0.3):
    """Greedy IoU pairing of source faces to render faces in one frame."""
    pairs, used = [], set()
    for si, (sb, _) in sorted(enumerate(src_faces), key=lambda t: -(t[1][0][2] - t[1][0][0])):
        best, bj = thresh, None
        for oj, (ob, _) in enumerate(out_faces):
            if oj in used:
                continue
            v = iou(sb, ob)
            if v > best:
                best, bj = v, oj
        if bj is not None:
            used.add(bj)
            pairs.append((si, bj))
    return pairs


def detect_stream(det, src_path, out_path, limit):
    """Detect on both clips one frame at a time. Never holds the clips: two
    4K clips of ~490 frames are ~24 GB, which is how grading once killed a
    bench on this machine (memory: bench-oom-not-harness-kill)."""
    a, b = cv2.VideoCapture(src_path), cv2.VideoCapture(out_path)
    src_faces, out_faces = [], []
    try:
        while len(src_faces) < limit:
            ok_a, fa = a.read()
            ok_b, fb = b.read()
            if not (ok_a and ok_b):
                break
            src_faces.append(detect(det, fa))
            out_faces.append(detect(det, fb))
    finally:
        a.release()
        b.release()
    return src_faces, out_faces


def track(per_frame, thresh=0.3):
    """Link source faces across frames by IoU; returns {track_id: {t: idx}}."""
    tracks, last, next_id = {}, {}, 0
    for t, faces in enumerate(per_frame):
        claimed = set()
        new_last = {}
        for i, (box, _) in enumerate(faces):
            best, bid = thresh, None
            for tid, (lt, lbox) in last.items():
                if tid in claimed or t - lt > 1:
                    continue
                v = iou(box, lbox)
                if v > best:
                    best, bid = v, tid
            if bid is None:
                bid, next_id = next_id, next_id + 1
                tracks[bid] = {}
            claimed.add(bid)
            tracks[bid][t] = i
            new_last[bid] = (t, box)
        last = new_last
    return tracks


def second_diff(a, b, c, iod):
    return float(np.mean(np.linalg.norm(c - 2 * b + a, axis=1))) / iod


def analyse(src_path, out_path, limit):
    from roop.face_util import solve_pose_5pt
    det = load_detector()
    src_faces, out_faces = detect_stream(det, src_path, out_path, limit)
    n = len(src_faces)
    matched = [dict(pair(s, o)) for s, o in zip(src_faces, out_faces)]

    rows = []
    for tid, frames in track(src_faces).items():
        for t in sorted(frames):
            if t - 1 not in frames or t + 1 not in frames:
                continue
            idx = [frames[t - 1], frames[t], frames[t + 1]]
            outs = [matched[t + d].get(i) for d, i in zip((-1, 0, 1), idx)]
            if any(o is None for o in outs):
                continue
            s = [src_faces[t + d][i][1] for d, i in zip((-1, 0, 1), idx)]
            o = [out_faces[t + d][j][1] for d, j in zip((-1, 0, 1), outs)]
            iod = float(np.linalg.norm(s[1][0] - s[1][1]))
            if iod < 8:
                continue
            pose = solve_pose_5pt(s[1])
            yaw = abs(float(pose[0])) if pose is not None else 0.0
            rows.append({
                "track": tid, "t": t, "yaw": yaw,
                "box": [round(float(x), 1) for x in src_faces[t][idx[1]][0]],
                "orig": second_diff(*s, iod),
                "out": second_diff(*o, iod),
                "rel": second_diff(*(oo - ss for oo, ss in zip(o, s)), iod),
            })
    return n, rows


def tag_from_run_rows(rows, csv_path):
    """Label each sample from the render's own rows.csv (same frame, best IoU):
    'gap_filled' = the pipeline swapped this face from INTERPOLATED landmarks,
    'occluded' = masked around an object. Unmatched samples get neither."""
    import csv
    by_frame = {}
    with open(csv_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                box = [float(r[k]) for k in ("x0", "y0", "x1", "y1")]
                by_frame.setdefault(int(r["frame"]), []).append((box, r.get("why", "")))
            except (TypeError, ValueError):
                continue
    for row in rows:
        best, why = 0.3, None
        for box, reason in by_frame.get(row["t"], []):
            v = iou(row["box"], box)
            if v > best:
                best, why = v, reason
        row["matched"] = why is not None
        row["gap_filled"] = bool(why and "gap-filled" in why)
        row["occluded"] = bool(why and "behind an object" in why)
    return rows


GROUPS = (("detected", lambda r: r.get("matched") and not r.get("gap_filled")),
          ("gap-filled", lambda r: r.get("gap_filled")),
          ("occluded", lambda r: r.get("occluded")),
          ("clear", lambda r: r.get("matched") and not r.get("occluded")))


def summarise(rows):
    out = {}
    slices = [(name, (lambda r, lo=lo, hi=hi: lo <= r["yaw"] < hi))
              for lo, hi, name in YAW_BUCKETS + ((0.0, 181.0, "all"),)]
    if any("matched" in r for r in rows):
        slices += list(GROUPS)
    for name, keep in slices:
        sel = [r for r in rows if keep(r)]
        if not sel:
            continue
        entry = {"n": len(sel)}
        for key in ("orig", "out", "rel"):
            v = np.array([r[key] for r in sel]) * 100.0
            entry[key] = {"median": round(float(np.median(v)), 3),
                          "p95": round(float(np.percentile(v, 95)), 3)}
        out[name] = entry
    return out


def encode_floor(src_path, dest):
    from roop.ffmpeg_path import ffmpeg_binary
    subprocess.run([ffmpeg_binary(), "-v", "error", "-y", "-i", src_path, "-an",
                    "-c:v", "libx264", "-crf", "12", "-preset", "faster",
                    "-pix_fmt", "yuv420p", dest], check=True)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="a two_face_video.py output dir (holds work/clip.mp4 + the render)")
    ap.add_argument("--floor", action="store_true",
                    help="compare the source with itself re-encoded (noise floor)")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--json", default=None)
    ap.add_argument("--rows", action="store_true",
                    help="split by the render's rows.csv: gap-filled vs detected, occluded vs clear")
    args = ap.parse_args()

    work = os.path.join(args.run_dir, "work")
    src = os.path.join(work, "clip.mp4")
    if args.floor:
        out = encode_floor(src, os.path.join(work, "floor_reencode.mp4"))
    else:
        renders = sorted(p for p in glob.glob(os.path.join(work, "clip_*.mp4")))
        if not renders:
            raise SystemExit(f"no render beside {src}")
        out = renders[-1]
    n, rows = analyse(src, out, args.limit)
    if args.rows and not args.floor:
        tag_from_run_rows(rows, os.path.join(args.run_dir, "rows.csv"))
    result = {"source": src, "render": out, "frames": n, "samples": len(rows),
              "floor": bool(args.floor), "buckets": summarise(rows)}
    print(json.dumps(result, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({**result, "rows": rows}, fh)


if __name__ == "__main__":
    main()
