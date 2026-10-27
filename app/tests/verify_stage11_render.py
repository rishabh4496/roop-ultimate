"""Prove the Stage 11 hooks EXECUTE inside a real render, on real footage, and measure what they change.

AGENTS.md: "before believing no effect, prove the code path executes ... count the stage, not the wrapper".
The pure functions have unit tests; this checks they are CALLED, by the production frame path, with a real
detector's keypoints and a real swap, and that what comes out is what the maths says.

Four arms through `core.batch_process_with_options` (the regression benchmark's own setup: the user's
config.yaml models, `config_sync`), on the first N frames of a real clip:

    base   every Stage 11 control at its default                       (the null: nothing here may run)
    fid    pupil_gaze_retention 1.0, micro_expression_restore 1.0, inner_mouth_preserve on
    roto   one frame-fixed exclusion zone on the face, and one zone PINNED to the face (nose + chin)
    lock   identity_lock on

Each new function is wrapped by a recorder that counts calls and, per call, whether it changed a pixel and by
how much (measured inside the call, so the GPU's own run-to-run noise cannot leak into it). The arms are also
compared as video: the exclusion zone's interior must sit at the plate (to encoder noise) where the baseline
has the swapped face, and everywhere else the arm must sit at the pixel noise floor.

Run from app/, one render at a time, nothing else on the GPU:
    env/Scripts/python.exe tests/verify_stage11_render.py [--clip PATH] [--frames 30] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)
os.chdir(APP)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

NOISE_FLOOR_MEAN = 0.7142        # /255, two renders of one unchanged config (AGENTS.md)


class Recorder:
    """Counts calls of one wrapped function and, per call, whether (and how much) it changed its input."""

    def __init__(self, name):
        self.name, self.calls, self.changed, self.max_delta, self.noops = name, 0, 0, 0, 0
        self.extra = {}

    def summary(self):
        return {"calls": self.calls, "changed_pixels_in_call": self.changed, "max_abs_change": self.max_delta,
                **self.extra}


def wrap_inplace(module, name, rec, result_index=0):
    """Wrap a function that edits its first argument in place and returns it."""
    orig = getattr(module, name)

    def wrapper(*a, **k):
        before = a[result_index].copy() if isinstance(a[result_index], np.ndarray) else None
        out = orig(*a, **k)
        rec.calls += 1
        if before is not None and isinstance(out, np.ndarray) and out.shape == before.shape:
            d = np.abs(out.astype(np.int16) - before.astype(np.int16))
            n = int((d.max(axis=-1) > 0).sum())
            if n:
                rec.changed += 1
                rec.max_delta = max(rec.max_delta, int(d.max()))
            else:
                rec.noops += 1
        return out

    setattr(module, name, wrapper)
    return orig


def decode(path, n):
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def region_diff(a, b, box):
    x0, y0, x1, y1 = box
    k = min(len(a), len(b))
    return [float(np.abs(a[i][y0:y1, x0:x1].astype(np.int16) - b[i][y0:y1, x0:x1].astype(np.int16)).mean())
            for i in range(k)]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None, help="a real clip (default: angle.mp4, Weeds.mp4, Love.mp4 from the media dir, first with a face)")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--start", type=float, default=None, help="seconds into the clip (default: try 0, 5, 15, 30)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None, help="keep the renders and the comparison crop here")
    args = ap.parse_args()

    from roop.benchmark import regression as rb
    log = lambda m: print(m, flush=True)  # noqa: E731
    work = args.out or tempfile.mkdtemp(prefix="stage11_")
    os.makedirs(work, exist_ok=True)

    media = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(APP))), "roop-keep")
    candidates = [args.clip] if args.clip else [os.path.join(media, n) for n in ("angle.mp4", "Weeds.mp4", "Love.mp4")]
    candidates = [c for c in candidates if c and os.path.isfile(c)]
    if not candidates:
        raise SystemExit("no clip found; pass --clip")
    setup = rb.prepare_pipeline(args.threads, rb.default_source(), log)
    g = setup.g
    # The first clip (at the first offset) in which the detector finds a face within the trimmed window.
    src = clip = target = tframe = None
    for cand in candidates:
        for start_s in ((args.start,) if args.start is not None else (0.0, 5.0, 15.0, 30.0)):
            trial = os.path.join(work, "clip.mp4")
            subprocess.run([rb._ffmpeg(), "-y", "-v", "error", "-ss", str(start_s), "-i", cand, "-frames:v",
                            str(args.frames), "-an", "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
                            "-pix_fmt", "yuv420p", trial], check=True)
            try:
                target, tframe = rb.capture_target(trial, window=args.frames)
            except RuntimeError:
                log("[verify] no face in %s @ %.0fs" % (os.path.basename(cand), start_s))
                continue
            src, clip = cand, trial
            break
        if clip:
            break
    if not clip:
        raise SystemExit("no face found in any candidate clip")
    plate = decode(clip, args.frames)
    H, W = plate[0].shape[:2]
    log("[verify] clip %s: %d frames %dx%d; target face bbox %s" % (os.path.basename(src or clip), len(plate), W, H,
                                                                   [int(v) for v in target.bbox]))

    from roop import expression_fidelity as ef, identity_lock as il, roto_polygons as rp
    rec = {
        "retain_iris_pupil": Recorder("retain_iris_pupil"),
        "restore_micro_expression": Recorder("restore_micro_expression"),
        "apply_exclusions": Recorder("apply_exclusions"),
    }
    wrap_inplace(ef, "retain_iris_pupil", rec["retain_iris_pupil"])
    wrap_inplace(ef, "restore_micro_expression", rec["restore_micro_expression"])
    wrap_inplace(rp, "apply_exclusions", rec["apply_exclusions"])
    placed = Recorder("placed_outline")
    orig_placed = rp.placed_outline

    def placed_wrapper(poly, faces=None):
        out = orig_placed(poly, faces)
        placed.calls += 1
        if poly.get("pin") is not None:
            placed.extra["pinned_calls"] = placed.extra.get("pinned_calls", 0) + 1
            if out[1]:
                placed.extra["pinned_and_tracked"] = placed.extra.get("pinned_and_tracked", 0) + 1
        return out

    rp.placed_outline = placed_wrapper
    lock_rec = Recorder("TrackLock.update")
    orig_update = il.TrackLock.update

    def update_wrapper(self, tid, frame, bbox, instant_veto, excess=0.0):
        v = orig_update(self, tid, frame, bbox, instant_veto, excess)
        lock_rec.calls += 1
        lock_rec.extra["vetoes_seen"] = lock_rec.extra.get("vetoes_seen", 0) + int(bool(instant_veto))
        lock_rec.extra["held"] = lock_rec.extra.get("held", 0) + int(bool(v.hold))
        lock_rec.extra["phases"] = sorted(set(lock_rec.extra.get("phases", [])) | {v.phase})
        return v

    il.TrackLock.update = update_wrapper

    # ── the face, in the clip's own pixels: what the zones are drawn on ──
    bx0, by0, bx1, by1 = [int(v) for v in target.bbox]
    bw, bh = bx1 - bx0, by1 - by0
    fixed = [[bx0 + 0.10 * bw, by0 + 0.55 * bh], [bx0 + 0.34 * bw, by0 + 0.55 * bh],
             [bx0 + 0.34 * bw, by0 + 0.82 * bh], [bx0 + 0.10 * bw, by0 + 0.82 * bh]]
    anchors = rp.face_anchors(target)
    pinned = [[bx0 + 0.66 * bw, by0 + 0.55 * bh], [bx0 + 0.90 * bw, by0 + 0.55 * bh],
              [bx0 + 0.90 * bw, by0 + 0.82 * bh], [bx0 + 0.66 * bw, by0 + 0.82 * bh]]
    zones = [{"id": "fixed", "mode": "exclude", "feather": 0, "points": fixed}]
    if anchors is not None:
        zones.append({"id": "pinned", "mode": "exclude", "feather": 0, "points": pinned,
                      "pin": {"ref_nose": anchors[0].tolist(), "ref_chin": anchors[1].tolist()}})
    zone_box = (int(min(p[0] for p in fixed)) + 2, int(min(p[1] for p in fixed)) + 2,
                int(max(p[0] for p in fixed)) - 2, int(max(p[1] for p in fixed)) - 2)
    face_box = (max(0, bx0), max(0, by0), min(W, bx1), min(H, by1))

    def reset():
        g.pupil_gaze_retention = 0.0
        g.micro_expression_restore = 0.0
        g.inner_mouth_preserve = False
        g.identity_lock = False
        g.identity_pins = []
        g.roto_polygons = []
        g.roto_preview_frame = None

    arms = {}

    def run(arm, **settings):
        reset()
        for k, v in settings.items():
            setattr(g, k, v)
        for r in list(rec.values()) + [placed, lock_rec]:
            r.calls = r.changed = r.max_delta = r.noops = 0
            r.extra = {}
        out, swaplog = rb.render(setup, clip, target, os.path.join(work, arm))
        swapped = sum(1 for v in swaplog.values() if v)
        arms[arm] = {"video": out, "swapped_frames": swapped,
                     "recorders": {k: r.summary() for k, r in rec.items()},
                     "placed_outline": placed.summary(), "lock": lock_rec.summary()}
        log("[verify] %-5s swapped %d/%d  %s" % (arm, swapped, len(plate),
                                                  json.dumps({k: v for k, v in arms[arm]["recorders"].items() if v["calls"]})))
        return out

    run("base")                                    # also pays the engine build; the null below is base vs base2
    run("base2")
    run("fid", pupil_gaze_retention=1.0, micro_expression_restore=1.0, inner_mouth_preserve=True)
    run("roto", roto_polygons=rp.sanitize(zones))
    run("lock", identity_lock=True)

    vids = {k: decode(v["video"], len(plate)) for k, v in arms.items() if v["video"]}
    report = {"frames": len(plate), "arms": {k: {kk: vv for kk, vv in v.items() if kk != "video"} for k, v in arms.items()}}

    def frame_mean(a, b, box):
        d = region_diff(vids[a], vids[b], box)
        return float(np.mean(d)) if d else float("nan")

    null_face = frame_mean("base", "base2", face_box)
    report["null_face_region_mean"] = null_face
    for arm in ("fid", "roto", "lock"):
        report[f"{arm}_vs_base_face_region_mean"] = frame_mean("base", arm, face_box)

    # exclusion zone: where the baseline put the swapped face, the plate must come back
    base_vs_plate = region_diff(vids["base"], plate, zone_box)
    roto_vs_plate = region_diff(vids["roto"], plate, zone_box)
    touched = [i for i, d in enumerate(base_vs_plate) if d > 6.0]       # frames where the swap reached the zone
    report["zone"] = {
        "frames_where_the_swap_reaches_the_zone": len(touched),
        "baseline_vs_plate_mean": float(np.mean([base_vs_plate[i] for i in touched])) if touched else None,
        "roto_vs_plate_mean": float(np.mean([roto_vs_plate[i] for i in touched])) if touched else None,
    }
    log("[verify] " + json.dumps(report, indent=2, default=float))
    json.dump(report, open(os.path.join(work, "report.json"), "w"), indent=2, default=float)

    # a picture to LOOK at: face crop, plate | base | fid | roto
    i = min(len(plate) - 1, max(0, len(plate) // 2))
    pad = int(0.35 * max(bw, bh))
    cx0, cy0, cx1, cy1 = max(0, bx0 - pad), max(0, by0 - pad), min(W, bx1 + pad), min(H, by1 + pad)
    tiles = [plate[i], vids["base"][i], vids["fid"][i], vids["roto"][i]]
    row = np.hstack([t[cy0:cy1, cx0:cx1] for t in tiles])
    cv2.imwrite(os.path.join(work, "compare_plate_base_fid_roto.png"), row)
    log("[verify] wrote %s" % os.path.join(work, "compare_plate_base_fid_roto.png"))

    # ── verdicts ────────────────────────────────────────────────────────────
    failures = []
    if arms["base"]["swapped_frames"] == 0:
        failures.append("the baseline swapped nothing: the run proves nothing")
    for name in rec:
        if arms["base"]["recorders"][name]["calls"] or arms["base2"]["recorders"][name]["calls"]:
            failures.append(f"{name} ran with every control at its default (must be a no-op)")
    for name in ("retain_iris_pupil", "restore_micro_expression"):
        s = arms["fid"]["recorders"][name]
        if not s["calls"]:
            failures.append(f"{name} never ran in the real render")
        elif not s["changed_pixels_in_call"]:
            failures.append(f"{name} ran {s['calls']}x and changed nothing")
    if not arms["roto"]["recorders"]["apply_exclusions"]["calls"]:
        failures.append("apply_exclusions never ran in the real render")
    z = report["zone"]
    if z["baseline_vs_plate_mean"] is None:
        failures.append("the swap never reached the exclusion zone: choose another zone")
    elif not (z["roto_vs_plate_mean"] < 0.35 * z["baseline_vs_plate_mean"] and z["roto_vs_plate_mean"] < 5.0):
        failures.append("the exclusion zone did not restore the plate: roto %.2f vs baseline %.2f"
                        % (z["roto_vs_plate_mean"], z["baseline_vs_plate_mean"]))
    if anchors is not None and not arms["roto"]["placed_outline"].get("pinned_and_tracked"):
        failures.append("the pinned zone was never tracked to a face")
    if not arms["lock"]["lock"]["calls"]:
        failures.append("the identity lock was on and TrackLock.update never ran")
    if abs(report["lock_vs_base_face_region_mean"] - null_face) > max(NOISE_FLOOR_MEAN, 2 * null_face):
        failures.append("the lock changed the picture on a clip with no crossing (face-region mean %.3f vs null %.3f)"
                        % (report["lock_vs_base_face_region_mean"], null_face))
    for f in failures:
        log("[verify] FAIL: " + f)
    log("[verify] %s" % ("ALL CHECKS PASSED" if not failures else "%d CHECK(S) FAILED" % len(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
