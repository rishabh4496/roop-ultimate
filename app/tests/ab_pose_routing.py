"""A/B: does pose-adaptive source routing change the swapped identity, by yaw band?

Arms OFF / ON / OFF / ON on one clip with one multi-angle source faceset, real
renders through the regression benchmark's setup (config.yaml via config_sync).
Per output frame: the target face's yaw is solved on the ORIGINAL frame
(solve_pose_5pt); the swapped face is re-detected in the output at that spot
and embedded with the app's recogniser; its cosine to the source identity is
recorded against two references:
  * fused  - the portfolio's fused frontal/quarter vector (what ON conditions
             frontal frames on);
  * mean   - the unit mean of all the source's own face vectors (closer to what
             OFF conditions on for a V1 faceset: FaceSet.AverageEmbeddings).
Caveat, stated: the recogniser that scores identity is the one whose vectors
condition the swap, so a small shift toward a reference can be the metric
reading its own input. Read the band split, not one number.

Run from app/ (one render at a time, nothing else on the GPU):
    env/Scripts/python.exe tests/ab_pose_routing.py --clip ../../../roop-keep/angle.mp4 \
        --faceset facesets/anshita.fsz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)
os.chdir(APP)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def unit(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def band(yaw):
    a = abs(yaw)
    return "frontal<=35" if a <= 35 else ("35-60" if a <= 60 else ">60")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--faceset", required=True)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from roop.benchmark import regression as rb
    from roop.face_util import get_all_faces, solve_pose_5pt
    from roop.source_portfolio import build_source_portfolio
    import roop.globals as g
    import source_gallery

    log = lambda m: print(m, flush=True)  # noqa: E731
    clip = os.path.abspath(args.clip)
    setup = rb.prepare_pipeline(args.threads, rb.default_source(), log)
    g.INPUT_FACESETS = []
    source_gallery._ingest_faceset(os.path.abspath(args.faceset))
    setup.faceset = g.INPUT_FACESETS[-1]
    pf = build_source_portfolio(setup.faceset)
    if pf is None:
        raise SystemExit("no portfolio from that faceset")
    print("[ab] portfolio:", json.dumps(pf.summary()), flush=True)
    own = [setup.faceset.embeddings_backup if i == 0 and setup.faceset.embeddings_backup is not None
           else f.embedding for i, f in enumerate(setup.faceset.faces)]
    ref_mean = unit(np.mean([unit(v) for v in own], axis=0))
    ref_fused = unit(pf.fused)

    target, _ = rb.capture_target(clip)

    # Target geometry per frame, from the ORIGINAL clip: the face nearest the
    # captured one (largest IoU with the previous frame's pick).
    orig = []
    cap = cv2.VideoCapture(clip)
    prev = [float(v) for v in target.bbox]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = get_all_faces(frame) or []
        pick = max(faces, key=lambda f: iou(prev, [float(v) for v in f.bbox]), default=None)
        if pick is not None and iou(prev, [float(v) for v in pick.bbox]) > 0.05:
            pose = solve_pose_5pt(pick.kps)
            box = [float(v) for v in pick.bbox]
            orig.append((box, None if pose is None else float(pose[0])))
            prev = box
        else:
            orig.append((None, None))
    cap.release()
    print(f"[ab] target posed on {sum(1 for b, y in orig if y is not None)}/{len(orig)} frames", flush=True)

    work = tempfile.mkdtemp(prefix="ab_routing_")
    results = {}
    for arm in ("OFF1", "ON1", "OFF2", "ON2"):
        on = arm.startswith("ON")
        setup.faceset.angle_portfolio = pf if on else None
        out, _log = rb.render(setup, clip, target, os.path.join(work, arm))
        from roop import core
        routes = core.process_mgr.angle_route_summary() if getattr(core, "process_mgr", None) else {}
        rows = []
        cap = cv2.VideoCapture(out)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or idx >= len(orig):
                break
            box, yaw = orig[idx]
            idx += 1
            if box is None or yaw is None:
                continue
            faces = get_all_faces(frame) or []
            hit = max(faces, key=lambda f: iou(box, [float(v) for v in f.bbox]), default=None)
            if hit is None or iou(box, [float(v) for v in hit.bbox]) < 0.3 or getattr(hit, "embedding", None) is None:
                rows.append((yaw, None, None))
                continue
            e = unit(hit.embedding)
            rows.append((yaw, float(e @ ref_fused), float(e @ ref_mean)))
        cap.release()
        results[arm] = {"rows": rows, "routes": routes}
        print(f"[ab] {arm}: frames scored {sum(1 for r in rows if r[1] is not None)}/{len(rows)} routes={routes}",
              flush=True)
    setup.faceset.angle_portfolio = None

    table = {}
    for arm, res in results.items():
        for yaw, s_f, s_m in res["rows"]:
            key = (arm[:-1], band(yaw))
            t = table.setdefault(key, {"n": 0, "lost": 0, "fused": [], "mean": []})
            t["n"] += 1
            if s_f is None:
                t["lost"] += 1
            else:
                t["fused"].append(s_f)
                t["mean"].append(s_m)
    print("\n[ab] identity to source (cosine), both repeats pooled")
    print(f"{'band':<12} {'arm':<4} {'n':>5} {'lost':>5} {'vs fused':>9} {'vs mean':>9}")
    summary = {}
    for b in ("frontal<=35", "35-60", ">60"):
        for arm in ("OFF", "ON"):
            t = table.get((arm, b))
            if not t:
                continue
            f = float(np.mean(t["fused"])) if t["fused"] else float("nan")
            m = float(np.mean(t["mean"])) if t["mean"] else float("nan")
            summary[f"{b}/{arm}"] = {"n": t["n"], "lost": t["lost"], "fused": f, "mean": m}
            print(f"{b:<12} {arm:<4} {t['n']:>5} {t['lost']:>5} {f:>9.4f} {m:>9.4f}")
    # repeat-to-repeat spread = the noise to judge ON-OFF against
    for arm in ("OFF", "ON"):
        a = [r[1] for r in results[arm + "1"]["rows"] if r[1] is not None]
        b = [r[1] for r in results[arm + "2"]["rows"] if r[1] is not None]
        if a and b:
            print(f"[ab] {arm} repeat spread (mean vs fused): {abs(np.mean(a) - np.mean(b)):.4f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "routes": {k: v["routes"] for k, v in results.items()},
                       "portfolio": pf.summary()}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
