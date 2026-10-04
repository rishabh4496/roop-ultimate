"""retinaface_r50_gpu vs the baseline detector: is any face lost, and what does it cost?

    env/Scripts/python.exe tests/ab_gpu_engine.py --phase recall
    env/Scripts/python.exe tests/ab_gpu_engine.py --phase speed

PHASE recall -- the acceptance question "no face the old engine found is lost".
For every frame of each baseline window (d1 whole, d4 0..600, d6 whole, Love 1100..1700;
d9.mp4 no longer exists: the roster was retired and d6 replaced it) the pipeline's own
`_detect_faces` (rescue ladder, expected_count=2, exactly the pre-pass call) runs under the
BASELINE engine (the live config's: scrfd) and under `retinaface_r50_gpu`. Each baseline face
must be matched by a new face with IoU >= 0.5. Faces the new engine finds that the baseline
did not are counted and listed too (they are what the identity lock later has to refuse or
accept). Rescue counters per engine are kept: an engine that needs the ladder less is part
of the cost story.

PHASE speed -- DETECTOR-only cost (`_detect_faces_raw(aux=False)`, frames pre-decoded in
memory so decode is not in it) for scrfd / retinaface_r50 / retinaface_r50_gpu, from 1 and
from 2 concurrent callers, ABBA-style engine order, through `angle_bench.init_pipeline` as
AGENTS.md requires. A one-frame upload of a 4K frame is in the GPU engine's number on purpose.
"""
import argparse
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("ROOP_PROFILE", "1")

import fixtures                      # noqa: E402

CLIPS = [("d1", "double/d1.mp4", 0, 418), ("d4", "double/d4.mp4", 0, 600),
         ("d6", "double/d6.mp4", 0, 488), ("Love", "Love.mp4", 1100, 1700)]
NEW = "retinaface_r50_gpu"


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def init():
    from settings import Settings
    import angle_bench as ab
    cfg = Settings(os.path.join(APP, "config.yaml"))
    g = ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)
    g.g_desired_face_analysis = ["landmark_3d_68", "landmark_2d_106", "detection", "recognition"]
    g.lm68_lazy = False
    return g, cfg


def frames(lo, hi, rel):
    cap = cv2.VideoCapture(fixtures.clip(rel, required=True))
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    for idx in range(lo, hi):
        ok, fr = cap.read()
        if not ok:
            break
        yield idx, fr
    cap.release()


def recall(args):
    g, cfg = init()
    from roop import face_util as fu
    from roop import baseline_probe as bp
    baseline_engine = str(cfg.detector_engine)
    # scrfd (the live config) is the acceptance baseline; the EXISTING retinaface_r50 engine is
    # the control: it is the same network, so scrfd->r50 differences are the model's and
    # r50->gpu differences are the port's (letterbox, padding, pyramid, NMS path).
    engines = [baseline_engine, "retinaface_r50", NEW]
    if args.engines:
        engines = [e.replace("BASELINE", baseline_engine) for e in args.engines.split(",")]
    print("[ab] engines=%s expected_count=%d tag=%r" % (engines, args.expected, args.tag), flush=True)
    cache_dir = os.path.join(APP, "output", "gpu_recall_cache")
    os.makedirs(cache_dir, exist_ok=True)

    def key_of(engine):
        return engine + (args.tag if engine == NEW else "")
    store = {}          # (engine, clip) -> {"faces": {idx: [(bbox, score)]}, "seconds": float, "counters": {...}}
    for engine in engines:
        g.detector_engine = engine
        for clip, rel, lo, hi in CLIPS:
            if args.only and clip not in args.only.split(","):
                continue
            hi = min(hi, lo + args.limit) if args.limit else hi
            cache = os.path.join(cache_dir, "%s__%s__%d.json" % (key_of(engine), clip, args.limit))
            if os.path.exists(cache) and not args.fresh:
                with open(cache, encoding="utf-8") as fh:
                    c = json.load(fh)
                c["faces"] = {int(k): v for k, v in c["faces"].items()}
                store[(key_of(engine), clip)] = c
                print("[ab] cached %s %s" % (key_of(engine), clip), flush=True)
                continue
            before = bp.snapshot()
            faces, secs, t_frames = {}, 0.0, 0
            for idx, fr in frames(lo, hi, rel):
                t0 = time.perf_counter()
                out = fu._detect_faces(fr, expected_count=args.expected)
                secs += time.perf_counter() - t0
                t_frames += 1
                faces[idx] = [([float(v) for v in f.bbox], float(f.det_score)) for f in out]
                if t_frames % 100 == 0:
                    print("[ab] %s %s %d frames, %.1f ms/frame" % (engine, clip, t_frames, 1000 * secs / t_frames), flush=True)
            after = bp.snapshot()
            delta = {k: sum(after[k].values()) - sum(before.get(k, {}).values()) for k in after
                     if k.startswith(("rescue.", "gpudet.", "detect.", "raw."))
                     and sum(after[k].values()) != sum(before.get(k, {}).values())}
            store[(key_of(engine), clip)] = {"faces": faces, "seconds": secs, "frames": t_frames, "counters": delta}
            with open(cache, "w", encoding="utf-8") as fh:
                json.dump(store[(key_of(engine), clip)], fh)
            print("[ab] DONE %s %s: %d frames, %d faces, %.1f ms/frame" % (
                engine, clip, t_frames, sum(len(v) for v in faces.values()), 1000 * secs / max(1, t_frames)), flush=True)

    def compare(ea, eb, clip):
        a, b = store.get((ea, clip)), store.get((eb, clip))
        if not (a and b):
            return None
        lost, extra, same_count, base_total, matched = [], [], 0, 0, 0
        ious = []
        for idx, base_faces in a["faces"].items():
            new_faces = b["faces"].get(idx, [])
            same_count += int(len(base_faces) == len(new_faces))
            for bb, sc in base_faces:
                base_total += 1
                best = max((iou(bb, nb) for nb, _ in new_faces), default=0.0)
                if best >= 0.5:
                    matched += 1
                    ious.append(best)
                else:
                    lost.append({"frame": idx, "bbox": [round(v, 1) for v in bb], "score": round(sc, 3),
                                 "size": [round(bb[2] - bb[0]), round(bb[3] - bb[1])], "best_iou": round(best, 3)})
            for nb, sc in new_faces:
                if max((iou(nb, bb) for bb, _ in base_faces), default=0.0) < 0.5:
                    extra.append({"frame": idx, "bbox": [round(v, 1) for v in nb], "score": round(sc, 3),
                                  "size": [round(nb[2] - nb[0]), round(nb[3] - nb[1])]})
        # "lost" splits in two: no overlapping box at all (really not found) vs a box of the
        # same face with a different extent (found, localised differently)
        not_found = [x for x in lost if x["best_iou"] < 0.1]
        return {
            "frames": a["frames"], "from_faces": base_total, "to_faces": sum(len(v) for v in b["faces"].values()),
            "matched_iou_0.5": matched, "LOST": len(lost),
            "lost_not_found_at_all_iou<0.1": len(not_found),
            "lost_found_but_mislocalised_0.1<=iou<0.5": len(lost) - len(not_found),
            "lost_not_found_median_score": round(statistics.median([x["score"] for x in not_found]), 3) if not_found else None,
            "extra_in_to": len(extra), "frames_with_same_face_count": same_count,
            "median_iou_of_matches": round(statistics.median(ious), 3) if ious else None,
            "min_iou_of_matches": round(min(ious), 3) if ious else None,
            "ms_per_frame_from": round(1000 * a["seconds"] / max(1, a["frames"]), 2),
            "ms_per_frame_to": round(1000 * b["seconds"] / max(1, b["frames"]), 2),
            "counters_from": a["counters"], "counters_to": b["counters"],
            "lost_faces": lost[:120], "extra_faces": extra[:120]}

    report = {"baseline_engine": baseline_engine, "new_engine": NEW, "expected_count": args.expected,
              "acceptance_scrfd_to_gpu": {}, "control_scrfd_to_r50": {}, "port_r50_to_gpu": {}}
    for clip, *_ in CLIPS:
        for key, (ea, eb) in (("acceptance_scrfd_to_gpu", (baseline_engine, key_of(NEW))),
                              ("control_scrfd_to_r50", (baseline_engine, "retinaface_r50")),
                              ("port_r50_to_gpu", ("retinaface_r50", key_of(NEW)))):
            r = compare(ea, eb, clip)
            if r:
                report[key][clip] = r
    report["clips"] = report["acceptance_scrfd_to_gpu"]
    for key in ("acceptance_scrfd_to_gpu", "control_scrfd_to_r50", "port_r50_to_gpu"):
        print("==", key)
        for c, r in report[key].items():
            print(c, json.dumps({k: v for k, v in r.items() if k not in ("lost_faces", "extra_faces", "counters_from", "counters_to")}))
    path = os.path.join(REPO, "docs", "perf", "gpu_engine_recall_%s%s.json" % (args.date, args.tag.replace("[", "_").replace("]", "")))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", path)
    return 0


def speed(args):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    g, cfg = init()
    from roop import face_util as fu
    engines = ["scrfd", "retinaface_r50", NEW]
    sets = {}
    for clip, rel, lo, hi in CLIPS:
        if args.only and clip not in args.only.split(","):
            continue
        n = 30 if clip == "d6" else 60
        step = max(1, (hi - lo) // n)
        sets[clip] = [fr for i, (idx, fr) in enumerate(frames(lo, hi, rel)) if i % step == 0][:n]
        print("[ab] speed set %s: %d frames %dx%d" % (clip, len(sets[clip]), sets[clip][0].shape[1], sets[clip][0].shape[0]), flush=True)

    def run(engine, clip, threads):
        g.detector_engine = engine
        work = sets[clip]
        for fr in work[:3]:
            fu._detect_faces_raw(fr, aux=False)                     # warm + build the engine's pool
        def one(fr):
            fu._detect_faces_raw(fr, aux=False)
        t0 = time.perf_counter()
        if threads == 1:
            for fr in work:
                one(fr)
        else:
            with ThreadPoolExecutor(max_workers=threads) as ex:
                list(ex.map(one, work))
        return 1000.0 * (time.perf_counter() - t0) / len(work)

    order = engines + engines[::-1]                                  # A B C C B A
    res = {}
    for threads in [int(t) for t in args.threads.split(',')]:
        for clip in sets:
            ms = {e: [] for e in engines}
            for e in order:
                ms[e].append(run(e, clip, threads))
            res["%s x%d" % (clip, threads)] = {e: {"runs": [round(x, 2) for x in v], "mean_ms": round(statistics.mean(v), 2)}
                                               for e, v in ms.items()}
            print("[ab] %s x%d: %s" % (clip, threads, {e: res["%s x%d" % (clip, threads)][e]["mean_ms"] for e in engines}), flush=True)
    path = os.path.join(REPO, "docs", "perf", "gpu_engine_speed_%s.json" % args.date)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", path)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("recall", "speed"), required=True)
    ap.add_argument("--date", default="2026-10-04")
    ap.add_argument("--expected", type=int, default=2)
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", default="1,2", help="speed phase: concurrent callers")
    ap.add_argument("--engines", default="", help="comma list; BASELINE = the live config's engine")
    ap.add_argument("--tag", default="", help="suffix naming a variant of the NEW engine, e.g. [squash]")
    ap.add_argument("--fresh", action="store_true", help="ignore the per-engine cache")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    return recall(args) if args.phase == "recall" else speed(args)


if __name__ == "__main__":
    sys.exit(main())
