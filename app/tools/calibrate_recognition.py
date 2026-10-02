"""Calibrate recognisers against each other on real footage: same-person vs different-person distance.

Generalises tools/calibrate_identity.py (w600k vs AdaFace only) to any registered recognition model
and fixes what makes its numbers optimistic:

  * SAME person is a TRACK, not two adjacent frames. Faces in consecutive samples are linked only when
    they are mutually each other's best box (IoU >= 0.5) with no close second candidate, and a scene cut
    (thumbnail change) breaks every link. Every pair INSIDE a track counts, so pairs eight seconds apart
    (pose, light and expression moved) are in the distribution, which is what a tracker has to survive.
  * DIFFERENT people are two faces in the same frame whose boxes are not duplicates (IoU > 0.35, the
    pipeline's own duplicate rule; touching heads sit near 0.2 and stay in, they are the hard case). Cross-clip pairs are NOT used: nothing proves two
    clips hold different people.
  * Faces under --min-face-px are left out of both sides.

The detector is initialised the way the user's config runs it (tests/angle_bench.init_pipeline with
sync_config), so the faces, landmarks and w600k vectors are the pipeline's own. Every other model embeds
the SAME aligned crop (repo align_crop, arcface_112_v2) through RecognitionInferenceEngine.

The cross-model question is not "whose gap is wider in its own units" (a distance scale is arbitrary) but
rank-based: AUC, the equal-error rate, and the false-accept rate at the false-reject rate the model in
production runs at today.

Usage (from app/, in the venv, with the GPU otherwise idle):
    python tools/calibrate_recognition.py --models default,adaface,glintr100 --models-dir <dir> \
        --clips ../../roop-keep/double ../../roop-keep/single --out result.json
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "tests"))

CUT_THUMB = (64, 36)
CUT_DIFF = 25.0           # mean abs gray difference (0-255) between consecutive samples that means "a new shot"
LINK_IOU = 0.5
SECOND_IOU = 0.2          # a rival candidate above this makes a link ambiguous, so it is not made
OVERLAP_IOU = 0.35        # the pipeline's own duplicate-detection rule. Closer-standing PEOPLE (two heads touching)
                          # sit around IoU 0.2 and are the hard different-person case, so they must stay in.


# --------------------------------------------------------------------------- pure analysis

def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_faces(boxes_a: Sequence[Sequence[float]], boxes_b: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    """Unambiguous identity links between two consecutive samples: mutual best box, IoU >= LINK_IOU, no rival."""
    if not boxes_a or not boxes_b:
        return []
    m = np.array([[iou(a, b) for b in boxes_b] for a in boxes_a])
    links = []
    for i in range(m.shape[0]):
        j = int(m[i].argmax())
        if int(m[:, j].argmax()) != i or m[i, j] < LINK_IOU:
            continue
        rival_row = np.delete(m[i], j).max(initial=0.0)
        rival_col = np.delete(m[:, j], i).max(initial=0.0)
        if rival_row > SECOND_IOU or rival_col > SECOND_IOU:
            continue
        links.append((i, j))
    return links


def build_tracks(samples: List[Dict[str, Any]]) -> List[List[Tuple[int, int]]]:
    """samples: [{'boxes': [...], 'cut_before': bool, 'new_window': bool}] in order.
    Returns tracks as lists of (sample_index, face_index)."""
    tracks: List[List[Tuple[int, int]]] = []
    open_track: Dict[int, List[Tuple[int, int]]] = {}          # face index in the previous sample -> its track
    for s, sample in enumerate(samples):
        nxt: Dict[int, List[Tuple[int, int]]] = {}
        if s > 0 and not sample["cut_before"] and not sample["new_window"]:
            for i, j in link_faces(samples[s - 1]["boxes"], sample["boxes"]):
                if i in open_track:
                    open_track[i].append((s, j))
                    nxt[j] = open_track[i]
        for k in range(len(sample["boxes"])):
            if k not in nxt:
                track = [(s, k)]
                tracks.append(track)
                nxt[k] = track
        open_track = nxt
    return tracks


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(1.0 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def rates_at(threshold: float, same: np.ndarray, diff: np.ndarray) -> Tuple[float, float]:
    """(false-reject rate of same pairs, false-accept rate of different pairs) for 'match if distance <= t'."""
    return float((same > threshold).mean()), float((diff <= threshold).mean())


def auc(same: np.ndarray, diff: np.ndarray) -> float:
    """P(a random same-person distance is smaller than a random different-person one)."""
    combined = np.concatenate([same, diff])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(len(combined))
    ranks[order] = np.arange(1, len(combined) + 1)
    # average ranks over ties
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    n_s, n_d = len(same), len(diff)
    rank_sum_same = ranks[:n_s].sum()
    return float(1.0 - (rank_sum_same - n_s * (n_s + 1) / 2) / (n_s * n_d))


def equal_error(same: np.ndarray, diff: np.ndarray) -> Tuple[float, float]:
    """(threshold, rate) where false-reject == false-accept."""
    candidates = np.unique(np.concatenate([same, diff]))
    best_t, best_gap, best_rate = float(candidates[0]), 9.0, 1.0
    for t in candidates:
        frr, far = rates_at(t, same, diff)
        if abs(frr - far) < best_gap:
            best_t, best_gap, best_rate = float(t), abs(frr - far), (frr + far) / 2
    return best_t, best_rate


def threshold_for_frr(target_frr: float, same: np.ndarray) -> float:
    """Smallest threshold whose false-reject rate on the same-person pairs is <= target."""
    return float(np.percentile(same, 100.0 * (1.0 - target_frr)))


def summarise(same: Sequence[float], diff: Sequence[float], current: Optional[float]) -> Dict[str, Any]:
    s, d = np.asarray(same), np.asarray(diff)
    out: Dict[str, Any] = {"n_same": len(s), "n_diff": len(d)}
    if len(s) < 20 or len(d) < 20:
        out["error"] = "too few pairs"
        return out
    s95, d05 = pct(s, 95), pct(d, 5)
    eer_t, eer = equal_error(s, d)
    out.update({
        "same": {"median": pct(s, 50), "p90": pct(s, 90), "p95": s95, "p99": pct(s, 99), "max": float(s.max())},
        "diff": {"min": float(d.min()), "p1": pct(d, 1), "p5": d05, "median": pct(d, 50)},
        "gap_p95_p5": d05 - s95, "midpoint_threshold": (s95 + d05) / 2,
        "auc": auc(s, d), "eer": eer, "eer_threshold": eer_t,
    })
    if current is not None:
        frr, far = rates_at(current, s, d)
        out["at_current_default"] = {"threshold": current, "frr": frr, "far": far}
    return out


def clip_bootstrap(same_by_clip: Dict[str, Dict[str, Sequence[float]]], diff_by_clip: Dict[str, Dict[str, Sequence[float]]],
                   models: Sequence[str], baseline: str, n_boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
    """95% interval of (model - baseline) AUC and EER, resampling whole CLIPS with replacement.

    Pairs inside one track are strongly correlated, so a pair-level interval would be far too narrow; a clip
    is the smallest unit that is independent of the others. `frac_better` is the share of resamples where the
    model beats the baseline (higher AUC, lower EER)."""
    rng = np.random.RandomState(seed)
    clips = sorted(set(c for m in models for c in list(same_by_clip[m]) + list(diff_by_clip[m])))
    draws: Dict[str, Dict[str, List[float]]] = {m: {"auc": [], "eer": []} for m in models}
    for _ in range(n_boot):
        pick = [clips[i] for i in rng.randint(0, len(clips), len(clips))]
        stats = {}
        for m in models:
            s = np.concatenate([np.asarray(same_by_clip[m].get(c, []), np.float64) for c in pick])
            d = np.concatenate([np.asarray(diff_by_clip[m].get(c, []), np.float64) for c in pick])
            if len(s) < 20 or len(d) < 20:
                stats = None
                break
            stats[m] = (auc(s, d), equal_error(s, d)[1])
        if stats is None:
            continue
        for m in models:
            draws[m]["auc"].append(stats[m][0] - stats[baseline][0])
            draws[m]["eer"].append(stats[m][1] - stats[baseline][1])
    out: Dict[str, Any] = {"baseline": baseline, "resamples": len(draws[baseline]["auc"]), "clips": len(clips)}
    for m in models:
        if m == baseline or not draws[m]["auc"]:
            continue
        a, e = np.asarray(draws[m]["auc"]), np.asarray(draws[m]["eer"])
        out[m] = {"delta_auc": float(np.median(a)), "auc_ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                  "delta_eer": float(np.median(e)), "eer_ci": [float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))],
                  "frac_better_auc": float((a > 0).mean()), "frac_better_eer": float((e < 0).mean())}
    return out


# --------------------------------------------------------------------------- footage / embedding

def clip_paths(entries: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for e in entries:
        if os.path.isdir(e):
            paths += sorted(glob.glob(os.path.join(e, "*.mp4")))
        else:
            paths += sorted(glob.glob(e))
    return paths


def window_starts(total: int, windows: int, per_window: int, step: int) -> List[int]:
    span = per_window * step
    if total <= windows * span:
        return [0] if total <= span else list(range(0, total - span + 1, span))[:windows]
    return [int(i * (total - span) / max(1, windows - 1)) for i in range(windows)]


def sample_clip(path: str, windows: int, per_window: int, step: int):
    """Yield (frame, window_index, first_in_window). Sequential reads inside a window, one seek per window."""
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for w, start in enumerate(window_starts(total, windows, per_window, step)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for k in range(per_window):
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, w, k == 0
            for _ in range(step - 1):
                if not cap.grab():
                    break
    cap.release()


def thumb(frame) -> np.ndarray:
    import cv2
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), CUT_THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)


def collect(paths: Sequence[str], models: Sequence[str], models_dir: str, windows: int, per_window: int,
            step: int, min_face_px: float, log) -> Dict[str, Any]:
    import cv2
    from roop.face_util import align_crop, get_all_faces
    from roop.recognition_engine import RecognitionInferenceEngine
    # "<model>@crop" runs that model on the SAME aligned crop every engine-based model gets. "default@crop" is the
    # control: w600k's own vector comes from the detector's alignment, the others from align_crop on the final
    # landmarks, and this says how much of any gap is the alignment path rather than the model.
    engines = {m: RecognitionInferenceEngine(m.split("@")[0], models_dir, "cuda")
               for m in models if m not in ("default", "adaface")}
    ada = None
    if "adaface" in models:
        os.environ.setdefault("ROOP_ADAFACE", "1")
        from roop import recognizer_adaface as ada
        ada.download()
    per_clip: Dict[str, Any] = {}
    t0, n_frames = time.time(), 0
    for path in paths:
        name = os.path.basename(path)
        samples: List[Dict[str, Any]] = []
        prev_thumb = None
        for frame, window, first in sample_clip(path, windows, per_window, step):
            th = thumb(frame)
            cut = prev_thumb is not None and not first and float(np.abs(th - prev_thumb).mean()) > CUT_DIFF
            prev_thumb = th
            faces = [f for f in (get_all_faces(frame) or [])
                     if min(float(f["bbox"][2] - f["bbox"][0]), float(f["bbox"][3] - f["bbox"][1])) >= min_face_px]
            boxes, embs = [], []
            for f in faces:
                crop, _ = align_crop(frame, np.asarray(f["kps"], np.float32), 112, mode="arcface_112_v2")
                e = {}
                for m in models:
                    if m == "default":
                        e[m] = np.asarray(f["embedding"], np.float32)
                    elif m == "adaface":
                        v = ada.face_embedding(f, frame)
                        e[m] = None if v is None else np.asarray(v, np.float32)
                    else:
                        e[m] = engines[m].compute_embedding(crop)[0]
                boxes.append([float(x) for x in f["bbox"]])
                embs.append(e)
            samples.append({"boxes": boxes, "embs": embs, "cut_before": bool(cut), "new_window": bool(first)})
            n_frames += 1
            if n_frames % 100 == 0:
                log("  %d frames sampled, %.1f frames/s" % (n_frames, n_frames / (time.time() - t0)))
        per_clip[name] = samples
        log("%s: %d samples, %d faces" % (name, len(samples), sum(len(s["boxes"]) for s in samples)))
    return per_clip


def distances(per_clip: Dict[str, Any], models: Sequence[str], max_same_per_track: int, seed: int):
    rng = np.random.RandomState(seed)
    same = {m: {} for m in models}
    diff = {m: {} for m in models}
    stats = {}
    for name, samples in per_clip.items():
        tracks = build_tracks(samples)
        n_same_tracks = 0
        for track in tracks:
            if len(track) < 2:
                continue
            n_same_tracks += 1
            pairs = [(a, b) for ai, a in enumerate(track) for b in track[ai + 1:]]
            if len(pairs) > max_same_per_track:
                pairs = [pairs[i] for i in rng.choice(len(pairs), max_same_per_track, replace=False)]
            for (sa, fa), (sb, fb) in pairs:
                for m in models:
                    ea, eb = samples[sa]["embs"][fa][m], samples[sb]["embs"][fb][m]
                    if ea is not None and eb is not None:
                        same[m].setdefault(name, []).append(cosine_distance(ea, eb))
        for sample in samples:
            n = len(sample["boxes"])
            for i in range(n):
                for j in range(i + 1, n):
                    if iou(sample["boxes"][i], sample["boxes"][j]) > OVERLAP_IOU:
                        continue
                    for m in models:
                        ea, eb = sample["embs"][i][m], sample["embs"][j][m]
                        if ea is not None and eb is not None:
                            diff[m].setdefault(name, []).append(cosine_distance(ea, eb))
        stats[name] = {"tracks_with_pairs": n_same_tracks, "samples": len(samples),
                       "faces": sum(len(s["boxes"]) for s in samples)}
    return same, diff, stats


def run(args, log=print) -> Dict[str, Any]:
    models = [m for m in args.models.split(",") if m]
    paths = clip_paths(args.clips)
    if not paths:
        raise SystemExit("no clips found")
    import yaml
    with open(os.path.join(APP, "config.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    import angle_bench
    angle_bench.init_pipeline(cfg.get("provider", "cuda"), cfg.get("swap_model", "hyperswap"), "none", "none", sync_config=True)
    if args.raw_kps:
        # Keep the detector's own 5 keypoints (what buffalo_l's recogniser embeds on) instead of the ones the
        # pipeline later derives from the 68 landmarks. Diagnostic: shows how much of a model gap is alignment.
        import roop.globals as _g
        _g.refine_landmarks = False
    log("detector as configured: provider=%s detector=%s, %d clips, models=%s" % (
        cfg.get("provider"), cfg.get("detector_engine"), len(paths), models))
    per_clip = collect(paths, models, args.models_dir, args.windows, args.per_window, args.step, args.min_face_px, log)
    same, diff, stats = distances(per_clip, models, args.max_same_per_track, args.seed)
    current = {"default": float(cfg.get("max_face_distance", 0.75)), "adaface": 0.5}
    report: Dict[str, Any] = {"args": vars(args), "clips": stats, "models": {}, "per_clip": {}}
    for m in models:
        s = np.concatenate([np.asarray(v) for v in same[m].values()]) if same[m] else np.array([])
        d = np.concatenate([np.asarray(v) for v in diff[m].values()]) if diff[m] else np.array([])
        report["models"][m] = summarise(s, d, current.get(m))
        report["per_clip"][m] = {
            c: {"n_same": len(same[m].get(c, [])), "n_diff": len(diff[m].get(c, [])),
                "same_p95": pct(same[m].get(c, []), 95), "diff_p5": pct(diff[m].get(c, []), 5)}
            for c in stats}
        report.setdefault("_raw", {})[m] = {"same": s.tolist(), "diff": d.tolist()}
        report.setdefault("_raw_by_clip", {})[m] = {"same": same[m], "diff": diff[m]}
    if len(models) > 1 and args.bootstrap > 0:
        baseline = "default@crop" if "default@crop" in models else ("default" if "default" in models else models[0])
        report["bootstrap"] = clip_bootstrap({m: same[m] for m in models}, {m: diff[m] for m in models}, models,
                                             baseline, args.bootstrap, args.seed)
    # What each model does at the false-reject rate production runs at today (w600k at its max_face_distance).
    if "default" in report["models"] and "at_current_default" in report["models"]["default"]:
        target = report["models"]["default"]["at_current_default"]["frr"]
        report["matched_frr"] = {"target_frr": target, "models": {}}
        for m in models:
            s, d = np.asarray(report["_raw"][m]["same"]), np.asarray(report["_raw"][m]["diff"])
            if len(s) >= 20 and len(d) >= 20:
                t = threshold_for_frr(target, s)
                frr, far = rates_at(t, s, d)
                report["matched_frr"]["models"][m] = {"threshold": t, "frr": frr, "far": far}
    return report


def markdown(report: Dict[str, Any]) -> str:
    lines = ["| Model | same p95 | diff p5 | gap | AUC | EER | EER thr | midpoint thr | at current default (FRR / FAR) |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |"]
    for m, r in report["models"].items():
        if "error" in r:
            lines.append("| `%s` | %s |" % (m, r["error"]))
            continue
        cur = r.get("at_current_default")
        lines.append("| `%s` | %.3f | %.3f | %+.3f | %.4f | %.2f%% | %.3f | %.3f | %s |" % (
            m, r["same"]["p95"], r["diff"]["p5"], r["gap_p95_p5"], r["auc"], 100 * r["eer"], r["eer_threshold"],
            r["midpoint_threshold"], "—" if not cur else "%.2f: %.2f%% / %.2f%%" % (cur["threshold"], 100 * cur["frr"], 100 * cur["far"])))
    mf = report.get("matched_frr")
    if mf:
        lines += ["", "At the false-reject rate w600k runs at today (%.2f%%):" % (100 * mf["target_frr"]), "",
                  "| Model | threshold | FRR | FAR |", "| :--- | ---: | ---: | ---: |"]
        for m, r in mf["models"].items():
            lines.append("| `%s` | %.3f | %.2f%% | %.2f%% |" % (m, r["threshold"], 100 * r["frr"], 100 * r["far"]))
    bs = report.get("bootstrap")
    if bs and bs.get("resamples"):
        lines += ["", "Versus `%s`, resampling whole clips (%d clips, %d resamples; 95%% interval):" % (
            bs["baseline"], bs["clips"], bs["resamples"]), "",
            "| Model | dAUC (median, 95% CI) | dEER in pp (median, 95% CI) | resamples where better (AUC / EER) |",
            "| :--- | :--- | :--- | :--- |"]
        for m, r in bs.items():
            if isinstance(r, dict) and "delta_auc" in r:
                lines.append("| `%s` | %+.4f (%+.4f, %+.4f) | %+.2f (%+.2f, %+.2f) | %.0f%% / %.0f%% |" % (
                    m, r["delta_auc"], r["auc_ci"][0], r["auc_ci"][1], 100 * r["delta_eer"], 100 * r["eer_ci"][0],
                    100 * r["eer_ci"][1], 100 * r["frac_better_auc"], 100 * r["frac_better_eer"]))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", default="default,adaface,glintr100")
    p.add_argument("--models-dir", default=os.path.join(APP, "models"))
    p.add_argument("--clips", nargs="+", required=True, help="video files, globs, or folders of .mp4")
    p.add_argument("--windows", type=int, default=6, help="windows per clip (short clips use contiguous windows)")
    p.add_argument("--per-window", type=int, default=40, help="samples per window")
    p.add_argument("--step", type=int, default=6, help="frames between samples inside a window")
    p.add_argument("--min-face-px", type=float, default=48.0)
    p.add_argument("--max-same-per-track", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=1000, help="clip-level bootstrap resamples (0 to skip)")
    p.add_argument("--raw-kps", action="store_true", help="align on the detector's own 5 keypoints (refine_landmarks off)")
    p.add_argument("--out", default="", help="write the full JSON (incl. raw distances) here")
    args = p.parse_args(argv)
    report = run(args)
    print("\n" + markdown(report))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
