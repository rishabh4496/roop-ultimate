"""Side-by-side comparison of every registered recognition model on one video, with a face swap.

    python tools/compare_recognition_swap.py collect --video V --work W     # detect + embed with all six models, timed
    python tools/compare_recognition_swap.py analyze --video V --work W     # subject, per-model decisions, metrics
    python tools/compare_recognition_swap.py swap    --video V --work W --source facesets/x.fsz   # swapped face crops
    python tools/compare_recognition_swap.py render  --video V --work W --out grid.mp4            # 3x2 grid + summary card

WHAT IT MEASURES. The six panels show the same video. Each model decides, per detected face, "is this the main
subject?" by cosine distance to a reference built from its OWN embeddings of the subject's exemplar faces; faces it
accepts get the source faceset swapped in, the rest stay untouched. So what differs between panels is purely the
recognition model: who it swapped (hits), who it missed, and which bystanders it swapped by mistake.

HONEST LIMITS, printed on the summary card too:
  * The subject is the person with the most large faces in the clip: long tracks are joined by majority vote of the five
    distinct models, and exemplars are spread over the cluster. Nobody is identified by face; whether two scenes show one
    person is the models' vote, not ground truth.
  * There is no per-face ground truth. The objective proxies are: recall on the subject's own continuous track
    (faces in the subject's tracks, minus the exemplars), double matches (two different faces in one frame cannot both be the
    subject), and label flips inside a track (a track is one person).
  * Decisions are per frame with no temporal smoothing, which is what exposes a recogniser; the real pipeline
    adds tracking on top and would hide some of these differences. Only w600k and AdaFace are wired into the real
    pipeline's matching; the other four are not, and this is an offline emulation of what wiring them would decide.
  * The swap itself is always the swapper's w600k identity (roop/swap_identity.py); recognition only gates it.
  * All six models embed the SAME 112 crop (detector keypoints), including w600k, so the comparison is like for like
    (w600k with its own detector-time alignment is slightly better still: see docs/development/RECOGNIZER_CALIBRATION.md).
  * Speed is per-face recognition cost on TensorRT FP16 with batch 1, plus the shared detection cost; it excludes the
    swap, which dominates a real render (see the regression benchmark).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "tests"))
sys.path.insert(0, os.path.join(APP, "tools"))

MODELS = ["default", "adaface", "glintr100", "antelopev2", "mobilefacenet", "facerecognizersf"]
TITLE = {"default": "w600k_r50 (default)", "adaface": "AdaFace IR-101", "glintr100": "Glint-R100",
         "antelopev2": "antelopev2 (= Glint-R100 file)", "mobilefacenet": "MobileFaceNet",
         "facerecognizersf": "OpenCV SFace"}
# Equal-error cosine-distance thresholds measured on the 16-clip calibration (raw keypoints, identical crops):
# docs/development/RECOGNIZER_CALIBRATION.md. Not tuned on the video under test.
THRESHOLD = {"default": 0.700, "adaface": 0.698, "glintr100": 0.704, "antelopev2": 0.704,
             "mobilefacenet": 0.631, "facerecognizersf": 0.557}
DISTINCT = ["default", "adaface", "glintr100", "mobilefacenet", "facerecognizersf"]   # antelopev2 is glintr100's file
MIN_FACE_PX = 48
CUT_DIFF = 25.0
DUP_IOU = 0.35
GREEN, GRAY, RED = (70, 210, 70), (150, 150, 150), (60, 60, 230)


# --------------------------------------------------------------------------- pure helpers (unit-tested)

def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64).reshape(-1)
    return (v / np.linalg.norm(v)).astype(np.float32)


def rows_by_frame(frames: np.ndarray) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for row, f in enumerate(frames):
        out.setdefault(int(f), []).append(row)
    return out


def double_match_rate(frames: np.ndarray, boxes: np.ndarray, label: np.ndarray) -> Tuple[float, int, int]:
    """(rate, frames with a violation, frames with at least one pair of distinct faces).

    Two faces in one frame whose boxes are not duplicates cannot both be the subject; a frame where a model
    accepts both is a false accept (or a missed duplicate-suppression, which the 0.35 IoU rule excludes)."""
    pairs_frames = bad = 0
    for rows in rows_by_frame(frames).values():
        distinct = [(i, j) for a, i in enumerate(rows) for j in rows[a + 1:] if iou(boxes[i], boxes[j]) <= DUP_IOU]
        if not distinct:
            continue
        pairs_frames += 1
        if any(label[i] and label[j] for i, j in distinct):
            bad += 1
    return (bad / pairs_frames if pairs_frames else 0.0), bad, pairs_frames


def flips_per_1000(tracks: Sequence[Sequence[int]], label: np.ndarray, min_len: int = 10) -> Tuple[float, int, int]:
    """Label changes along tracks of at least `min_len` faces, per 1000 faces in those tracks."""
    flips = faces = 0
    for t in tracks:
        if len(t) < min_len:
            continue
        faces += len(t)
        flips += int(np.sum(label[list(t)][1:] != label[list(t)][:-1]))
    return (1000.0 * flips / faces if faces else 0.0), flips, faces


def track_recall(track: Sequence[int], exclude: Sequence[int], label: np.ndarray) -> Tuple[float, int]:
    rest = [r for r in track if r not in set(exclude)]
    return (float(np.mean(label[rest])) if rest else 0.0), len(rest)


def agreement(label: np.ndarray, consensus: np.ndarray) -> float:
    return float(np.mean(label == consensus))


def roi_box(bbox: Sequence[float], width: int, height: int, margin: float = 0.25) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(width, int(x1 + mx)), min(height, int(y1 + my)))


def feather_mask(h: int, w: int) -> np.ndarray:
    """Soft ellipse covering the face inside a margin-expanded ROI (so the paste has no hard edge)."""
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (max(1, int(w * 0.42)), max(1, int(h * 0.44))), 0, 0, 360, 1.0, -1)
    sigma = max(1.0, 0.07 * min(h, w))
    return cv2.GaussianBlur(m, (0, 0), sigma)[:, :, None]


def paste_roi(base: np.ndarray, roi: np.ndarray, x0: int, y0: int) -> np.ndarray:
    h, w = roi.shape[:2]
    m = feather_mask(h, w)
    region = base[y0:y0 + h, x0:x0 + w].astype(np.float32)
    base[y0:y0 + h, x0:x0 + w] = (roi.astype(np.float32) * m + region * (1.0 - m)).astype(np.uint8)
    return base


def speed_table(det_time: np.ndarray, model_time: Dict[str, np.ndarray], skip: int = 30) -> Dict[str, Dict[str, float]]:
    """Per-model recognition cost and the end-to-end analysis fps (detect + this model), after `skip` warm-up frames."""
    det_per_frame = float(np.mean(det_time[skip:])) if len(det_time) > skip else float(np.mean(det_time))
    out = {}
    for m, t in model_time.items():
        t = np.asarray(t, np.float64)
        faces = len(t)
        t_rec = float(t.sum())
        rec_per_frame = t_rec / max(1, len(det_time))
        out[m] = {"ms_per_face": 1000.0 * t_rec / max(1, faces), "faces_per_s": faces / t_rec if t_rec else 0.0,
                  "p95_ms": 1000.0 * float(np.percentile(t, 95)) if faces else 0.0,
                  "pipeline_fps": 1.0 / (det_per_frame + rec_per_frame) if (det_per_frame + rec_per_frame) else 0.0}
    return out


# --------------------------------------------------------------------------- phase 1: collect

def _init_pipeline(sync_provider: Optional[str] = None):
    import yaml
    with open(os.path.join(APP, "config.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    import angle_bench
    angle_bench.init_pipeline(sync_provider or cfg.get("provider", "cuda"), cfg.get("swap_model", "hyperswap"),
                              "none", "none", sync_config=True)
    return cfg


def _gpu_state() -> str:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,clocks.sm",
                                       "--format=csv,noheader,nounits"], text=True, timeout=10).strip().splitlines()[0]
        return out
    except Exception:
        return "n/a"


def collect(a) -> int:
    from roop.recognition_engine import RecognitionInferenceEngine
    cfg = _init_pipeline()
    import roop.globals as g
    g.refine_landmarks = False            # detector keypoints are the final ones: every model sees the same ideal crop
    from roop.face_util import align_crop, get_all_faces

    engines = {}
    for m in MODELS:
        e = RecognitionInferenceEngine(m, a.models_dir, "tensorrt")
        if e.degraded or e.active_providers[0] != "TensorrtExecutionProvider":
            raise SystemExit("%s is not on TensorRT (%s): %s" % (m, e.active_providers, e.fallback_log))
        engines[m] = e
    cap = cv2.VideoCapture(a.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(total, a.frames) if a.frames else total
    print("video %s: %d frames (%d used), %.2f fps, GPU at start: %s" % (
        os.path.basename(a.video), total, n_frames, cap.get(cv2.CAP_PROP_FPS), _gpu_state()), flush=True)

    warm = np.random.RandomState(0).randint(0, 256, (112, 112, 3), dtype=np.uint8)
    t_end = time.time() + 3.0                                  # ramp the GPU clocks before anything is timed
    while time.time() < t_end:
        for e in engines.values():
            e.compute_embedding(warm)

    cols: Dict[str, list] = {"frame": [], "bbox": [], "det": [], "kps": []}
    emb = {m: [] for m in MODELS}
    mtime = {m: [] for m in MODELS}
    det_time = np.zeros(n_frames)
    cut = np.zeros(n_frames, bool)
    prev, started, last_report = None, time.time(), time.time()
    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            n_frames = fi
            break
        th = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        cut[fi] = prev is not None and float(np.abs(th - prev).mean()) > CUT_DIFF
        prev = th
        t0 = time.perf_counter()
        faces = get_all_faces(frame) or []
        det_time[fi] = time.perf_counter() - t0
        for f in faces:
            b = np.asarray(f["bbox"], np.float32)
            if min(b[2] - b[0], b[3] - b[1]) < MIN_FACE_PX:
                continue
            kps = np.asarray(f["kps"], np.float32)
            crop, _ = align_crop(frame, kps, 112, mode="arcface_112_v2")
            cols["frame"].append(fi); cols["bbox"].append(b); cols["kps"].append(kps)
            cols["det"].append(float(getattr(f, "det_score", 0.0) or 0.0))
            for m in MODELS:
                t0 = time.perf_counter()
                v, _ = engines[m].compute_embedding(crop)
                mtime[m].append(time.perf_counter() - t0)
                emb[m].append(v)
        if time.time() - last_report >= 180 or fi == n_frames - 1:
            el = time.time() - started
            print("frame %d/%d | %.1f fps overall | %d faces | elapsed %.0fs, eta %.0fs" % (
                fi + 1, n_frames, (fi + 1) / el, len(cols["frame"]), el, el * (n_frames - fi - 1) / max(1, fi + 1)), flush=True)
            last_report = time.time()
    meta = {"video": a.video, "frames": n_frames, "fps": cap.get(cv2.CAP_PROP_FPS), "width": int(cap.get(3)),
            "height": int(cap.get(4)), "gpu_start": _gpu_state(), "provider": "TensorRT FP16, batch 1",
            "wall_s": time.time() - started, "refine_landmarks": False, "min_face_px": MIN_FACE_PX}
    os.makedirs(a.work, exist_ok=True)
    np.savez_compressed(os.path.join(a.work, "collect.npz"), frame=np.asarray(cols["frame"], np.int32),
                        bbox=np.asarray(cols["bbox"]), kps=np.asarray(cols["kps"]), det=np.asarray(cols["det"]),
                        det_time=det_time[:n_frames], cut=cut[:n_frames], meta=json.dumps(meta),
                        **{"emb_" + m: np.asarray(emb[m], np.float32) for m in MODELS},
                        **{"time_" + m: np.asarray(mtime[m]) for m in MODELS})
    st = speed_table(det_time[:n_frames], {m: np.asarray(mtime[m]) for m in MODELS})
    print("done: %d faces in %d frames, wall %.0fs" % (len(cols["frame"]), n_frames, meta["wall_s"]))
    for m in MODELS:
        print("  %-17s %.2f ms/face  %7.0f faces/s  pipeline %.2f fps" % (m, st[m]["ms_per_face"], st[m]["faces_per_s"], st[m]["pipeline_fps"]))
    return 0


# --------------------------------------------------------------------------- phase 2: analyze

def stitch_tracks(tracks: List[List[int]], frames: np.ndarray, boxes: np.ndarray, cut: np.ndarray,
                  max_gap: int = 5, min_iou: float = 0.5) -> List[List[int]]:
    """Join a track's end to another track's start when it begins within `max_gap` frames, overlaps its last box
    and no scene cut lies between (a detector miss must not split one person's track in two)."""
    tracks = sorted(([list(t) for t in tracks]), key=lambda t: int(frames[t[0]]))
    used = [False] * len(tracks)
    merged: List[List[int]] = []
    starts = [int(frames[t[0]]) for t in tracks]
    for i, t in enumerate(tracks):
        if used[i]:
            continue
        cur = list(t)
        used[i] = True
        while True:
            end_f, end_box = int(frames[cur[-1]]), boxes[cur[-1]]
            best, best_iou = None, min_iou
            for j in range(len(tracks)):
                if used[j] or not (end_f < starts[j] <= end_f + max_gap):
                    continue
                if cut[end_f + 1:starts[j] + 1].any():
                    continue
                v = iou(end_box, boxes[tracks[j][0]])
                if v >= best_iou:
                    best, best_iou = j, v
            if best is None:
                break
            used[best] = True
            cur += tracks[best]
        merged.append(cur)
    return merged


def pick_subject(tracks: Sequence[Sequence[int]], boxes: np.ndarray, det: np.ndarray, min_px: float = 80.0,
                 n_exemplars: int = 16) -> Tuple[int, List[int]]:
    """The longest track (by faces at least `min_px` wide) and evenly spaced confident exemplars from it."""
    def size(r):
        return float(min(boxes[r][2] - boxes[r][0], boxes[r][3] - boxes[r][1]))
    scores = [sum(1 for r in t if size(r) >= min_px) for t in tracks]
    k = int(np.argmax(scores))
    good = [r for r in tracks[k] if size(r) >= min_px and det[r] >= 0.6]
    if len(good) < 4:
        good = [r for r in tracks[k] if size(r) >= min_px] or list(tracks[k])
    idx = np.unique(np.linspace(0, len(good) - 1, min(n_exemplars, len(good))).astype(int))
    return k, [good[i] for i in idx]


def subject_cluster(tracks: Sequence[Sequence[int]], boxes: np.ndarray, det: np.ndarray, emb: Dict[str, np.ndarray],
                    min_track_faces: int = 20, min_px: float = 80.0, max_exemplars: int = 24
                    ) -> Tuple[List[int], List[int]]:
    """The person with the most large faces across the video, found WITHOUT trusting any single model.

    Long tracks are agglomerated by majority vote: two track groups are the same person when at least three of the five
    distinct models put their (mean) embeddings within that model's threshold. The cluster with the most large faces is
    the subject; exemplars are spread over its tracks so the reference is not one scene's lighting and styling.
    Returns (track indices of the cluster, exemplar rows)."""
    def size(r):
        return float(min(boxes[r][2] - boxes[r][0], boxes[r][3] - boxes[r][1]))
    cand = []
    for k, t in enumerate(tracks):
        big = [r for r in t if size(r) >= min_px]
        if len(big) >= min_track_faces:
            cand.append((k, big))
    if not cand:
        raise SystemExit("no track has %d faces of at least %d px; lower --min-track-faces" % (min_track_faces, min_px))
    groups = []                                   # {"tracks": [k], "n": faces, "cent": {model: unit vector}}
    for k, big in cand:
        groups.append({"tracks": [k], "n": len(big),
                       "cent": {m: unit(np.mean([unit(emb[m][r]) for r in big], axis=0)) for m in DISTINCT}})
    while True:
        best, best_score = None, 0.0
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                votes = [1.0 - float(groups[i]["cent"][m] @ groups[j]["cent"][m]) <= THRESHOLD[m] for m in DISTINCT]
                if sum(votes) >= 3:
                    score = sum(THRESHOLD[m] - (1.0 - float(groups[i]["cent"][m] @ groups[j]["cent"][m])) for m in DISTINCT)
                    if best is None or score > best_score:
                        best, best_score = (i, j), score
        if best is None:
            break
        i, j = best
        gi, gj = groups[i], groups[j]
        n = gi["n"] + gj["n"]
        cent = {m: unit((gi["n"] * gi["cent"][m] + gj["n"] * gj["cent"][m]) / n) for m in DISTINCT}
        groups[i] = {"tracks": gi["tracks"] + gj["tracks"], "n": n, "cent": cent}
        del groups[j]
    main = max(groups, key=lambda g: g["n"])
    exemplars: List[int] = []
    for k in sorted(main["tracks"], key=lambda k: -len(tracks[k])):
        big = [r for r in tracks[k] if size(r) >= min_px and det[r] >= 0.6] or [r for r in tracks[k] if size(r) >= min_px]
        pick = np.unique(np.linspace(0, len(big) - 1, min(3, len(big))).astype(int))
        exemplars += [big[i] for i in pick]
        if len(exemplars) >= max_exemplars:
            break
    return sorted(main["tracks"]), sorted(exemplars[:max_exemplars], key=lambda r: r)


def analyze(a) -> int:
    import calibrate_recognition as cr
    d = np.load(os.path.join(a.work, "collect.npz"), allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    frames, boxes, det, cut = d["frame"], d["bbox"], d["det"], d["cut"]
    nfr = meta["frames"]
    per = rows_by_frame(frames)
    samples = [{"boxes": [boxes[r].tolist() for r in per.get(f, [])], "cut_before": bool(cut[f]), "new_window": f == 0}
               for f in range(nfr)]
    raw = cr.build_tracks(samples)
    tracks = [[per[s][k] for s, k in t] for t in raw]
    tracks = stitch_tracks(tracks, frames, boxes, cut)
    emb = {m: d["emb_" + m] for m in MODELS}
    subj_tracks, exemplars = subject_cluster(tracks, boxes, det, emb, a.min_track_faces)
    subject_rows = sorted(r for k in subj_tracks for r in tracks[k])
    subj = max(subj_tracks, key=lambda k: len(tracks[k]))
    ref = {m: unit(np.mean([unit(emb[m][r]) for r in exemplars], axis=0)) for m in MODELS}
    dist = {m: 1.0 - emb[m] @ ref[m] for m in MODELS}
    label = {m: dist[m] <= THRESHOLD[m] for m in MODELS}
    consensus = np.sum([label[m] for m in DISTINCT], axis=0) >= 3
    speed = speed_table(d["det_time"], {m: d["time_" + m] for m in MODELS})

    metrics: Dict[str, Any] = {}
    for m in MODELS:
        dm, dm_bad, dm_frames = double_match_rate(frames, boxes, label[m])
        fl, fl_n, fl_faces = flips_per_1000(tracks, label[m])
        rec, rec_n = track_recall(subject_rows, exemplars, label[m])
        metrics[m] = {"subject_faces": int(label[m].sum()), "subject_pct": float(100 * label[m].mean()),
                      "track_recall_pct": 100 * rec, "track_recall_n": rec_n,
                      "double_match_pct": 100 * dm, "double_match_frames": dm_bad, "double_match_of": dm_frames,
                      "flips_per_1000": fl, "flips": fl_n, "flip_faces": fl_faces,
                      "agree_consensus_pct": 100 * agreement(label[m], consensus), **speed[m]}
    union = np.zeros(len(frames), bool)
    for m in MODELS:
        union |= label[m]
    out = {"meta": meta, "faces": int(len(frames)), "frames_with_faces": int(len(per)), "tracks": len(tracks),
           "tracks_ge10": int(sum(1 for t in tracks if len(t) >= 10)), "subject_track": subj,
           "subject_tracks": [int(k) for k in subj_tracks], "subject_track_len": len(subject_rows),
           "subject_frames": [int(frames[subject_rows].min()), int(frames[subject_rows].max())],
           "exemplar_rows": [int(r) for r in exemplars], "threshold": THRESHOLD, "metrics": metrics,
           "union_faces": int(union.sum()), "consensus_subject_faces": int(consensus.sum())}
    np.savez_compressed(os.path.join(a.work, "analysis.npz"), union=union, consensus=consensus,
                        **{"dist_" + m: dist[m] for m in MODELS}, **{"label_" + m: label[m] for m in MODELS})
    with open(os.path.join(a.work, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump({**out, "tracks_rows": [list(map(int, t)) for t in tracks]}, fh)
    _contact_sheet(a, frames, boxes, tracks, subj_tracks, os.path.join(a.work, "tracks_sheet.png"))
    print("faces %d in %d frames | %d tracks (%d with >=10 faces) | SUBJECT: %d tracks, %d faces, frames %s, %d exemplars | union to swap: %d" % (
        out["faces"], out["frames_with_faces"], out["tracks"], out["tracks_ge10"], len(subj_tracks), len(subject_rows),
        out["subject_frames"], len(exemplars), out["union_faces"]))
    print("%-17s %6s %8s %8s %9s %8s %8s | %7s %7s %6s" % ("model", "subj%", "recall%", "dbl%", "flips/1k", "agree%", "thr", "ms/face", "faces/s", "fps"))
    for m in MODELS:
        x = metrics[m]
        print("%-17s %6.1f %8.1f %8.1f %9.1f %8.1f %8.3f | %7.2f %7.0f %6.1f" % (
            m, x["subject_pct"], x["track_recall_pct"], x["double_match_pct"], x["flips_per_1000"], x["agree_consensus_pct"],
            THRESHOLD[m], x["ms_per_face"], x["faces_per_s"], x["pipeline_fps"]))
    return 0


def _contact_sheet(a, frames, boxes, tracks, subj_tracks, path, n_subject: int = 7, n_other: int = 4, per_track: int = 8) -> None:
    """Exemplar crops: the subject cluster's longest tracks (labelled SUBJECT), then the longest tracks outside it,
    so a person can confirm who the pipeline took to be the main subject."""
    inside = sorted(subj_tracks, key=lambda k: -len(tracks[k]))[:n_subject]
    outside = [k for k in sorted(range(len(tracks)), key=lambda k: -len(tracks[k])) if k not in set(subj_tracks)][:n_other]
    cap = cv2.VideoCapture(a.video)
    rows = []
    for k in inside + outside:
        t = tracks[k]
        picks = [t[i] for i in np.unique(np.linspace(0, len(t) - 1, min(per_track, len(t))).astype(int))]
        tiles = []
        for r in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[r]))
            ok, fr = cap.read()
            if not ok:
                continue
            x0, y0, x1, y1 = [int(v) for v in boxes[r]]
            m = int(0.2 * (x1 - x0))
            crop = fr[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
            tiles.append(cv2.resize(crop, (140, 140)))
        while len(tiles) < per_track:
            tiles.append(np.zeros((140, 140, 3), np.uint8))
        strip = np.hstack(tiles)
        is_subject = k in set(subj_tracks)
        label = "track %d%s: %d faces, frames %d-%d" % (k, "  SUBJECT" if is_subject else "  (not subject)", len(t), frames[t[0]], frames[t[-1]])
        cv2.putText(strip, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if is_subject else (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(strip)
    cv2.imwrite(path, np.vstack(rows))


# --------------------------------------------------------------------------- phase 3: swap

def _roi_path(work: str, fi: int, row: int) -> str:
    return os.path.join(work, "swapped", "%d_%d.png" % (fi, row))


def swap(a) -> int:
    """Swap the source faceset onto every face at least one model accepted, each face INDEPENDENTLY from the original
    frame (so a neighbour's swap can never leak into its crop), through the real ProcessMgr.process_face (swapper,
    mask, enhancer and paste as configured). Saves lossless ROI crops; nothing is re-encoded."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from roop.benchmark import regression as rg
    d = np.load(os.path.join(a.work, "collect.npz"), allow_pickle=False)
    an = np.load(os.path.join(a.work, "analysis.npz"), allow_pickle=False)
    info = json.load(open(os.path.join(a.work, "analysis.json"), encoding="utf-8"))
    frames, boxes, union = d["frame"], d["bbox"], an["union"]
    W, H, nfr = info["meta"]["width"], info["meta"]["height"], info["meta"]["frames"]
    row_ids = np.nonzero(union)[0]
    per_frame: Dict[int, List[int]] = {}
    for r in row_ids:
        per_frame.setdefault(int(frames[r]), []).append(int(r))
    source_image = os.path.splitext(a.source)[0] + ".png"
    setup = rg.prepare_pipeline(a.threads, source_image, print)
    from source_gallery import _ingest_faceset
    import roop.globals as g
    faceset = _ingest_faceset(a.source)
    from roop.face_util import get_all_faces
    cap = cv2.VideoCapture(a.video)
    ex = info["exemplar_rows"][0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[ex]))
    ok, fr = cap.read()
    cand = get_all_faces(fr) or []
    target = max(cand, key=lambda f: iou(f["bbox"], boxes[ex]))
    from roop.ProcessMgr import ProcessMgr
    from roop.target_selection import normalize_target_selection
    g.INPUT_FACESETS, g.TARGET_FACES, g.TARGET_FACE_GROUP = [faceset], [target], [0]
    setup.options.selection_state = normalize_target_selection({"selection_mode": "multi_person", "person_ids": [0]}, person_count=1)
    mgr = ProcessMgr(None)
    mgr.initialize([faceset], [target], setup.options)
    os.makedirs(os.path.join(a.work, "swapped"), exist_ok=True)

    stats = {"swapped": 0, "unmatched": 0, "skipped_existing": 0}
    lock = threading.Lock()

    def work(fi: int, frame: np.ndarray, rows: List[int]) -> None:
        faces = get_all_faces(frame) or []
        for r in rows:
            path = _roi_path(a.work, fi, r)
            if os.path.exists(path):
                with lock:
                    stats["skipped_existing"] += 1
                continue
            best, best_iou = None, 0.6
            for f in faces:
                v = iou(f["bbox"], boxes[r])
                if v > best_iou:
                    best, best_iou = f, v
            if best is None:
                with lock:
                    stats["unmatched"] += 1
                continue
            out = mgr.process_face(0, best, frame.copy(), plate=frame)
            x0, y0, x1, y1 = roi_box(boxes[r], W, H)
            cv2.imwrite(path, out[y0:y1, x0:x1])
            with lock:
                stats["swapped"] += 1

    cap = cv2.VideoCapture(a.video)
    last_frame = max(per_frame) if per_frame else -1
    gate = threading.Semaphore(a.threads * 3)
    started, last_report = time.time(), time.time()
    pending = []
    with ThreadPoolExecutor(a.threads) as pool:
        for fi in range(min(nfr, last_frame + 1)):
            if fi not in per_frame:
                cap.grab()
                continue
            ok, frame = cap.read()
            if not ok:
                break
            gate.acquire()
            fut = pool.submit(work, fi, frame, per_frame[fi])
            fut.add_done_callback(lambda _f: gate.release())
            pending.append(fut)
            if time.time() - last_report >= 180:
                done = stats["swapped"] + stats["skipped_existing"]
                el = time.time() - started
                print("frame %d/%d | %d/%d faces swapped | %.1f faces/s | elapsed %.0fs" % (
                    fi, last_frame, done, len(row_ids), done / el, el), flush=True)
                last_report = time.time()
        for f in pending:
            f.result()
    el = time.time() - started
    print("done: %d swapped, %d unmatched (re-detection found no face at that box), %d reused, %.0fs (%.1f faces/s)" % (
        stats["swapped"], stats["unmatched"], stats["skipped_existing"], el, stats["swapped"] / max(1, el)))
    with open(os.path.join(a.work, "swap_stats.json"), "w") as fh:
        json.dump({**stats, "faces_requested": int(len(row_ids)), "wall_s": el, "threads": a.threads,
                   "source": a.source}, fh)
    mgr.release_resources()
    return 0


# --------------------------------------------------------------------------- phase 4: render

CELL_W, CELL_H, TITLE_H, HEADER_H = 640, 360, 44, 36


def _text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def summary_card(info: Dict[str, Any], width: int, height: int) -> np.ndarray:
    img = np.full((height, width, 3), 24, np.uint8)
    _text(img, "Recognition models on %s: harjot faceset swapped onto the main subject only" % info["meta"]["video"].split("/")[-1],
          (30, 44), 0.8, (255, 255, 255), 2)
    cols = [("model", 30), ("ms/face", 330), ("faces/s", 430), ("pipeline fps", 540), ("subject %", 680), ("track recall %", 790),
            ("double-match %", 940), ("flips/1000", 1100), ("agree %", 1220), ("threshold", 1330)]
    y = 92
    for name, x in cols:
        _text(img, name, (x, y), 0.55, (0, 220, 255), 1)
    for m in MODELS:
        y += 36
        x = info["metrics"][m]
        vals = ["%.2f" % x["ms_per_face"], "%.0f" % x["faces_per_s"], "%.1f" % x["pipeline_fps"], "%.1f" % x["subject_pct"],
                "%.1f" % x["track_recall_pct"], "%.1f" % x["double_match_pct"], "%.1f" % x["flips_per_1000"],
                "%.1f" % x["agree_consensus_pct"], "%.3f" % info["threshold"][m]]
        _text(img, TITLE[m], (30, y), 0.55)
        for (name, cx), v in zip(cols[1:], vals):
            _text(img, v, (cx, y), 0.55)
    notes = [
        "track recall: share of the faces in the subject's tracks (minus the exemplars) that the model accepted (higher is better).",
        "double-match: frames where two different faces were both accepted as the subject (lower is better; they cannot both be).",
        "flips/1000: accept/reject changes inside a track per 1000 faces (lower is better; a track is one person).",
        "agree: share of faces where the model matches the majority of the five distinct models.   pipeline fps = detection + this model.",
        "Subject = the person with the most large faces in the clip: %d tracks (%d faces) joined by majority vote of the five distinct models" % (
            len(info["subject_tracks"]), info["subject_track_len"]),
        "(nobody is identified by face; whether two scenes are one person is the models' vote).  Decisions are per frame, no temporal smoothing.",
        "Thresholds are the equal-error cosine distances from the 16-clip calibration, not tuned on this video. TensorRT FP16, batch 1.",
        "The swap is always the swapper's w600k identity; recognition only chooses WHICH faces get it. antelopev2 is Glint-R100's file.",
    ]
    y += 56
    for n in notes:
        _text(img, n, (30, y), 0.5, (200, 200, 200))
        y += 28
    return img


def render(a) -> int:
    d = np.load(os.path.join(a.work, "collect.npz"), allow_pickle=False)
    an = np.load(os.path.join(a.work, "analysis.npz"), allow_pickle=False)
    info = json.load(open(os.path.join(a.work, "analysis.json"), encoding="utf-8"))
    frames, boxes = d["frame"], d["bbox"]
    label = {m: an["label_" + m] for m in MODELS}
    dist = {m: an["dist_" + m] for m in MODELS}
    subject_rows = set(r for k in info["subject_tracks"] for r in info["tracks_rows"][k])
    meta = info["meta"]
    W, H, fps = meta["width"], meta["height"], meta["fps"]
    nfr = min(meta["frames"], a.frames) if a.frames else meta["frames"]
    per = rows_by_frame(frames)
    sx, sy = CELL_W / W, CELL_H / H
    cum = {m: np.zeros(nfr + 1, np.int32) for m in MODELS}
    for m in MODELS:
        np.add.at(cum[m], np.minimum(frames[label[m]] + 1, nfr), 1)
        cum[m] = np.cumsum(cum[m])
    out_w, out_h = 3 * CELL_W, HEADER_H + 2 * (TITLE_H + CELL_H)
    from roop.ffmpeg_path import ffmpeg_binary
    # Written under a temporary name and moved into place only after ffmpeg exits cleanly: an MP4 has no playable index
    # until the end, so a half-written file at the final path is unplayable AND would replace a good earlier copy.
    part = os.path.splitext(a.out)[0] + ".part.mp4"
    cmd = [ffmpeg_binary(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (out_w, out_h), "-r", "%.4f" % fps,
           "-i", "-", "-i", a.video, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-f", "mp4", part]
    errlog = open(os.path.join(a.work, "ffmpeg.log"), "w")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=errlog, stderr=errlog)
    cap = cv2.VideoCapture(a.video)
    header = np.full((HEADER_H, out_w, 3), 24, np.uint8)
    _text(header, "%s | harjot faceset swapped onto the main subject | the recognition model decides who is swapped | "
          "green = swapped   red = subject-track face MISSED   gray = not swapped" % meta["video"].split("/")[-1],
          (12, 24), 0.55)
    missing_roi = 0
    started, last_report = time.time(), time.time()
    for fi in range(nfr):
        ok, frame = cap.read()
        if not ok:
            break
        rows = per.get(fi, [])
        rois = {}
        for r in rows:
            p = _roi_path(a.work, fi, r)
            if os.path.exists(p):
                rois[r] = cv2.imread(p)
        cells = []
        for m in MODELS:
            img = frame.copy()
            for r in rows:
                if label[m][r]:
                    if r in rois:
                        x0, y0, _, _ = roi_box(boxes[r], W, H)
                        img = paste_roi(img, rois[r], x0, y0)
                    else:
                        missing_roi += 1
            small = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
            for r in rows:
                b = boxes[r]
                p0, p1 = (int(b[0] * sx), int(b[1] * sy)), (int(b[2] * sx), int(b[3] * sy))
                if label[m][r]:
                    color, th = GREEN, 2
                elif r in subject_rows:
                    color, th = RED, 2
                else:
                    color, th = GRAY, 1
                cv2.rectangle(small, p0, p1, color, th)
                _text(small, "%.2f" % dist[m][r], (p0[0], max(12, p0[1] - 4)), 0.4, color)
            x = info["metrics"][m]
            title = np.full((TITLE_H, CELL_W, 3), 40, np.uint8)
            _text(title, "%s   thr %.3f" % (TITLE[m], THRESHOLD[m]), (8, 17), 0.5, (255, 255, 255), 1)
            _text(title, "%.2f ms/face | %.0f faces/s | pipeline %.1f fps | swapped %d" % (
                x["ms_per_face"], x["faces_per_s"], x["pipeline_fps"], cum[m][fi + 1]), (8, 36), 0.45, (0, 220, 255), 1)
            cells.append(np.vstack([title, small]))
        row1, row2 = np.hstack(cells[:3]), np.hstack(cells[3:])
        proc.stdin.write(np.vstack([header, row1, row2]).tobytes())
        if time.time() - last_report >= 180:
            print("render frame %d/%d | %.1f fps" % (fi + 1, nfr, (fi + 1) / (time.time() - started)), flush=True)
            last_report = time.time()
    card = summary_card(info, out_w, out_h).tobytes()
    for _ in range(int(fps * 8)):
        proc.stdin.write(card)
    proc.stdin.close()
    code = proc.wait()
    errlog.close()
    if code == 0 and os.path.exists(part):
        os.replace(part, a.out)
    elif os.path.exists(part):
        print("ffmpeg failed (exit %d); the partial file %s was NOT moved over %s (see ffmpeg.log)" % (code, part, a.out))
    print("render done: %s (ffmpeg exit %d), %.0fs, %d accepted faces had no swapped crop" % (a.out, code, time.time() - started, missing_roi))
    with open(os.path.join(a.work, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({k: info[k] for k in ("meta", "faces", "frames_with_faces", "tracks", "subject_track_len", "subject_frames",
                                        "threshold", "metrics", "union_faces")}, fh, indent=1)
    return code


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("phase", choices=["collect", "analyze", "swap", "render"])
    p.add_argument("--video", required=True)
    p.add_argument("--work", required=True, help="working folder for intermediate data")
    p.add_argument("--models-dir", default=os.path.join(APP, "models"))
    p.add_argument("--source", default=os.path.join(APP, "facesets", "harjot.fsz"), help="faceset (.fsz) to swap in")
    p.add_argument("--threads", type=int, default=4, help="swap workers")
    p.add_argument("--min-track-faces", type=int, default=20, help="analyze: faces (>=80 px) a track needs to join the subject search")
    p.add_argument("--frames", type=int, default=0, help="limit to the first N frames (0 = all)")
    p.add_argument("--out", default="", help="render: output mp4")
    a = p.parse_args(argv)
    return globals()[a.phase](a)


if __name__ == "__main__":
    sys.exit(main())
