"""Phase 11 video-level enhancer matrix.

This extends ``compare_enhancers_video`` rather than creating another render
pipeline. Each arm renders the same locked clip with one manual enhancer or
the Adaptive selector, then measures runtime, observed memory, temporal change,
plate-referenced texture/edge behavior, and source-embedding similarity.

Example:
    env/Scripts/python.exe tests/bench_adaptive_enhancer_video.py \
      --clip G:/pinokio/roop-keep/double/d4.mp4 --source harjot \
      --enhancers "Adaptive,GPEN 256 Pro,GPEN Realistic,UltraMax"

The quality values are screening metrics, not a substitute for retained-output
visual review. ``identity_similarity`` is measured from output detections
against the selected FaceSet embedding; ``detail_ratio`` and ``edge_ratio``
compare the swapped region with the original plate.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compare_enhancers_video as compare  # noqa: E402
from roop.face_util import get_all_faces  # noqa: E402


def _rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2 ** 30
    except Exception:
        return None


def _vram_mb():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def _embedding_similarity(a, b):
    try:
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / den) if den > 1e-6 else None
    except Exception:
        return None


def _face_mask(plate, output):
    """Find the swapped region from output/plate difference, like the existing grader."""
    d = cv2.cvtColor(cv2.absdiff(output, plate), cv2.COLOR_BGR2GRAY)
    mask = (cv2.GaussianBlur(d, (0, 0), 3) > 12).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask.astype(bool)


def measure_video(path, clip, source_embedding, stride=5, limit=0):
    out_cap, plate_cap = cv2.VideoCapture(path), cv2.VideoCapture(clip)
    rows = []
    prev = None
    i = 0
    while True:
        ok_a, out = out_cap.read()
        ok_b, plate = plate_cap.read()
        if not (ok_a and ok_b):
            break
        if limit and i >= limit:
            break
        if i % max(1, stride) == 0:
            if out.shape[:2] != plate.shape[:2]:
                out = cv2.resize(out, (plate.shape[1], plate.shape[0]),
                                 interpolation=cv2.INTER_AREA)
            mask = _face_mask(plate, out)
            if int(mask.sum()) >= 500:
                go = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
                gp = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
                hf_o = go - cv2.GaussianBlur(go, (0, 0), 1.1)
                hf_p = gp - cv2.GaussianBlur(gp, (0, 0), 1.1)
                eo = np.abs(cv2.Laplacian(go, cv2.CV_32F, ksize=3))
                ep = np.abs(cv2.Laplacian(gp, cv2.CV_32F, ksize=3))
                face = mask & (ep < np.percentile(ep[mask], 65))
                structure = mask & (ep > np.percentile(ep[mask], 75))
                row = {
                    "frame": i,
                    "detail_ratio": float(hf_o[face].std() /
                                            max(float(hf_p[face].std()), 1e-6))
                    if int(face.sum()) else None,
                    "edge_ratio": float(eo[structure].mean() /
                                          max(float(ep[structure].mean()), 1e-6))
                    if int(structure.sum()) else None,
                    "temporal_delta": (float(np.abs(out.astype(np.int16) - prev.astype(np.int16)).mean())
                                       if prev is not None else None),
                    "identity_similarity": None,
                }
                try:
                    faces = get_all_faces(out)
                    if faces:
                        detected = max(faces, key=lambda f: float(
                            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
                        row["identity_similarity"] = _embedding_similarity(
                            getattr(detected, "embedding", None), source_embedding)
                except Exception:
                    pass
                rows.append(row)
            prev = out.copy()
        i += 1
    out_cap.release()
    plate_cap.release()

    def mean(key):
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {"frames": i, "sampled": len(rows),
            "detail_ratio": mean("detail_ratio"),
            "edge_ratio": mean("edge_ratio"),
            "temporal_delta": mean("temporal_delta"),
            "identity_similarity": mean("identity_similarity")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--enhancers", default="Adaptive,GPEN 256 Pro,GPEN Realistic,UltraMax")
    ap.add_argument("--adaptive-profile", default="BALANCED",
                    choices=("FAST", "BALANCED", "REALISTIC", "MAX QUALITY"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # The shared loader extracts ordinary FaceSet images through the detector.
    # compare.render initializes that detector before loading its own FaceSet;
    # defer this independent grading load until after the first render so the
    # benchmark does not accidentally grade an empty source archive.
    source_embedding = None
    labels = [x.strip() for x in args.enhancers.split(",") if x.strip()]
    invalid = [x for x in labels if x not in compare.VALID_ENHANCERS]
    if invalid:
        raise SystemExit("unknown enhancer(s): %s" % invalid)
    out_base = args.out or os.path.join(APP, "output", "adaptive_enhancer_video")
    os.makedirs(out_base, exist_ok=True)
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    threads = int(getattr(cfg, "max_threads", 1) or 1)
    swapper = getattr(cfg, "swap_model", "realswap") or "realswap"
    mask_ui = getattr(cfg, "mask_engine", "RealityUX") or "RealityUX"
    mask = {"RealityUX": "mask_realityux", "DFL XSeg": "mask_xseg",
            "None": "None"}.get(mask_ui, mask_ui)
    rows = {}
    for label in labels:
        before_rss, before_vram = _rss_gb(), _vram_mb()
        started = time.perf_counter()
        output, elapsed = compare.render(
            args.clip, args.source, label, os.path.join(out_base, label.replace(" ", "_")),
            swapper, mask, threads,
            adaptive_profile=args.adaptive_profile)
        if source_embedding is None:
            fs = compare.load_library_faceset(args.source)
            if not getattr(fs, "faces", None):
                raise SystemExit("selected FaceSet has no source faces")
            source_embedding = getattr(fs.faces[0], "embedding", None)
        elapsed = max(float(elapsed), time.perf_counter() - started)
        after_rss, after_vram = _rss_gb(), _vram_mb()
        quality = measure_video(output, args.clip, source_embedding,
                               stride=args.stride, limit=args.limit)
        rows[label] = {
            "status": "measured", "elapsed_sec": round(elapsed, 3),
            "fps": round(quality["frames"] / elapsed, 3) if elapsed else None,
            "rss_before_gb": before_rss, "rss_after_gb": after_rss,
            "vram_before_mb": before_vram, "vram_after_mb": after_vram,
            "output": output, **quality,
        }
        print("%-20s %7.3f fps  detail=%s edge=%s temporal=%s identity=%s" % (
            label, rows[label]["fps"], quality["detail_ratio"],
            quality["edge_ratio"], quality["temporal_delta"],
            quality["identity_similarity"]), flush=True)
    result = {"clip": args.clip, "source": args.source,
              "adaptive_profile": args.adaptive_profile, "rows": rows}
    path = os.path.join(out_base, "results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
