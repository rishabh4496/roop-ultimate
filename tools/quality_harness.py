#!/usr/bin/env python
"""Quality harness: how far is a candidate configuration from a full-FP32 reference render, measured on real footage.

NO simulated, constant, random or hard-coded metric lives in this file. Every number in its output is computed from
pixels or embeddings the pipeline actually produced, and the run FAILS if the numbers cannot be told apart.

    env/Scripts/python.exe tools/quality_harness.py run \\
        --clips d1,d4,d6,Love,s7 --frames 300 \\
        --candidate "mixed|swap_model=hyperswap|trt_precision=mixed" \\
        --candidate "inswapper|swap_model=inswapper_128|trt_precision=mixed" \\
        --out output/quality_2026-10-09

What it does
------------
1. REFERENCE. Each clip is rendered for ``--frames`` frames (default 300) at ``trt_precision=fp32`` with
   ``ROOP_SWAP_FP32=1``: every TensorRT engine FP32, the swapper forced FP32. The run VERIFIES it was really FP32 from
   the live session records (no FP16 TensorRT session may exist) and refuses to continue otherwise.
2. CANDIDATES. The same clips, same window, same pinned stabilizer geometry, the candidate's model / precision / env.
   Everything renders through ``tests/two_face_video.py`` (the app's real path, live ``config.yaml``) with a LOSSLESS
   encode (x264 CRF 0), because a rate-controlled encoder turns one sub-pixel difference into a whole-frame difference.
3. METRICS, each against the reference unless stated:
   * identity cosine to the SOURCE with AdaFace (``adaface_ir101.onnx``) - a recogniser the pipeline does not use for
     swapping or matching (w600k); the run refuses if ``ROOP_ADAFACE`` is on, because then it would not be independent.
     Controls: the same face in the untouched PLATE (negative), and the source images against each other (positive).
   * SSIM / PSNR of the composited face region (landmark hull of the plate's 106 landmarks).
   * XSeg mask IoU: the FP32 reference XSeg vs the candidate's precision on identical aligned crops.
   * detection recall (IoU >= 0.5) of the reference's faces, from each render's own per-frame detections; track count.
   * swap-audit counts, parsed from each render's own audit block.
4. DISTINCTNESS GUARD. If a metric that must depend on the swap model is identical across two different models (or two
   model names resolve to the same network files), the run exits non-zero. So a constant, a stub, or one ONNX behind two
   names cannot produce a table. ``--strict-all-metrics`` applies the rule literally to every metric.
5. PER-MODEL LOG: the real provider, ``trt_fp16`` and device memory before / after the FIRST inference of every ONNX
   session in every render (``tests/first_inference_probe.py``).

Metrics that cannot depend on the swap model (detection recall, track count, XSeg IoU, audit counts) are reported and
exempt from the guard by default: they are properties of the detector, tracker and mask net, so two swappers sharing
them are SUPPOSED to match. That is a choice about reading "any metric", stated here rather than hidden.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP = os.path.join(ROOT, "app")
TESTS = os.path.join(APP, "tests")
for _p in (APP, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REFERENCE_NAME = "reference_fp32"
DEFAULT_CLIPS = "d1,d4,d6,Love,s7"
DEFAULT_FRAMES = 300
PSNR_CAP_DB = 100.0                 # identical regions have infinite PSNR; capped so means stay finite (and flagged)
# The metrics that MUST differ between two different swap models. Everything else is reported but exempt (see docstring).
MODEL_DEPENDENT = ("identity_cos", "identity_delta_vs_reference", "ssim", "psnr_db", "eye_drift_px",
                   "mouth_drift_px", "skin_detail", "identity_jitter")
FLAT_KEYS = ("identity_cos", "identity_delta_vs_reference", "ssim", "psnr_db", "eye_drift_px", "mouth_drift_px",
             "skin_detail", "identity_jitter")
YAW_BINS = ((0.0, 20.0), (20.0, 45.0), (45.0, 75.0), (75.0, 181.0))      # degrees of |yaw|
# A swapped face is graded only where the reference swapped it and its crop is not shared with a neighbour
# (the same exclusion tests/two_face_video.py applies, for the same reason).
EXTRA_CLIPS = {"s7": {"rel": "single/s7.mp4", "sources": "harjot", "window": (0, 600),
                      "capture": None, "capture_face": None}}


# ═══════════════════════════ pure metric functions (no GPU, unit-tested) ═══════════════════════════

def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a[:4]]
    bx0, by0, bx1, by1 = [float(v) for v in b[:4]]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def detection_recall(ref_boxes: Sequence[Sequence[float]], cand_boxes: Sequence[Sequence[float]],
                     thr: float = 0.5) -> Tuple[int, int]:
    """(reference faces matched by a candidate box at IoU >= thr, reference faces). One candidate box matches at most
    one reference face (greedy by IoU), so a single big box cannot "recall" two people."""
    pairs = sorted(((box_iou(r, c), i, j) for i, r in enumerate(ref_boxes) for j, c in enumerate(cand_boxes)),
                   reverse=True)
    used_r, used_c, hit = set(), set(), 0
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        hit += 1
    return hit, len(ref_boxes)


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return float("nan")
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _luma(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel SSIM on luma (Wang et al. 2004: 11x11 Gaussian window, sigma 1.5)."""
    x = _luma(a).astype(np.float64)
    y = _luma(b).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda im: cv2.GaussianBlur(im, (11, 11), 1.5)                       # noqa: E731
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    sel = np.asarray(mask) > 0
    if a.shape[:2] != b.shape[:2] or not sel.any():
        return float("nan")
    return float(ssim_map(a, b)[sel].mean())


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    sel = np.asarray(mask) > 0
    if a.shape != b.shape or not sel.any():
        return float("nan")
    d = a.astype(np.float64)[sel] - b.astype(np.float64)[sel]
    mse = float(np.mean(d * d))
    return PSNR_CAP_DB if mse <= 1e-12 else min(PSNR_CAP_DB, 10.0 * math.log10(255.0 ** 2 / mse))


def mask_iou(a: np.ndarray, b: np.ndarray, thr: float = 0.5) -> float:
    ma, mb = np.asarray(a) >= thr, np.asarray(b) >= thr
    union = int((ma | mb).sum())
    # Two EMPTY masks are not "perfect agreement", they are "no face found in either": not measurable, so NaN, which
    # `summarize` drops and the report counts. Returning 1.0 here would hand a free perfect score to a mask net that
    # finds nothing.
    return float((ma & mb).sum() / union) if union else float("nan")


def yaw_bin(yaw: Optional[float]) -> Optional[str]:
    """Label of the |yaw| band a head falls in, or None when the pose is unknown."""
    if yaw is None or (isinstance(yaw, float) and math.isnan(yaw)):
        return None
    a = abs(float(yaw))
    for lo, hi in YAW_BINS:
        if lo <= a < hi:
            return "%d-%d" % (lo, hi) if hi <= 90 else "%d+" % lo
    return None


def masked_laplacian_variance(img: np.ndarray, mask: np.ndarray) -> float:
    """High-frequency detail: variance of the Laplacian of the luma, over the masked pixels only."""
    sel = np.asarray(mask) > 0
    if not sel.any():
        return float("nan")
    lap = cv2.Laplacian(_luma(img), cv2.CV_64F)
    return float(np.var(lap[sel]))


def identity_jitter(frames: Sequence[int], values: Sequence[float]) -> float:
    """Mean |change| of a per-frame score between CONSECUTIVE frames: how much the identity flickers.
    Only adjacent frames count (a gap is not a flicker); NaN if there is no adjacent pair."""
    order = sorted(range(len(frames)), key=lambda k: frames[k])
    diffs = []
    for a, b in zip(order, order[1:]):
        if frames[b] - frames[a] == 1 and not (math.isnan(values[a]) or math.isnan(values[b])):
            diffs.append(abs(values[b] - values[a]))
    return float(np.mean(diffs)) if diffs else float("nan")


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], np.float64)
    if v.size == 0:
        return {"n": 0, "mean": None, "median": None, "p05": None, "min": None}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p05": float(np.percentile(v, 5)), "min": float(v.min())}


_AUDIT_ROW = re.compile(r"^\s+(\S.*?)\s{2,}(\d+)\s+([\d.]+)%\s*$")
_AUDIT_NOFACE = re.compile(r"(\d+) frames of (\d+) \(([\d.]+)% of frames\) had NO face detected")


def parse_swap_audit(text: str) -> Dict[str, Any]:
    """Counts from a render's own SWAP AUDIT block: {label: count} plus the no-face-frame figures."""
    out: Dict[str, Any] = {}
    m = re.search(r"==== SWAP AUDIT[^\n]*\n(.*?)\n={10,}", text, re.S)
    if not m:
        return out
    for line in m.group(1).splitlines():
        r = _AUDIT_ROW.match(line)
        if r:
            out[r.group(1).strip()] = int(r.group(2))
        n = _AUDIT_NOFACE.search(line)
        if n:
            out["frames with no face detected at all"] = int(n.group(1))
            out["frames considered"] = int(n.group(2))
    return out


def capture_fixture(lines: Sequence[str]) -> Tuple[Any, ...]:
    """The DECISION a render's target capture made, without its diagnostics.

    What defines the fixture is how many people were captured and the frame each person was captured from (or the
    manually pinned frame). Two things the capture also prints are NOT part of it: the scan time and floats (separation,
    off-axis degrees), and the SEED frame. The seed is only where the search for a separated pair starts; it moved from
    456 to 452 between an FP32 and an FP16 detector on d6 while both captured the same two people from the same two
    frames, so comparing it raised a false "different fixture". `capture_seed` returns it for the report."""
    people, frames, pinned = None, [], None
    for line in lines or []:
        m = re.search(r"auto-capture: (\d+) people", line)
        if m:
            people = int(m.group(1))
        m = re.search(r"\]\s+person (\d+): frame (\d+)", line)
        if m:
            frames.append((int(m.group(1)), int(m.group(2))))
        m = re.search(r"target faces? captured from frame (\d+)", line) or re.search(r"target: face \d+ .*? of frame (\d+)", line)
        if m:
            pinned = int(m.group(1))
    return (people, tuple(sorted(frames)), pinned)


def capture_seed(lines: Sequence[str]) -> Optional[int]:
    for line in lines or []:
        m = re.search(r"auto-capture: \d+ people, seed frame (\d+)", line)
        if m:
            return int(m.group(1))
    return None


# Markers of a render that did not finish cleanly even though the process returned 0. Calibrated against every healthy
# render log of the first full run: none of them appears in one, all appear in the render whose encoder died.
RENDER_ERROR_PATTERNS = ("write thread failed", "frame reader failed", "processing worker failed",
                         "Traceback (most recent call last)", "processing failed", "RAISED during processing",
                         "DETECTOR FAILED", "encoder process exited")


def render_errors(log_text: str) -> List[str]:
    return [p for p in RENDER_ERROR_PATTERNS if p in (log_text or "")]


def count_video_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def render_problems(log_text: str, output_file: Optional[str], frames: int) -> List[str]:
    """Everything wrong with a finished render, or []. A render that wrote a short or damaged video is not a reference
    or a candidate, whatever its return code said."""
    problems = ["log reports: " + e for e in render_errors(log_text)]
    if not output_file or not os.path.exists(output_file):
        problems.append("no output video")
    else:
        n = count_video_frames(output_file)
        if n != frames:
            problems.append("output has %d frames, expected %d" % (n, frames))
    return problems


def parse_track_count(text: str) -> Optional[int]:
    m = re.search(r"\[Track\] (\d+) tracks over (\d+) frames", text)
    return int(m.group(1)) if m else None


def normalize_input_sig(sig: str) -> str:
    """'a:Nonex3x192x192,b:Nx512' and 'b:unk__7x512,a:?x3x192x192' describe the same inputs: every symbolic dimension
    becomes N and the items are sorted, so a session can be matched across the different ways the app prints shapes."""
    items = []
    for item in [i for i in (sig or "").split(",") if i]:
        name, _, shape = item.rpartition(":")
        toks = [t if t.isdigit() else "N" for t in shape.split("x")]
        items.append("%s:%s" % (name, "x".join(toks)))
    return ",".join(sorted(items))


def parse_sessions(text: str) -> List[Dict[str, str]]:
    rows = []
    pat = r"^\[Session\] (\S+) file=(\S+) provider=(\S+) trt_fp16=(\S+)(?: input=(\S*))?"
    for m in re.finditer(pat, text, re.M):
        rows.append({"tag": m.group(1), "file": m.group(2), "provider": m.group(3), "trt_fp16": m.group(4),
                     "input": normalize_input_sig(m.group(5) or "")})
    return rows


# ═══════════════════════════ candidates ═══════════════════════════

@dataclass
class Variant:
    name: str
    swap_model: str
    trt_precision: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    is_reference: bool = False

    def config_key(self) -> str:
        return json.dumps([self.swap_model, self.trt_precision, sorted(self.env.items())])


def parse_candidate(spec: str) -> Variant:
    """``NAME|swap_model=hyperswap|trt_precision=mixed|env.ROOP_X=1``"""
    parts = [p for p in spec.split("|") if p.strip()]
    if not parts or "=" in parts[0]:
        raise ValueError("candidate spec must start with a name: %r" % spec)
    v = Variant(name=parts[0].strip(), swap_model="")
    for p in parts[1:]:
        k, _, val = p.partition("=")
        k, val = k.strip(), val.strip()
        if k == "swap_model":
            v.swap_model = val
        elif k == "trt_precision":
            v.trt_precision = val.lower()
        elif k.startswith("env."):
            v.env[k[4:]] = val
        else:
            raise ValueError("unknown candidate key %r in %r" % (k, spec))
    if not v.swap_model:
        raise ValueError("candidate %r has no swap_model" % v.name)
    if v.name == REFERENCE_NAME:
        raise ValueError("%r is reserved for the reference" % REFERENCE_NAME)
    return v


def reference_variant(swap_model: str) -> Variant:
    return Variant(name=REFERENCE_NAME, swap_model=swap_model, trt_precision="fp32",
                   env={"ROOP_SWAP_FP32": "1"}, is_reference=True)


# ═══════════════════════════ the distinctness guard (pure) ═══════════════════════════

def _num(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)) else None


def distinctness_violations(variants: Dict[str, Dict[str, Any]], strict_all: bool = False,
                            dependent_prefixes: Sequence[str] = MODEL_DEPENDENT) -> List[str]:
    """Reasons the run must FAIL. ``variants[name]`` = {'swap_model': str, 'model_key': str, 'metrics': {key: float}}.

    ``model_key`` is built from the CONTENT HASH of every network file the render actually loaded (see ``model_key``),
    never from the name, so:
      * two candidates with different names but the same files are ONE network behind two names - a violation by
        itself (this is what a missing ``hyperswap_1b`` silently falling back to ``hyperswap_1a`` looks like);
      * two candidates with different files are different models, and any model-dependent metric that is identical
        in both is a violation, as is one that has the same value in every candidate (a constant);
      * the same files under the same name (e.g. two precisions of one model) are the same model and not compared.
    A metric key is ``<clip or ALL>.<metric>``.
    """
    problems: List[str] = []
    names = sorted(variants)
    dep = lambda key: strict_all or key.split(".", 1)[-1].startswith(tuple(dependent_prefixes))      # noqa: E731
    for n in names:
        if "?" in variants[n]["model_key"]:
            problems.append("%s: a swapper file could not be located or hashed (%s), so which network ran is unknown"
                            % (n, variants[n]["model_key"]))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = variants[a], variants[b]
            if va["model_key"] == vb["model_key"]:
                if va.get("swap_model") != vb.get("swap_model"):
                    problems.append("%s (swap_model=%s) and %s (swap_model=%s) loaded the SAME network files [%s]: "
                                    "one model behind two names" % (a, va.get("swap_model"), b, vb.get("swap_model"),
                                                                    va["model_key"]))
                continue
            for key in sorted(set(va["metrics"]) & set(vb["metrics"])):
                if not dep(key):
                    continue
                x, y = _num(va["metrics"][key]), _num(vb["metrics"][key])
                if x is None or y is None:
                    continue
                if x == y or math.isclose(x, y, rel_tol=0.0, abs_tol=1e-12):
                    problems.append("metric %s is identical (%r) in %s and %s, which are different models" % (key, x, a, b))
    keys = set().union(*(set(v["metrics"]) for v in variants.values())) if variants else set()
    distinct_models = {v["model_key"] for v in variants.values()}
    for key in sorted(keys):
        if not dep(key) or len(distinct_models) < 2:
            continue
        vals = [_num(v["metrics"].get(key)) for v in variants.values()]
        vals = [x for x in vals if x is not None]
        if len(vals) >= 2 and len({round(x, 12) for x in vals}) == 1:
            problems.append("metric %s has the same value (%r) in all %d candidates" % (key, vals[0], len(vals)))
    seen, unique = set(), []
    for p in problems:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def model_key(files: Sequence[Tuple[str, Optional[str]]]) -> str:
    """The identity of the network(s) a render loaded: the content hash of every swapper file, and nothing else.

    A file that is not on disk contributes ``?`` so the guard can refuse to guess. The model NAME is deliberately not
    part of the key: the registry falls back to another model when a name's file is missing, and the key has to see
    through that."""
    return ",".join("%s=%s" % (f, (h or "?")[:16]) for f, h in sorted(files)) or "?"


def file_sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════ rendering ═══════════════════════════

def clip_table() -> Dict[str, Dict[str, Any]]:
    import baseline_snapshot as bs
    table = dict(bs.CLIPS)
    table.update(EXTRA_CLIPS)
    return table


def clean_env(chunk_mb: int) -> Dict[str, str]:
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("ROOP_BENCH_") or k in ("ROOP_SWAP_FP32", "ROOP_SWAP_FP16", "ROOP_TRT_BOUND", "ROOP_FIRST_INFERENCE_LOG"):
            env.pop(k)
    env["ROOP_PROFILE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["ROOP_STAB_CHUNK_MB"] = str(chunk_mb)          # one stabilizer geometry for every variant (pixels depend on it)
    return env


def render(variant: Variant, clip: str, spec: Dict[str, Any], frames: int, cfg: Any, out_root: str,
           threads: int, chunk_mb: int, reuse: bool, fixture_in: Optional[str] = None,
           fixture_out: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """One render. ``fixture_out``: save the captured target people here (the reference does). ``fixture_in``: load them
    instead of capturing (a candidate whose own capture picked different frames). ``force``: ignore a stored render."""
    import argparse as _ap
    import baseline_snapshot as bs
    import baseline_controlled as bc
    label = "%s__%s" % (variant.name, clip)
    outdir = os.path.join(out_root, label)
    meta_path = os.path.join(outdir, "harness_render.json")
    if reuse and not force and os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        # A stored render is trusted only if it validates NOW: a clean log, the right frame count, the lossless /
        # first-inference records. (The first full run stored a reference whose encoder died at frame 299.)
        if (meta.get("returncode") == 0 and os.path.exists(meta.get("output_file") or "")
                and "lossless_encode" in meta and os.path.exists(meta.get("first_inference_log") or "")
                and os.path.exists(meta.get("log") or "")
                and not render_problems(open(meta["log"], encoding="utf-8", errors="ignore").read(),
                                        meta["output_file"], frames)):
            return meta
        print("  [harness] stored render %s is missing or invalid: rendering again" % label, flush=True)
    os.makedirs(out_root, exist_ok=True)
    cfg2 = copy.copy(cfg)
    cfg2.swap_model = variant.swap_model
    env = clean_env(chunk_mb)
    env["ROOP_BENCH_CODEC"] = "libx264"
    env["ROOP_BENCH_CRF"] = "0"
    env["ROOP_BENCH_LAUNCHER"] = os.path.join(TESTS, "first_inference_probe.py")
    fi_log = os.path.join(out_root, label + ".first_inference.jsonl")
    if os.path.exists(fi_log):
        os.remove(fi_log)
    env["ROOP_FIRST_INFERENCE_LOG"] = fi_log
    if variant.trt_precision:
        env["ROOP_BENCH_TRT_PRECISION"] = variant.trt_precision
    if fixture_in:
        env["ROOP_BENCH_FIXTURE_IN"] = fixture_in
    elif fixture_out:
        env["ROOP_BENCH_FIXTURE_OUT"] = fixture_out
    env.update(variant.env)
    env = bc.ensure_ffmpeg(env)
    os.environ["PATH"] = env["PATH"]
    start = int(spec["window"][0])
    # baseline_snapshot.build_cmd reads ROOP_BENCH_LAUNCHER from THIS process's environment (it builds the command
    # here, then hands `env` to the child), so the launcher has to be set on both or it silently never runs and the
    # per-model first-inference log comes back empty (found on the first smoke run).
    prev_launcher = os.environ.get("ROOP_BENCH_LAUNCHER")
    os.environ["ROOP_BENCH_LAUNCHER"] = env["ROOP_BENCH_LAUNCHER"]
    attempts, problems = 0, []
    try:
        while True:
            attempts += 1
            rec = bs.run_one(label, clip, spec, cfg2, _ap.Namespace(detector_engine=None, governor=False), env,
                             out_root, threads, window=(start, start + frames))
            text = open(rec["log"], encoding="utf-8", errors="ignore").read()
            out_file = (rec.get("output") or {}).get("file")
            problems = render_problems(text, out_file, frames) if rec["returncode"] == 0 else ["return code %s" % rec["returncode"]]
            if fixture_in and "[bench] fixture pinned from" not in text:
                problems.append("the render did not load the pinned capture fixture")
            if not problems or attempts >= 2:
                break
            # The failure seen so far is the app's own encoder dying at the end of a long lossless encode ("usually Smart
            # App Control blocking an unsigned ffmpeg DLL ... re-run the job"): a flake, so one retry, not a silent pass.
            print("  [harness] %s is not a valid render (%s); rendering it once more" % (label, "; ".join(problems)), flush=True)
    finally:
        if prev_launcher is None:
            os.environ.pop("ROOP_BENCH_LAUNCHER", None)
        else:
            os.environ["ROOP_BENCH_LAUNCHER"] = prev_launcher
    if problems:
        raise SystemExit("%s is not a valid render after %d attempts: %s (log: %s)" % (label, attempts, "; ".join(problems), rec["log"]))
    meta = {
        "label": label, "variant": variant.name, "clip": clip, "swap_model": variant.swap_model,
        "trt_precision": variant.trt_precision, "returncode": rec["returncode"], "window": [start, start + frames],
        "log": rec["log"], "output_file": out_file, "outdir": outdir,
        "rows_csv": os.path.join(outdir, "rows.csv"), "plate_video": os.path.join(outdir, "work", "clip.mp4"),
        "capture_lines": rec.get("capture_lines") or [], "swap_audit": parse_swap_audit(text),
        "tracks": parse_track_count(text), "sessions": parse_sessions(text),
        "lossless_encode": verify_lossless(out_file), "attempts": attempts,
        "fixture_mode": "pinned" if fixture_in else ("saved" if fixture_out else "natural"),
        "fixture_file": fixture_in or fixture_out,
        "first_inference_log": fi_log, "frame_loop_fps": rec.get("frame_loop_fps"),
        "faces_per_s": rec.get("faces_per_s"), "output_sha256": (rec.get("output") or {}).get("sha256"),
        "failed_frames": rec.get("failed_frames"), "wall_seconds": rec.get("wall_seconds"),
        "bench_config": rec.get("bench_config"),
    }
    os.makedirs(outdir, exist_ok=True)
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=1, default=str)
    return meta


def is_lossless_encode(ffmpeg_banner: str) -> bool:
    """True when an `ffmpeg -i file` banner describes x264 CRF 0: H.264 only uses the High 4:4:4 Predictive profile
    for a lossless stream (a lossy 4:2:0 encode reports High / Main / Baseline)."""
    return bool(re.search(r"Video: h264 \(High 4:4:4 Predictive\)", ffmpeg_banner or ""))


def verify_lossless(path: str) -> bool:
    import shutil
    import baseline_controlled as bc
    env = bc.ensure_ffmpeg(dict(os.environ))
    ff = shutil.which("ffmpeg", path=env["PATH"])
    if not ff or not path or not os.path.exists(path):
        return False
    p = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True, errors="ignore")
    return is_lossless_encode(p.stderr)


def load_first_inference(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def verify_reference_is_fp32(first_inference: List[Dict[str, Any]], sessions: List[Dict[str, str]]) -> List[str]:
    """A reference that is not FP32 invalidates every number measured against it."""
    bad = []
    for rec in first_inference:
        if rec.get("provider") == "TensorrtExecutionProvider" and rec.get("trt_fp16") != "off":
            bad.append("%s ran TensorRT with trt_fp16=%s" % (rec.get("file"), rec.get("trt_fp16")))
    for s in sessions:
        if s["provider"] == "TensorrtExecutionProvider" and s["trt_fp16"] != "off":
            bad.append("%s (%s) bound TensorRT with trt_fp16=%s" % (s["file"], s["tag"], s["trt_fp16"]))
    return sorted(set(bad))


def swapper_files(sessions: List[Dict[str, str]]) -> List[Tuple[str, Optional[str]]]:
    # Every session that belongs to the swap stage. A composite swapper (realswap = hyperswap_1a + a HiFiFace band)
    # loads more than one network, and its identity is ALL of them; matching only the first would make it look like
    # plain hyperswap to the guard.
    files = sorted({s["file"] for s in sessions
                    if any(t in s["tag"].lower() for t in ("swap", "hififace", "crossface"))})
    models = os.path.join(APP, "models")
    return [(f, file_sha256(os.path.join(models, f))) for f in files]


# ═══════════════════════════ scoring (needs the pipeline: detector, AdaFace) ═══════════════════════════

def read_video(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def read_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def _row_box(r: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])


class Scorer:
    """Holds the pipeline (fixed FP32 instrument) and everything cached per clip."""

    def __init__(self, cfg: Any, sources_dir: str) -> None:
        os.environ["ROOP_BENCH_TRT_PRECISION"] = "fp32"        # the RULER does not change with the candidate
        import angle_bench as ab
        ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), None, None, sync_config=True)
        from roop import recognizer_adaface as ada
        if ada.enabled():
            raise SystemExit("ROOP_ADAFACE is on: the pipeline would be using AdaFace for matching, so it is not an "
                             "independent judge of identity. Unset it and re-run.")
        from roop.face_util import align_crop, get_all_faces
        self.ada, self.align_crop, self.get_all_faces = ada, align_crop, get_all_faces
        self._plate_faces: Dict[str, List[List[Any]]] = {}
        self._source_emb: Dict[str, Dict[str, Any]] = {}

    # ── sources ────────────────────────────────────────────────────────────────────────────────
    def source_embeddings(self, name: str) -> Dict[str, Any]:
        if name in self._source_emb:
            return self._source_emb[name]
        import two_face_video as tfv
        fs = tfv.load_library_faceset(name)
        embs = []
        for face, img in zip(fs.faces, fs.ref_images):
            crop, _ = self.align_crop(img, face.kps, 112, mode=self.ada.ALIGN_MODE)
            e = self.ada.embed_crop(crop)
            if e is not None:
                embs.append(np.asarray(e, np.float64) / max(1e-9, np.linalg.norm(e)))
        if not embs:
            raise SystemExit("no AdaFace embedding could be computed for faceset %s" % name)
        mean = np.mean(embs, axis=0)
        pos = [cosine(a, b) for i, a in enumerate(embs) for b in embs[i + 1:]]
        self._source_emb[name] = {"mean": mean / np.linalg.norm(mean), "all": embs, "n": len(embs),
                                  "within_source_cosine": summarize(pos)}
        return self._source_emb[name]

    # ── plate detections (the same for every candidate) ────────────────────────────────────────
    def plate_faces(self, clip: str, plate_video: str, plates: List[np.ndarray]) -> List[List[Any]]:
        if clip not in self._plate_faces:
            self._plate_faces[clip] = [list(self.get_all_faces(p) or []) for p in plates]
        return self._plate_faces[clip]

    def embed_at(self, frame: np.ndarray, kps: np.ndarray) -> Optional[np.ndarray]:
        crop, _ = self.align_crop(frame, kps, 112, mode=self.ada.ALIGN_MODE)
        e = self.ada.embed_crop(crop)
        return None if e is None else np.asarray(e, np.float64)

    # ── one candidate on one clip ──────────────────────────────────────────────────────────────
    def score(self, clip: str, spec: Dict[str, Any], ref: Dict[str, Any], cand: Dict[str, Any],
              frames: int, csv_path: str) -> Dict[str, Any]:
        import two_face_video as tfv
        from roop.procmgr_masking import landmark_hull
        names = [s.strip() for s in spec["sources"].split(",")]
        src = [self.source_embeddings(n) for n in names]
        plates = read_video(ref["plate_video"])
        ref_out = read_video(ref["output_file"])
        cand_out = read_video(cand["output_file"])
        if not (len(plates) == len(ref_out) == len(cand_out) == frames):
            raise SystemExit("%s: frame counts differ (plate %d, reference %d, candidate %d, wanted %d)" % (
                clip, len(plates), len(ref_out), len(cand_out), frames))
        start = ref["window"][0]
        faces_by_frame = self.plate_faces(clip, ref["plate_video"], plates)
        ref_rows = [r for r in read_rows(ref["rows_csv"])]
        cand_rows = [r for r in read_rows(cand["rows_csv"])]
        by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for r in ref_rows:
            by_frame.setdefault(int(r["frame"]) - start, []).append(r)
        cand_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for r in cand_rows:
            cand_by_frame.setdefault(int(r["frame"]) - start, []).append(r)

        from roop.face_util import solve_pose_5pt
        from roop.hyperswap_optimizer import HyperSwapQualityAuditor
        out_faces: Dict[int, List[Any]] = {}
        ref_faces: Dict[int, List[Any]] = {}
        per_face: List[Dict[str, Any]] = []
        det_hit = det_total = 0
        for i in range(frames):
            rrows = by_frame.get(i, [])
            h, t = detection_recall([_row_box(r) for r in rrows],
                                    [_row_box(r) for r in cand_by_frame.get(i, [])], 0.5)
            det_hit += h
            det_total += t
            for r in rrows:
                if r.get("src", "") in ("", None):
                    continue                                             # not swapped in the reference
                if r.get("contam", "") != "" and float(r["contam"]) >= tfv.GRADE_CONTAM_MAX:
                    continue                                             # shared crop: identity unreadable, as in the harness
                s = int(float(r["src"]))
                if not (0 <= s < len(src)):
                    continue
                box = _row_box(r)
                face = max(faces_by_frame[i], key=lambda f: box_iou(box, f.bbox), default=None)
                if face is None or box_iou(box, face.bbox) < 0.5:
                    continue
                kps = np.asarray(face.kps, np.float32)
                e_c, e_r, e_p = (self.embed_at(cand_out[i], kps), self.embed_at(ref_out[i], kps),
                                 self.embed_at(plates[i], kps))
                mask_full = np.zeros(plates[i].shape[:2], np.uint8)
                lm = getattr(face, "landmark_2d_106", None)
                if lm is not None:
                    hull = landmark_hull(lm, kps)[0]
                    cv2.fillConvexPoly(mask_full, np.asarray(hull, np.int32).reshape(-1, 2), 255)
                else:
                    x0, y0, x1, y1 = [int(v) for v in face.bbox]
                    mask_full[max(0, y0):y1, max(0, x0):x1] = 255
                ys, xs = np.nonzero(mask_full)
                if ys.size < 64:
                    continue
                sl = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
                m = mask_full[sl]
                a, b = cand_out[i][sl], ref_out[i][sl]
                # Geometry: the same keypoints detected on the PLATE and on the swapped OUTPUT (the pipeline's own
                # detector, FP32). A swap that moves the eyes or the mouth shows up here; a face that cannot be
                # re-detected in the output is counted, not scored.
                if i not in out_faces:
                    out_faces[i] = list(self.get_all_faces(cand_out[i]) or [])
                twin = max(out_faces[i], key=lambda f: box_iou(box, f.bbox), default=None)
                if twin is not None and box_iou(box, twin.bbox) >= 0.5:
                    geo = HyperSwapQualityAuditor.evaluate_geometric_alignment_error(kps, np.asarray(twin.kps, np.float32))
                    eye_px, mouth_px, redetected = geo["eye_error_px"], geo["mouth_error_px"], 1
                else:
                    eye_px = mouth_px = float("nan")
                    redetected = 0
                if i not in ref_faces:
                    ref_faces[i] = list(self.get_all_faces(ref_out[i]) or [])
                rtwin = max(ref_faces[i], key=lambda f: box_iou(box, f.bbox), default=None)
                if rtwin is not None and box_iou(box, rtwin.bbox) >= 0.5:
                    rgeo = HyperSwapQualityAuditor.evaluate_geometric_alignment_error(kps, np.asarray(rtwin.kps, np.float32))
                    ref_eye_px, ref_mouth_px = rgeo["eye_error_px"], rgeo["mouth_error_px"]
                else:
                    ref_eye_px = ref_mouth_px = float("nan")
                pose = solve_pose_5pt(kps)
                yaw = float(pose[0]) if pose is not None else float("nan")
                per_face.append({
                    "clip": clip, "frame": i, "box": list(box), "src": s, "source": names[s],
                    "yaw_deg": yaw, "yaw_bin": yaw_bin(yaw) or "",
                    "eye_drift_px": eye_px, "mouth_drift_px": mouth_px, "redetected": redetected,
                    "eye_drift_reference_px": ref_eye_px, "mouth_drift_reference_px": ref_mouth_px,
                    "skin_detail": masked_laplacian_variance(a, m),
                    "skin_detail_reference": masked_laplacian_variance(b, m),
                    "id_cos_candidate": cosine(e_c, src[s]["mean"]), "id_cos_reference": cosine(e_r, src[s]["mean"]),
                    "id_cos_plate": cosine(e_p, src[s]["mean"]),
                    "ssim": masked_ssim(a, b, m), "psnr_db": masked_psnr(a, b, m),
                    "plate_vs_reference_ssim": masked_ssim(plates[i][sl], b, m),
                    "region_px": int((m > 0).sum())})
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if per_face:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(per_face[0].keys()))
                w.writeheader()
                w.writerows(per_face)
        col = lambda k: [p[k] for p in per_face]                                        # noqa: E731
        delta = [c - r for c, r in zip(col("id_cos_candidate"), col("id_cos_reference"))]
        jit = []
        for sidx in sorted({p["src"] for p in per_face}):
            rows_s = [p for p in per_face if p["src"] == sidx]
            jit.append(identity_jitter([p["frame"] for p in rows_s], [p["id_cos_candidate"] for p in rows_s]))
        by_yaw = {}
        for lo, hi in YAW_BINS:
            label = yaw_bin(lo + 0.001)
            rows_b = [p for p in per_face if p["yaw_bin"] == label]
            by_yaw[label] = {"faces": len(rows_b),
                             "identity_cos": summarize([p["id_cos_candidate"] for p in rows_b]),
                             "ssim": summarize([p["ssim"] for p in rows_b]),
                             "identity_delta_vs_reference": summarize(
                                 [p["id_cos_candidate"] - p["id_cos_reference"] for p in rows_b])}
        detail_ratio = [p["skin_detail"] / p["skin_detail_reference"] for p in per_face
                        if p["skin_detail_reference"] and not math.isnan(p["skin_detail"])]
        return {
            "faces_scored": len(per_face),
            "eye_drift_px": summarize(col("eye_drift_px")), "mouth_drift_px": summarize(col("mouth_drift_px")),
            "eye_drift_reference_px_control": summarize(col("eye_drift_reference_px")),
            "mouth_drift_reference_px_control": summarize(col("mouth_drift_reference_px")),
            "output_face_redetected": {"redetected": int(sum(col("redetected"))), "of": len(per_face)},
            "skin_detail": summarize(col("skin_detail")),
            "skin_detail_ratio_vs_reference": summarize(detail_ratio),
            "identity_jitter": summarize(jit), "by_yaw": by_yaw,
            "identity_cos": summarize(col("id_cos_candidate")),
            "identity_cos_reference": summarize(col("id_cos_reference")),
            "identity_cos_plate_control": summarize(col("id_cos_plate")),
            "identity_delta_vs_reference": summarize(delta),
            "ssim": summarize(col("ssim")), "psnr_db": summarize(col("psnr_db")),
            "plate_vs_reference_ssim_control": summarize(col("plate_vs_reference_ssim")),
            "detection_recall": {"matched": det_hit, "reference_faces": det_total,
                                 "recall": (det_hit / det_total) if det_total else None},
            "source_within_cosine_control": {n: s["within_source_cosine"] for n, s in zip(names, src)},
            "per_face_csv": csv_path}

    # ── aligned crops for the XSeg stage ───────────────────────────────────────────────────────
    def xseg_crops(self, clip: str, ref: Dict[str, Any], frames: int, every: int = 10) -> np.ndarray:
        from roop.face_analyser import canonicalize_face_alignment
        plates = read_video(ref["plate_video"])
        faces_by_frame = self.plate_faces(clip, ref["plate_video"], plates)
        crops = []
        for i in range(0, frames, every):
            for f in faces_by_frame[i]:
                aligned, _, _ = canonicalize_face_alignment(plates[i], f, 256, "arcface")
                crops.append(np.ascontiguousarray(aligned))
        return np.stack(crops) if crops else np.zeros((0, 256, 256, 3), np.uint8)


def xseg_masks_subprocess(precision: str, crops_path: str, out_path: str) -> Dict[str, Any]:
    env = dict(os.environ)
    env["ROOP_BENCH_TRT_PRECISION"] = precision
    cmd = [sys.executable, os.path.abspath(__file__), "_xseg", "--crops", crops_path, "--out", out_path,
           "--precision", precision]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if p.returncode != 0:
        raise SystemExit("xseg stage (%s) failed:\n%s" % (precision, (p.stdout + p.stderr)[-2000:]))
    return json.load(open(out_path + ".json", encoding="utf-8"))


def _xseg_main(args: argparse.Namespace) -> int:
    """Child process: XSeg over saved crops under ONE precision; records the real provider, trt_fp16, VRAM."""
    os.environ["ROOP_BENCH_TRT_PRECISION"] = args.precision
    import angle_bench as ab
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), None, None, sync_config=True)
    import onnxruntime
    import torch
    import roop.globals as g
    from roop import baseline_probe
    from roop.precision_policy import providers_for
    from roop.utilities import get_onnx_session_options, get_small_card_safe_providers, resolve_relative_path
    crops = np.load(args.crops)["crops"]
    path = resolve_relative_path("../models/xseg.onnx")
    base = get_small_card_safe_providers(g.execution_providers, model_path=path, stage="mask:xseg")
    providers, _ = providers_for("masking:xseg", base, path)
    sess = onnxruntime.InferenceSession(path, get_onnx_session_options(), providers=providers)
    name = sess.get_inputs()[0].name

    def used():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1048576.0

    masks, before, after = [], None, None
    for k, c in enumerate(crops):
        x = cv2.resize(c, (256, 256), interpolation=cv2.INTER_CUBIC).astype(np.float32)[None] / 255.0
        if k == 0:
            before = used()
        y = sess.run(None, {name: x})[0]
        if k == 0:
            after = used()
        masks.append(np.asarray(y)[0].reshape(256, 256).astype(np.float32))
    np.savez_compressed(args.out, masks=np.stack(masks) if masks else np.zeros((0, 256, 256), np.float32))
    json.dump({"precision": args.precision, "provider": sess.get_providers()[0],
               "trt_fp16": baseline_probe.trt_fp16_state(sess), "crops": int(len(crops)),
               "vram_before_first_inference_mib": before, "vram_after_first_inference_mib": after},
              open(args.out + ".json", "w", encoding="utf-8"))
    return 0


# ═══════════════════════════ orchestration, report ═══════════════════════════

def per_model_log(meta_by_variant: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Real provider, trt_fp16 and VRAM before / after the FIRST inference, per session, per variant (all clips).

    A session built from in-memory model bytes (the swappers rewrite their graph first) has no path, so it is named by
    matching its input signature against the render's own `[Session]` lines; an ambiguous or unmatched one stays `?`
    rather than being guessed. Sessions are grouped by (file, provider, inputs) so two different sessions are never
    averaged into one row."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for variant, metas in meta_by_variant.items():
        by_sig: Dict[str, set] = {}
        for m in metas:
            for srow in m.get("sessions") or []:
                if srow.get("input"):
                    by_sig.setdefault(srow["input"], set()).add(srow["file"])
        groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for m in metas:
            for rec in load_first_inference(m["first_inference_log"]):
                sig = normalize_input_sig(rec.get("input_signature", ""))
                f = rec["file"]
                if f == "?":
                    cands = by_sig.get(sig, set())
                    f = next(iter(cands)) if len(cands) == 1 else "?"
                groups.setdefault((f, rec["provider"], sig), []).append(rec)
        rows = []
        for (f, provider, sig), recs in sorted(groups.items()):
            rows.append({"file": f, "provider": provider, "inputs": sig,
                         "trt_fp16": sorted({r["trt_fp16"] for r in recs}), "renders_observed": len(recs),
                         "vram_before_mib": summarize([r["vram_before_mib"] for r in recs if r.get("vram_before_mib") is not None]),
                         "vram_after_mib": summarize([r["vram_after_mib"] for r in recs if r.get("vram_after_mib") is not None]),
                         "first_inference_delta_mib": summarize([r["vram_delta_mib"] for r in recs if r.get("vram_delta_mib") is not None]),
                         "first_call_ms": summarize([r["first_call_ms"] for r in recs])})
        out[variant] = rows
    return out


def flatten_metrics(per_clip: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """{'<clip>.<metric>': value} plus ALL.* (mean over clips weighted by faces) for the guard."""
    flat: Dict[str, float] = {}
    for clip, m in per_clip.items():
        for k in FLAT_KEYS:
            v = (m.get(k) or {}).get("mean")
            if v is not None:
                flat["%s.%s" % (clip, k)] = v
    for k in FLAT_KEYS:
        num = den = 0.0
        for m in per_clip.values():
            s = m.get(k) or {}
            if s.get("mean") is not None:
                num += s["mean"] * s["n"]
                den += s["n"]
        if den:
            flat["ALL.%s" % k] = num / den
    return flat


def run(args: argparse.Namespace) -> int:
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    table = clip_table()
    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    for c in clips:
        if c not in table:
            raise SystemExit("unknown clip %r (known: %s)" % (c, ",".join(sorted(table))))
    threads = args.threads or int(cfg.max_threads)
    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)
    ref_v = reference_variant(args.reference_swap_model or str(cfg.swap_model))
    candidates = [parse_candidate(c) for c in args.candidate]
    if not candidates:
        raise SystemExit("give at least one --candidate")
    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise SystemExit("candidate names must be unique")
    t0 = time.time()
    # 1/2. renders ----------------------------------------------------------------------------------------------
    fixture_notes: List[str] = []
    metas: Dict[str, Dict[str, Dict[str, Any]]] = {REFERENCE_NAME: {}}
    for v in candidates:
        metas[v.name] = {}
    for clip in clips:
        spec = table[clip]
        fx = os.path.join(out_root, "%s.fixture.pkl" % clip)
        metas[REFERENCE_NAME][clip] = render(ref_v, clip, spec, args.frames, cfg, out_root, threads, args.chunk_mb, args.reuse,
                                             fixture_out=fx)
        r = metas[REFERENCE_NAME][clip]
        if r["returncode"] != 0:
            raise SystemExit("reference render of %s failed (rc %s): %s" % (clip, r["returncode"], r["log"]))
        if not r["lossless_encode"]:
            raise SystemExit("the reference render of %s is not a lossless encode, so pixel metrics would measure the "
                             "encoder: %s" % (clip, r["output_file"]))
        if not load_first_inference(r["first_inference_log"]):
            raise SystemExit("the first-inference probe wrote nothing for %s (%s): the launcher did not run, so VRAM "
                             "before / after the first inference cannot be reported" % (clip, r["first_inference_log"]))
        bad = verify_reference_is_fp32(load_first_inference(r["first_inference_log"]), r["sessions"])
        if bad:
            raise SystemExit("the reference render of %s is NOT full FP32:\n  %s" % (clip, "\n  ".join(bad)))
        for v in candidates:
            metas[v.name][clip] = render(v, clip, spec, args.frames, cfg, out_root, threads, args.chunk_mb, args.reuse)
            c = metas[v.name][clip]
            if c["returncode"] != 0:
                raise SystemExit("candidate %s render of %s failed (rc %s): %s" % (v.name, clip, c["returncode"], c["log"]))
            if not c["lossless_encode"]:
                raise SystemExit("candidate %s render of %s is not a lossless encode: %s" % (v.name, clip, c["output_file"]))
            if c.get("fixture_mode") != "pinned" and capture_seed(c["capture_lines"]) != capture_seed(r["capture_lines"]):
                fixture_notes.append("%s/%s: seed frame %s vs the reference's %s (same people, same capture frames)" % (
                    v.name, clip, capture_seed(c["capture_lines"]), capture_seed(r["capture_lines"])))
            if c.get("fixture_mode") != "pinned" and capture_fixture(c["capture_lines"]) != capture_fixture(r["capture_lines"]):
                # The candidate's own auto-capture chose other people / frames than the reference's (a near-tie broken
                # differently by an FP16 vs FP32 detector). A different TARGET is a confound in every metric, so the
                # candidate is re-rendered from the reference's captured fixture.
                natural = capture_fixture(c["capture_lines"])
                if not os.path.exists(fx):
                    old_sha = r["output_sha256"]
                    r = metas[REFERENCE_NAME][clip] = render(ref_v, clip, spec, args.frames, cfg, out_root, threads,
                                                             args.chunk_mb, args.reuse, fixture_out=fx, force=True)
                    if r["output_sha256"] != old_sha:
                        raise SystemExit("the %s reference is not bit-reproducible (re-rendered to save its fixture: sha256 "
                                         "%s != %s), so candidates scored against the first one are not comparable" % (
                                             clip, str(r["output_sha256"])[:12], str(old_sha)[:12]))
                c = metas[v.name][clip] = render(v, clip, spec, args.frames, cfg, out_root, threads, args.chunk_mb, args.reuse,
                                                 fixture_in=fx, force=True)
                if c["returncode"] != 0 or not c["lossless_encode"]:
                    raise SystemExit("pinned candidate %s render of %s failed" % (v.name, clip))
                fixture_notes.append("%s/%s: its own capture chose %s but the reference chose %s; re-rendered from the "
                                     "reference's captured fixture" % (v.name, clip, natural, capture_fixture(r["capture_lines"])))
    # 3. scoring ------------------------------------------------------------------------------------------------
    scorer = Scorer(cfg, os.path.join(APP, "facesets"))
    results: Dict[str, Dict[str, Any]] = {v.name: {"per_clip": {}} for v in candidates}
    xseg_by_precision: Dict[str, Dict[str, Any]] = {}
    crops_path = os.path.join(out_root, "xseg_crops.npz")
    all_crops = []
    crop_index: List[Tuple[str, int]] = []
    for clip in clips:
        cr = scorer.xseg_crops(clip, metas[REFERENCE_NAME][clip], args.frames)
        all_crops.append(cr)
        crop_index.append((clip, len(cr)))
    np.savez_compressed(crops_path, crops=np.concatenate(all_crops) if all_crops else np.zeros((0, 256, 256, 3), np.uint8))
    ref_x = xseg_masks_subprocess("fp32", crops_path, os.path.join(out_root, "xseg_masks_fp32.npz"))
    ref_masks = np.load(os.path.join(out_root, "xseg_masks_fp32.npz"))["masks"]
    precisions = sorted({(v.trt_precision or str(cfg.trt_precision)).lower() for v in candidates})
    for p in precisions:
        if p == "fp32":
            xseg_by_precision[p] = {"info": ref_x, "iou": summarize([1.0] * len(ref_masks)),
                                    "note": "same precision as the reference: IoU is 1 by definition"}
            continue
        info = xseg_masks_subprocess(p, crops_path, os.path.join(out_root, "xseg_masks_%s.npz" % p))
        m = np.load(os.path.join(out_root, "xseg_masks_%s.npz" % p))["masks"]
        ious = [mask_iou(a, b) for a, b in zip(m, ref_masks)]
        xseg_by_precision[p] = {"info": info, "iou": summarize(ious),
                                "below_0.95": int(sum(1 for x in ious if x < 0.95)), "crops": len(ious)}
    for v in candidates:
        for clip in clips:
            sc = scorer.score(clip, table[clip], metas[REFERENCE_NAME][clip], metas[v.name][clip], args.frames,
                              os.path.join(out_root, "scores", "%s__%s.csv" % (v.name, clip)))
            ref_m, cand_m = metas[REFERENCE_NAME][clip], metas[v.name][clip]
            sc["tracks"] = {"reference": ref_m["tracks"], "candidate": cand_m["tracks"]}
            sc["swap_audit"] = {"reference": ref_m["swap_audit"], "candidate": cand_m["swap_audit"]}
            sc["render"] = {"fps": cand_m["frame_loop_fps"], "fps_note": "lossless x264 encode (CPU-bound): not model or render speed", "output_sha256": cand_m["output_sha256"],
                            "failed_frames": cand_m["failed_frames"]}
            results[v.name]["per_clip"][clip] = sc
    # 4. guard --------------------------------------------------------------------------------------------------
    guard_in: Dict[str, Dict[str, Any]] = {}
    for v in candidates:
        files = swapper_files(metas[v.name][clips[0]]["sessions"])
        results[v.name]["swapper_files"] = [{"file": f, "sha256": h} for f, h in files]
        guard_in[v.name] = {"swap_model": v.swap_model, "model_key": model_key(files),
                            "metrics": flatten_metrics(results[v.name]["per_clip"])}
        results[v.name]["model_key"] = guard_in[v.name]["model_key"]
        results[v.name]["variant"] = {"swap_model": v.swap_model, "trt_precision": v.trt_precision, "env": v.env}
    violations = distinctness_violations(guard_in, strict_all=args.strict_all_metrics)
    # instrument controls: the harness must be able to see a swap at all
    controls = {}
    for v in candidates:
        for clip, sc in results[v.name]["per_clip"].items():
            ssim_plate = (sc.get("plate_vs_reference_ssim_control") or {}).get("mean")
            plate_id = (sc.get("identity_cos_plate_control") or {}).get("mean")
            ref_id = (sc.get("identity_cos_reference") or {}).get("mean")
            controls["%s.%s" % (v.name, clip)] = {"plate_vs_reference_ssim": ssim_plate, "identity_plate": plate_id,
                                                  "identity_reference": ref_id}
            if ssim_plate is not None and ssim_plate > 0.995:
                violations.append("%s/%s: the plate and the reference render are indistinguishable by SSIM (%.4f), so "
                                  "the face region metric cannot see a swap" % (v.name, clip, ssim_plate))
            if plate_id is not None and ref_id is not None and ref_id <= plate_id:
                violations.append("%s/%s: the swapped reference is not closer to the source (AdaFace %.4f) than the "
                                  "untouched plate (%.4f): the identity metric is not detecting a swap" % (
                                      v.name, clip, ref_id, plate_id))
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "frames": args.frames, "clips": clips,
        "reference": {"swap_model": ref_v.swap_model, "trt_precision": "fp32", "env": ref_v.env,
                      "renders": {c: {k: metas[REFERENCE_NAME][c][k] for k in ("frame_loop_fps", "tracks", "swap_audit", "output_sha256")}
                                  for c in clips}},
        "candidates": results, "xseg_iou_by_precision": xseg_by_precision,
        "per_model_runtime_log": {"reference": per_model_log({REFERENCE_NAME: list(metas[REFERENCE_NAME].values())})[REFERENCE_NAME],
                                  **per_model_log({v.name: list(metas[v.name].values()) for v in candidates})},
        "instrument_controls": controls, "fixture_notes": fixture_notes,
        "guard": {"passed": not violations, "violations": violations, "strict_all_metrics": bool(args.strict_all_metrics)},
        "wall_minutes": round((time.time() - t0) / 60.0, 1),
    }
    path = os.path.join(out_root, getattr(args, "report_name", None) or "quality_report.json")
    json.dump(report, open(path, "w", encoding="utf-8"), indent=1, default=str)
    print_report(report)
    print("wrote", path)
    if violations:
        print("\nQUALITY HARNESS FAILED - the numbers above cannot be trusted:")
        for p in violations:
            print("  - " + p)
        return 2
    return 0


def print_report(rep: Dict[str, Any]) -> None:
    print("\n=== QUALITY vs FP32 REFERENCE (%d frames/clip; AdaFace identity, masked SSIM/PSNR on the face hull) ===" % rep["frames"])
    for name, res in rep["candidates"].items():
        print("\n%s  [%s]" % (name, res.get("model_key")))
        print("  %-6s %6s %9s %9s %9s %9s %8s %7s %6s" % ("clip", "faces", "id_cos", "id_ref", "id_plate", "d_id", "ssim", "psnr", "recall"))
        for clip, m in res["per_clip"].items():
            g = lambda k, f="mean": (m.get(k) or {}).get(f)                                   # noqa: E731
            rc = (m.get("detection_recall") or {}).get("recall")
            fmt = lambda x, w, d: ("%*.*f" % (w, d, x)) if isinstance(x, (int, float)) else "%*s" % (w, "-")   # noqa: E731
            print("  %-6s %6d %s %s %s %s %s %s %s" % (
                clip, m["faces_scored"], fmt(g("identity_cos"), 9, 4), fmt(g("identity_cos_reference"), 9, 4),
                fmt(g("identity_cos_plate_control"), 9, 4), fmt(g("identity_delta_vs_reference"), 9, 4),
                fmt(g("ssim"), 8, 4), fmt(g("psnr_db"), 7, 2), fmt(rc, 6, 3)))
    print("\nXSeg IoU vs the FP32 reference mask, by precision:")
    for p, x in rep["xseg_iou_by_precision"].items():
        print("  %-6s provider %s trt_fp16 %s | IoU mean %s min %s" % (
            p, x["info"].get("provider"), x["info"].get("trt_fp16"),
            None if x["iou"]["mean"] is None else round(x["iou"]["mean"], 4),
            None if x["iou"]["min"] is None else round(x["iou"]["min"], 4)))
    print("\nPer-model runtime (real provider, trt_fp16, device MiB before -> after the FIRST inference):")
    for variant, rows in rep["per_model_runtime_log"].items():
        print("  [%s]" % variant)
        for r in rows:
            b, a, d = (r["vram_before_mib"].get("median"), r["vram_after_mib"].get("median"),
                       r["first_inference_delta_mib"].get("median"))
            print("    %-30s %-24s fp16=%-4s %s -> %s MiB (%s)" % (
                r["file"], r["provider"].replace("ExecutionProvider", ""), ",".join(r["trt_fp16"]),
                "-" if b is None else "%.0f" % b, "-" if a is None else "%.0f" % a, "-" if d is None else "%+.0f" % d))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="render reference + candidates, score, guard, report")
    r.add_argument("--clips", default=DEFAULT_CLIPS)
    r.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    r.add_argument("--candidate", action="append", default=[], metavar="SPEC",
                   help='"NAME|swap_model=hyperswap|trt_precision=mixed|env.K=V" (repeatable)')
    r.add_argument("--reference-swap-model", default=None, help="default: swap_model from config.yaml")
    r.add_argument("--out", default=os.path.join(APP, "output", "quality_" + time.strftime("%Y-%m-%d")))
    r.add_argument("--threads", type=int, default=None)
    r.add_argument("--chunk-mb", type=int, default=1500, help="pinned ROOP_STAB_CHUNK_MB for every render")
    r.add_argument("--reuse", action="store_true", help="reuse renders already on disk in --out")
    r.add_argument("--strict-all-metrics", action="store_true", help="apply the identical-metric rule to EVERY metric")
    r.add_argument("--report-name", default="quality_report.json", help="file name of the report inside --out (so a second pass can reuse the renders)")
    x = sub.add_parser("_xseg")
    x.add_argument("--crops", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--precision", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "_xseg":
        return _xseg_main(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
