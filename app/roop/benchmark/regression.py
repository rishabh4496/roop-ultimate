"""Headless performance + quality regression suite.

    python run.py --benchmark --benchmark-mode regression [--benchmark-frames 300]
        [--benchmark-clip PATH] [--benchmark-source IMG] [--benchmark-threads N]
        [--benchmark-update-baseline]

WHAT RUNS.  A REAL render through `core.batch_process_with_options` -- the
worker pool, swap batcher, stabilizers and ffmpeg writer the Start button uses
-- configured from the user's config.yaml (tests/config_sync), not the
`BenchmarkRunner` preview loop, which is `process_frame` x1 thread and never
reaches those stages. A short warm-up render first pays the TensorRT engine
build and model load, so the timed render measures steady state.

WHAT IT MEASURES.
  * raw decode / raw encode fps: the app's ffmpeg alone, clip -> pipe, pipe -> file;
  * per-stage throughput from the render itself, through the `_prof` stage
    sink: decode, detect, swap, enhance, mask, encode -- calls and busy
    seconds. The CALL COUNTS are also the proof each stage executed (this
    project's recurring failure is a stage that silently never runs);
  * end-to-end fps, p50/p99 per-frame latency and frame-time variance
    (`frame_total`, one sample per frame), and the output cadence (interval
    between successive encodes);
  * peak VRAM and GPU utilisation from NVML, sampled every 50 ms;
  * quality against a stored golden render: full-frame PSNR, and SSIM + PSNR
    on the face crops -- where a swap lives. A whole-frame metric alone cannot
    see a face that stopped being swapped: the face is ~2% of a 1080p frame;
  * swap coverage two ways: frames the pipeline DECIDED to paste (swap log)
    and frames whose face region actually CHANGED against the input by more
    than 5x the measured render noise floor (0.71/255 mean).

WHAT FAILS IT (exit 1): output frame count != input, swap coverage below the
baseline's, or any quality metric below its threshold. FPS is ADVISORY: a
300-frame window measures warm-up-contaminated numbers (AGENTS.md: 600 frames
minimum for an acceptance claim), so an fps drop is printed as a warning and
never fails the run. Exit 3 = baseline not comparable (different config,
clip or source); exit 2 = the suite itself could not run.

Baselines are per machine AND per configuration: `<repo>/.roop/regression/
<gpu>_<config-hash>/` holds baseline.json and golden.mp4. The first run on a
new key records one; `--benchmark-update-baseline` replaces it.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from roop.degrade import swallowed as _swallowed

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(APP_DIR)
STORE_DIR = os.path.join(REPO_DIR, ".roop", "regression")
DEFAULT_FRAMES = 300
WARMUP_FRAMES = 60
SCHEMA = 1

# Face-region mean absolute difference (0-255) above which a face counts as
# CHANGED by the render. The measured noise floor between two renders of one
# unchanged config is 0.7142 mean (AGENTS.md); 4.0 is 5.6x that.
CHANGED_MAD = 4.0

DEFAULT_THRESHOLDS = {
    # Against the golden render of the SAME config on the SAME machine. GPU
    # reduction order is non-deterministic, so an unchanged pipeline does not
    # reproduce bit-exact; these sit below the measured null-control spread
    # (see docs/development/REGRESSION_BENCHMARK.md for the numbers).
    "face_ssim_mean_min": 0.97,
    "face_ssim_frame_min": 0.90,
    "face_psnr_mean_min": 35.0,
    "frame_psnr_mean_min": 40.0,
    # Coverage may not fall more than this below the baseline's (fraction).
    "swap_coverage_drop_max": 0.02,
    # Advisory only.
    "fps_drop_warn": 0.15,
}

# The settings that decide what the output LOOKS like. A baseline recorded
# with different values is not comparable; perf-only knobs (threads, pools,
# batch sizes) are deliberately absent -- changing them must not change the
# picture, and catching it when it does is part of the point.
SIGNATURE_KEYS = (
    "provider", "trt_precision", "swap_model", "selected_enhancer",
    "adaptive_enhancer_profile", "mask_engine", "mask_engine_2", "blend_ratio",
    "face_mask_blend", "mouth_mask_blend", "swap_model_mask_strength",
    "color_transfer_mode", "color_match_after_enhance", "enhancer_align",
    "detector_engine", "face_detector_size", "face_detector_threshold",
    "face_detector_nms", "subsample_upscale", "stabilize_face",
    "stabilize_method", "stabilize_min_cutoff", "stabilize_beta",
    "stabilize_enhancer", "stabilize_enhancer_strength", "stabilize_mask",
    "stabilize_mask_strength", "stabilize_landmarks", "stabilize_hf_texture",
    "stabilize_hf_texture_weight", "merger_hist_match", "merger_sharpen",
    "merger_motion_blur", "merger_grain_match", "merger_degrade",
    "merger_clarity", "eyes_blend_amount", "eyes_feather_blend",
    "output_video_codec", "video_quality", "track_identities",
    "temporal_detection", "autorotate_faces",
)

STAGES = ("decode", "detect", "swap", "mask", "enhance", "encode")

# A reported stage that `_prof` records under more than one name. Detection
# runs per frame as 'detect', but with temporal_detection on (the default) it
# runs in the tracking pre-pass instead, as 'detection' inside 'track_detect'
# (which adds the GPU-guard wait, so it would double-count).
STAGE_SOURCES = {"detect": ("detect", "detection")}


# ── small pure helpers (unit-tested) ─────────────────────────────────────────

def percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-10:
        return 100.0
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM on luma (Wang et al. 2004: 11x11 Gaussian, sigma 1.5)."""
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda img: cv2.GaussianBlur(img, (11, 11), 1.5)  # noqa: E731
    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx
    syy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    num = (2 * mx * my + c1) * (2 * sxy + c2)
    den = (mx * mx + my * my + c1) * (sxx + syy + c2)
    return float(np.mean(num / den))


def expand_box(box, width: int, height: int, margin: float = 0.15) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, int(x1 - mw)), max(0, int(y1 - mh)),
            min(width, int(x2 + mw)), min(height, int(y2 + mh)))


def engine_list(mapped) -> List[str]:
    """`api.map_mask_engines` returns None, ONE engine as a plain string, or a
    list of two. Iterating the string form gives its characters."""
    if isinstance(mapped, str):
        return [mapped]
    return [e for e in (mapped or []) if e]


def config_signature(cfg) -> Dict[str, object]:
    return {key: getattr(cfg, key, None) for key in SIGNATURE_KEYS}


def signature_hash(signature: dict) -> str:
    return hashlib.sha256(json.dumps(signature, sort_keys=True, default=str)
                          .encode()).hexdigest()[:10]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(text: str) -> str:
    keep = "".join(ch if ch.isalnum() else "-" for ch in (text or "cpu").lower())
    return "-".join(part for part in keep.split("-") if part)[:40] or "cpu"


def compare_to_baseline(current: dict, baseline: dict, thresholds: dict) -> dict:
    """Pass/fail verdicts. Pure: no I/O, so the gates are unit-testable."""
    failures, warnings = [], []
    q = current["quality"]
    if current["frames_out"] != current["frames_in"]:
        failures.append("output has %d frames, input %d"
                        % (current["frames_out"], current["frames_in"]))
    for key, metric in (("face_ssim_mean_min", "face_ssim_mean"),
                        ("face_ssim_frame_min", "face_ssim_min"),
                        ("face_psnr_mean_min", "face_psnr_mean"),
                        ("frame_psnr_mean_min", "frame_psnr_mean")):
        value = q.get(metric)
        if value is None:
            failures.append("%s not measured (no face in the golden frames?)" % metric)
        elif value < thresholds[key]:
            failures.append("%s %.4f < %.4f" % (metric, value, thresholds[key]))
    for key in ("changed_coverage", "decided_coverage"):
        now, then = current["swap"][key], baseline["swap"][key]
        if now < then - thresholds["swap_coverage_drop_max"]:
            failures.append("%s fell %.1f%% -> %.1f%%" % (key, 100 * then, 100 * now))
    fps_now, fps_then = current["perf"]["e2e_fps"], baseline["perf"]["e2e_fps"]
    if fps_then and fps_now < fps_then * (1.0 - thresholds["fps_drop_warn"]):
        warnings.append("end-to-end fps %.2f vs baseline %.2f (%.1f%%) -- advisory: a "
                        "300-frame window is not an acceptance measurement"
                        % (fps_now, fps_then, 100.0 * (fps_now / fps_then - 1.0)))
    return {"passed": not failures, "failures": failures, "warnings": warnings}


# ── frame I/O through the app's own ffmpeg ──────────────────────────────────

def _ffmpeg() -> str:
    from roop.ffmpeg_path import ffmpeg_binary
    return ffmpeg_binary()


def probe(path: str) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return width, height, fps, count


class FrameReader:
    """bgr24 frames over an ffmpeg pipe -- the same decoder the render uses."""

    def __init__(self, path: str, limit: Optional[int] = None):
        self.width, self.height, _fps, _n = probe(path)
        command = [_ffmpeg(), "-v", "error", "-i", path]
        if limit:
            command += ["-frames:v", str(int(limit))]
        command += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=1 << 24)
        self.size = self.width * self.height * 3

    def __iter__(self):
        while True:
            raw = self.proc.stdout.read(self.size)
            if len(raw) < self.size:
                break
            yield np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 3)
        self.close()

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


def count_frames(path: str) -> int:
    """Decoded frame count (exact), not the container's estimate."""
    from roop.ffmpeg_path import ffprobe_binary
    res = subprocess.run([ffprobe_binary(), "-v", "error", "-count_frames",
                          "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", path],
                         capture_output=True, text=True)
    try:
        return int(res.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return probe(path)[3]


def trim_clip(src: str, dst: str, frames: int) -> str:
    """First `frames` frames, re-encoded lossless so the cut is exact."""
    subprocess.run([_ffmpeg(), "-y", "-v", "error", "-i", src, "-frames:v", str(frames),
                    "-an", "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", dst], check=True)
    return dst


def raw_decode_fps(path: str, frames: int) -> float:
    reader = FrameReader(path, frames)
    start = time.perf_counter()
    n = sum(1 for _ in reader)
    return n / max(1e-9, time.perf_counter() - start)


def raw_encode_fps(path: str, frames: int, work_dir: str, codec: str, quality: int) -> dict:
    """The app's FFMPEG_VideoWriter with the configured codec, fed decoded frames."""
    from roop.ffmpeg_writer import FFMPEG_VideoWriter
    reader = FrameReader(path, frames)
    target = os.path.join(work_dir, "encode_probe.mp4")
    writer = FFMPEG_VideoWriter(target, (reader.width, reader.height), 30.0,
                                codec=codec, crf=quality)
    busy, n = 0.0, 0
    try:
        for frame in reader:
            start = time.perf_counter()
            writer.write_frame(frame)
            busy += time.perf_counter() - start
            n += 1
    finally:
        start = time.perf_counter()
        writer.close()
        busy += time.perf_counter() - start
    return {"fps": n / max(1e-9, busy), "codec": writer.codec, "frames": n}


# ── observation of the real render ──────────────────────────────────────────

class StageClock:
    """`procmgr_runtime.set_stage_sink` target: per-stage calls and seconds."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = defaultdict(int)
        self.seconds = defaultdict(float)
        self.frame_ms: List[float] = []
        self.encode_times: List[float] = []

    def __call__(self, stage: str, dt: float) -> None:
        with self._lock:
            self.calls[stage] += 1
            self.seconds[stage] += dt
            if stage == "frame_total":
                self.frame_ms.append(dt * 1000.0)
            elif stage == "encode":
                self.encode_times.append(time.perf_counter())

    def stage_table(self) -> Dict[str, dict]:
        table = {}
        for stage in sorted(set(self.calls) | set(STAGES)):
            sources = STAGE_SOURCES.get(stage, (stage,))
            calls = sum(self.calls.get(name, 0) for name in sources)
            secs = sum(self.seconds.get(name, 0.0) for name in sources)
            table[stage] = {
                "calls": calls,
                "busy_s": round(secs, 4),
                "ms_per_call": round(1000.0 * secs / calls, 3) if calls else None,
                # Calls per busy second of the stage, summed across the worker
                # threads that ran it -- per-thread capacity, NOT a wall-clock
                # share, and not a speedup budget (AGENTS.md).
                "per_thread_fps": round(calls / secs, 2) if secs > 0 else None,
            }
        return table


class VramSampler:
    """NVML device-wide used memory and utilisation, every `period` seconds."""

    def __init__(self, period: float = 0.05, device: int = 0):
        self.period, self.device = period, device
        self.samples: List[Tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="regression_vram")
        self._nvml = None
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        except Exception as exc:
            _swallowed("roop/benchmark/regression.py:VramSampler", exc,
                       "VRAM and utilisation unavailable")

    @property
    def available(self) -> bool:
        return self._handle is not None

    def used_mb(self) -> Optional[float]:
        if not self.available:
            return None
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2 ** 20

    def _run(self):
        while not self._stop.is_set():
            try:
                util = float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
                self.samples.append((util, self.used_mb()))
            except Exception as exc:
                _swallowed("roop/benchmark/regression.py:VramSampler._run", exc,
                           "sample dropped")
            self._stop.wait(self.period)

    def start(self):
        if self.available:
            self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if not self.samples:
            return {"vram_peak_mb": None, "gpu_util_mean_pct": None, "gpu_util_peak_pct": None}
        utils = [u for u, _ in self.samples]
        return {
            "vram_peak_mb": round(max(m for _, m in self.samples), 1),
            "gpu_util_mean_pct": round(statistics.fmean(utils), 1),
            "gpu_util_peak_pct": round(max(utils), 1),
            "samples": len(self.samples),
        }


# ── the pipeline, configured as the user runs it ────────────────────────────

@dataclass
class Setup:
    g: object
    options: object
    faceset: object
    signature: dict
    threads: int
    gpu_name: str = ""
    notes: List[str] = field(default_factory=list)


def _tests_on_path():
    tests = os.path.join(APP_DIR, "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)


def load_source_faceset(image_path: str):
    from roop.FaceSet import FaceSet
    from roop.face_util import extract_face_images
    from source_gallery import _mask_offsets_from_cfg
    faces = extract_face_images(image_path, (False, 0))
    if not faces:
        raise RuntimeError("no face detected in the source image %s" % image_path)
    frame = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    faceset = FaceSet()
    face = faces[0][0]
    face.mask_offsets = _mask_offsets_from_cfg()
    faceset.faces.append(face)
    faceset.ref_images.append(frame)
    return faceset


def prepare_pipeline(threads: Optional[int], source_image: str, log) -> Setup:
    """Headless init through the harness helpers every bench shares.

    `init_pipeline(sync_config=True)` copies config.yaml onto roop.globals
    first (tests/config_sync.py -- the one fix for "the bench did not run the
    stack the user runs", violated three times before it existed).
    """
    _tests_on_path()
    os.chdir(APP_DIR)
    from settings import Settings
    import angle_bench as ab
    cfg = Settings("config.yaml")
    g = ab.init_pipeline(cfg.provider, cfg.swap_model, cfg.selected_enhancer,
                         None, swap_model_mask_strength=float(cfg.swap_model_mask_strength),
                         sync_config=True)
    g.execution_threads = int(threads or g.CFG.max_threads)
    g.face_swap_mode = "selected"
    g.track_identities = bool(getattr(g.CFG, "track_identities", True))
    g.temporal_detection = bool(getattr(g.CFG, "temporal_detection", g.track_identities))
    g.video_encoder = g.CFG.output_video_codec
    g.video_quality = int(g.CFG.video_quality)
    g.subsample_size = int(getattr(g.CFG, "subsample_size", g.subsample_size) or 256)

    from api import map_mask_engines
    from roop.core import get_processing_plugins
    engines = engine_list(map_mask_engines(
        g.CFG.mask_engine, getattr(g.CFG, "mask_engine_2", "None"), ""))
    options = ab.build_options(
        g, cfg.swap_model, engines[0] if engines else None,
        stabilize_mask=bool(g.CFG.stabilize_mask),
        stabilize_mask_strength=float(g.CFG.stabilize_mask_strength),
        stabilize_face=bool(g.CFG.stabilize_face),
        stabilize_enhancer=bool(g.CFG.stabilize_enhancer))
    if len(engines) > 1:
        options.processors = get_processing_plugins(engines, swap_model=cfg.swap_model)

    faceset = load_source_faceset(source_image)
    gpu_name = ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        _swallowed("roop/benchmark/regression.py:gpu_name", exc, "gpu name unknown")
    log("[regression] swap=%s enhancer=%s masks=%s provider=%s threads=%d codec=%s gpu=%s"
        % (cfg.swap_model, cfg.selected_enhancer, engines or "none", cfg.provider,
           g.execution_threads, g.video_encoder, gpu_name or "none"))
    return Setup(g=g, options=options, faceset=faceset, signature=config_signature(g.CFG),
                 threads=g.execution_threads, gpu_name=gpu_name)


def capture_target(clip: str, window: int = 150, stride: int = 5):
    """The most confident face in the first `window` frames, as the selected person.

    NOT the first face found: b1.mp4 has nobody on screen until frame ~30, and
    "first box in the first 30 frames" captured a 0.80-score false box at the
    frame edge, so every real track was refused as somebody else and the run
    swapped nothing. A real face scores well above a false one.
    """
    from roop.face_util import get_all_faces
    best, best_key, best_index = None, None, -1
    for index, frame in enumerate(FrameReader(clip, window)):
        if index % stride:
            continue
        for face in get_all_faces(frame.copy()) or []:
            area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            key = (round(float(getattr(face, "det_score", 0.0) or 0.0), 2), area)
            if best_key is None or key > best_key:
                best, best_key, best_index = face, key, index
    if best is None:
        raise RuntimeError("no face detected in the first %d frames of %s" % (window, clip))
    return best, best_index


def render(setup: Setup, clip: str, target, out_dir: str) -> Tuple[Optional[str], dict]:
    """One real render. Returns (output path, swap log {frame: [(bbox, src)]})."""
    import roop.globals as g
    from roop import ProcessMgr as pm
    from roop.core import batch_process_with_options
    from roop.ProcessEntry import ProcessEntry
    from roop.target_selection import normalize_target_selection

    os.makedirs(out_dir, exist_ok=True)
    g.INPUT_FACESETS = [setup.faceset]
    g.TARGET_FACES = [target]
    g.TARGET_FACE_GROUP = [0]
    g.output_path = out_dir
    # A direct caller has no API request; state the selection explicitly so
    # the render cannot quietly select nobody (the Stage 13 trap).
    setup.options.selection_state = normalize_target_selection(
        {"selection_mode": "multi_person", "person_ids": [0]}, person_count=1)
    pm._SWAP_LOG = {}
    before = set(os.listdir(out_dir))
    entry = ProcessEntry(clip, 0, 0, probe(clip)[2])
    try:
        batch_process_with_options([entry], setup.options, None)
    finally:
        log = pm._SWAP_LOG or {}
        pm._SWAP_LOG = None
    if entry.finalname and os.path.exists(entry.finalname):
        return entry.finalname, log
    fresh = sorted((f for f in os.listdir(out_dir)
                    if f not in before and f.lower().endswith(".mp4") and not f.startswith(".")),
                   key=lambda f: os.path.getmtime(os.path.join(out_dir, f)))
    return (os.path.join(out_dir, fresh[-1]) if fresh else None), log


def face_boxes(clip: str, frames: int) -> List[List[List[float]]]:
    from roop.face_util import get_all_faces
    boxes = []
    for frame in FrameReader(clip, frames):
        boxes.append([[float(v) for v in f.bbox] for f in (get_all_faces(frame.copy()) or [])])
    return boxes


def quality_pass(clip: str, output: str, golden: Optional[str], boxes, frames: int) -> dict:
    """One lock-step walk over input, output and golden frames."""
    readers = [FrameReader(clip, frames), FrameReader(output, frames)]
    if golden:
        readers.append(FrameReader(golden, frames))
    face_ssim, face_psnr, frame_psnr = [], [], []
    face_frames = changed = 0
    width, height = readers[0].width, readers[0].height
    for index, row in enumerate(zip(*readers)):
        src, out = row[0], row[1]
        ref = row[2] if golden else None
        if ref is not None:
            frame_psnr.append(psnr(out, ref))
        frame_boxes = boxes[index] if index < len(boxes) else []
        if not frame_boxes:
            continue
        face_frames += 1
        frame_changed = False
        for box in frame_boxes:
            x1, y1, x2, y2 = expand_box(box, width, height)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            crop_out = out[y1:y2, x1:x2]
            mad = float(np.mean(np.abs(crop_out.astype(np.int16) - src[y1:y2, x1:x2].astype(np.int16))))
            frame_changed = frame_changed or mad > CHANGED_MAD
            if ref is not None:
                face_ssim.append(ssim(crop_out, ref[y1:y2, x1:x2]))
                face_psnr.append(psnr(crop_out, ref[y1:y2, x1:x2]))
        changed += int(frame_changed)
    for reader in readers:
        reader.close()
    mean = lambda v: round(float(statistics.fmean(v)), 4) if v else None  # noqa: E731
    return {
        "face_frames": face_frames,
        "changed_frames": changed,
        "face_ssim_mean": mean(face_ssim),
        "face_ssim_min": round(min(face_ssim), 4) if face_ssim else None,
        "face_psnr_mean": mean(face_psnr),
        "frame_psnr_mean": mean(frame_psnr),
    }


# ── orchestration ───────────────────────────────────────────────────────────

def default_clip(frames: int, log) -> str:
    from roop.benchmark.asset_manager import BenchmarkAssetManager, WorkloadMode
    manager = BenchmarkAssetManager()
    path = manager.asset_dir / ("regression_solo_%df_1080p.mp4" % frames)
    if not path.is_file() or count_frames(str(path)) != frames:
        log("[regression] generating the %d-frame 1080p clip %s" % (frames, path.name))
        manager.generate_benchmark_clip(path, WorkloadMode.SOLO,
                                        duration_seconds=frames / 30.0)
    return str(path)


def default_source() -> str:
    """A person other than the clip's own face (plate 0 = the first PNG)."""
    from roop.benchmark.asset_manager import BenchmarkAssetManager
    manager = BenchmarkAssetManager()
    pngs = sorted(manager.facesets_dir.glob("*.png")) if manager.facesets_dir.is_dir() else []
    if len(pngs) >= 2:
        return str(pngs[1])
    return str(manager.ensure_source_reference())


def _print_report(report: dict, log) -> None:
    p, q, s = report["perf"], report["quality"], report["swap"]
    log("")
    log("=" * 74)
    log(" REGRESSION BENCHMARK  %s frames  %s  threads=%s"
        % (report["frames_in"], report["gpu"] or "cpu", report["threads"]))
    log("=" * 74)
    log(" raw decode   %8.1f fps     raw encode  %8.1f fps (%s)"
        % (p["raw_decode_fps"], p["raw_encode_fps"], p["encode_codec"]))
    log(" end-to-end   %8.2f fps     wall %.2f s" % (p["e2e_fps"], p["render_s"]))
    log(" frame time   p50 %.1f ms  p99 %.1f ms  var %.1f ms^2   output p99 interval %.1f ms"
        % (p["frame_ms_p50"], p["frame_ms_p99"], p["frame_ms_var"], p["output_interval_ms_p99"]))
    log(" VRAM peak    %s MB (before render %s MB)   GPU util mean %s%% peak %s%%"
        % (p["vram_peak_mb"], p["vram_before_mb"], p["gpu_util_mean_pct"], p["gpu_util_peak_pct"]))
    log(" stage (in the render)   calls   busy s   ms/call   per-thread fps")
    for stage in STAGES:
        row = report["stages"].get(stage, {})
        log("   %-9s %13s %8s %9s %12s" % (stage, row.get("calls", 0), row.get("busy_s", 0),
                                          row.get("ms_per_call"), row.get("per_thread_fps")))
    log(" swap         decided %d/%d (%.1f%%)   changed %d/%d (%.1f%%)"
        % (s["decided_frames"], report["frames_in"], 100 * s["decided_coverage"],
           s["changed_frames"], s["face_frames"], 100 * s["changed_coverage"]))
    log(" quality      face SSIM mean %s min %s   face PSNR %s dB   frame PSNR %s dB"
        % (q.get("face_ssim_mean"), q.get("face_ssim_min"), q.get("face_psnr_mean"),
           q.get("frame_psnr_mean")))


def run_regression(frames: int = DEFAULT_FRAMES, clip: Optional[str] = None,
                   source: Optional[str] = None, threads: Optional[int] = None,
                   update_baseline: bool = False,
                   log: Callable[[str], None] = print) -> int:
    started = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(STORE_DIR, "runs", started)
    os.makedirs(run_dir, exist_ok=True)

    clip = os.path.abspath(clip) if clip else default_clip(frames, log)
    available = count_frames(clip)
    if available < frames:
        log("[regression] %s has %d frames, fewer than the %d asked for" % (clip, available, frames))
        return 2
    if available > frames:
        clip = trim_clip(clip, os.path.join(run_dir, "clip.mp4"), frames)
    source = os.path.abspath(source) if source else default_source()

    setup = prepare_pipeline(threads, source, log)
    signature = dict(setup.signature, clip_sha256=file_sha256(clip),
                     source_sha256=file_sha256(source), frames=frames)
    key = "%s_%s" % (slug(setup.gpu_name), signature_hash(signature))
    base_dir = os.path.join(STORE_DIR, key)
    baseline_path = os.path.join(base_dir, "baseline.json")
    baseline = None
    if os.path.isfile(baseline_path) and not update_baseline:
        with open(baseline_path, "r", encoding="utf-8") as handle:
            baseline = json.load(handle)

    target, target_frame = capture_target(clip)
    log("[regression] target face captured from frame %d (score %.2f); source %s"
        % (target_frame, float(getattr(target, "det_score", 0.0) or 0.0), os.path.basename(source)))

    log("[regression] raw decode / encode passes")
    decode_fps = raw_decode_fps(clip, frames)
    encode = raw_encode_fps(clip, frames, run_dir, setup.g.video_encoder, setup.g.video_quality)

    log("[regression] warm-up render (%d frames): TensorRT engines, model load" % WARMUP_FRAMES)
    warm_clip = trim_clip(clip, os.path.join(run_dir, "warmup.mp4"), WARMUP_FRAMES)
    render(setup, warm_clip, target, os.path.join(run_dir, "warmup_out"))

    from roop import procmgr_runtime
    clock, sampler = StageClock(), VramSampler()
    vram_before = sampler.used_mb()
    log("[regression] timed render (%d frames)" % frames)
    procmgr_runtime.set_stage_sink(clock)
    sampler.start()
    t0 = time.perf_counter()
    try:
        output, swap_log = render(setup, clip, target, os.path.join(run_dir, "render"))
    finally:
        render_s = time.perf_counter() - t0
        gpu = sampler.stop()
        procmgr_runtime.set_stage_sink(None)
    if not output:
        log("[regression] FAIL: the render produced no output file")
        return 1

    boxes = baseline["face_boxes"] if baseline else face_boxes(clip, frames)
    golden = os.path.join(base_dir, "golden.mp4") if baseline else None
    quality = quality_pass(clip, output, golden, boxes, frames)
    frames_out = count_frames(output)
    intervals = [1000.0 * (b - a) for a, b in zip(clock.encode_times, clock.encode_times[1:])]
    face_frames = max(1, quality["face_frames"])
    report = {
        "schema": SCHEMA, "started": started, "key": key, "gpu": setup.gpu_name,
        "host": platform.node(), "threads": setup.threads, "clip": clip, "source": source,
        "frames_in": frames, "frames_out": frames_out, "signature": signature,
        "perf": {
            "raw_decode_fps": round(decode_fps, 1),
            "raw_encode_fps": round(encode["fps"], 1), "encode_codec": encode["codec"],
            "e2e_fps": round(frames / render_s, 3), "render_s": round(render_s, 2),
            "frame_ms_p50": round(percentile(clock.frame_ms, 50), 2),
            "frame_ms_p99": round(percentile(clock.frame_ms, 99), 2),
            "frame_ms_var": round(statistics.pvariance(clock.frame_ms), 2) if clock.frame_ms else 0.0,
            "output_interval_ms_p50": round(percentile(intervals, 50), 2),
            "output_interval_ms_p99": round(percentile(intervals, 99), 2),
            "vram_before_mb": round(vram_before, 1) if vram_before is not None else None,
            **gpu,
        },
        "stages": clock.stage_table(),
        "swap": {
            "decided_frames": len(swap_log),
            "decided_coverage": round(len(swap_log) / frames, 4),
            "face_frames": quality["face_frames"],
            "changed_frames": quality["changed_frames"],
            "changed_coverage": round(quality["changed_frames"] / face_frames, 4),
        },
        "quality": {k: quality[k] for k in ("face_ssim_mean", "face_ssim_min",
                                            "face_psnr_mean", "frame_psnr_mean")},
    }
    _print_report(report, log)

    # The stages have to have RUN, not merely exist (AGENTS.md failure class).
    missing = [s for s in ("decode", "detect", "swap", "encode") if not report["stages"][s]["calls"]]
    if clock.calls.get("frame_total", 0) == 0:
        missing.append("frame_total")
    thresholds = dict(DEFAULT_THRESHOLDS, **(baseline or {}).get("thresholds", {}))

    if baseline is None:
        problems = []
        if frames_out != frames:
            problems.append("output has %d frames, input %d" % (frames_out, frames))
        if missing:
            problems.append("stages never executed: %s" % ", ".join(missing))
        if report["swap"]["changed_coverage"] < 0.5 or report["swap"]["decided_coverage"] < 0.5:
            problems.append("the render swapped too little to be a baseline (decided %.1f%%, "
                            "changed %.1f%%)" % (100 * report["swap"]["decided_coverage"],
                                                 100 * report["swap"]["changed_coverage"]))
        if problems:
            report["verdict"] = {"passed": False, "failures": problems, "warnings": []}
            _save(report, run_dir)
            log(" RESULT       FAIL -- refusing to record a baseline: " + "; ".join(problems))
            return 1
        os.makedirs(base_dir, exist_ok=True)
        shutil.copyfile(output, os.path.join(base_dir, "golden.mp4"))
        record = dict(report, face_boxes=boxes, thresholds=DEFAULT_THRESHOLDS)
        with open(baseline_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1)
        report["verdict"] = {"passed": True, "failures": [], "warnings": [],
                             "baseline_recorded": base_dir}
        _save(report, run_dir)
        log(" RESULT       BASELINE RECORDED -> %s" % base_dir)
        log("              run again to compare against it")
        return 0

    verdict = compare_to_baseline(report, baseline, thresholds)
    if missing:
        verdict["failures"].append("stages never executed: %s" % ", ".join(missing))
        verdict["passed"] = False
    report["verdict"] = verdict
    report["baseline"] = {"key": key, "recorded": baseline.get("started"),
                          "e2e_fps": baseline["perf"]["e2e_fps"]}
    _save(report, run_dir)
    for warning in verdict["warnings"]:
        log(" WARN         " + warning)
    if verdict["passed"]:
        log(" RESULT       PASS (baseline %s, fps %.2f -> %.2f)"
            % (baseline.get("started"), baseline["perf"]["e2e_fps"], report["perf"]["e2e_fps"]))
        return 0
    for failure in verdict["failures"]:
        log(" FAIL         " + failure)
    log(" RESULT       REGRESSION (report %s)" % run_dir)
    return 1


def _save(report: dict, run_dir: str) -> None:
    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, default=str)


def run_regression_cli(frames: int = DEFAULT_FRAMES, clip: Optional[str] = None,
                       source: Optional[str] = None, threads: Optional[int] = None,
                       update_baseline: bool = False, log: Callable[[str], None] = print) -> int:
    try:
        return run_regression(frames=frames, clip=clip, source=source, threads=threads,
                              update_baseline=update_baseline, log=log)
    except KeyboardInterrupt:
        log("[regression] interrupted")
        return 2
    except Exception as exc:
        import traceback
        traceback.print_exc()
        log("[regression] could not run: %s: %s" % (type(exc).__name__, exc))
        return 2


def run_regression_from_args(args) -> int:
    """The one entry point run.py and core.py both call (flags are identical)."""
    return run_regression_cli(
        frames=int(getattr(args, "benchmark_frames", None) or DEFAULT_FRAMES),
        clip=getattr(args, "benchmark_clip", None),
        source=getattr(args, "benchmark_source", None),
        threads=getattr(args, "benchmark_threads", None),
        update_baseline=bool(getattr(args, "benchmark_update_baseline", False)))
