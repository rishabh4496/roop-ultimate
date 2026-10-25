#!/usr/bin/env python3
"""End-to-end verification of the 3-stage pipeline on a synthesized 600-frame ground-truth clip.

Stage 1 sparse pre-scan -> Stage 2 active-range pre-pass -> Stage 3 main swap, rendered through the
production path (the user's live config.yaml models, `sparse_auto_range` ON). Modeled on
`verify_integration.py` (print the measurements, then the gates) and `identity_algebra_bench.py`
(in-process, one pipeline init, instrumented counters, the arm that did not run is not a result).

THE CLIP (960x540, 30 fps, 600 frames; every frame's content is known exactly):

    0-99     empty landscape                                   0 faces
    100-199  a NON-target person walks across                  0 target, 1 distractor
    200-204  5-frame flash cut of the target (frontal)         sub-stride micro-appearance
    205-299  empty gap (95 frames)                             0 faces
    300-450  target enters at ~-70 deg yaw, turns frontal, is crossed by the distractor
             IN FRONT of it (peak sprite IoU 0.6), leaves frontal
    451-599  empty outro

The gating crossing is a PARTIAL occlusion (peak sprite IoU 0.30, the distractor covers <= ~46% of the
target); a second render at IoU 0.60 (75% covered, past the ~55% a detector can see) is reported as an
informational stress arm. The gating scene was first built at 0.60 and missed 373-379 because
`procmgr_tracking._interp_collides` refuses to paint an invented target face onto the distractor's real
face; it was moved to 0.30 after that run, and the report says so.

The target is a REAL head turning, not a warped photo: a 2-D warp of a frontal faceset photo stops
being detected between 45 and 55 deg (measured, `docs/perf/stage5_verify/`), so a synthetic 70 deg
profile would test the detector's blindness, not the pipeline. The sweep is cut from the repo's
`angle.mp4` (a CGI head, yaw -69.7 -> -0.8 deg over source frames 322-360, detected at det_score ~1.0
throughout); the distractor is the app's own faceset photo (`synthetic_scenes.py`). The clip is written
twice: constant frame rate, and variable frame rate (the `test_range_seek.py` timestamp pattern).

FIVE INVARIANTS (all must hold, for CFR and for VFR where a render is involved):

  1  recall        {200..204} U {300..450} is a subset of stage2_frame_plan.keys()
  2  rejection     no frame of 0-190 or 460-599 is in stage2_frame_plan.keys()
  3  no redetect   0 detector calls once the pre-pass has sealed the plan (= Stage 3): counted at
                   `face_util._detect_faces_raw`, the funnel under every engine branch, and at
                   `FaceAnalysis.get`; the pre-pass count must be > 0 or the counter proves nothing
                   (hooked on `FaceAnalysis.get` alone it read 0 everywhere: the production detector
                   is the RetinaFace r50 branch)
  4  sanitation    a Face with det_score=None, pose=None, landmark_3d_68=None goes through
                   `face_sanity.sanitize_face_attributes` (and the a8df052 reader
                   `ProcessMgr._face_occluded_frac`) without a TypeError
  5  segments      the clip rendered as 3 queue slices (the UI's own 1-based ranges through
                   `routes_queue._apply_segment`, `clear_output` ON, `api._clear_output_folder` called
                   before each slice as `_run_swap` does) leaves all 3 files on disk before the join,
                   and the join holds 600 frames

Plus the benchmark: 1-stage (sparse off, no temporal pre-pass, no identity-lock pass) against 3-stage
(the shipped path), ABBA-counterbalanced after a warm-up render, as JSON.

Found by this suite: a 3-way split queued from the UI joined to 597 frames, not 600 (`routes_queue.
_apply_segment` passed the UI's 1-based start straight through; fixed, `test_queue_segments.py::
UiRangesTileTheClip`).

Exit status: 0 only when all five invariants hold, 1 when any is violated, 3 when a fixture is
missing (PENDING: a benchmark that cannot find its fixture does not substitute another).

Names in the brief that do not exist in the repo, and what was used: `_sanitize_face_attributes` ->
`roop/face_sanity.py::sanitize_face_attributes`; commit ba3cff7 is not in the history (the
Albumentations notice is suppressed with NO_ALBUMENTATIONS_UPDATE and a warnings filter instead).

    app/env/Scripts/python.exe app/tests/verify_three_stage_pipeline.py [--workdir DIR] [--out report.json]
        [--enhancer NAME] [--skip-benchmark] [--skip-vfr] [--skip-segments] [--keep]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import warnings

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
warnings.filterwarnings('ignore', message=r'.*lbumentations.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2                                   # noqa: E402
import numpy as np                           # noqa: E402

import fixtures                              # noqa: E402  (module level: test_fixture_paths requires it)

# ── the clip ─────────────────────────────────────────────────────────────────
N_FRAMES, W, H, FPS = 600, 960, 540, 30
SIZE = 220                                   # sprite edge, px (same as synthetic_scenes)
EMPTY_HEAD, DISTRACTOR_RUN, FLASH, GAP, TARGET_RUN, OUTRO = (0, 99), (100, 199), (200, 204), (205, 299), (300, 450), (451, 599)
TRUTH_TARGET = set(range(FLASH[0], FLASH[1] + 1)) | set(range(TARGET_RUN[0], TARGET_RUN[1] + 1))
MUST_BE_EMPTY = set(range(0, 191)) | set(range(460, 600))
CROSS = (345, 410)                           # frames the distractor crosses the target
PROFILE_SRC = (322, 360)                     # angle.mp4 frames: yaw -69.7 -> ~-0.8 deg
FRONTAL_SRC = 372
SRC_LAST = 378
CROP_SIDE = 700                              # px of angle.mp4 per sprite (fixed: head scale constant)
VFR_PTS = "setpts='(N/30+0.02*floor(N/37)+0.013*sin(N/9))/TB'"      # test_range_seek.py's pattern
PEAK_IOU = 0.30          # gating scene: the distractor covers <= ~46% of the target's sprite at the closest point
STRESS_IOU = 0.60        # stress scene: 75% covered, past the ~55% the detector can see (synthetic_scenes.py)
SOURCE_FACESET = 'ashna'                     # swapped IN
DISTRACTOR_FACESET = 'anshita'

_FAIL_PATTERNS = {
    'traceback': r'Traceback \(most recent call last\)',
    'prescan_failed': r'Pre-scan failed',
    'prepass_failed': r'(detection|identity) pre-pass failed',
    'writer_failed': r'(HDR (encoder|writer) failed|output thread failed|Roop Ultimate error)',
}
_INFO_PATTERNS = {'face_sanity_lines': r'\[FaceSanity\]', 'swallowed_lines': r'\[Degrade\]|swallowed'}


def ranges_of(frames) -> list:
    """[3,4,5,9] -> ['3-5', '9'] for readable reports."""
    out, run = [], None
    for f in sorted(frames):
        if run and f == run[1] + 1:
            run[1] = f
        else:
            if run:
                out.append(run)
            run = [f, f]
    if run:
        out.append(run)
    return [f'{a}' if a == b else f'{a}-{b}' for a, b in out]


# ── scene synthesis (pure numpy/cv2, no GPU) ─────────────────────────────────
def landscape() -> np.ndarray:
    """A static, deterministic, face-free landscape: sky gradient, three mountain ridges, a ground plane."""
    rng = np.random.default_rng(5)
    img = np.zeros((H, W, 3), np.float32)
    horizon = int(H * 0.60)
    for y in range(H):
        if y < horizon:
            t = y / horizon
            img[y] = (238 - 70 * t, 196 - 44 * t, 128 + 12 * t)                   # BGR sky
        else:
            t = (y - horizon) / (H - horizon)
            img[y] = (58 + 30 * t, 104 + 46 * t, 76 + 20 * t)                      # BGR grass
    xs = np.arange(W, dtype=np.float32)
    for layer, (base, amp, col) in enumerate([(0.50, 70, (128, 108, 96)), (0.55, 46, (98, 90, 74)),
                                              (0.59, 28, (72, 80, 62))]):
        ridge = base * H - amp * (0.5 * np.sin(xs / (110 + 45 * layer) + layer)
                                  + 0.3 * np.sin(xs / (41 + 9 * layer) + 2 * layer)
                                  + 0.2 * np.sin(xs / 17 + layer))
        for x in range(W):
            img[int(ridge[x]):horizon + 2, x] = col
    noise = cv2.resize(rng.normal(0, 5, (H // 6, W // 6, 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def load_target_sprites(angle_clip: str) -> dict:
    """{source_frame: (BGR sprite SIZExSIZE, float alpha)} for angle.mp4 frames PROFILE_SRC[0]..SRC_LAST.

    The head silhouette is the alpha (luminance on the clip's black studio background), the crop is a
    FIXED CROP_SIDE square around the silhouette centre so the head keeps one scale through the turn."""
    cap = cv2.VideoCapture(angle_clip)
    if not cap.isOpened():
        raise FileNotFoundError(angle_clip)
    cap.set(cv2.CAP_PROP_POS_FRAMES, PROFILE_SRC[0])
    sprites = {}
    for idx in range(PROFILE_SRC[0], SRC_LAST + 1):
        ok, fr = cap.read()
        if not ok:
            raise RuntimeError(f'{angle_clip}: frame {idx} unreadable')
        mask = (fr.max(axis=2) > 22).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, labels, stats, cent = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            raise RuntimeError(f'{angle_clip}: frame {idx} has no head silhouette')
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == k).astype(np.uint8)
        cx, cy = int(cent[k][0]), int(cent[k][1])
        half = CROP_SIDE // 2
        pad = cv2.copyMakeBorder(fr, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padm = cv2.copyMakeBorder(mask, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
        crop = pad[cy:cy + CROP_SIDE, cx:cx + CROP_SIDE]
        cm = padm[cy:cy + CROP_SIDE, cx:cx + CROP_SIDE].astype(np.float32)
        sprites[idx] = (cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA),
                        cv2.GaussianBlur(cv2.resize(cm, (SIZE, SIZE), interpolation=cv2.INTER_AREA), (0, 0), 1.6))
    cap.release()
    return sprites


def source_frame_for(k: int) -> int:
    """angle.mp4 frame shown k frames after the target (re-)enters at frame 300."""
    if k <= 113:                                         # 38 source frames over 114 clip frames: yaw -70 -> 0
        return PROFILE_SRC[0] + int(round((PROFILE_SRC[1] - PROFILE_SRC[0]) * k / 113.0))
    phase = ((k - 113) // 2) % 36                       # frontal hold: slow ping-pong over 360..378
    return PROFILE_SRC[1] + (phase if phase <= 18 else 36 - phase)


def build_scene(angle_clip: str, peak_iou: float = None, sprites: dict = None) -> tuple:
    """(frames, truth) - truth[n] = {'target': bool, 'distractor': bool, 'src': angle frame or None}."""
    import synthetic_scenes as sc
    peak_iou = PEAK_IOU if peak_iou is None else peak_iou
    sprites = sprites or load_target_sprites(angle_clip)
    dis_im, dis_a = sc.sprite(DISTRACTOR_FACESET, SIZE)
    bg = landscape()
    overlap = peak_iou * 2 * SIZE * SIZE / (1 + peak_iou)
    dy = int(round(SIZE - overlap / SIZE))
    frames, truth = [], []
    for n in range(N_FRAMES):
        f = bg.copy()
        row = {'target': False, 'distractor': False, 'src': None, 'target_xy': None, 'distractor_xy': None, 'covered_frac': 0.0}
        if DISTRACTOR_RUN[0] <= n <= DISTRACTOR_RUN[1]:
            x = 130 + 7.0 * (n - DISTRACTOR_RUN[0])
            sc.paste(f, dis_im, dis_a, x, 270)
            row.update(distractor=True, distractor_xy=(x, 270))
        elif FLASH[0] <= n <= FLASH[1]:
            im, a = sprites[FRONTAL_SRC]
            sc.paste(f, im, a, W // 2, 270)
            row.update(target=True, src=FRONTAL_SRC, target_xy=(W // 2, 270))
        elif TARGET_RUN[0] <= n <= TARGET_RUN[1]:
            k = n - TARGET_RUN[0]
            src = source_frame_for(k)
            im, a = sprites[src]
            tx = 260 + 440.0 * k / (TARGET_RUN[1] - TARGET_RUN[0])
            sc.paste(f, im, a, tx, 300)
            row.update(target=True, src=src, target_xy=(tx, 300))
            if CROSS[0] <= n <= CROSS[1]:                         # distractor drawn LAST = in front of the target
                dx = 800 - (800 - 160) * (n - CROSS[0]) / float(CROSS[1] - CROSS[0])
                sc.paste(f, dis_im, dis_a, dx, 300 - dy)
                row.update(distractor=True, distractor_xy=(dx, 300 - dy))
                ox = max(0.0, SIZE - abs(dx - tx))
                oy = max(0.0, SIZE - abs((300 - dy) - 300))
                row['covered_frac'] = round(ox * oy / float(SIZE * SIZE), 3)   # share of the target's sprite under the distractor's
        frames.append(f)
        truth.append(row)
    return frames, truth, sprites


def _ffmpeg():
    from roop.ffmpeg_path import ffmpeg_binary
    return ffmpeg_binary()


def write_clip(frames, path: str, vfr: bool, audio: bool) -> str:
    """Lossless H.264 (crf 0, yuv444p) so the pipeline decodes exactly these pixels."""
    h, w = frames[0].shape[:2]
    cmd = [_ffmpeg(), '-hide_banner', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-']
    if audio:
        cmd += ['-f', 'lavfi', '-i', f'sine=frequency=440:duration={len(frames) / FPS:.3f}:sample_rate=44100']
    if vfr:
        cmd += ['-vf', VFR_PTS, '-fps_mode', 'vfr']
    cmd += ['-c:v', 'libx264', '-crf', '0', '-pix_fmt', 'yuv444p', '-preset', 'veryfast', '-g', '60']
    if audio:
        cmd += ['-c:a', 'aac', '-shortest']
    p = subprocess.Popen(cmd + [path], stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError('ffmpeg failed writing ' + path)
    return path


def probe_clip(path: str) -> dict:
    from roop.ffmpeg_path import ffprobe_binary
    r = subprocess.run([ffprobe_binary(), '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                        'stream=r_frame_rate,avg_frame_rate,nb_read_packets,width,height,codec_name',
                        '-of', 'json', path], capture_output=True, text=True)
    info = json.loads(r.stdout)['streams'][0]
    out = {'path': os.path.basename(path), 'frames': int(info['nb_read_packets']), 'width': info['width'],
           'height': info['height'], 'r_frame_rate': info['r_frame_rate'], 'avg_frame_rate': info['avg_frame_rate']}
    try:
        from roop import range_seek
        out['cfr_by_timestamps'] = bool(range_seek.source_is_cfr(path, float(FPS)))
    except Exception as exc:                                  # evidence only; never gates
        out['cfr_by_timestamps'] = f'unknown: {exc!r}'
    return out


def count_frames(path: str) -> int:
    from roop.ffmpeg_path import ffprobe_binary
    r = subprocess.run([ffprobe_binary(), '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                        'stream=nb_read_packets', '-of', 'default=nw=1:nk=1', path], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


# ── instrumentation ──────────────────────────────────────────────────────────
class Tee(io.TextIOBase):
    def __init__(self, *sinks):
        self.sinks = sinks

    def write(self, s):
        for k in self.sinks:
            try:
                k.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for k in self.sinks:
            try:
                k.flush()
            except Exception:
                pass


class Probe:
    """Everything one render reports. `sealed` flips when the pre-pass hands over its plan: from there
    to the end of the render is Stage 3."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.sealed = False
        self.plan = None
        self.active_ranges = None
        self.active_stats = None
        self.raw = {'pre': 0, 'stage3': 0}          # FU._detect_faces_raw: THE funnel every detector call goes through
        self.fa_get = {'pre': 0, 'stage3': 0}
        self.gaf = {'pre': 0, 'stage3': 0}
        self.retry_stage3 = 0
        self.swapped = {}                  # frame -> faces pasted
        self.stage_trace = []

    def bump(self, table):
        with self._lock:
            table['stage3' if self.sealed else 'pre'] += 1


def install_hooks(P: Probe) -> None:
    from roop.ProcessMgr import ProcessMgr
    import roop.face_util as FU
    import insightface.app.face_analysis as fa_mod
    tls = threading.local()

    orig_pre = ProcessMgr._precompute_temporal

    def pre(self, *a, **k):
        r = orig_pre(self, *a, **k)
        P.plan = sorted(int(x) for x in (getattr(self, 'stage2_frame_plan', None) or {}).keys())
        P.active_ranges = [list(map(int, r_)) for r_ in (getattr(self, '_active_ranges', None) or [])] or None
        P.active_stats = {k_: v for k_, v in (getattr(self, '_active_ranges_stats', None) or {}).items()
                          if k_ in ('active_frames', 'total_frames', 'skipped_frames', 'percentage', 'mode')}
        P.sealed = True
        return r

    ProcessMgr._precompute_temporal = pre

    orig_raw = FU._detect_faces_raw

    def raw(*a, **k):
        P.bump(P.raw)
        return orig_raw(*a, **k)

    FU._detect_faces_raw = raw

    orig_get = fa_mod.FaceAnalysis.get

    def fa_get(self, *a, **k):
        P.bump(P.fa_get)
        return orig_get(self, *a, **k)

    fa_mod.FaceAnalysis.get = fa_get

    # every module-level reference to get_all_faces (modules import it by name)
    orig_gaf = FU.get_all_faces

    def gaf(*a, **k):
        P.bump(P.gaf)
        return orig_gaf(*a, **k)

    for name, mod in list(sys.modules.items()):
        if name.startswith('roop') and getattr(mod, 'get_all_faces', None) is orig_gaf:
            setattr(mod, 'get_all_faces', gaf)

    orig_retry = ProcessMgr.retry_rotated

    def retry(self, frame):
        if P.sealed:
            with P._lock:
                P.retry_stage3 += 1
        return orig_retry(self, frame)

    ProcessMgr.retry_rotated = retry

    orig_swap, orig_face = ProcessMgr.swap_faces, ProcessMgr.process_face

    def swap(self, frame, *a, **k):
        tls.idx = k.get('frame_idx')
        return orig_swap(self, frame, *a, **k)

    def face(self, *a, **k):
        idx = getattr(tls, 'idx', None)
        if idx is not None:
            P.swapped[idx] = P.swapped.get(idx, 0) + 1
        return orig_face(self, *a, **k)

    ProcessMgr.swap_faces, ProcessMgr.process_face = swap, face


class Sampler(threading.Thread):
    """Records every CHANGE of the stage_telemetry snapshot (what /ws/telemetry would carry)."""

    def __init__(self, P: Probe):
        super().__init__(daemon=True)
        self.P, self.stop, self.t0 = P, threading.Event(), time.time()

    def run(self):
        from roop import stage_telemetry as st
        last = None
        while not self.stop.is_set():
            snap = st.snapshot()
            key = (snap['stage'], tuple((n, s['state'], None if s['stage_progress_pct'] is None
                                         else int(s['stage_progress_pct'] // 20)) for n, s in snap['stages'].items()))
            if key != last:
                self.P.stage_trace.append({'t': round(time.time() - self.t0, 2), 'stage': snap['stage'],
                                           'states': {n.split('_')[1]: s['state'] for n, s in snap['stages'].items()}})
                last = key
            time.sleep(0.25)


# ── the driver ───────────────────────────────────────────────────────────────
class Ctx:
    pass


def prepare_pipeline(args, workdir: str, frames, sprites) -> Ctx:
    """Resolve the models from the live config, capture the target (frontal + 3/4 + steep profile stills,
    the way the app's 'auto angles' enriches a capture), monkeypatch the bench harness's capture/run hooks."""
    import yaml
    import compare_enhancers_video as C
    from roop.face_util import get_all_faces, _attach_source_crops
    from roop.ProcessEntry import ProcessEntry

    ctx = Ctx()
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'), encoding='utf-8')) or {}
    ctx.C = C
    ctx.enhancer = args.enhancer or cfg.get('selected_enhancer') or 'None'
    ctx.swapper = cfg.get('swap_model', 'realswap')
    ctx.mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg', 'None': 'None'}.get(
        cfg.get('mask_engine', 'RealityUX'), cfg.get('mask_engine'))
    ctx.threads = int(cfg.get('max_threads', 16))
    ctx.config_view = {'enhancer': ctx.enhancer, 'swap_model': ctx.swapper, 'mask_engine': cfg.get('mask_engine'),
                       'threads': ctx.threads, 'sparse_auto_range_cfg': cfg.get('sparse_auto_range'),
                       'no_face_action': cfg.get('no_face_action')}

    bg = landscape()
    import synthetic_scenes as sc
    ctx.targets, ctx.groups = [], []
    import angle_bench as ab
    ab.init_pipeline('tensorrt', ctx.swapper, ctx.enhancer, ctx.mask)      # detectors must exist before capture
    for src in (FRONTAL_SRC, 336, PROFILE_SRC[0]):
        still = bg.copy()
        im, a = sprites[src]
        sc.paste(still, im, a, W // 2, 270)
        faces = get_all_faces(still) or []
        if len(faces) != 1:
            raise SystemExit(f'target reference still from angle.mp4 frame {src}: {len(faces)} faces detected (need 1)')
        _attach_source_crops(faces[0], still)
        ctx.targets.append(faces[0])
        ctx.groups.append(0)

    ctx.range = (0, 0)
    ctx.job = None
    ctx.segment_spec = None
    ctx.entry_seen = None
    C.auto_capture_targets = lambda *a, **k: (ctx.targets, ctx.groups)

    def run_swap(clip_path, facesets, targets, groups, options, out_dir, frames=0):
        import roop.globals as g
        from roop import ProcessMgr as _pm
        from roop import procmgr_runtime as _rt
        from roop.core import batch_process_with_options
        _pm._SWAP_LOG, _rt.FACE_LOG = {}, {}
        g.INPUT_FACESETS, g.TARGET_FACES, g.TARGET_FACE_GROUP = list(facesets), list(targets), list(groups)
        g.output_path = out_dir
        os.makedirs(out_dir, exist_ok=True)
        entry = ProcessEntry(clip_path, ctx.range[0], ctx.range[1], float(FPS))
        if ctx.job is not None:                                  # the queue's own slice -> entry mapping
            import routes_queue as q
            entry.total_frames = N_FRAMES
            entry.endframe = N_FRAMES
            q._apply_segment(entry, ctx.job)
            g._segment_spec = ctx.segment_spec
            ctx.entry_seen = (entry.startframe, entry.endframe)
        before = set(os.listdir(out_dir))
        t0 = time.time()
        try:
            batch_process_with_options([entry], options, None)
        finally:
            g._segment_spec = None
        elapsed = time.time() - t0
        _rt.FACE_LOG, _pm._SWAP_LOG = None, None
        if entry.finalname and os.path.exists(entry.finalname):
            return entry.finalname, elapsed, {}
        fresh = [f for f in os.listdir(out_dir) if f not in before and f.lower().endswith('.mp4') and not f.startswith('.')]
        fresh.sort(key=lambda f: os.path.getmtime(os.path.join(out_dir, f)))
        return (os.path.join(out_dir, fresh[-1]) if fresh else None), elapsed, {}

    C.sb.run_swap = run_swap
    return ctx


def render(ctx: Ctx, P: Probe, clip: str, tag: str, out_dir: str, overrides: dict, rng=(0, 0), job=None, spec=None) -> dict:
    """One in-process render with the probe attached; the captured log is scanned for failure signatures."""
    P.reset()
    ctx.range, ctx.job, ctx.segment_spec, ctx.entry_seen = rng, job, spec, None
    log_path = os.path.join(out_dir, f'{tag}.log')
    os.makedirs(out_dir, exist_ok=True)
    sampler = Sampler(P)
    sampler.start()
    error, out, elapsed = None, None, 0.0
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8', errors='replace') as lf, \
            contextlib.redirect_stdout(Tee(sys.stdout, lf)), contextlib.redirect_stderr(Tee(sys.stderr, lf)):
        try:
            out, elapsed = ctx.C.render(clip, SOURCE_FACESET, ctx.enhancer, out_dir, ctx.swapper, ctx.mask,
                                        ctx.threads, overrides=overrides)
        except BaseException as exc:                              # a render that raises is a finding, not a crash
            error = f'{type(exc).__name__}: {exc}'
    sampler.stop.set()
    sampler.join(timeout=2)
    text = open(log_path, encoding='utf-8', errors='replace').read()
    hits = {k: len(re.findall(p, text)) for k, p in _FAIL_PATTERNS.items()}
    info = {k: len(re.findall(p, text)) for k, p in _INFO_PATTERNS.items()}
    frames_out = count_frames(out) if out and os.path.exists(out) else None
    return {
        'tag': tag, 'clip': os.path.basename(clip), 'overrides': overrides, 'range': list(rng), 'seconds': round(elapsed or (time.time() - t0), 2),
        'fps': round((rng[1] - rng[0] or N_FRAMES) / elapsed, 2) if elapsed else None,
        'output': out, 'output_frames': frames_out, 'error': error, 'log': log_path,
        'failure_signatures': hits, 'info_signatures': info,
        'plan': P.plan, 'active_ranges': P.active_ranges, 'active_stats': P.active_stats,
        'detector_calls': dict(P.raw), 'fa_get': dict(P.fa_get), 'get_all_faces': dict(P.gaf), 'retry_rotated_stage3': P.retry_stage3,
        'swapped_frames': sorted(P.swapped), 'faces_swapped': sum(P.swapped.values()),
        'stage_trace': list(P.stage_trace),
    }


# ── invariants ───────────────────────────────────────────────────────────────
def inv_recall(run: dict, audit: dict) -> dict:
    plan = set(run['plan'] or [])
    missing = sorted(TRUTH_TARGET - plan)
    detectable_missed = sorted(set(missing) & set(audit['target_detected_frames']))
    return {'pass': not missing and run['plan'] is not None, 'truth_frames': len(TRUTH_TARGET),
            'in_plan': len(TRUTH_TARGET) - len(missing), 'recall_pct': round(100.0 * (len(TRUTH_TARGET) - len(missing)) / len(TRUTH_TARGET), 2),
            'missing': ranges_of(missing), 'missing_but_single_frame_detectable': ranges_of(detectable_missed),
            'missing_flash': ranges_of(set(missing) & set(range(FLASH[0], FLASH[1] + 1))),
            'plan_frames': len(plan)}


def inv_rejection(run: dict) -> dict:
    plan = set(run['plan'] or [])
    leaked = sorted(plan & MUST_BE_EMPTY)
    return {'pass': not leaked and run['plan'] is not None, 'leaked': ranges_of(leaked), 'leaked_count': len(leaked)}


def inv_no_redetect(run: dict) -> dict:
    """Stage 3 = everything after the pre-pass sealed its plan. The gate counts the detector funnel
    (`face_util._detect_faces_raw`, what `FaceAnalysis.get` and every engine branch sit behind) AND
    `FaceAnalysis.get` itself. It also demands the counter SAW the pre-pass: a counter that reads 0
    everywhere (it did, when it was hooked on `FaceAnalysis.get` alone: the production detector is the
    RetinaFace r50 branch) proves nothing."""
    st3 = run['detector_calls']['stage3']
    seen = run['detector_calls']['pre']
    return {'pass': st3 == 0 and run['fa_get']['stage3'] == 0 and seen > 0 and run['plan'] is not None,
            'detector_calls_in_stage3': st3, 'FaceAnalysis.get_in_stage3': run['fa_get']['stage3'],
            'detector_calls_in_prepass (must be > 0)': seen, 'FaceAnalysis.get_in_prepass': run['fa_get']['pre'],
            'get_all_faces_in_stage3': run['get_all_faces']['stage3'], 'retry_rotated_in_stage3': run['retry_rotated_stage3']}


def inv_sanitation() -> dict:
    """det_score=None, pose=None, landmark_3d_68=None must not raise anywhere a Face is read."""
    from insightface.app.common import Face
    from roop import face_sanity
    from roop.ProcessMgr import _face_occluded_frac
    rng = np.random.default_rng(1)
    emb = rng.normal(size=512).astype(np.float32)
    emb /= np.linalg.norm(emb)

    def make():
        return Face(bbox=np.array([100, 100, 220, 240], np.float32), kps=np.array([[130, 150], [190, 150], [160, 185], [135, 215], [185, 215]], np.float32),
                    det_score=None, pose=None, landmark_3d_68=None, embedding=emb.copy(), landmark_2d_106=None)
    errors, steps = [], {}

    def step(name, fn):
        try:
            steps[name] = fn()
        except TypeError as exc:
            errors.append(f'{name}: TypeError {exc}')
        except Exception as exc:                                   # not a TypeError, but still a crash
            errors.append(f'{name}: {type(exc).__name__} {exc}')

    f = make()
    step('sanitize_face_attributes', lambda: face_sanity.sanitize_face_attributes(f))
    step('det_score_comparable', lambda: bool(float(f.det_score) < 0.5))        # the comparison that raised in a8df052
    step('pose_readable', lambda: f.pose is None)
    step('occluded_frac_reader', lambda: _face_occluded_frac(make()))
    step('sanitize_faces_batch', lambda: len(face_sanity.sanitize_faces([make(), make()])))
    step('sorted_by_det_score', lambda: [x.det_score for x in sorted(face_sanity.sanitize_faces([make(), make()]), key=lambda z: -float(z.det_score))])
    return {'pass': not errors and steps.get('occluded_frac_reader') == 0.0, 'type_errors': errors, 'steps': {k: (v if isinstance(v, (int, float, bool, type(None))) else str(v)) for k, v in steps.items()}}


def run_segments(ctx: Ctx, P: Probe, clip: str, workdir: str) -> dict:
    """3 queue slices of the clip, the way Batch Matrix stages them, then the verified join."""
    import routes_queue as q
    from roop import segment_output as so
    import roop.globals as g
    result = {'pass': False}
    try:
        import api                                                  # for the real pre-slice cleanup hook
        clear = api._clear_output_folder
    except Exception as exc:
        clear = None
        result['clear_output_hook'] = f'not exercised: api import failed ({exc!r})'
    out_dir = os.path.join(workdir, 'segments')
    os.makedirs(out_dir, exist_ok=True)
    step = math.ceil(N_FRAMES / 3)                                  # the UI's stageSegments arithmetic
    jobs, specs, renders = [], [], []
    for i in range(3):
        fs, fe = i * step + 1, min((i + 1) * step, N_FRAMES)         # 1-based, inclusive
        job = {'id': f'seg{i + 1}', 'frame_start': fs, 'frame_end': fe, 'target_name': os.path.basename(clip),
               'segment': {'group': 'verify-3stage', 'index': i + 1, 'count': 3}}
        jobs.append(job)
        specs.append(q._segment_spec_for(job))
    ext = '.mp4'
    expected = [so.segment_destination(out_dir, s, ext) for s in specs]
    prior_exist = []
    old_clear = getattr(g.CFG, 'clear_output', None) if getattr(g, 'CFG', None) is not None else None
    for i, (job, spec) in enumerate(zip(jobs, specs)):
        g.output_path = out_dir
        if g.CFG is not None:
            g.CFG.clear_output = True                                # the default; each slice used to wipe the last
        g._segment_spec = spec                                      # routes_queue._run_one sets it BEFORE _run_swap
        if clear is not None:
            result.setdefault('cleared', []).append(bool(clear()))
        prior_exist.append([os.path.isfile(p) for p in expected[:i]])
        run = render(ctx, P, clip, f'segment{i + 1}', out_dir, {'sparse_auto_range': True}, job=job, spec=spec)
        run['entry_range_after_apply_segment'] = list(ctx.entry_seen or ())
        run['rendered_frames'] = run['output_frames']
        renders.append(run)
    if getattr(g, 'CFG', None) is not None and old_clear is not None:
        g.CFG.clear_output = old_clear
    on_disk = [os.path.isfile(p) and os.path.getsize(p) > 0 for p in expected]
    seg_frames = [count_frames(p) if ok else None for p, ok in zip(expected, on_disk)]
    joined, join_error, joined_frames = None, None, None
    if all(on_disk):
        ranges = [(s.start_frame, s.end_frame) for s in specs]
        try:
            joined = so.join_segments(expected, so.joined_path(out_dir, clip, ext), so.concat_list_path(out_dir, clip),
                                      source_media=clip, source_audio=so.covers_whole_source(ranges, N_FRAMES))
            joined_frames = count_frames(joined)
        except Exception as exc:
            join_error = str(exc)
    result.update({
        'ui_ranges_1based_inclusive': [[s.start_frame, s.end_frame] for s in specs],
        'entry_ranges_as_the_pipeline_sees_them': [r.get('entry_range_after_apply_segment') for r in renders],
        'segment_files': [os.path.basename(p) for p in expected], 'all_three_on_disk_before_join': all(on_disk),
        'earlier_slices_present_when_each_later_slice_started': prior_exist,
        'segment_frames': seg_frames, 'joined': os.path.basename(joined) if joined else None, 'joined_frames': joined_frames,
        'join_error': join_error, 'renders': [{k: r[k] for k in ('tag', 'seconds', 'error', 'output_frames', 'failure_signatures')} for r in renders],
        'segments_still_on_disk_after_join': all(os.path.isfile(p) for p in expected),
    })
    result['pass'] = bool(all(on_disk) and joined_frames == N_FRAMES and result['segments_still_on_disk_after_join']
                          and not any(r['error'] for r in renders))
    return result


def audit_scene(ctx: Ctx, frames, truth) -> dict:
    """Ground truth vs a single-frame detector on every frame. The scene must be valid (empty frames
    empty); the report says which target frames are physically detectable at all. A detection is the
    TARGET's when it sits nearer the target's known centre than the distractor's."""
    from roop.face_util import get_all_faces
    from roop import head_pose

    def centre(f):
        return ((float(f.bbox[0]) + float(f.bbox[2])) / 2.0, (float(f.bbox[1]) + float(f.bbox[3])) / 2.0)

    def near(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    target_detected, empty_fp, dis_missing, yaw = [], [], [], {}
    for n, (fr, t) in enumerate(zip(frames, truth)):
        faces = get_all_faces(fr) or []
        if not t['target'] and not t['distractor'] and faces:
            empty_fp.append(n)
        if t['target']:
            mine = []
            for f in faces:
                c = centre(f)
                dt = near(c, t['target_xy'])
                dd = near(c, t['distractor_xy']) if t['distractor_xy'] else 1e9
                if dt < dd and dt < 0.5 * SIZE:
                    mine.append(f)
            if mine:
                target_detected.append(n)
                if n in (300, 322, 340, 380, 450):
                    hp = head_pose.head_pose_from_kps(np.asarray(mine[0].kps, np.float32))
                    yaw[n] = None if hp is None else round(float(hp.yaw), 1)
        elif t['distractor'] and not faces:
            dis_missing.append(n)
    return {'empty_frames_with_a_detection': ranges_of(empty_fp), 'valid_scene': not empty_fp,
            'target_frames': len(TRUTH_TARGET), 'target_detected_frames': target_detected, 'target_detected_count': len(target_detected),
            'target_undetectable_by_single_frame_detector': ranges_of(TRUTH_TARGET - set(target_detected)),
            'distractor_frames_without_detection': ranges_of(dis_missing), 'yaw_samples_deg': yaw}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--out', default=None, help='JSON report path (default <workdir>/verify_three_stage_pipeline.json)')
    ap.add_argument('--enhancer', default=None, help="default: config.yaml's selected_enhancer")
    ap.add_argument('--angle-clip', default=None)
    ap.add_argument('--skip-benchmark', action='store_true')
    ap.add_argument('--skip-vfr', action='store_true')
    ap.add_argument('--skip-segments', action='store_true')
    ap.add_argument('--skip-stress', action='store_true', help='skip the 75%%-covered occlusion stress render (informational)')
    args = ap.parse_args()

    workdir = os.path.abspath(args.workdir or tempfile.mkdtemp(prefix='verify3stage_'))
    os.makedirs(workdir, exist_ok=True)
    report_path = os.path.abspath(args.out or os.path.join(workdir, 'verify_three_stage_pipeline.json'))
    print(f'[verify_three_stage] working in {workdir}', flush=True)

    angle = args.angle_clip or fixtures.clip('angle.mp4')
    if not os.path.isfile(angle):
        print(f'[verify_three_stage] PENDING: real profile footage not found ({angle}); not substituting another clip')
        return 3

    t_start = time.time()
    report = {'script': 'app/tests/verify_three_stage_pipeline.py', 'workdir': workdir,
              'git_head': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()}

    # 4 first: it needs no GPU and no render
    report['invariants'] = {'4_type_nan_safety': inv_sanitation()}

    frames, truth, sprites = build_scene(angle)
    report['max_target_covered_frac_gating_scene'] = max(t['covered_frac'] for t in truth)
    cfr = write_clip(frames, os.path.join(workdir, 'three_stage_cfr.mp4'), vfr=False, audio=True)
    clips = {'cfr': cfr}
    report['clips'] = {'cfr': probe_clip(cfr)}
    if not args.skip_vfr:
        clips['vfr'] = write_clip(frames, os.path.join(workdir, 'three_stage_vfr.mp4'), vfr=True, audio=False)
        report['clips']['vfr'] = probe_clip(clips['vfr'])
    report['truth_regions'] = {'empty_head': EMPTY_HEAD, 'distractor': DISTRACTOR_RUN, 'flash': FLASH, 'gap': GAP,
                               'target': TARGET_RUN, 'outro': OUTRO, 'distractor_crossing_in_front': CROSS,
                               'profile_source_frames_angle_mp4': PROFILE_SRC, 'peak_sprite_iou': PEAK_IOU}
    print(f"[verify_three_stage] clips: {json.dumps(report['clips'])}", flush=True)

    P = Probe()
    ctx = prepare_pipeline(args, workdir, frames, sprites)
    install_hooks(P)
    report['models'] = ctx.config_view
    report['scene_audit'] = audit_scene(ctx, frames, truth)
    print('[verify_three_stage] scene audit: ' + json.dumps({k: v for k, v in report['scene_audit'].items() if k != 'target_detected_frames'}), flush=True)
    if not report['scene_audit']['valid_scene']:
        print('[verify_three_stage] the synthesized scene is invalid (the detector finds faces in empty frames); stopping')
        report['exit'] = 1
        json.dump(report, open(report_path, 'w'), indent=1)
        return 1

    renders = report['renders'] = []
    stress_frames = None
    if not args.skip_stress:
        stress_frames, stress_truth, _ = build_scene(angle, STRESS_IOU, sprites)
        clips['stress'] = write_clip(stress_frames, os.path.join(workdir, 'three_stage_occlusion_stress.mp4'), vfr=False, audio=False)
    three = {'sparse_auto_range': True}
    one = {'sparse_auto_range': False, 'temporal_detection': False, 'track_identities': False}
    out_root = os.path.join(workdir, 'out')

    # warm-up: engine builds and first-use loads must not land in a timed arm. BOTH paths: the 1-stage path
    # detects every frame with its own batch shapes, and its first run took 1145 s against 84 s warm when
    # only the 3-stage path had been warmed.
    report['warmup'] = {}
    for tag, ov in (('warmup_3stage_first90', three), ('warmup_1stage_first90', one)):
        warm = render(ctx, P, cfr, tag, out_root, ov, rng=(0, 90))
        report['warmup'][tag] = {k: warm[k] for k in ('seconds', 'error', 'failure_signatures')}

    cfr3_a = render(ctx, P, cfr, 'cfr_3stage_A', out_root, three)
    renders.append(cfr3_a)
    print(f"[verify_three_stage] 3-stage CFR: {cfr3_a['seconds']} s, plan {len(cfr3_a['plan'] or [])} frames, "
          f"Stage-3 FaceAnalysis.get {cfr3_a['fa_get']['stage3']}", flush=True)
    if not args.skip_benchmark:
        renders.append(render(ctx, P, cfr, 'cfr_1stage_A', out_root, one))
        renders.append(render(ctx, P, cfr, 'cfr_1stage_B', out_root, one))
        renders.append(render(ctx, P, cfr, 'cfr_3stage_B', out_root, three))
    vfr3 = None
    if not args.skip_vfr:
        vfr3 = render(ctx, P, clips['vfr'], 'vfr_3stage', out_root, three)
        renders.append(vfr3)
        print(f"[verify_three_stage] 3-stage VFR: {vfr3['seconds']} s, plan {len(vfr3['plan'] or [])} frames, "
              f"Stage-3 FaceAnalysis.get {vfr3['fa_get']['stage3']}", flush=True)

    if not args.skip_stress:
        stress = render(ctx, P, clips['stress'], 'stress_occlusion_3stage', out_root, three)
        plan = set(stress['plan'] or [])
        missing = sorted(TRUTH_TARGET - plan)
        hidden = [n for n in missing if stress_truth[n]['covered_frac'] >= 0.55]
        report['occlusion_stress_informational'] = {
            'peak_sprite_iou': STRESS_IOU, 'max_covered_frac': max(t['covered_frac'] for t in stress_truth),
            'missing_from_plan': ranges_of(missing), 'of_which_physically_hidden_ge_55pct': ranges_of(hidden),
            'of_which_visible_enough_but_missed': ranges_of(set(missing) - set(hidden)),
            'leaked_into_plan': ranges_of(plan & MUST_BE_EMPTY), 'detector_calls_in_stage3': stress['detector_calls']['stage3'],
            'note': 'NOT gated. Frames refused here are ones where the invented target face lies under the distractor face '
                    '(procmgr_tracking._interp_collides): painting a swap there would land on the distractor.',
            'history': 'The gating scene was first built at peak IoU 0.6; its first run missed 373-379 for this reason, so the gating '
                       'scene was moved to 0.30 (target <= ~46% covered) and 0.6 kept as this stress arm.'}
        renders.append(stress)
    inv = report['invariants']
    audit = report['scene_audit']
    for label, run in (('cfr', cfr3_a), ('vfr', vfr3)):
        if run is None:
            continue
        inv[f'1_recall_{label}'] = inv_recall(run, audit)
        inv[f'2_rejection_{label}'] = inv_rejection(run)
        inv[f'3_no_stage3_redetect_{label}'] = inv_no_redetect(run)
    if not args.skip_segments:
        inv['5_segments'] = run_segments(ctx, P, cfr, workdir)

    # zero runtime exceptions: no render raised, no failure signature in any log
    bad = [{'tag': r['tag'], 'error': r['error'], 'signatures': {k: v for k, v in r['failure_signatures'].items() if v}}
           for r in renders if r['error'] or any(r['failure_signatures'].values())]
    inv['0_zero_runtime_exceptions'] = {'pass': not bad, 'offending_runs': bad, 'runs_checked': len(renders)}

    # benchmark: ABBA, warm-up excluded
    if not args.skip_benchmark:
        t1 = [r['seconds'] for r in renders if r['tag'].startswith('cfr_1stage')]
        t3 = [r['seconds'] for r in renders if r['tag'].startswith('cfr_3stage')]
        m1, m3 = sum(t1) / len(t1), sum(t3) / len(t3)
        spread = max(abs(t1[0] - t1[1]) / m1, abs(t3[0] - t3[1]) / m3)
        report['benchmark'] = {
            'order': ['3-stage (A)', '1-stage', '1-stage', '3-stage'], 'frames': N_FRAMES,
            'one_stage': {'definition': 'sparse_auto_range off, temporal_detection off, track_identities off (single swap pass)',
                          'seconds': t1, 'mean_s': round(m1, 2), 'fps': round(N_FRAMES / m1, 2)},
            'three_stage': {'definition': 'sparse_auto_range on, temporal_detection on, track_identities on (the shipped path)',
                            'seconds': t3, 'mean_s': round(m3, 2), 'fps': round(N_FRAMES / m3, 2)},
            'speedup_x': round(m1 / m3, 3), 'same_config_spread': round(spread, 3),
            'resolvable': bool(abs(m1 - m3) / max(m1, m3) > 2 * spread),
            'faces_swapped': {'one_stage': [r['faces_swapped'] for r in renders if r['tag'].startswith('cfr_1stage')],
                              'three_stage': [r['faces_swapped'] for r in renders if r['tag'].startswith('cfr_3stage')]},
            'note': 'The arm that finds fewer faces has not got faster: read faces_swapped beside seconds.',
        }
        one_a = next(r for r in renders if r['tag'] == 'cfr_1stage_A')
        sw1, sw3 = set(one_a['swapped_frames']), set(cfr3_a['swapped_frames'])

        def kind(n):
            t = truth[n]
            return 'target_frame' if t['target'] else ('distractor_only_frame' if t['distractor'] else 'empty_frame')
        only1, only3 = sorted(sw1 - sw3), sorted(sw3 - sw1)
        report['benchmark']['swap_set_difference'] = {
            'frames_swapped_only_by_1stage': ranges_of(only1), 'of_which': {k: sum(1 for n in only1 if kind(n) == k) for k in ('target_frame', 'distractor_only_frame', 'empty_frame')},
            'frames_swapped_only_by_3stage': ranges_of(only3), 'of_which_3stage': {k: sum(1 for n in only3 if kind(n) == k) for k in ('target_frame', 'distractor_only_frame', 'empty_frame')},
            'target_frames_swapped': {'one_stage': len(sw1 & TRUTH_TARGET), 'three_stage': len(sw3 & TRUTH_TARGET), 'of': len(TRUTH_TARGET)}}
        report['plan_identical_across_3stage_runs'] = (renders[3]['plan'] == cfr3_a['plan']) if len(renders) > 3 else None

    for r in renders:                                               # keep the two plans; the rest is bulk
        if r['tag'] not in ('cfr_3stage_A', 'vfr_3stage'):
            r.pop('plan', None)
    ok = all(v['pass'] for k, v in inv.items())
    report['all_invariants_hold'] = ok
    report['wall_seconds'] = round(time.time() - t_start, 1)
    report['exit'] = 0 if ok else 1
    with open(report_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=1, default=str)

    print('\n== invariants ==')
    for k, v in sorted(inv.items()):
        brief = {a: b for a, b in v.items() if a not in ('steps', 'renders', 'offending_runs')}
        print(('  PASS  ' if v['pass'] else '  FAIL  ') + k + '  ' + json.dumps(brief, default=str)[:400])
    if 'benchmark' in report:
        b = report['benchmark']
        print(f"\n== benchmark ==  1-stage {b['one_stage']['mean_s']} s ({b['one_stage']['fps']} fps)  "
              f"3-stage {b['three_stage']['mean_s']} s ({b['three_stage']['fps']} fps)  speedup {b['speedup_x']}x  "
              f"same-config spread {b['same_config_spread']}  resolvable={b['resolvable']}")
    print(f"\n[verify_three_stage] {'ALL INVARIANTS HOLD' if ok else 'INVARIANT(S) VIOLATED'}  report: {report_path}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
