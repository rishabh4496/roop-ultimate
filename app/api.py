"""FastAPI backend for the React UI (react-ui/).

This runs in a daemon thread alongside the Gradio app (see run.py) and shares the
same roop.globals state. It mirrors the Gradio Face Swap / Face Manager / Settings /
Extras handlers as plain HTTP endpoints so the React front-end can reach full parity.

The Gradio UI (app/ui/) is the frozen legacy/backup UI and is NOT touched here.
"""

import os
import io
import collections
import hashlib
from collections import OrderedDict
import sys
import json
import shutil
import subprocess
import threading
import time
import traceback

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response

import api_state as state
from source_gallery import (
    _mask_offsets_from_cfg,
    estimate_face_pose_from_kps,
    _sources_append,
    _sources_pop,
    _sources_move,
    _sources_clear,
    _sources_desync,
    _get_source_faces_info,
    _source_faces_payload,
    _ingest_faceset,
)
from api_media import (_save_upload, _rgb_to_dataurl, _bgr_to_dataurl,
                       _bgr_to_preview_dataurl, _dataurl_to_bgr)
import roop.globals as roop_globals
from roop import utilities as util
from roop.face_util import extract_face_images, get_all_faces
from roop.FaceSet import FaceSet
from roop.capturer import get_video_frame, get_video_frame_total, get_image_frame
from roop.ProcessEntry import ProcessEntry
from roop import capture_seed
from roop import segment_writer
from roop import live_preview
from roop import procmgr_runtime as _procmgr_runtime
import ui.globals as ui_globals

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def mapped_facesets(mapping, swap_mode=""):
    """Reorder the loaded source facesets into person order for a swap.

    `mapping[rank]` is the source faceset index chosen for target person `rank`,
    so the swap can address a source by person rank. Returns None when there is
    no mapping (the caller then uses INPUT_FACESETS as-is).

    Not every swap mode addresses sources by person. "All input faces" walks the
    detected faces and the source list in lockstep by POSITION — its whole
    contract is "one input face per detected face, in gallery order" — so a
    person-ordered list is meaningless to it and, worse, is only as long as the
    number of captured people: with three sources loaded and one person
    captured it silently swapped one face instead of three. That mode therefore
    opts out and keeps the gallery order. The modes that DO use the mapping are
    "Selected face" (indexes by person rank) and the single-source modes (whose
    gallery index is translated by mapped_selected_index below).

    This returns a NEW list and never touches roop_globals.INPUT_FACESETS. An
    earlier version swapped the global out for the duration of the swap and
    restored it afterwards — but a swap holds that window for seconds (preview)
    to minutes (a run), and /api/source/* mutates the same global from another
    thread. An upload landing inside the window appended to the temporary list
    and was then thrown away by the restore, while its thumbnail (a separate
    list) survived: the gallery showed the new face but the backend no longer
    had it, so that person swapped with an empty faceset — i.e. stayed
    un-swapped — until the face was removed and re-added outside the window.
    """
    if not isinstance(mapping, list) or len(mapping) == 0 or swap_mode == "all_input":
        return None
    facesets = list(roop_globals.INPUT_FACESETS)
    mapped = []
    for x in mapping:
        try:
            src_idx = int(x)
        except (ValueError, TypeError):
            src_idx = -1
        if 0 <= src_idx < len(facesets):
            mapped.append(facesets[src_idx])
        else:
            mapped.append(FaceSet())
    return mapped


def mapped_selected_index(mapping, mapped, selected):
    """Translate the gallery-selected source index into the mapped list's space.

    `selected_input_face_index` counts positions in the input-faces gallery, but
    once a mapping is active the swap sees the person-ordered list instead — the
    two only agree when the mapping is the identity. The modes that use a single
    source for every face ("All faces", "First found", gender) index it with
    `selected_index`, so an untranslated gallery index either picks the wrong
    source or (with fewer persons than source faces) runs off the end, in which
    case the face is skipped and nothing is swapped at all. Falls back to the
    first person's source when the selected face is mapped to nobody.
    """
    if mapped is None:
        return selected
    # Per-entry coercion, matching mapped_facesets: one unparseable entry must
    # not throw away the translation for every other face (a bare
    # `[int(x) for x in mapping]` would, and silently pin every mode to source 0).
    for rank, x in enumerate(mapping):
        try:
            if int(x) == int(selected):
                return rank
        except (ValueError, TypeError):
            continue
    return 0

API_TEMP = os.path.join(os.getcwd(), "temp", "api_uploads")
os.makedirs(API_TEMP, exist_ok=True)

# ── Multi-angle target bank gates ────────────────────────────────────────────
# All in scipy cosine distance (0..2). A same-person PROFILE is routinely 0.7-1.0
# from a frontal angle, so these have to stay loose or the bank ends up holding
# only near-frontal angles and the person stops being recognised the moment they
# turn. Wrong-person capture is prevented by the relative cross-person checks at
# the call sites, not by these absolute cutoffs.
_ANGLE_MANUAL_MAX = float(os.environ.get('ROOP_ANGLE_MANUAL_MAX', '0.90'))  # /target/add_angle
_ANGLE_ACCEPT     = float(os.environ.get('ROOP_ANGLE_ACCEPT', '0.60'))      # /target/auto_angles
_ANGLE_SEED_MAX   = float(os.environ.get('ROOP_ANGLE_SEED_MAX', '0.85'))    # /target/auto_angles

# ── Auto-angles intake, for the frames where the relative guards go silent ───
# The cross-person and runner-up checks in consider_frame are RELATIVE, which is
# why they can be strict without punishing a hard pose. But both need a second
# face to compare against: the runner-up check is literally `len(scored) > 1`,
# and the cross-person one needs ANOTHER captured person. On a frame holding one
# face — thousands of them in a long clip, and by definition every frame where
# the target has walked off — neither can run, and the only things left are two
# absolute distances against the target's own bank.
#
# A typical stranger (0.93-1.07 measured) fails those. What crosses them is a
# DEGRADED frame, where the embedding collapses toward the middle of the space
# and lands near everybody. Hence the two gates below, which cost nothing on a
# clean frame:
#
#   BLUR_FRAC / MIN_PX / MIN_QUALITY — refuse to bank a blurred or tiny crop at
#     all. Judged on image quality only (face_quality.image_quality), never on
#     pose, or the gate would reject the profiles this feature exists to collect.
#
#     Blur is gated RELATIVE to what this clip actually offers — a candidate
#     less than BLUR_FRAC of the median sharpness of the faces scanned so far.
#     An absolute cutoff cannot work at both ends: measured on face-like crops,
#     a clean face scores 0.96-1.00 on the sharpness axis and a mildly blurred
#     one 0.03, but the numbers move wholesale with grain, compression and
#     focus, so a fixed line either passes everything on a crisp clip or rejects
#     everything on a soft one. The composite score cannot carry this either —
#     a heavily blurred 200px face still totals 0.505, because size and detector
#     confidence prop up exactly the frame whose embedding is worthless. Hence a
#     dedicated blur gate, with MIN_QUALITY left as a weak backstop.
#   LONE_ACCEPT — on those unguarded frames, require the face to be closer to
#     the bank than the usual ACCEPT. Deliberately tightens the BANK distance
#     rather than the seed distance: a genuine profile arrives by chaining, so
#     it sits near an intermediate angle already banked and its bank distance is
#     small, while a marginal admission sits out near the limit. Tightening the
#     seed distance instead would have hit the profiles hardest.
#
# Pollution matters more than one bad thumbnail: every swap-time gate takes the
# MINIMUM over a person's angles, so one wrong angle makes a stranger's whole
# track measure ~0 to that person and inherit their source.
_ANGLE_MIN_QUALITY = float(os.environ.get('ROOP_ANGLE_MIN_QUALITY', '0.35'))
_ANGLE_MIN_PX      = float(os.environ.get('ROOP_ANGLE_MIN_PX', '64'))
# ROOP_ANGLE_BLUR_FRAC / ROOP_ANGLE_BLUR_WARMUP are owned by face_quality, which
# applies them; re-reading them here would be a second source of truth for the
# same gate. Only a per-request override is resolved below, and None means "use
# the module default".
_ANGLE_LONE_ACCEPT = float(os.environ.get('ROOP_ANGLE_LONE_ACCEPT', '0.45'))
# Banked angles this far from the seed are the ones nearest the drift limit, so
# they are listed for review. NOT a rejection — a true extreme profile lands
# here too; the point is that a polluted bank stops being invisible.
_ANGLE_REVIEW      = float(os.environ.get('ROOP_ANGLE_REVIEW', '0.70'))

# ── Shared server-side state (mirrors the Gradio module globals) ──────────────
list_files_process: list = []          # list[ProcessEntry] – the target media queue

fm_selected_index = -1

# Live progress, polled by the React UI
_progress = {"processing": False, "paused": False, "progress": 0.0, "desc": "", "error": ""}
_last_output = {"path": "", "kind": ""}

# Per-run accumulator, reset at the start of every swap and read back when the
# run is recorded into history. Wall time + the highest "done / total" frame
# count seen in the status line let the Run History view show how long a run
# took and its average throughput — data that used to evaporate the instant the
# job finished. Populated centrally in get_progress() so it captures every stage
# without touching the pipeline. `start == 0` means no run has begun this session.
import re as _re
_FRAME_RE = _re.compile(r"(\d[\d,]*)\s*/\s*(\d[\d,]*)")
_run_stats = {"start": 0.0, "frames_done": 0, "frames_total": 0}

# Rolling terminal-style log tail surfaced to the preview box while a job runs.
# Capture is centralized in get_progress() (see below) so it mirrors _progress["desc"]
# regardless of which stage produced it (swap / upscale / interpolate / combine).
from collections import deque as _deque
_log_lines = _deque(maxlen=250)
_log_state = {"last": "", "last_ts": 0.0, "seq": 0, "last_err": "", "status": "",
              "parts_seen": 0, "counter_shapes": set()}


# A status line the UI pins and rewrites in place rather than scrolling. Matches
# "12 / 300", "45%", "20.3 fps" — anything whose whole content is a counter that
# changes every poll. Keeping these OUT of the log is what makes the history
# readable: one run used to bury its stage changes under thousands of near
# identical frame lines, which is the same information the progress bar already
# shows, once per frame instead of once.
_COUNTER_RE = _re.compile(r"\d[\d,]*\s*/\s*\d[\d,]*|\bfps\b|\d+\s*%", _re.I)

# The same line with every number blanked. Two counter updates from the SAME
# stage share a shape; a different stage does not. That distinction is what
# keeps the history honest: dropping counter lines outright also dropped the
# only evidence a stage ever ran, because "Processing frame N / M" is the ONLY
# thing the swap stage ever says — a video run's history went straight from
# "Analyzing faces" to "Combining", with the swap itself invisible.
#
# Parenthesised numbers are stripped BEFORE blanking because the rate suffix is
# not always there: tqdm reports "(20.3 FPS)" only once it has a rate, and drops
# it again on a stall. Without this, that suffix appearing and disappearing
# reads as the stage starting over and over, and the spam comes straight back.
_COUNTER_SHAPE_RE = _re.compile(r"\d[\d,.]*")
_COUNTER_PAREN_RE = _re.compile(r"\s*\([^)]*\d[^)]*\)")


def is_counter_line(msg):
    """True for a live counter (frames done / fps / percent) — pinned, not logged."""
    return bool(msg) and bool(_COUNTER_RE.search(msg))


def counter_shape(msg):
    """Identity of a counter line independent of its numbers."""
    return _COUNTER_SHAPE_RE.sub('#', _COUNTER_PAREN_RE.sub('', msg or ''))


def _push_log(msg, force=False, part=None):
    """Append a line to the rolling terminal feed, de-duped.

    A counter line (frames done / fps / percent) is kept ONCE — the first time
    its shape appears, which is the moment that stage started — and every later
    update of the same shape is dropped, because it is the pinned status line's
    job to show those. So the history reads as one line per thing that happened:
    stages beginning, parts finalized, warnings, errors, done. `force=True`
    keeps a line whatever it looks like.
    """
    import time as _time
    if not msg:
        return
    msg = str(msg).strip()
    if not msg or msg == _log_state["last"]:
        return
    if not force and is_counter_line(msg):
        # Kept once per shape per PHASE. The phase ends when something real is
        # logged (a stage message, a finalized part, an error), which is also
        # what separates one file from the next in a batch — so file 2's swap
        # announces itself again rather than being mistaken for file 1's.
        shape = counter_shape(msg)
        if shape in _log_state["counter_shapes"]:
            return
        _log_state["counter_shapes"].add(shape)
    else:
        _log_state["counter_shapes"] = set()
    _log_state["last"] = msg
    _log_state["last_ts"] = _time.time()
    _log_state["seq"] += 1
    # Tag the line with the output part that was open when it arrived, so the
    # console can group its history by part (the [1✓][2✓][3•] tabs). An explicit
    # `part` is for lines ABOUT a part, which are emitted just after it closed.
    if part is None:
        try:
            part = segment_writer.current_part_index()
        except Exception:
            part = 0
    _log_lines.append({"t": _time.strftime("%H:%M:%S"), "msg": msg,
                       "seq": _log_state["seq"], "part": part})
# Set by /api/stop, reset at the start of each run — lets the post-swap upscale
# pass know the run was aborted (so it doesn't start a long upscale on a
# deliberately-stopped output).
_stop_requested = {"flag": False}

no_face_choices = ["Use untouched original frame", "Retry rotated", "Skip Frame",
                   "Skip Frame if no similar face", "Use last swapped"]














def _update_mask_offsets_from_payload(payload: dict):
    keys = ["mask_top", "mask_bottom", "mask_left", "mask_right",
            "face_mask_blend", "mouth_mask_blend",
            "mouth_top_scale", "mouth_bottom_scale", "mouth_left_scale", "mouth_right_scale"]
    
    # Update CFG so they are persisted. Only hit the disk when a value actually
    # changed — /api/preview calls this on every scrub/slider debounce.
    if roop_globals.CFG:
        cfg_changed = False
        for k in keys:
            if k in payload and payload[k] is not None:
                if getattr(roop_globals.CFG, k, None) != payload[k]:
                    setattr(roop_globals.CFG, k, payload[k])
                    cfg_changed = True
        if cfg_changed:
            roop_globals.CFG.save()

    # Update current faceset face offsets
    face_index = state.selected_input_face_index
    if len(roop_globals.INPUT_FACESETS) > face_index:
        faceset = roop_globals.INPUT_FACESETS[face_index]
        if faceset.faces:
            face = faceset.faces[0]
            if not hasattr(face, "mask_offsets") or face.mask_offsets is None:
                face.mask_offsets = _mask_offsets_from_cfg()
            offs = list(face.mask_offsets)
            while len(offs) < 10:
                offs.append(1.0)
            
            mapping = {
                "mask_top": 0,
                "mask_bottom": 1,
                "mask_left": 2,
                "mask_right": 3,
                "face_mask_blend": 4,
                "mouth_mask_blend": 5,
                "mouth_top_scale": 6,
                "mouth_bottom_scale": 7,
                "mouth_left_scale": 8,
                "mouth_right_scale": 9
            }
            updated = False
            for k, idx in mapping.items():
                if k in payload and payload[k] is not None:
                    try:
                        offs[idx] = float(payload[k])
                        updated = True
                    except (ValueError, TypeError):
                        pass
            if updated:
                face.mask_offsets = offs


def translate_swap_mode(text):
    return {"Selected face": "selected", "First found": "first",
            "All input faces": "all_input", "All female": "all_female",
            "All male": "all_male"}.get(text, "all")


def index_of_no_face_action(text):
    try:
        return no_face_choices.index(text)
    except ValueError:
        return 0


def map_mask_engine(selected_mask_engine, clip_text):
    if selected_mask_engine == "Clip2Seg":
        return "mask_clip2seg" if clip_text else None
    if selected_mask_engine == "DFL XSeg":
        return "mask_xseg"
    if selected_mask_engine == "Face Parser (BiSeNet)":
        return "mask_faceparser"
    if selected_mask_engine == "RealityUX":
        return "mask_realityux"
    if selected_mask_engine == "Face Occluder":
        return "mask_occluder"
    if selected_mask_engine == "Face Occluder v3 (XSeg-3)":
        return "mask_xseg3"
    if selected_mask_engine == "Segment Anything (MobileSAM)":
        return "mask_mobilesam"
    if selected_mask_engine == "Segment Anything (FastSAM)":
        return "mask_fastsam"
    if selected_mask_engine == "Segment Anything 2 (tracked)":
        return "mask_sam2"
    return None


def map_mask_engines(primary, secondary, clip_text):
    """Both occlusion engines, as the list `get_processing_plugins` takes.

    Two rather than one because a single engine failing to recognise a
    particular object is the usual reason a hand, a mug or a microphone ends up
    with a face painted over it. The engines were trained on different data and
    compose as a union of "not face", so a second one can only ever restore MORE
    of the original — it cannot eat into the swap.

    Returns a plain string when there is only one, so the common case is
    byte-for-byte the old call.
    """
    engines = []
    for name in (primary, secondary):
        mapped = map_mask_engine(name, clip_text)
        if mapped and mapped not in engines:
            engines.append(mapped)
    if not engines:
        return None
    return engines[0] if len(engines) == 1 else engines


class ApiProgress:
    """gradio.Progress-compatible shim that records progress into _progress."""
    def __call__(self, value=0, desc="", total=None, unit=None):
        try:
            if isinstance(value, (tuple, list)) and len(value) == 2 and value[1]:
                _progress["progress"] = float(value[0]) / float(value[1])
            else:
                _progress["progress"] = float(value)
        except Exception:
            pass
        if desc:
            _progress["desc"] = desc

        # Dynamic VRAM Safety Guard & Memory Flush
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info(0)
                free_gb = free_bytes / (1024 ** 3)
                if free_gb < 0.8:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
        except Exception:
            pass

    def tqdm(self, iterable=None, *a, **k):
        return iterable if iterable is not None else []


# ── Settings ─────────────────────────────────────────────────────────────────
@app.get("/api/settings")
def get_settings():
    if roop_globals.CFG:
        return roop_globals.CFG.__dict__
    return {}


@app.get("/api/settings/defaults")
def get_settings_defaults():
    """The values a fresh install would have, for the UI's "changed" markers.

    Settings.load() falls back to its hardcoded default for every key it cannot
    read out of the config file, so pointing a throwaway instance at a path that
    does not exist yields exactly the default set — no duplicated table to drift
    out of step with the real one. Nothing is written: only load() runs.

    A couple of defaults are computed rather than constant (max_threads scales
    with VRAM), which is the right answer here too — "default" means what this
    machine would have started with.
    """
    from settings import Settings
    try:
        return Settings(os.path.join(os.path.dirname(__file__), '__nonexistent_defaults__.yaml')).__dict__
    except Exception:
        return {}


@app.post("/api/settings")
def save_settings(settings: dict = Body(...)):
    _update_mask_offsets_from_payload(settings)
    if roop_globals.CFG:
        for k, v in settings.items():
            if hasattr(roop_globals.CFG, k):
                setattr(roop_globals.CFG, k, v)
        roop_globals.CFG.save()
    return {"status": "success"}


# ── Hardware benchmark ───────────────────────────────────────────────────────
# The measuring is all in roop/bench.py; this is transport. See that module's
# docstring for what is measured and for what the previous version of this
# endpoint was actually measuring (a batched matmul, reported as "FPS", wired
# into Settings.resolve_threads).
import threading

_benchmark_lock = threading.Lock()
_benchmark_cancel_requested = False
_benchmark_state = {
    "running": False,
    "progress": 0,
    "elapsed_sec": 0,
    "total_sec": 330,
    "profile": "full",
    "phase": "idle",
    "current_mode": "Idle",
    "current_threads": 0,
    "current_fps": 0.0,
    "current_vram_gb": 0.0,
    "status_msg": "Idle",
    "logs": [],
    "result": None,
    "error": "",
}


def _bench_report(**kw):
    """Progress callback handed to roop.bench — the only thing this layer does.

    Keeps the state dict the polling endpoint returns, including a bounded log
    ring so a long full-profile run cannot grow the response without limit.
    """
    with _benchmark_lock:
        if kw.get("status"):
            _benchmark_state["status_msg"] = kw["status"]
        if kw.get("phase"):
            _benchmark_state["phase"] = kw["phase"]
        if kw.get("pct") is not None:
            _benchmark_state["progress"] = max(0, min(99, int(kw["pct"])))
        if kw.get("stage"):
            _benchmark_state["current_mode"] = kw["stage"]
        if kw.get("threads") is not None:
            _benchmark_state["current_threads"] = int(kw["threads"])
        if kw.get("fps") is not None:
            _benchmark_state["current_fps"] = float(kw["fps"])
        if kw.get("vram") is not None:
            _benchmark_state["current_vram_gb"] = float(kw["vram"])
        if kw.get("log"):
            _benchmark_state["logs"].append(
                f"[{time.strftime('%H:%M:%S')}] {kw['log']}")
            if len(_benchmark_state["logs"]) > 400:
                del _benchmark_state["logs"][:-400]
        _benchmark_state["elapsed_sec"] = int(time.time() - _benchmark_started_at[0])


_benchmark_started_at = [0.0]


def _run_benchmark_worker(profile: str, faces: float):
    from roop import bench
    try:
        result = bench.run_benchmark(
            profile=profile,
            faces_per_frame=faces,
            report=_bench_report,
            cancelled=lambda: _benchmark_cancel_requested,
            apply_to_config=True,
        )
        with _benchmark_lock:
            _benchmark_state["result"] = result
            _benchmark_state["progress"] = 100
            _benchmark_state["status_msg"] = "Benchmark complete"
    except bench.Cancelled:
        with _benchmark_lock:
            _benchmark_state["status_msg"] = "Cancelled"
            _benchmark_state["logs"].append("Cancelled by user.")
    except Exception as e:                                  # noqa: BLE001
        import traceback
        traceback.print_exc()
        with _benchmark_lock:
            _benchmark_state["error"] = f"{type(e).__name__}: {e}"
            _benchmark_state["status_msg"] = f"Failed: {type(e).__name__}"
            _benchmark_state["logs"].append(f"ERROR {type(e).__name__}: {e}")
    finally:
        with _benchmark_lock:
            _benchmark_state["running"] = False
            _benchmark_state["elapsed_sec"] = int(time.time() - _benchmark_started_at[0])


@app.get("/api/settings/benchmark_status")
def get_benchmark_status():
    with _benchmark_lock:
        state = dict(_benchmark_state)
        # `logs` has to be COPIED, not aliased. FastAPI serialises the return
        # value after this function has released the lock, and the worker thread
        # appends to that same list roughly once a second — a shallow dict()
        # hands the serialiser a list that is still being mutated, which raises
        # mid-response and shows up as the poll silently failing.
        # Only the tail is sent: the ring holds 400 lines, the panel shows the
        # recent ones, and shipping all of them once a second is pure weight.
        state["logs"] = list(_benchmark_state["logs"][-80:])
        return state


@app.post("/api/settings/benchmark_threads")
def benchmark_threads(payload: dict = Body(default=None)):
    global _benchmark_cancel_requested
    payload = payload or {}
    profile = str(payload.get("profile", "full")).lower()

    from roop import bench
    if profile not in bench.PROFILES:
        profile = "full"

    # Refuse to share the GPU with a render. Both would be wrong: the render
    # slows down, and the benchmark measures a card that is already busy and
    # recommends thread counts for a machine nobody has.
    if _progress["processing"]:
        return JSONResponse(status_code=409, content={
            "message": "A render is in progress — benchmarking now would measure "
                       "a busy GPU and slow the render down."})

    import torch
    if not torch.cuda.is_available():
        return JSONResponse(status_code=400, content={
            "message": "No CUDA device. The benchmark measures GPU stage cost, "
                       "pool scaling and VRAM headroom, none of which exist on CPU."})

    with _benchmark_lock:
        if _benchmark_state["running"]:
            return {"status": "running", "message": "Benchmark already in progress."}
        _benchmark_cancel_requested = False
        _benchmark_started_at[0] = time.time()
        _benchmark_state.update({
            "running": True, "progress": 0, "elapsed_sec": 0,
            "total_sec": bench.PROFILES[profile]["est_sec"],
            "profile": profile, "phase": "probe",
            "current_mode": "Initializing", "current_threads": 0,
            "current_fps": 0.0, "current_vram_gb": 0.0,
            "status_msg": "Loading models…", "logs": [], "result": None,
            "error": "",
        })

    faces = float(payload.get("faces_per_frame", 1.0) or 1.0)
    threading.Thread(target=_run_benchmark_worker, args=(profile, faces),
                     daemon=True).start()
    return {"status": "started", "profile": profile,
            "est_sec": bench.PROFILES[profile]["est_sec"]}


@app.post("/api/settings/benchmark_cancel")
def benchmark_cancel():
    global _benchmark_cancel_requested
    _benchmark_cancel_requested = True
    return {"status": "cancelling", "message": "Cancellation requested"}


def _get_git_version() -> str:
    import subprocess
    try:
        # Get active branch name
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
        # Get tag or short commit
        version = subprocess.check_output(["git", "describe", "--tags", "--always"], stderr=subprocess.DEVNULL).decode("ascii").strip()
        return f"{branch}@{version}"
    except Exception:
        try:
            version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
            return f"main@{version}"
        except Exception:
            return "main"


@app.get("/api/meta")
def get_meta():
    """Choice lists + current galleries for the React UI to render."""
    try:
        from roop.core import suggest_execution_providers
        providers = suggest_execution_providers()
    except Exception:
        providers = ["cpu"]
    return {
        "git_version": _get_git_version(),
        "providers": providers,
        "trt_precisions": ["fp32", "fp16", "mixed"],
        "enhancers": ["None", "Codeformer", "Codeformer (fp16)", "DMDNet",
                       "GFPGAN", "GPEN 256", "GPEN 256 Pro", "GPEN Realistic", "GPEN", "GPEN 1024",
                       "GPEN 2048",
                       "Restoreformer++", "UltraMax", "KEEP (sidecar)"],
        "swap_models": ["realswap", "inswapper", "reswapper", "hyperswap",
                         "hyperswap_1b", "hyperswap_1c", "ghost_1", "ghost_2",
                         "ghost_3", "simswap", "simswap_512", "hififace",
                         "blendswap", "uniface", "instyleswapper_a",
                         "instyleswapper_b", "instyleswapper_c", "cscs"],
        "face_detection_modes": ["First found", "All input faces", "All female",
                                  "All male", "All faces", "Selected face"],
        "mask_engines": ["None", "Clip2Seg", "DFL XSeg", "Face Parser (BiSeNet)",
                          "RealityUX", "Face Occluder", "Face Occluder v3 (XSeg-3)",
                          "Segment Anything (MobileSAM)", "Segment Anything (FastSAM)",
                          "Segment Anything 2 (tracked)"],
        "sam2_model_sizes": ["tiny", "small", "base_plus", "large"],
        "color_transfer_modes": ["none", "rct", "lct", "mkl", "idt"],
        "detector_engines": ["scrfd", "yoloface", "retinaface", "retinaface_r50", "yunet"],
        "encoder_presets": ["auto", "ultrafast", "superfast", "veryfast", "faster",
                             "fast", "medium", "slow", "slower", "veryslow"],
        "pool_sizes": ["auto", "1", "2", "3", "4", "5", "6", "7", "8",
                        "10", "12", "16"],
        "tristate": ["auto", "on", "off"],
        # Identity & tracking selects. Kept here rather than hardcoded in the
        # panel for the same reason every other list is: the dropdown and the
        # backend that has to accept the value stay defined in one place.
        "recognizers": ["default", "adaface"],
        # Exactly the keys keep_awake._PRIORITY_CLASSES accepts. It falls back
        # to 'high' for anything it does not recognise, so an option outside
        # this list would look like a working choice and do nothing.
        "priorities": ["auto", "high", "above_normal", "normal"],
        "no_face_actions": no_face_choices,
        "upscale": ["128px", "256px", "512px"],
        "video_methods": ["Extract Frames to media", "In-Memory processing"],
        "output_methods": ["File", "Virtual Camera", "Both"],
        "image_formats": ["jpg", "png", "webp"],
        "video_formats": ["avi", "mkv", "mp4", "webm"],
        "video_codecs": _available_video_codecs(),
    }


_VIDEO_CODECS = ["libx264", "libx265", "libvpx-vp9", "h264_nvenc", "hevc_nvenc"]
_codec_cache = {"list": None}


def _available_video_codecs():
    """The offered codecs, minus any this ffmpeg build cannot actually encode.

    The bundled ffmpeg here has no libvpx-vp9, so picking VP9 could only ever end
    in "Unknown encoder" — the run does fail fast (core.py probes the selected
    encoder before rendering), but offering a choice that cannot work is a
    trap. `ffmpeg -encoders` is asked once per process and cached; if the query
    itself fails we fall back to the full list rather than hiding everything.
    """
    if _codec_cache["list"] is not None:
        return _codec_cache["list"]
    out = _VIDEO_CODECS
    try:
        from roop.ffmpeg_writer import FFMPEG_BINARY
        kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
        proc = subprocess.run([FFMPEG_BINARY, "-hide_banner", "-encoders"],
                              capture_output=True, timeout=20, **kwargs)
        blob = ((proc.stdout or b"") + (proc.stderr or b"")).decode("utf-8", "replace")
        found = [c for c in _VIDEO_CODECS if f" {c} " in blob]
        if found:
            missing = [c for c in _VIDEO_CODECS if c not in found]
            if missing:
                print(f"[Encoders] not available in this ffmpeg build, hidden from "
                      f"the codec list: {', '.join(missing)}", flush=True)
            out = found
    except Exception:
        pass
    _codec_cache["list"] = out
    return out




# ── Fine pose bins (coverage accounting, NOT display) ────────────────────────
# estimate_face_pose_from_kps() above is the human-readable label shown in the
# UI and is deliberately coarse: its "Front" bucket spans roughly ±45° of yaw,
# and there are only 9 labels in total. That is fine for a caption but useless
# as a coverage measure — six mildly-turned faces fill the "Front" bucket and
# then block further capture while true profiles are still missing entirely.
# So auto-angle harvesting bins on the raw log-ratios instead: symmetric about
# 0, one bin per meaningful step of yaw/pitch, 9 × 5 possible bins.
_YAW_EDGES = (-1.20, -0.75, -0.43, -0.15, 0.15, 0.43, 0.75, 1.20)
_PITCH_EDGES = (-0.75, -0.43, 0.37, 0.75)


def _bin_index(value, edges):
    i = 0
    for e in edges:
        if value >= e:
            i += 1
    return i


def _pose_transition_intervals(timeline):
    """Frame ranges worth re-scanning at fine stride, best first.

    `timeline` is [(frame_pos, pose_bin | None), ...] from a coarse sweep. A pose
    that never landed on a coarse sample has to lie BETWEEN two samples whose
    bins differ, so those gaps — and only those — are where extra decoding pays
    off. Ranked by how far apart the endpoint bins are: the widest jumps skipped
    over the most unseen intermediate poses. A gap with the person on one side
    only still counts (they entered, left, or turned too far to match) but ranks
    last, because a plain scene cut looks identical and refining it is wasted."""
    intervals = []
    for (fa, ba), (fb, bb) in zip(timeline, timeline[1:]):
        if fb - fa <= 1 or ba == bb:
            continue
        if ba is None and bb is None:
            continue
        span = 1 if (ba is None or bb is None) else abs(ba[0] - bb[0]) + abs(ba[1] - bb[1])
        intervals.append((span, fa, fb))
    intervals.sort(key=lambda x: -x[0])
    return intervals


def _pose_bin(kps):
    """Fine (yaw_bin, pitch_bin) coverage bucket for a face, or None if unusable."""
    try:
        (lex, ley), (rex, rey), (nx, ny), (_lmx, lmy), (_rmx, rmy) = [tuple(p) for p in kps[:5]]
        yaw = float(np.log((abs(nx - lex) + 1e-6) / (abs(rex - nx) + 1e-6)))
        eye_y = (ley + rey) * 0.5
        mouth_y = (lmy + rmy) * 0.5
        pitch = float(np.log((abs(ny - eye_y) + 1e-6) / (abs(mouth_y - ny) + 1e-6)))
    except Exception:
        return None
    return (_bin_index(yaw, _YAW_EDGES), _bin_index(pitch, _PITCH_EDGES))


# ── The source gallery is TWO parallel lists ─────────────────────────────────
# `roop_globals.INPUT_FACESETS` holds what the swap actually uses;
# `ui_globals.ui_input_thumbs` holds the picture of it the gallery draws. They
# are positional — index i in one must be index i in the other — and until now
# every endpoint kept them in step by hand, which is how a face could show in
# the gallery that the backend no longer had: the swap then ran with an empty
# faceset for that person and simply left them un-swapped, with nothing on
# screen to say so. These four helpers are the only supported way to change the
# gallery, so the two lists cannot drift apart one edit at a time.













@app.get("/api/state")
def get_state():
    """Rehydrate the UI: current source/target galleries and target queue."""
    targets = [_target_entry_dict(entry) for entry in list_files_process]
    desync = _sources_desync()
    return {
        **({"desync": desync} if desync else {}),
        "source_faces": [_rgb_to_dataurl(t) for t in ui_globals.ui_input_thumbs],
        "source_faces_info": _get_source_faces_info(),
        "target_faces": [_rgb_to_dataurl(t) for t in ui_globals.ui_target_thumbs],
        "target_groups": _target_groups_ranked(),
        "target_faces_info": _target_faces_info(),
        "target_names": _target_names_ranked(),
        "targets": targets,
        "selected_target_index": state.selected_target_index,
        "faceset_count": len(roop_globals.INPUT_FACESETS),
    }


# ── Source faces ─────────────────────────────────────────────────────────────
# ── Why the upload handlers are sync `def`, not `async def` ─────────────────
# FastAPI runs a sync endpoint in its worker threadpool but an async one ON the
# event loop. Every upload handler here does blocking work — writing the file,
# running face detection, and in the Extras case downloading models and doing
# whole-video inference — and awaits nothing at all. Declared `async`, each of
# them therefore held the single event loop for its entire duration, so while a
# file was being ingested NOTHING else async got served: the progress poll, the
# telemetry poll and the log feed all queued up behind it and the UI looked
# frozen. Sync `def` costs a threadpool hop and gives the loop back.
@app.post("/api/source/add")
def source_add(files: list[UploadFile] = File(...)):
    for f in files:
        path = _save_upload(f)
        try:
            if path.lower().endswith("fsz"):
                _ingest_faceset(path)
            elif util.has_image_extension(path):
                roop_globals.source_path = path
                faces_data = extract_face_images(path, (False, 0))
                for fd in faces_data:
                    fs = FaceSet()
                    face = fd[0]
                    face.mask_offsets = _mask_offsets_from_cfg()
                    fs.faces.append(face)
                    _sources_append(fs, util.convert_to_gradio(fd[1]))
        except Exception:
            traceback.print_exc()
    return _source_faces_payload()




@app.post("/api/lipsync/audio/add")
def lipsync_audio_add(file: UploadFile = File(...)):
    """Ingest a dub track for lip-sync's 'upload' audio source.

    A separate small endpoint rather than folding this into /api/swap so that
    endpoint stays JSON-only, like every other setting on it — the returned
    path is referenced from the JSON payload afterward via lipsync_audio_path.
    """
    path = _save_upload(file)
    return {"path": path}


@app.post("/api/source/remove")
def source_remove(payload: dict = Body(...)):
    _sources_pop(int(payload.get("index", -1)))
    return _source_faces_payload()


@app.post("/api/source/move")
def source_move(payload: dict = Body(...)):
    idx = int(payload.get("index", -1))
    offset = -1 if payload.get("direction", "right") == "left" else 1
    _sources_move(idx, idx + offset)
    return _source_faces_payload()


@app.post("/api/source/clear")
def source_clear():
    _sources_clear()
    return _source_faces_payload()


@app.post("/api/source/select")
def source_select(payload: dict = Body(...)):
    state.selected_input_face_index = int(payload.get("index", 0))
    return {"selected": state.selected_input_face_index}


@app.post("/api/source/refresh_thumbs")
def source_refresh_thumbs():
    """Recompute the gallery thumbnail for each loaded multi-angle faceset to the
    most frontal face, without needing to clear + reload the source."""
    for idx, fs in enumerate(roop_globals.INPUT_FACESETS):
        refs = getattr(fs, "ref_images", None) or []
        if not refs:
            continue
        crop = _frontal_crop_from_images(refs)
        if crop is not None and idx < len(ui_globals.ui_input_thumbs):
            ui_globals.ui_input_thumbs[idx] = util.convert_to_gradio(crop)
    return _source_faces_payload()


# ── Faceset library (persistent, named .fsz facesets) ─────────────────────────
# The library is a real folder on disk holding named `<name>.fsz` facesets plus a
# `<name>.png` thumbnail sidecar for instant previews. Facesets saved here survive
# restarts so sources never need re-uploading, can be renamed/exported/imported,
# and — by pointing the folder at OneDrive/Dropbox/Google Drive (Settings →
# "Faceset library folder") — sync across devices automatically.













































# ── Target media ─────────────────────────────────────────────────────────────
def _is_usable_target(path):
    """Exactly what the pipeline itself will accept as a target.

    Asked of roop.utilities rather than of a list of extensions kept here. A
    private list is guaranteed to drift: it started out missing .ts, .mts and
    .mxf — all of which /api/target/add would have taken happily, because an
    upload is never checked at all — so the path route would have been the
    stricter of the two ways to add the same file.

    is_image() deliberately returns False for animated GIF and WebP, which ARE
    valid targets; _refresh_target_frames makes the same three-way test.
    """
    return (util.is_video(path) or util.is_image(path)
            or path.lower().endswith('.gif') or util.is_animated_webp(path))


@app.post("/api/target/add_path")
def target_add_path(payload: dict = Body(...)):
    """Add targets that are ALREADY on this machine, by path — no upload.

    The server runs on 127.0.0.1, on the same disk as the media. Posting a
    4 GB video to it over HTTP and then writing a second 4 GB copy into
    temp/ (which is what /api/target/add does, via _save_upload) is pure
    waste: it costs minutes of transfer and doubles the disk footprint of
    every clip, to arrive at a file the process could simply have opened.

    Referencing the original also means the file is not silently duplicated
    when you re-add it later, and the run history's target name is the real
    one rather than a temp copy.

    Paths are taken as given. That is the same trust model the app already
    operates under — /api/file serves an arbitrary path and /api/reveal opens
    one — and it is bounded by the server being loopback-only. The checks
    below are for MISTAKES, not for an attacker: a directory, a typo, or a
    .txt would otherwise be appended and fail much later, mid-render.
    """
    raw = payload.get("paths") or []
    if isinstance(raw, str):
        raw = [raw]

    first_new = len(list_files_process)
    added, rejected = [], []
    for p in raw:
        path = os.path.abspath(os.path.expanduser(str(p).strip().strip('"')))
        if not os.path.isfile(path):
            rejected.append({"path": p, "why": "not a file on this machine"})
            continue
        if not _is_usable_target(path):
            rejected.append({"path": p, "why": "not a supported image or video"})
            continue
        list_files_process.append(ProcessEntry(path, 0, 0, 0))
        added.append(path)

    # Nothing landed — leave the selection alone. Unlike the upload route, this
    # one can legitimately add zero entries (every path was a typo), and the
    # expression below would then read `first_new == len(...)` and fall to 0,
    # yanking the user off the target they were working on as the reward for
    # mistyping a path.
    if not added:
        out = _target_list_payload()
        out["added"] = []
        out["rejected"] = rejected
        return out

    for i in range(first_new, len(list_files_process)):
        _refresh_target_frames(i)
    state.selected_target_index = first_new
    out = _target_list_payload()
    out["added"] = added
    out["rejected"] = rejected
    return out


@app.post("/api/target/add")
def target_add(files: list[UploadFile] = File(...)):
    first_new = len(list_files_process)
    for f in files:
        path = _save_upload(f)
        list_files_process.append(ProcessEntry(path, 0, 0, 0))
    # Refresh every new entry (not just index 0) so videos report their real
    # frame count/fps immediately — the UI's auto-queue and labels rely on it.
    for i in range(first_new, len(list_files_process)):
        _refresh_target_frames(i)
    state.selected_target_index = first_new if first_new < len(list_files_process) else 0
    return _target_list_payload()


def _refresh_target_frames(idx):
    if idx >= len(list_files_process):
        return
    entry = list_files_process[idx]
    filename = entry.filename
    if util.is_video(filename) or filename.lower().endswith("gif") or util.is_animated_webp(filename):
        total = get_video_frame_total(filename) or 1
        # Animated WebP is detected too: utilities.detect_fps has a PIL path that
        # derives the rate from the frame durations. Skipping it here left a .webp
        # inheriting whatever fps the PREVIOUS target happened to have, so the
        # render came out at the wrong speed.
        try:
            state.current_video_fps = util.detect_fps(filename)
        except Exception:
            state.current_video_fps = 30
        entry.fps = state.current_video_fps
    else:
        total = 1
        entry.fps = 0
    entry.total_frames = total
    # Keep the user's trim markers on re-select; only (re)initialise when unset
    # or out of range.
    if not entry.endframe or entry.endframe > total:
        entry.endframe = total
    if entry.startframe > total:
        entry.startframe = 0


def _target_entry_dict(entry):
    total = getattr(entry, "total_frames", 0) or entry.endframe or 1
    return {
        "name": os.path.basename(entry.filename),
        "startframe": entry.startframe,
        "endframe": entry.endframe,
        "start_frame": entry.startframe,
        "end_frame": entry.endframe,
        "frames": total,
        "fps": entry.fps or 0,
    }


def _target_list_payload():
    targets = [_target_entry_dict(entry) for entry in list_files_process]
    return {"targets": targets, "selected_target_index": state.selected_target_index,
            "fps": state.current_video_fps}


@app.post("/api/target/select")
def target_select(payload: dict = Body(...)):
    idx = int(payload.get("index", 0))
    # Clamp: a negative index would silently wrap to the last entry via
    # Python list indexing in downstream helpers.
    state.selected_target_index = max(0, min(idx, max(0, len(list_files_process) - 1)))
    _refresh_target_frames(state.selected_target_index)
    return _target_list_payload()


@app.post("/api/target/remove")
def target_remove(payload: dict = Body(...)):
    """Remove a single target media item from the queue."""
    idx = int(payload.get("index", -1))
    if 0 <= idx < len(list_files_process):
        list_files_process.pop(idx)
    if state.selected_target_index >= len(list_files_process):
        state.selected_target_index = max(0, len(list_files_process) - 1)
    if list_files_process:
        _refresh_target_frames(state.selected_target_index)
    return _target_list_payload()


@app.post("/api/target/clear")
def target_clear():
    list_files_process.clear()
    roop_globals.TARGET_FACES.clear()
    roop_globals.TARGET_FACE_GROUP.clear()
    if getattr(roop_globals, 'TARGET_FACE_NAMES', None):
        roop_globals.TARGET_FACE_NAMES.clear()
    ui_globals.ui_target_thumbs.clear()
    state.selected_target_index = 0
    return _target_list_payload()


@app.post("/api/target/set_frame")
def target_set_frame(payload: dict = Body(...)):
    """Set start/end frame of the selected target (Set as Start / End)."""
    idx = state.selected_target_index
    which = payload.get("which", "start")
    frame = int(payload.get("frame", 1))
    if idx < len(list_files_process):
        entry = list_files_process[idx]
        total = getattr(entry, "total_frames", 0) or entry.endframe or 0
        if which == "start":
            entry.startframe = min(frame, entry.endframe or frame)
        else:
            if total:
                frame = min(frame, total)
            entry.endframe = max(frame, entry.startframe)
    return _target_list_payload()


def _frame_identity(filename: str) -> str:
    """Cheap content identity for a target file — path, size and mtime. Enough to
    build an ETag with, and it costs a stat rather than a decode."""
    try:
        st = os.stat(filename)
        return f"{filename}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return filename


def _frame_etag(filename: str, frame: int, width: int, fmt: str, quality: int) -> str:
    """Identity of one ENCODED still. Covers the file (path, size, mtime) and
    every parameter that changes the bytes, so it doubles as an HTTP validator
    and as the key of the encoded-response cache below."""
    return 'W/"' + hashlib.sha1(
        f"{_frame_identity(filename)}:{frame}:{width}:{fmt}:{quality}".encode("utf-8")
    ).hexdigest() + '"'


# Encoded stills, keyed by their ETag. The capturer caches decoded frames, which
# removes the seek; this removes the resize-and-encode on top of it, so a frame
# the UI has already asked for comes back in about a millisecond instead of
# fifteen.
#
# Bounded by BYTES, not by entry count: a 200px filmstrip still is ~8 KB but a
# full-resolution 4K one is ~1.5 MB, so any fixed number of entries is either
# uselessly small for the first or hundreds of megabytes for the second.
_ENCODED_CACHE_BYTES = 64 * 1024 * 1024
_encoded_cache = OrderedDict()          # tag -> (bytes, media_type)
_encoded_cache_used = 0
_encoded_cache_lock = threading.Lock()


def _encoded_get(tag: str):
    with _encoded_cache_lock:
        hit = _encoded_cache.get(tag)
        if hit is not None:
            _encoded_cache.move_to_end(tag)
        return hit


def _encoded_put(tag: str, data: bytes, media: str):
    global _encoded_cache_used
    if not data or len(data) > _ENCODED_CACHE_BYTES:
        return
    with _encoded_cache_lock:
        old = _encoded_cache.pop(tag, None)
        if old is not None:
            _encoded_cache_used -= len(old[0])
        _encoded_cache[tag] = (data, media)
        _encoded_cache_used += len(data)
        while _encoded_cache_used > _ENCODED_CACHE_BYTES and len(_encoded_cache) > 1:
            _, evicted = _encoded_cache.popitem(last=False)
            _encoded_cache_used -= len(evicted[0])


def _encode_frame(frame_img, width: int, fmt: str, quality: int = 90):
    """Downscale-then-encode used by every raw-frame endpoint. Returns
    (bytes, media_type) or (None, None)."""
    if width and width > 0 and frame_img.shape[1] > width:
        h = max(1, round(frame_img.shape[0] * width / frame_img.shape[1]))
        frame_img = cv2.resize(frame_img, (width, h), interpolation=cv2.INTER_AREA)
    if fmt == "png":
        ok, buf = cv2.imencode(".png", frame_img)
        media = "image/png"
    else:
        ok, buf = cv2.imencode(".jpg", frame_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        media = "image/jpeg"
    if not ok:
        return None, None
    return buf.tobytes(), media


def _read_target_frame(filename: str, frame: int):
    if util.is_video(filename) or filename.lower().endswith("gif") or util.is_animated_webp(filename):
        return get_video_frame(filename, frame)
    return get_image_frame(filename)


def _still(filename: str, frame: int, width: int, fmt: str, quality: int):
    """One encoded still, from cache when we have it. Returns (tag, data, media);
    data is None when the frame could not be read."""
    tag = _frame_etag(filename, frame, width, fmt, quality)
    hit = _encoded_get(tag)
    if hit is not None:
        return tag, hit[0], hit[1]
    frame_img = _read_target_frame(filename, frame)
    if frame_img is None:
        return tag, None, None
    data, media = _encode_frame(frame_img, width, fmt, quality)
    if data is not None:
        _encoded_put(tag, data, media)
    return tag, data, media


@app.get("/api/target/preview")
def target_preview(request: Request, index: int = 0, frame: int = 1, width: int = 0,
                   fmt: str = "jpg", quality: int = 90):
    """Return the raw target frame (no swap).

    Defaults to JPEG — PNG encode of an HD/4K frame is 5-10x slower and much
    larger, which made timeline scrubbing feel sluggish. Pass fmt=png for a
    lossless frame, width=N for a server-side downscale (storyboard / hover
    thumbnails don't need full-resolution frames).

    Answers conditional requests. A given (file, frame, width, format) is always
    the same picture, so the browser is told it may reuse what it has; scrubbing
    back across frames it has already shown then costs nothing at all, and the
    revalidation that does happen is answered from a stat rather than a decode.
    The validator covers the file's size and mtime, so a target replaced on disk
    (or a different clip landing on the same list index) still re-renders."""
    if index < 0 or index >= len(list_files_process):
        return JSONResponse(status_code=404, content={"message": "no target"})
    filename = list_files_process[index].filename
    quality = max(30, min(int(quality), 100))

    etag = _frame_etag(filename, frame, width, fmt, quality)
    # `no-cache` (revalidate, don't blind-trust) rather than a max-age: the list
    # index in the URL can come to mean a different file, and a stale first frame
    # of the wrong clip is exactly the kind of bug that reads as "the timeline is
    # broken". The ETag makes the revalidation free.
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)

    _, data, media = _still(filename, frame, width, fmt, quality)
    if data is None:
        return JSONResponse(status_code=404, content={"message": "no frame"})
    return Response(content=data, media_type=media, headers=headers)


@app.get("/api/target/preview_grid")
def target_preview_grid(index: int = 0, frames: str = "", width: int = 200,
                        quality: int = 78):
    """Return an ARBITRARY set of frames in ONE response (the timeline's
    filmstrip).

    The strip is twelve stills spread across the visible range, and it was twelve
    separate GETs — which is not just twelve round trips but twelve requests
    competing for the browser's six connections to this origin, so the filmstrip
    could hold up the one request that matters: the frame the playhead is on.

    Decode order is ascending regardless of the order asked for, because the
    capturer walks forward cheaply but seeks expensively; the response is in the
    order requested. Body is the same length-prefixed stream preview_seq uses:
        [4-byte big-endian JPEG length][JPEG bytes] ... repeated
    A zero length means that frame could not be read — the client keeps its
    place in the sequence rather than shifting every later still onto the wrong
    slot."""
    if index < 0 or index >= len(list_files_process):
        return JSONResponse(status_code=404, content={"message": "no target"})
    filename = list_files_process[index].filename
    wanted = []
    for part in (frames or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            wanted.append(int(part))
        except ValueError:
            continue
    if not wanted:
        return JSONResponse(status_code=400, content={"message": "no frames"})
    wanted = wanted[:64]
    quality = max(30, min(int(quality), 95))

    encoded = {}
    for f in sorted(set(wanted)):
        _, data, _media = _still(filename, f, width, "jpg", quality)
        if data is not None:
            encoded[f] = data

    parts = []
    for f in wanted:
        data = encoded.get(f, b"")
        parts.append(len(data).to_bytes(4, "big"))
        parts.append(data)
    return StreamingResponse(io.BytesIO(b"".join(parts)),
                             media_type="application/octet-stream")


@app.get("/api/target/preview_seq")
def target_preview_seq(index: int = 0, start: int = 1, count: int = 16,
                       width: int = 0, quality: int = 82):
    """Return `count` CONSECUTIVE target frames in ONE response.

    Timeline playback used to fetch frames one at a time, which capped the frame
    rate at 1/round-trip no matter how fast the decode was: the request had to be
    single-flight (parallel requests arrive out of order and break capturer.py's
    sequential fast path, costing a full long-GOP seek per frame), so every frame
    paid a fresh HTTP round trip, a FastAPI dispatch and a lock acquire on top of
    its ~1 ms sequential decode. Batching amortises all of that over `count`
    frames while keeping the decode strictly in order, which is exactly the
    access pattern the capturer is fastest at.

    Body is a length-prefixed stream so the client can slice it without base64:
        [4-byte big-endian JPEG length][JPEG bytes] ... repeated
    A short body simply means the clip ended; the client stops there.
    """
    if index < 0 or index >= len(list_files_process):
        return JSONResponse(status_code=404, content={"message": "no target"})
    filename = list_files_process[index].filename
    if not (util.is_video(filename) or filename.lower().endswith("gif")
            or util.is_animated_webp(filename)):
        return JSONResponse(status_code=400, content={"message": "not a video"})

    count = max(1, min(int(count), 64))
    quality = max(30, min(int(quality), 95))
    parts = []
    for i in range(count):
        frame_img = get_video_frame(filename, start + i)
        if frame_img is None:
            break
        data, _ = _encode_frame(frame_img, width, "jpg", quality)
        if data is None:
            break
        parts.append(len(data).to_bytes(4, "big"))
        parts.append(data)
    return StreamingResponse(io.BytesIO(b"".join(parts)),
                             media_type="application/octet-stream")


def _target_groups_ranked():
    """Map raw group ids in TARGET_FACE_GROUP to contiguous 0-based person ranks
    (sorted by group id) so person N → source faceset N regardless of removals.
    Also keeps the group list length in sync with TARGET_FACES."""
    faces = roop_globals.TARGET_FACES
    grp = roop_globals.TARGET_FACE_GROUP
    if len(grp) != len(faces):
        grp = list(range(len(faces)))   # default: each face its own person
        roop_globals.TARGET_FACE_GROUP = grp
    uniq = sorted(set(grp))
    rank = {g: r for r, g in enumerate(uniq)}
    return [rank[g] for g in grp]


def _target_faces_info():
    """Per-target-face pose label (parallel to TARGET_FACES / thumbs), so the UI
    can show what angle each captured face is and compute pose coverage."""
    info = []
    for face in roop_globals.TARGET_FACES:
        kps = getattr(face, 'kps', None)
        if kps is None and isinstance(face, dict):
            kps = face.get('kps')
        pose = estimate_face_pose_from_kps(kps) if kps is not None else "Front"
        info.append({"pose": pose})
    return info


def _target_names_ranked():
    """Person display names indexed by contiguous rank (parallel to the ranks
    returned by _target_groups_ranked). Empty string when unnamed."""
    grp = roop_globals.TARGET_FACE_GROUP
    uniq = sorted(set(grp))
    names = getattr(roop_globals, 'TARGET_FACE_NAMES', {}) or {}
    return [names.get(g, "") for g in uniq]


def _target_faces_payload(extra=None):
    out = {
        "target_faces": [_rgb_to_dataurl(t) for t in ui_globals.ui_target_thumbs],
        "target_groups": _target_groups_ranked(),
        "target_faces_info": _target_faces_info(),
        "target_names": _target_names_ranked(),
    }
    if extra:
        out.update(extra)
    return out


def _faces_from_frame(idx, frame):
    target_path = list_files_process[idx].filename
    roop_globals.target_path = target_path
    if util.is_image(target_path) and not target_path.lower().endswith("gif"):
        return extract_face_images(target_path, (False, 0))
    return extract_face_images(target_path, (True, frame))


def _face_data_at_index(idx, frame, fi):
    """Return ``[[face, crop]]`` for the *fi*-th face using the SAME detection
    and ordering (get_all_faces, sorted left-to-right by bbox[0]) that the
    preview overlay draws its numbered boxes from — so clicking box *fi*
    captures exactly that face.

    Why not reuse ``extract_face_images``: it iterates the same get_all_faces
    list but silently *skips* any face whose clamped crop is degenerate
    (``face_temp.size < 1``), which shifts every later index and makes a click
    capture the wrong person. Here we index into get_all_faces directly and
    never drop the selected face (fall back to the whole frame as the crop)."""
    from roop.face_util import get_all_faces, _attach_source_crops, clamp_cut_values
    target_path = list_files_process[idx].filename
    roop_globals.target_path = target_path
    if util.is_image(target_path) and not target_path.lower().endswith("gif"):
        img = get_image_frame(target_path)
    else:
        img = get_video_frame(target_path, frame)
    if img is None:
        return []
    faces = get_all_faces(img)
    if not (0 <= fi < len(faces)):
        return []
    face = faces[fi]
    (sx, sy, ex, ey) = face["bbox"].astype("int")
    sx, ex, sy, ey = clamp_cut_values(sx, ex, sy, ey, img)
    crop = img[sy:ey, sx:ex]
    if crop.size < 1:
        crop = img          # never drop the clicked face — fall back to full frame
    _attach_source_crops(face, img)
    return [[face, crop]]


@app.post("/api/target/use_face")
def target_use_face(payload: dict = Body(...)):
    """Add target faces from the current frame, each as a NEW person (group).

    If `face_index` is supplied, only that single detected face is added — the
    index is into the left-to-right detection order, which matches the numbered
    boxes drawn on the live-preview overlay, so clicking a box adds exactly that
    person to the target faces.
    """
    idx = int(payload.get("index", state.selected_target_index))
    frame = int(payload.get("frame", 1))
    if idx >= len(list_files_process):
        return {"target_faces": [], "target_groups": []}
    face_index = payload.get("face_index", None)
    if face_index is not None:
        # Single box clicked: select by get_all_faces order (matches the overlay
        # boxes exactly), NOT the extract_face_images list which can skip faces
        # and shift indices → capturing the wrong person.
        faces_data = _face_data_at_index(idx, frame, int(face_index))
    else:
        faces_data = _faces_from_frame(idx, frame)
    next_id = (max(roop_globals.TARGET_FACE_GROUP) + 1) if roop_globals.TARGET_FACE_GROUP else 0
    for fd in faces_data:
        roop_globals.TARGET_FACES.append(fd[0])
        roop_globals.TARGET_FACE_GROUP.append(next_id)
        ui_globals.ui_target_thumbs.append(util.convert_to_gradio(fd[1]))
        next_id += 1
    return _target_faces_payload({"count": len(faces_data)})


@app.post("/api/target/add_angle")
def target_add_angle(payload: dict = Body(...)):
    """Add another angle (lateral/profile/upside-down) of an EXISTING target
    person, picking the face in the current frame closest to that person so
    matching survives pose changes (anti-flicker, multi-angle tracking)."""
    person = int(payload.get("person", 0))     # 0-based person rank
    idx = int(payload.get("index", state.selected_target_index))
    frame = int(payload.get("frame", 1))
    if idx >= len(list_files_process):
        return _target_faces_payload({"count": 0})
    ranks = _target_groups_ranked()
    raw_group = next((roop_globals.TARGET_FACE_GROUP[i] for i, r in enumerate(ranks) if r == person), None)
    if raw_group is None:
        return _target_faces_payload({"count": 0, "message": "no such person"})
    faces_data = _faces_from_frame(idx, frame)
    if not faces_data:
        return _target_faces_payload({"count": 0, "message": "no face in frame"})
    existing = [roop_globals.TARGET_FACES[i] for i, g in enumerate(roop_globals.TARGET_FACE_GROUP) if g == raw_group]
    existing_embs = [getattr(e, 'embedding', None) for e in existing if getattr(e, 'embedding', None) is not None]
    if not existing_embs:
        return _target_faces_payload({"count": 0, "message": "no valid embeddings for this person"})
    best_fd, best_d = None, 1e9
    for fd in faces_data:
        fd_emb = getattr(fd[0], 'embedding', None)
        if fd_emb is None:
            continue
        d = min(util.compute_cosine_distance(ee, fd_emb) for ee in existing_embs)
        if d < best_d:
            best_d, best_fd = d, fd
    if best_fd is not None:
        # Absolute sanity gate for a MANUAL capture. Deliberately loose: the user
        # picked this face on purpose, and a profile/steep-pitch frame of the same
        # person sits 0.7-1.0 from a frontal angle in scipy cosine distance (range
        # 0..2), so a tight gate here rejects exactly the lateral angles the bank
        # exists to collect — and a bank with no profiles is why lateral frames
        # stop being recognised mid-render. The meaningful protection against
        # capturing the wrong person is the RELATIVE check below (is some other
        # captured person a better explanation of this face?), which does not
        # depend on an absolute cutoff.
        if best_d > _ANGLE_MANUAL_MAX:
            return _target_faces_payload({"count": 0, "message": "face does not match this person"})
        other_embeddings = [np.asarray(roop_globals.TARGET_FACES[i].embedding, dtype=np.float32)
                            for i, g in enumerate(roop_globals.TARGET_FACE_GROUP)
                            if g != raw_group and getattr(roop_globals.TARGET_FACES[i], 'embedding', None) is not None]
        if other_embeddings:
            other_d = min(float(util.compute_cosine_distance(oe, best_fd[0].embedding)) for oe in other_embeddings)
            if other_d < best_d:
                return _target_faces_payload({"count": 0, "message": "face belongs to another target person"})
        roop_globals.TARGET_FACES.append(best_fd[0])
        roop_globals.TARGET_FACE_GROUP.append(raw_group)
        ui_globals.ui_target_thumbs.append(util.convert_to_gradio(best_fd[1]))
    return _target_faces_payload({"count": 1, "distance": round(float(best_d), 3)})


@app.post("/api/target/auto_angles")
def target_auto_angles(payload: dict = Body(...)):
    """Auto-harvest many pose angles of an EXISTING captured person from across
    the whole video into their multi-angle bank — so their identity survives
    turns/profiles without hand-capturing 30 frames.

    Runs in two phases, both matching each frame's faces against the person's
    GROWING angle bank (so gradual turns chain: each newly-added angle extends
    coverage for the next):

      1. a coarse pass over evenly-spaced samples across the whole video, and
      2. a targeted refine pass that re-scans, at fine stride, only the
         intervals BETWEEN coarse samples whose pose changed.

    Phase 2 is what stops angles being missed. A head turn lasts well under a
    second, so at a coarse stride the profile frames fall between samples; but
    pose is continuous, so a pose that never landed on a sample must lie inside
    an interval whose endpoints differ. Spending the extra decode there — rather
    than on uniformly more frames, which mostly returns more of the same frontal
    look — is what finds the extremes. The intermediate angles it picks up also
    chain the bank outward, which is what lets a true profile pass ACCEPT at all.

    Budget is per pose BIN (see _pose_bin) rather than one global count, so easy
    frontal angles can no longer consume the whole allowance before the hard
    poses are reached. Wrong grabs (rare) are removable via the per-angle ✕ in
    the UI. The swap already matches min-distance across a person's angles, so a
    fuller bank directly improves pose robustness."""
    from roop.face_util import get_all_faces, _attach_source_crops, clamp_cut_values
    from roop.face_quality import image_quality, blur_outlier
    if _progress["processing"]:
        return JSONResponse(status_code=409, content={"message": "busy processing"})

    person = int(payload.get("person", 0))
    idx = int(payload.get("index", state.selected_target_index))
    if idx >= len(list_files_process):
        return _target_faces_payload({"count": 0, "message": "no target"})
    target_path = list_files_process[idx].filename
    if util.is_image(target_path) and not target_path.lower().endswith("gif"):
        return _target_faces_payload({"count": 0, "message": "auto-angles needs a video target"})

    ranks = _target_groups_ranked()
    raw_group = next((roop_globals.TARGET_FACE_GROUP[i] for i, r in enumerate(ranks) if r == person), None)
    if raw_group is None:
        return _target_faces_payload({"count": 0, "message": "no such person"})

    # Seed the growing bank + pose coverage from the person's existing angles.
    bank, bin_counts = [], {}
    for i, g in enumerate(roop_globals.TARGET_FACE_GROUP):
        if g != raw_group:
            continue
        f = roop_globals.TARGET_FACES[i]
        e = getattr(f, 'embedding', None)
        if e is not None:
            bank.append(np.asarray(e, dtype=np.float32))
        kps = getattr(f, 'kps', None)
        b = _pose_bin(kps) if kps is not None else None
        if b is not None:
            bin_counts[b] = bin_counts.get(b, 0) + 1
    if not bank:
        return _target_faces_payload({"count": 0, "message": "capture the person once first"})

    other_embeddings = [np.asarray(roop_globals.TARGET_FACES[i].embedding, dtype=np.float32)
                        for i, g in enumerate(roop_globals.TARGET_FACE_GROUP)
                        if g != raw_group and getattr(roop_globals.TARGET_FACES[i], 'embedding', None) is not None]

    # Immutable anchor: the angles that were ALREADY trusted for this person
    # before this run (hand-captured, or previously accepted). The bank below is
    # allowed to grow so gradual turns chain in, but every newly accepted angle
    # must stay within SEED_MAX of one of THESE — otherwise a long video lets the
    # bank drift angle-by-angle onto a look-alike/bystander, and each drifted
    # angle then widens the person's match region so OTHER faces get swapped.
    seed = list(bank)

    # Defaults are tuned so PROFILE angles still qualify. In scipy cosine distance
    # (0..2) a true profile of the same person sits ~0.7-1.0 from a frontal seed,
    # so a SEED_MAX near 0.6 silently harvests only near-frontal angles — the bank
    # then fails to recognise the person the moment they turn, which reads as
    # "doesn't recognise faces sometimes" and makes the swap blink on and off.
    # Bank pollution is held off by the cross-person checks below (other_d /
    # AMBIG_MARGIN), which are relative and therefore don't punish hard poses.
    ACCEPT = float(payload.get("accept", _ANGLE_ACCEPT))       # same-identity cosine gate
    SEED_MAX = float(payload.get("seed_max", _ANGLE_SEED_MAX))  # max drift from an original angle
    AMBIG_MARGIN = float(payload.get("ambig_margin", 0.10))  # min gap to the runner-up
    NOVELTY = float(payload.get("novelty", 0.15))      # skip near-duplicate embeddings
    # MAX_ADD is now only a safety stop — the real budget is PER_BIN_CAP, so a
    # run can no longer spend its whole allowance on near-frontal angles and quit
    # before the profiles. `per_pose_cap` is still accepted as the old spelling.
    MAX_ADD = int(payload.get("max_add", 40))
    MAX_SAMPLES = int(payload.get("samples", 150))
    # Intake gates for the frames where the relative guards go silent, and the
    # review cutoff for the report — see the constants for the reasoning.
    MIN_QUALITY = float(payload.get("min_quality", _ANGLE_MIN_QUALITY))
    MIN_PX = float(payload.get("min_px", _ANGLE_MIN_PX))
    BLUR_FRAC = payload.get("blur_frac")            # None -> face_quality default
    BLUR_WARMUP = payload.get("blur_warmup")
    LONE_ACCEPT = float(payload.get("lone_accept", _ANGLE_LONE_ACCEPT))
    REVIEW = float(payload.get("review", _ANGLE_REVIEW))
    PER_BIN_CAP = int(payload.get("per_bin_cap", payload.get("per_pose_cap", 3)))
    REFINE_STEPS = int(payload.get("refine_steps", 6))          # sub-samples per interval
    MAX_REFINE = int(payload.get("refine_intervals", 24))       # intervals to revisit
    TIME_BUDGET = float(payload.get("time_budget", 90.0))       # seconds, whole scan

    total = get_video_frame_total(target_path) or 0
    if total <= 1:
        return _target_faces_payload({"count": 0, "message": "not a video"})
    stride = max(1, total // MAX_SAMPLES)
    roop_globals.target_path = target_path

    t0 = time.time()
    added = 0
    scanned = 0
    # What was banked (for the review report) and why faces were turned away.
    # Both are reported: a run that adds nothing because every candidate was
    # blurred is a different problem from one that found no matching face, and
    # without the counts the two are indistinguishable from the toast.
    banked = []
    rejected = collections.Counter()
    sharp_samples = []      # sharpness of every candidate, for the blur median
    # Phase 1 gets a soft share of the budget. Past it, the coarse pass keeps
    # SCANNING (the timeline has to span the whole video or phase 2 is blind to
    # the tail) but stops ADDING, so the targeted pass always has room left to
    # capture the hard poses it goes looking for.
    P1_ADDS = max(1, (MAX_ADD * 2) // 3)
    P1_TIME = TIME_BUDGET * 0.6

    def out_of_budget(time_cap=None):
        return added >= MAX_ADD or (time.time() - t0) > (TIME_BUDGET if time_cap is None else time_cap)

    def consider_frame(fpos, allow_add=True):
        """Detect on one frame; add the person's face if it fills a pose gap.

        Returns the person's pose bin on that frame (None if they weren't found),
        which phase 2 uses to decide which intervals are worth refining. Note the
        bin is returned even when nothing is added — a frame can be a useful
        *signpost* for a pose transition without itself being worth capturing."""
        nonlocal added, scanned
        img = get_video_frame(target_path, fpos)
        if img is None:
            return None
        scanned += 1
        faces = get_all_faces(img)
        if not faces:
            return None
        scored = []
        for f in faces:
            e = getattr(f, 'embedding', None)
            if e is None:
                continue
            e = np.asarray(e, dtype=np.float32)
            d = min(float(util.compute_cosine_distance(e, b)) for b in bank)
            scored.append((d, f, e))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        best_d, best_f, best_e = scored[0]
        if best_d > ACCEPT:
            return None
        # Anti-drift anchor: chaining against the growing bank follows gradual
        # turns, but is unbounded — over a long/crowded video it can walk onto a
        # different identity. Require every accepted angle to also stay within
        # SEED_MAX of an ORIGINAL angle, so the bank covers new poses but can
        # never drift to a new person (the cause of "other faces get swapped").
        seed_d = min(float(util.compute_cosine_distance(best_e, s)) for s in seed)
        if seed_d > SEED_MAX:
            return None
        # Check against other captured target persons: do not capture a face that
        # belongs or is closer to another target person already in TARGET_FACES.
        if other_embeddings:
            other_d = min(float(util.compute_cosine_distance(best_e, oe)) for oe in other_embeddings)
            if other_d <= best_d + AMBIG_MARGIN:
                return None
        # Ambiguity guard: in a multi-person frame, only accept when the winner
        # is CLEARLY the closest. A near-tie means look-alikes are present and
        # grabbing the wrong one would poison the bank for the whole swap run.
        if len(scored) > 1 and scored[1][0] < best_d + AMBIG_MARGIN:
            return None
        # Neither guard above could run: one face in the frame and no other
        # captured person to compare against. Nothing relative is watching, so
        # require a closer match to the bank — see _ANGLE_LONE_ACCEPT.
        if len(scored) == 1 and not other_embeddings and best_d > LONE_ACCEPT:
            rejected['unguarded frame'] += 1
            return None
        kps = getattr(best_f, 'kps', None)
        pose_bin = _pose_bin(kps) if kps is not None else None
        if pose_bin is None:
            return None
        (sx, sy, ex, ey) = best_f["bbox"].astype("int")
        sx, ex, sy, ey = clamp_cut_values(sx, ex, sy, ey, img)
        crop = img[sy:ey, sx:ex]
        if crop.size < 1:
            return pose_bin
        # Intake quality — the picture, never the pose. A blurred or tiny crop
        # has an embedding that sits near everybody, which is the realistic way
        # a face that is not this person gets past the absolute gates above.
        #
        # Measured for EVERY identity-accepted candidate, and deliberately
        # BEFORE the allow_add / novelty / per-bin-cap returns below. Those skip
        # most frames on a clip without much pose variety, so sampling after
        # them fed the blur median a handful of values, the warm-up count was
        # never reached, and the gate sat inert while blurred frames were banked
        # — verified by driving the endpoint, not by reading it. The median has
        # to describe what the clip OFFERS, not what survived the pose budget.
        # Cost is a crop slice and a Laplacian on an image already decoded, next
        # to the detection that produced it.
        qual, qbits = image_quality(best_f, crop)
        sharp = float(qbits.get('sharpness', 1.0))
        sharp_samples.append(sharp)
        if not allow_add:
            return pose_bin
        if qbits.get('face_px', 0) < MIN_PX:
            rejected['too small'] += 1
            return pose_bin
        if blur_outlier(sharp, sharp_samples, BLUR_FRAC, BLUR_WARMUP):
            rejected['blurred'] += 1
            return pose_bin
        if qual < MIN_QUALITY:
            rejected['low quality'] += 1
            return pose_bin
        # Skip only when it's BOTH a near-duplicate embedding AND that bin is
        # already represented — otherwise a new pose is always worth adding.
        if best_d < NOVELTY and bin_counts.get(pose_bin, 0) >= 1:
            return pose_bin
        if bin_counts.get(pose_bin, 0) >= PER_BIN_CAP:
            return pose_bin
        _attach_source_crops(best_f, img)
        roop_globals.TARGET_FACES.append(best_f)
        roop_globals.TARGET_FACE_GROUP.append(raw_group)
        ui_globals.ui_target_thumbs.append(util.convert_to_gradio(crop))
        bank.append(best_e)
        bin_counts[pose_bin] = bin_counts.get(pose_bin, 0) + 1
        banked.append({
            "index": len(roop_globals.TARGET_FACES) - 1,
            "seed_d": round(seed_d, 3),
            "quality": round(qual, 3),
            "face_px": qbits.get('face_px', 0),
            "frame": int(fpos),
        })
        added += 1
        return pose_bin

    # ── Phase 1: coarse pass over the whole video ────────────────────────────
    timeline = []   # (frame_pos, pose_bin | None) in time order
    for fpos in range(1, total + 1, stride):
        if out_of_budget(P1_TIME):
            break
        timeline.append((fpos, consider_frame(fpos, allow_add=added < P1_ADDS)))

    # ── Phase 2: refine the intervals where the pose actually changed ────────
    intervals = _pose_transition_intervals(timeline)
    for _span, fa, fb in intervals[:MAX_REFINE]:
        if out_of_budget():
            break
        sub = max(1, (fb - fa) // (REFINE_STEPS + 1))
        for fpos in range(fa + sub, fb, sub):
            if out_of_budget():
                break
            consider_frame(fpos)

    # ── Review report ────────────────────────────────────────────────────────
    # Pollution used to be silent: a wrong angle looks like any other thumbnail,
    # and its damage only shows up much later as the wrong person being swapped
    # (every swap-time gate takes the MINIMUM over a person's angles, so one bad
    # entry makes a stranger's whole track measure ~0 to this person). Surfacing
    # the angles nearest the drift limit turns "check all 30 thumbnails" into
    # "check these two". A true extreme profile lands here as well — this ranks
    # what to look at, it does not claim anything is wrong.
    review = sorted((b for b in banked if b["seed_d"] > REVIEW),
                    key=lambda b: -b["seed_d"])
    seed_ds = sorted(b["seed_d"] for b in banked)
    if banked:
        print(f'[AutoAngles] +{added} angles over {len(bin_counts)} pose bins, '
              f'{scanned} frames scanned; distance from the original capture '
              f'min {seed_ds[0]:.2f} / median {seed_ds[len(seed_ds) // 2]:.2f} / '
              f'max {seed_ds[-1]:.2f}'
              + (f'; {len(review)} angle(s) past {REVIEW} — worth a look'
                 if review else ''))
    if rejected:
        print('[AutoAngles] turned away: '
              + ', '.join(f'{n} {why}' for why, n in rejected.most_common()))

    return _target_faces_payload({
        "count": added,
        "scanned": scanned,
        "bins": len(bin_counts),
        "refined": min(len(intervals), MAX_REFINE),
        "seconds": round(time.time() - t0, 1),
        "banked": banked,
        "review": review,
        "rejected": dict(rejected),
        "seed_span": ({"min": seed_ds[0],
                       "median": seed_ds[len(seed_ds) // 2],
                       "max": seed_ds[-1]} if seed_ds else None),
    })


@app.post("/api/target/auto_capture")
def target_auto_capture(payload: dict = Body(...)):
    """Find each person in the clip and capture them from their own best frame.

    Replaces "scrub to a frame, click the boxes" as the way to get target
    people, because that flow picks the seed frame by whatever the user happens
    to be looking at — and on footage where people interact, the frames that
    obviously show two boxes are the frames where they TOUCH, which is the worst
    possible reference (roop/capture_seed.py has the measurements).

    Additive by default so a user can auto-capture and then keep hand-adding
    angles; pass `replace: true` to clear the existing people first, which is
    what the UI button does since re-running it otherwise duplicates everyone.
    """
    idx = int(payload.get("index", state.selected_target_index))
    if idx >= len(list_files_process):
        return _target_faces_payload({"count": 0, "message": "no target selected"})
    target_path = list_files_process[idx].filename
    if not util.is_video(target_path):
        return _target_faces_payload({"count": 0, "message": "auto-capture needs a video target"})
    roop_globals.target_path = target_path

    expect = payload.get("people")
    expect = int(expect) if expect else None
    result = capture_seed.auto_capture(
        target_path,
        expect=expect,
        time_budget=float(payload.get("time_budget", 90.0)),
    )
    if not result["targets"]:
        return _target_faces_payload({"count": 0,
                                      "message": "; ".join(result["notes"]) or "found nobody",
                                      "notes": result["notes"]})

    if bool(payload.get("replace", True)):
        roop_globals.TARGET_FACES.clear()
        roop_globals.TARGET_FACE_GROUP.clear()
        if getattr(roop_globals, 'TARGET_FACE_NAMES', None):
            roop_globals.TARGET_FACE_NAMES.clear()
        ui_globals.ui_target_thumbs.clear()

    next_id = (max(roop_globals.TARGET_FACE_GROUP) + 1) if roop_globals.TARGET_FACE_GROUP else 0
    from roop.face_util import _attach_source_crops, clamp_cut_values
    for face, grp, info in zip(result["targets"], result["groups"], result["per_person"]):
        # The scan keeps faces, not frames (487 frames of 1280x632 would be a
        # gigabyte). Re-decode just the handful actually chosen, for the crop the
        # thumbnail needs and for the source crops the image-source swap models
        # consume — both must come from the frame this face's kps belong to.
        #
        # +1 is not slop: capture_seed counts frames the way cv2 does
        # (CAP_PROP_POS_FRAMES, 0-based) while get_video_frame takes the UI's
        # 1-based number and subtracts one itself. Without the conversion every
        # crop comes from the frame BEFORE the face, which on a moving head
        # misplaces the thumbnail and silently warps _attach_source_crops'
        # aligned crops off the actual face.
        img = get_video_frame(target_path, int(info["frame"]) + 1)
        if img is None:
            continue
        sx, sy, ex, ey = face["bbox"].astype("int")
        sx, ex, sy, ey = clamp_cut_values(sx, ex, sy, ey, img)
        crop = img[sy:ey, sx:ex]
        if crop.size < 1:
            crop = img
        _attach_source_crops(face, img)
        roop_globals.TARGET_FACES.append(face)
        roop_globals.TARGET_FACE_GROUP.append(next_id + grp)
        ui_globals.ui_target_thumbs.append(util.convert_to_gradio(crop))

    # ── Grow each person into a multi-angle bank ─────────────────────────────
    # Measured on the full d6.mp4, same harness, same 8344 faces, only the
    # capture differing:
    #
    #     hand capture from a contact frame        25.0% not swapped
    #     one best-frontal face per person         27.5%
    #     best-frontal + this enrichment           15.8%
    #
    # So the angles are where the win is, not the seed: the enriched run above
    # used a WORSE seed by every measure this module reports (a frame flagged
    # "no clearly separated faces", one person captured at 40.6 degrees
    # off-axis against 11.1) and still beat the un-enriched one by 12 points,
    # purely from banking 21 more angles over 11 pose bins. "Track matched but
    # has no source" went 21.3% -> 11.7% on that change alone, which is more
    # than any assignment-gate change managed.
    #
    # A single reference face only matches the poses near it, and a person in
    # motion spends most of the clip somewhere else. Capturing one face per
    # person and stopping is the shape of the bug this endpoint was written to
    # fix, one level up.
    #
    # Same handler the UI's per-person "Auto-capture angles" button calls, so
    # this is the button doing for the user what they would otherwise do by
    # hand, not a second implementation of it.
    enriched = 0
    if bool(payload.get("enrich", True)):
        before = len(roop_globals.TARGET_FACES)
        for rank in range(len(set(result["groups"]))):
            try:
                target_auto_angles({"person": rank, "index": idx})
            except Exception as e:
                result["notes"].append(f"angle harvest failed for person {rank + 1}: {e}")
        enriched = len(roop_globals.TARGET_FACES) - before

    return _target_faces_payload({
        "count": len(result["targets"]),
        "angles_added": enriched,
        "separation": None if result["separation"] is None else round(float(result["separation"]), 3),
        "seed_frame": result["seed_index"],
        "per_person": result["per_person"],
        "scanned": result["scanned"],
        "seconds": result["seconds"],
        "complete": result["complete"],
        "notes": result["notes"],
    })


@app.post("/api/target/remove_face")
def target_remove_face(payload: dict = Body(...)):
    idx = int(payload.get("index", -1))
    if 0 <= idx < len(roop_globals.TARGET_FACES):
        roop_globals.TARGET_FACES.pop(idx)
    if 0 <= idx < len(roop_globals.TARGET_FACE_GROUP):
        roop_globals.TARGET_FACE_GROUP.pop(idx)
    if 0 <= idx < len(ui_globals.ui_target_thumbs):
        ui_globals.ui_target_thumbs.pop(idx)
    return _target_faces_payload()


@app.post("/api/target/clear_faces")
def target_clear_faces():
    """Remove every captured target person/angle, but keep the target media
    queue intact (unlike /api/target/clear which also drops the videos/images).
    Backs the 'Reset' button in the Target Faces panel."""
    roop_globals.TARGET_FACES.clear()
    roop_globals.TARGET_FACE_GROUP.clear()
    if getattr(roop_globals, 'TARGET_FACE_NAMES', None):
        roop_globals.TARGET_FACE_NAMES.clear()
    ui_globals.ui_target_thumbs.clear()
    return _target_faces_payload({"count": 0})


@app.post("/api/target/group")
def target_group(payload: dict = Body(...)):
    groups = payload.get("groups")
    if isinstance(groups, list):
        parsed = []
        for x in groups[:len(roop_globals.TARGET_FACES)]:
            try:
                parsed.append(int(x))
            except (ValueError, TypeError):
                parsed.append(0)
        roop_globals.TARGET_FACE_GROUP = parsed
    return _target_faces_payload()


@app.post("/api/target/name")
def target_name(payload: dict = Body(...)):
    """Give a person (by 0-based rank) a display name, e.g. 'Bride'. Stored by
    the person's stable raw group id so it survives rank shifts."""
    person = int(payload.get("person", 0))
    name = str(payload.get("name", "")).strip()[:40]
    ranks = _target_groups_ranked()
    raw_group = next((roop_globals.TARGET_FACE_GROUP[i]
                      for i, r in enumerate(ranks) if r == person), None)
    if raw_group is not None:
        if not hasattr(roop_globals, 'TARGET_FACE_NAMES') or roop_globals.TARGET_FACE_NAMES is None:
            roop_globals.TARGET_FACE_NAMES = {}
        if name:
            roop_globals.TARGET_FACE_NAMES[raw_group] = name
        else:
            roop_globals.TARGET_FACE_NAMES.pop(raw_group, None)
    return _target_faces_payload()


@app.post("/api/target/autocluster")
def target_autocluster(payload: dict = Body(...)):
    """Auto-assign every captured target face to a person by clustering their
    recognition embeddings — same identity within `threshold` cosine distance
    lands in one group. Replaces the current manual grouping."""
    threshold = float(payload.get("threshold", 0.55))
    faces = roop_globals.TARGET_FACES
    groups = [-1] * len(faces)
    next_id = 0
    for i, face in enumerate(faces):
        if groups[i] != -1:
            continue
        emb_i = getattr(face, 'embedding', None)
        groups[i] = next_id
        if emb_i is not None:
            for j in range(i + 1, len(faces)):
                if groups[j] != -1:
                    continue
                emb_j = getattr(faces[j], 'embedding', None)
                if emb_j is None:
                    continue
                try:
                    d = util.compute_cosine_distance(emb_i, emb_j)
                except Exception:
                    continue
                if d < threshold:
                    groups[j] = next_id
        next_id += 1
    roop_globals.TARGET_FACE_GROUP = groups
    # Names keyed by old raw ids are meaningless after a full re-cluster.
    if getattr(roop_globals, 'TARGET_FACE_NAMES', None):
        roop_globals.TARGET_FACE_NAMES.clear()
    return _target_faces_payload({"people": next_id})


# ── Run history (settings snapshot per produced output) ──────────────────────
# Every completed swap stores the exact settings payload next to the output
# names it produced, so any file in the Gallery can answer "how was this made?"
# and reload those settings for an A/B re-run.
HISTORY_FILE = "run_history.json"
_HISTORY_MAX = 200
# Payload keys that are per-run aliases/derived values, not settings — loading
# them back into the settings object would be stale or redundant.
_HISTORY_STRIP = {"face_mapping", "enhancer", "detection", "video_method",
                  "upscale", "clip_text", "face_distance", "autorotate"}


def _load_history() -> list:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_history(entries: list):
    try:
        tmp = HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entries[:_HISTORY_MAX], fh, indent=1)
        os.replace(tmp, HISTORY_FILE)
    except Exception:
        traceback.print_exc()


def _record_run_history(payload: dict, produced_files: list):
    """Called after a completed swap run. Never fatal."""
    try:
        if not produced_files:
            return
        snap = {k: v for k, v in payload.items() if k not in _HISTORY_STRIP}
        # Duration + throughput, best-effort from the run accumulator. Older
        # entries and runs where the status line never showed a frame count
        # simply omit these; the UI treats them as unknown.
        now = time.time()
        started = _run_stats.get("start") or 0.0
        duration_s = round(now - started, 1) if started and now > started else 0.0
        frames = _run_stats.get("frames_total") or _run_stats.get("frames_done") or 0
        fps = round(frames / duration_s, 1) if duration_s > 0 and frames else 0.0
        entry = {
            "id": int(now * 1000),
            "time": now,
            "outputs": [os.path.basename(f) for f in produced_files],
            "settings": snap,
            "duration_s": duration_s,
            "frames": frames,
            "fps": fps,
        }
        entries = _load_history()
        entries.insert(0, entry)
        _save_history(entries)
    except Exception:
        traceback.print_exc()


@app.get("/api/history")
def get_history():
    return {"entries": _load_history()}


@app.post("/api/history/delete")
def delete_history(payload: dict = Body(...)):
    try:
        rid = int(payload.get("id", 0))
    except (TypeError, ValueError):
        rid = 0
    entries = [e for e in _load_history() if e.get("id") != rid]
    _save_history(entries)
    return {"entries": entries}


# ── Profiles (persisted settings presets) ────────────────────────────────────
PROFILES_FILE = "profiles.json"


@app.get("/api/profiles")
def get_profiles():
    try:
        with open(PROFILES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {"profiles": data}
    except Exception:
        pass
    return {"profiles": []}


@app.post("/api/profiles")
def save_profiles(payload: dict = Body(...)):
    profiles = payload.get("profiles", [])
    if not isinstance(profiles, list):
        return JSONResponse(status_code=400, content={"message": "profiles must be a list"})
    try:
        tmp = PROFILES_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
        os.replace(tmp, PROFILES_FILE)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
    return {"status": "success", "count": len(profiles)}








# ── Person ids for the live-preview overlay ──────────────────────────────────
# The overlay numbers each detected face. To keep those numbers meaningful and
# stable, we bind each detected face to the nearest CAPTURED target person (the
# ones the user assigns sources to), so the box label matches that person's
# number and doesn't drift frame-to-frame. Faces matching no captured person are
# numbered after the known persons; with nothing captured yet we fall back to a
# deterministic left-to-right numbering. Ids are 0-based; the UI shows id + 1.
# Min cosine similarity to bind a detected face to a captured person. Kept low
# (0.20) so the SAME person keeps their number through hard frames (steep
# profile, motion blur, partial occlusion) where the embedding similarity dips —
# arcface same-person stays well above this while different people fall below it,
# so Person 1 remains Person 1 throughout without mislabeling strangers.
_PREVIEW_TARGET_MATCH_THRESH = 0.20


def _face_normed_emb(f):
    """Return a unit-length embedding for a detected face, or None."""
    e = None
    try:
        e = f["normed_embedding"]
    except Exception:
        e = getattr(f, "normed_embedding", None)
    if e is None:
        raw = None
        try:
            raw = f["embedding"]
        except Exception:
            raw = getattr(f, "embedding", None)
        if raw is not None:
            e = raw
    if e is None:
        return None
    e = np.asarray(e, dtype=np.float32).flatten()
    n = float(np.linalg.norm(e))
    return e / n if n > 0 else None


def _target_person_embs():
    """All captured embeddings grouped by person rank (multi-angle aware), so a
    detected face can be matched against each captured angle of a person rather
    than a blurred mean. Empty if nothing captured."""
    by_rank = {}
    try:
        captured = list(roop_globals.TARGET_FACES)
        if not captured:
            return by_rank
        ranks = _target_groups_ranked()
        for face, r in zip(captured, ranks):
            e = _face_normed_emb(face)
            if e is not None:
                by_rank.setdefault(r, []).append(e)
    except Exception:
        return {}
    return by_rank


def _preview_person_ids(idx, faces):
    """Number the detected faces for the overlay.

    Each detected face is bound to the captured target person whose closest
    captured angle best matches it, so the box label equals that person's
    number and stays put across the clip (one person per id per frame) even as
    they turn. Faces matching no captured person are numbered after the known
    persons. With nothing captured yet, fall back to deterministic left-to-right
    numbering. Ids are 0-based; the UI shows id + 1.
    """
    by_rank = _target_person_embs()
    if not by_rank:
        return list(range(len(faces)))          # positional, deterministic

    next_extra = max(by_rank) + 1
    ids, used = [], set()
    for f in faces:
        e = _face_normed_emb(f)
        pid = None
        if e is not None:
            best_r, best_s = None, _PREVIEW_TARGET_MATCH_THRESH
            for r, embs in by_rank.items():
                if r in used:
                    continue
                s = max(float(np.dot(e, b)) for b in embs)   # best-matching angle
                if s > best_s:
                    best_s, best_r = s, r
            if best_r is not None:
                pid = best_r
                used.add(best_r)
        if pid is None:
            pid = next_extra
            next_extra += 1
        ids.append(pid)
    return ids


def _apply_merger_settings(payload):
    """Push the DFL merger post-op knobs from a request onto roop.globals.

    Six numeric settings that /api/preview and /api/swap both need to apply
    identically — a preview that ignored them would be previewing something
    other than what the render produces. One helper rather than the same six
    lines twice, so the two paths cannot drift apart.
    """
    for key, default in (("merger_hist_match", 0.0),
                         ("merger_sharpen", 0.0),
                         ("merger_motion_blur", 0.0),
                         ("merger_grain_match", 0.0),
                         ("merger_degrade", 0.0),
                         ("merger_clarity", 0.0),
                         ("output_face_scale", 0.0)):
        fallback = getattr(roop_globals.CFG, key, default)
        try:
            value = float(payload.get(key, fallback))
        except (TypeError, ValueError):
            value = default
        setattr(roop_globals, key, value)


def _apply_eye_restore_settings(payload):
    """Push the eye-restore knobs from a request onto roop.globals.

    Same reasoning as _apply_merger_settings: /api/preview and /api/swap must
    apply them identically or the preview stops predicting the render. Kept off
    the ProcessOptions constructor that `restore_original_mouth` goes through,
    because that parameter is positional all the way down through
    core.batch_process_regular and the frozen Gradio tab.
    """
    fallback = getattr(roop_globals.CFG, "restore_original_eyes", False)
    roop_globals.restore_original_eyes = bool(payload.get("restore_original_eyes", fallback))
    for key, default in (("eyes_blend_amount", 1.0),
                         ("eyes_feather_blend", 25.0),
                         ("eyes_size_factor", 1.0),
                         ("eyes_radius_x", 1.0),
                         ("eyes_radius_y", 1.0)):
        fb = getattr(roop_globals.CFG, key, default)
        try:
            value = float(payload.get(key, fb))
        except (TypeError, ValueError):
            value = default
        setattr(roop_globals, key, value)


def _apply_enhancer_settings(payload):
    """Enhancer alignment + the second colour pass, onto roop.globals.

    Both are opt-in and both change pixels, so preview and run must agree —
    same reason as the other helpers here.
    """
    for key, default in (("enhancer_align", False),
                         ("color_match_after_enhance", False)):
        fallback = getattr(roop_globals.CFG, key, default)
        setattr(roop_globals, key, bool(payload.get(key, fallback)))


def _apply_lipsync_settings(payload):
    """Lip-sync (MuseTalk) toggle + audio source, onto roop.globals.

    lipsync_audio_path is deliberately NOT read from payload's own default —
    it comes only from CFG (there is no durable default for a per-job upload
    reference) or the payload itself when the client sends one.
    """
    for key, default in (("lipsync_enabled", False),
                         ("lipsync_audio_source", "original")):
        fallback = getattr(roop_globals.CFG, key, default)
        value = payload.get(key, fallback)
        setattr(roop_globals, key, bool(value) if key == "lipsync_enabled" else value)
    roop_globals.lipsync_audio_path = payload.get("lipsync_audio_path") or None


def _apply_parser_region_settings(payload):
    """Push the Face Parser region selection onto roop.globals.

    Its own helper rather than a tail on the eye one: these are not numbers —
    a list of group names and a {group: grow_px} map — and a helper whose name
    says "eye restore" while it also decides the mask is the kind of thing
    nobody finds until it is wrong.
    """
    regions = payload.get("parser_regions", getattr(roop_globals.CFG, "parser_regions", None))
    roop_globals.parser_regions = list(regions) if isinstance(regions, (list, tuple)) else None
    grow = payload.get("parser_region_grow", getattr(roop_globals.CFG, "parser_region_grow", None))
    roop_globals.parser_region_grow = dict(grow) if isinstance(grow, dict) else None


# ── Live preview swap ────────────────────────────────────────────────────────
@app.post("/api/preview")
def preview(payload: dict = Body(...)):
    """Render the selected target frame, optionally with a live face swap."""
    _update_mask_offsets_from_payload(payload)
    idx = int(payload.get("index", state.selected_target_index))
    frame = int(payload.get("frame", 1))
    fake = bool(payload.get("fake_preview", False))

    if idx >= len(list_files_process):
        return JSONResponse(status_code=404, content={"message": "no target"})

    filename = list_files_process[idx].filename
    if util.is_video(filename) or filename.lower().endswith("gif") or util.is_animated_webp(filename):
        current_frame = get_video_frame(filename, frame)
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None:
        return JSONResponse(status_code=404, content={"message": "no frame"})

    # Apply detection resolution before any detection so the face-box overlay and
    # the swap both use the chosen det_size (640 accurate / 320 fast).
    roop_globals.default_det_size = bool(payload.get("default_det_size", roop_globals.CFG.default_det_size))
    roop_globals.face_detector_size = str(payload.get("face_detector_size", roop_globals.CFG.face_detector_size))
    roop_globals.face_detector_threshold = float(payload.get("face_detector_threshold", roop_globals.CFG.face_detector_threshold))
    roop_globals.face_detector_nms = float(payload.get("face_detector_nms", roop_globals.CFG.face_detector_nms))
    roop_globals.refine_landmarks = bool(payload.get("refine_landmarks", getattr(roop_globals.CFG, "refine_landmarks", False)))
    roop_globals.swap_model_mask_strength = float(payload.get("swap_model_mask_strength", getattr(roop_globals.CFG, "swap_model_mask_strength", 0.0)))
    roop_globals.jaw_reshape = bool(payload.get("jaw_reshape", getattr(roop_globals.CFG, "jaw_reshape", False)))
    roop_globals.jaw_reshape_strength = float(payload.get("jaw_reshape_strength", getattr(roop_globals.CFG, "jaw_reshape_strength", 0.5)))
    roop_globals.detail_transfer_strength = float(payload.get("detail_transfer_strength", getattr(roop_globals.CFG, "detail_transfer_strength", 0.0)))
    roop_globals.expression_restore_strength = float(payload.get("expression_restore_strength", getattr(roop_globals.CFG, "expression_restore_strength", 0.0)))
    roop_globals.expression_restore_region = payload.get("expression_restore_region", getattr(roop_globals.CFG, "expression_restore_region", "all"))
    roop_globals.rescue_small_faces = bool(payload.get("rescue_small_faces", getattr(roop_globals.CFG, "rescue_small_faces", False)))
    roop_globals.detector_engine = payload.get("detector_engine", getattr(roop_globals.CFG, "detector_engine", "scrfd"))
    _apply_merger_settings(payload)
    _apply_eye_restore_settings(payload)
    _apply_parser_region_settings(payload)
    _apply_enhancer_settings(payload)
    _apply_lipsync_settings(payload)

    faces_list = []
    person_ids = []
    kps_list = []
    pose_list = []
    try:
        from roop.face_util import get_all_faces, solve_pose_5pt
        faces = get_all_faces(current_frame)
        if faces:
            for f in faces:
                bbox = f["bbox"].astype(int).tolist()
                faces_list.append(bbox)
                # 5-point keypoints, so a mask painted in frame space can carry
                # ref_kps and be warped into the aligned face crop.
                k = f.get("kps") if isinstance(f, dict) else getattr(f, "kps", None)
                kps = np.asarray(k).astype(float).tolist() if k is not None else None
                kps_list.append(kps)
                # Head pose for the debug overlay.
                #
                # Solved HERE rather than in the browser on purpose. The whole
                # value of showing an angle is that it is the same number the
                # pipeline gates on — the non-frontal mask router and the
                # angle-bank intake. A second implementation in
                # JS would drift from this one and quietly become a liar, which
                # is worse than showing nothing.
                pose = solve_pose_5pt(k) if k is not None else None
                pose_list.append([round(float(v), 1) for v in pose] if pose is not None else None)
            person_ids = _preview_person_ids(idx, faces)
    except Exception:
        pass

    if not fake or len(roop_globals.INPUT_FACESETS) < 1:
        return {"image": _bgr_to_preview_dataurl(current_frame), "faces": faces_list, "person_ids": person_ids, "kps": kps_list, "pose": pose_list}

    try:
        from roop.core import live_swap, get_processing_plugins
        from roop.ProcessOptions import ProcessOptions

        roop_globals.face_swap_mode = translate_swap_mode(payload.get("detection", "All faces"))
        roop_globals.selected_enhancer = payload.get("enhancer", "None")
        roop_globals.codeformer_fidelity = float(payload.get("codeformer_fidelity", 0.5))
        roop_globals.distance_threshold = float(payload.get("face_distance", roop_globals.CFG.max_face_distance))
        roop_globals.blend_ratio = float(payload.get("blend_ratio", 0.8))
        roop_globals.no_face_action = index_of_no_face_action(payload.get("no_face_action", "Retry rotated"))
        roop_globals.vr_mode = bool(payload.get("vr_mode", False))
        roop_globals.autorotate_faces = bool(payload.get("autorotate", True))
        roop_globals.subsample_size = int(str(payload.get("upscale", "256px"))[:3])
        roop_globals.execution_threads = roop_globals.CFG.max_threads
        roop_globals.color_transfer_mode = payload.get("color_transfer_mode", getattr(roop_globals.CFG, "color_transfer_mode", "rct"))
        roop_globals.sam2_model_size = payload.get("sam2_model_size", getattr(roop_globals.CFG, "sam2_model_size", "tiny"))
        # refine_landmarks / rescue_small_faces / detector_engine already set
        # above (before the box-overlay detection) so both paths agree.

        swap_model = payload.get("swap_model", "inswapper")
        mask_engine = map_mask_engines(payload.get("mask_engine", "None"),
                                       payload.get("mask_engine_2", "None"),
                                       payload.get("clip_text", ""))
        face_index = state.selected_input_face_index
        if len(roop_globals.INPUT_FACESETS) <= face_index:
            face_index = 0
        face_mapping = payload.get("face_mapping")
        mapped = mapped_facesets(face_mapping, roop_globals.face_swap_mode)
        face_index = mapped_selected_index(face_mapping, mapped, face_index)

        options = ProcessOptions(
            get_processing_plugins(mask_engine, swap_model=swap_model),
            roop_globals.distance_threshold, roop_globals.blend_ratio,
            roop_globals.face_swap_mode, face_index, payload.get("clip_text", ""),
            # Manual brush mask from the preview box. ProcessMgr parses this JSON
            # ({"<faceset>": {exclude, canonical, ref_kps}}); '' means none.
            payload.get("imagemask") or None,
            int(payload.get("num_swap_steps", 1)), roop_globals.subsample_size,
            bool(payload.get("show_mask_offsets", False)),
            bool(payload.get("restore_original_mouth", False)),
            use_3d_recon=bool(payload.get("use_3d_recon", False)),
            use_source_bank=bool(payload.get("use_source_bank", False)),
            use_frontalization=bool(payload.get("use_frontalization", False)),
            frontalization_threshold=float(payload.get("frontalization_threshold", 30.0)),
            swap_model=swap_model,
            stabilize_method=payload.get("stabilize_method", "one_euro"),
            stabilize_face=bool(payload.get("stabilize_face", False)))

        swapped = live_swap(current_frame, options, input_facesets=mapped)
        if swapped is None:
            return {"image": _bgr_to_preview_dataurl(current_frame), "faces": faces_list, "person_ids": person_ids, "kps": kps_list, "pose": pose_list}
        return {"image": _bgr_to_preview_dataurl(swapped), "faces": faces_list, "person_ids": person_ids, "kps": kps_list, "pose": pose_list}
    except Exception:
        traceback.print_exc()
        return {"image": _bgr_to_preview_dataurl(current_frame), "faces": faces_list, "person_ids": person_ids, "kps": kps_list, "pose": pose_list, "error": "swap failed"}


@app.post("/api/preview_upscale")
def preview_upscale(payload: dict = Body(...)):
    """AI-upscale a single already-swapped preview frame — a cheap spot-check of
    the final upscale quality on one frame. Operates on the image the client
    currently shows (passed in as a data-URL); it does NOT re-run the swap
    pipeline, so it can't race the live preview's GPU sessions."""
    img = _dataurl_to_bgr(payload.get("image", ""))
    if img is None:
        return JSONResponse(status_code=400, content={"message": "no image to upscale"})

    subtype = payload.get("subtype") or getattr(roop_globals.CFG, "upscale_model_after", "esrganx2")

    # Fast classical upscale (lanczos/fsr/spline/sinc): a plain resize, no model
    # — spot-check it instantly.
    spec = _classical_spec(subtype)
    if spec is not None:
        mode, scale = spec
        out = _classical_image_apply(img, mode, scale)
        oh, ow = out.shape[:2]
        return {"image": _bgr_to_dataurl(out), "width": int(ow), "height": int(oh)}

    if subtype not in _FRAME_UPSCALERS:
        subtype = "esrganx2"

    from ui.main import prepare_environment
    prepare_environment()   # ensure the Frame/* upscale model is downloaded
    try:
        proc = _make_frame_processor("upscale", subtype)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"failed to load model: {e}"})
    try:
        out = proc.Run(img)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"upscale failed: {e}"})
    finally:
        try:
            proc.Release()
        except Exception:
            pass

    h, w = out.shape[:2]
    return {"image": _bgr_to_dataurl(out), "width": int(w), "height": int(h)}


# ── Run the swap ─────────────────────────────────────────────────────────────
@app.post("/api/swap")
def trigger_swap(payload: dict = Body(...)):
    if _progress["processing"]:
        return JSONResponse(status_code=409, content={"message": "already processing"})
    # The benchmark holds several pools of TensorRT contexts and is timing them.
    # Starting a render on top of that gives the render a card that is already
    # busy AND makes the benchmark recommend settings for a machine nobody has —
    # the refusal is mutual (see /api/settings/benchmark_threads).
    if _benchmark_state["running"]:
        return JSONResponse(status_code=409, content={
            "message": "The hardware benchmark is running — cancel it first, or "
                       "wait for it to finish."})
    if len(list_files_process) < 1:
        return JSONResponse(status_code=400, content={"message": "no target media"})
    if len(roop_globals.INPUT_FACESETS) < 1:
        return JSONResponse(status_code=400, content={"message": "no source faces"})

    # Claim the processing flag synchronously — the worker thread also sets it,
    # but only after it starts, so two rapid POSTs could otherwise both pass
    # the guard above and run concurrently.
    _progress.update({"processing": True, "paused": False, "progress": 0.0, "desc": "Starting…", "error": ""})
    threading.Thread(target=_run_swap, args=(payload,), daemon=True).start()
    return {"status": "started"}


def _run_swap(payload):
    from ui.main import prepare_environment
    from roop.core import batch_process_regular

    roop_globals.pause = False
    _stop_requested["flag"] = False
    # Fresh terminal feed for this run.
    _log_lines.clear()
    _log_state.update({"last": "", "last_ts": 0.0, "seq": 0, "last_err": "",
                       "status": "", "parts_seen": 0, "counter_shapes": set()})
    # Parts are per-run: clear now rather than waiting for the writer, which is
    # only constructed once encoding starts (and never, for an image job).
    segment_writer.reset_parts()
    # Likewise the live frame — otherwise a new run opens showing the last
    # frame of the previous one.
    live_preview.reset()
    # And the previous run's "time left", so this one's opening seconds fall back
    # to the UI's own estimate rather than inheriting a finished run's figure.
    _procmgr_runtime.reset_eta()
    _push_log("▶ Starting job…", force=True)
    _progress.update({"processing": True, "paused": False, "progress": 0.0, "desc": "Starting…", "error": ""})
    _run_stats.update({"start": time.time(), "frames_done": 0, "frames_total": 0})
    try:
        # Inside the try so any failure (e.g. CFG.save() I/O error) still hits
        # the finally block and clears the processing flag.
        _update_mask_offsets_from_payload(payload)
        prepare_environment()
        if roop_globals.CFG.clear_output:
            shutil.rmtree(roop_globals.output_path, ignore_errors=True)
            os.makedirs(roop_globals.output_path, exist_ok=True)

        enhancer = payload.get("enhancer", roop_globals.CFG.selected_enhancer)
        detection = payload.get("detection", roop_globals.CFG.face_detection_mode)
        output_method = payload.get("output_method", roop_globals.CFG.output_method)
        processing_method = payload.get("video_method", roop_globals.CFG.video_swapping_method)
        upsample = payload.get("upscale", roop_globals.CFG.subsample_upscale)
        selected_mask_engine = payload.get("mask_engine", roop_globals.CFG.mask_engine)
        selected_mask_engine_2 = payload.get(
            "mask_engine_2", getattr(roop_globals.CFG, "mask_engine_2", "None"))
        clip_text = payload.get("clip_text", roop_globals.CFG.mask_clip_text)

        roop_globals.selected_enhancer = enhancer
        roop_globals.codeformer_fidelity = float(payload.get("codeformer_fidelity", getattr(roop_globals.CFG, "codeformer_fidelity", 0.5)))
        roop_globals.target_path = None
        roop_globals.distance_threshold = float(payload.get("face_distance", roop_globals.CFG.max_face_distance))
        roop_globals.blend_ratio = float(payload.get("blend_ratio", roop_globals.CFG.blend_ratio))
        roop_globals.keep_frames = bool(payload.get("keep_frames", roop_globals.CFG.keep_frames))
        roop_globals.wait_after_extraction = bool(payload.get("wait_after_extraction", roop_globals.CFG.wait_after_extraction))
        roop_globals.skip_audio = bool(payload.get("skip_audio", roop_globals.CFG.skip_audio))
        roop_globals.face_swap_mode = translate_swap_mode(detection)
        roop_globals.default_det_size = bool(payload.get("default_det_size", roop_globals.CFG.default_det_size))
        roop_globals.face_detector_size = str(payload.get("face_detector_size", roop_globals.CFG.face_detector_size))
        roop_globals.face_detector_threshold = float(payload.get("face_detector_threshold", roop_globals.CFG.face_detector_threshold))
        roop_globals.face_detector_nms = float(payload.get("face_detector_nms", roop_globals.CFG.face_detector_nms))
        roop_globals.sam2_model_size = payload.get("sam2_model_size", getattr(roop_globals.CFG, "sam2_model_size", "tiny"))
        roop_globals.track_identities = bool(payload.get("track_identities", getattr(roop_globals.CFG, "track_identities", False)))
        roop_globals.no_face_action = index_of_no_face_action(payload.get("no_face_action", roop_globals.CFG.no_face_action))
        roop_globals.vr_mode = bool(payload.get("vr_mode", roop_globals.CFG.vr_mode))
        roop_globals.autorotate_faces = bool(payload.get("autorotate", roop_globals.CFG.autorotate_faces))
        roop_globals.subsample_size = int(str(upsample)[:3])
        roop_globals.upscale_after_swap = bool(payload.get("upscale_after_swap", getattr(roop_globals.CFG, "upscale_after_swap", True)))
        roop_globals.upscale_model_after = payload.get("upscale_model_after", getattr(roop_globals.CFG, "upscale_model_after", "esrganx2"))
        if getattr(roop_globals.CFG, 'auto_thread_selection', True):
            _mode = 'standard'
            if enhancer and enhancer != 'None':
                _mode = 'enhanced'
            if selected_mask_engine and selected_mask_engine != 'None' and selected_mask_engine != 'DFL XSeg':
                _mode = 'heavy'
            if payload.get('expression_restore_strength', 0) > 0 or payload.get('lipsync_enabled'):
                _mode = 'heavy'
            roop_globals.execution_threads = roop_globals.CFG.resolve_threads(_mode)
            print(f"[Auto Thread Selection] Resolved {roop_globals.execution_threads} threads for '{_mode}' workload mode.", flush=True)
        else:
            roop_globals.execution_threads = roop_globals.CFG.max_threads
        roop_globals.color_transfer_mode = payload.get("color_transfer_mode", roop_globals.CFG.color_transfer_mode)
        roop_globals.refine_landmarks = bool(payload.get("refine_landmarks", roop_globals.CFG.refine_landmarks))
        roop_globals.swap_model_mask_strength = float(payload.get("swap_model_mask_strength", getattr(roop_globals.CFG, "swap_model_mask_strength", 0.0)))
        roop_globals.jaw_reshape = bool(payload.get("jaw_reshape", getattr(roop_globals.CFG, "jaw_reshape", False)))
        roop_globals.jaw_reshape_strength = float(payload.get("jaw_reshape_strength", getattr(roop_globals.CFG, "jaw_reshape_strength", 0.5)))
        roop_globals.detail_transfer_strength = float(payload.get("detail_transfer_strength", getattr(roop_globals.CFG, "detail_transfer_strength", 0.0)))
        roop_globals.expression_restore_strength = float(payload.get("expression_restore_strength", getattr(roop_globals.CFG, "expression_restore_strength", 0.0)))
        roop_globals.expression_restore_region = payload.get("expression_restore_region", getattr(roop_globals.CFG, "expression_restore_region", "all"))
        roop_globals.rescue_small_faces = bool(payload.get("rescue_small_faces", roop_globals.CFG.rescue_small_faces))
        roop_globals.detector_engine = payload.get("detector_engine", roop_globals.CFG.detector_engine)
        roop_globals.temporal_detection = bool(payload.get("temporal_detection", getattr(roop_globals.CFG, "temporal_detection", False)))
        _apply_merger_settings(payload)
        _apply_eye_restore_settings(payload)
        _apply_parser_region_settings(payload)
        _apply_enhancer_settings(payload)
        _apply_lipsync_settings(payload)
        roop_globals.video_encoder = roop_globals.CFG.output_video_codec
        roop_globals.video_quality = roop_globals.CFG.video_quality
        roop_globals.max_memory = roop_globals.CFG.memory_limit if roop_globals.CFG.memory_limit > 0 else None

        mask_engine = map_mask_engines(selected_mask_engine, selected_mask_engine_2, clip_text)

        if roop_globals.face_swap_mode == "selected" and len(roop_globals.TARGET_FACES) < 1:
            _progress.update({"processing": False, "error": "No target face selected"})
            return

        roop_globals.processing = True
        target_idx = payload.get("target_index")
        if target_idx is not None:
            try:
                target_idx = int(target_idx)
                if 0 <= target_idx < len(list_files_process):
                    files_to_process = [list_files_process[target_idx]]
                else:
                    files_to_process = list_files_process
            except Exception:
                files_to_process = list_files_process
        else:
            files_to_process = list_files_process

        # Flush VRAM and Python garbage collector before starting execution to ensure max headroom
        try:
            import torch
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Stash the settings signature so the completion hook (core.py) can
        # record actual ms/frame for the learned runtime estimator, using the
        # same signature the /api/runtime_estimate endpoint predicts from.
        try:
            from roop import runtime_calib
            roop_globals._run_signature = runtime_calib.signature_from_payload(
                payload, gpu=_gpu_name(),
                threads=roop_globals.CFG.max_threads,
                precision=getattr(roop_globals.CFG, 'trt_precision', 'mixed'))
        except Exception:
            roop_globals._run_signature = None

        # Snapshot the output dir so we can tell which files THIS run produces
        # (for the optional AI upscale second pass below).
        _pre_swap_outputs = _snapshot_output_mtimes()

        _stages = "analyze → swap"
        if roop_globals.upscale_after_swap:
            _stages += " → upscale → combine"
        print(f"\n===== SWAP PIPELINE: {_stages} =====", flush=True)
        print("[Stage 1/2] ANALYZE + SWAP (per-frame detection & swapping)…", flush=True)

        run_mapping = payload.get("face_mapping")
        run_facesets = mapped_facesets(run_mapping, roop_globals.face_swap_mode)
        batch_process_regular(
            output_method, files_to_process, mask_engine, clip_text,
            processing_method == "In-Memory processing",
            # imagemask — the preview box's brush mask. Was hardcoded None, so a
            # painted mask was silently dropped on the way to the render.
            payload.get("imagemask") or None,
            bool(payload.get("restore_original_mouth", roop_globals.CFG.restore_original_mouth)),
            int(payload.get("num_swap_steps", roop_globals.CFG.num_swap_steps)),
            ApiProgress(),
            mapped_selected_index(run_mapping, run_facesets, state.selected_input_face_index),
            use_3d_recon=bool(payload.get("use_3d_recon", roop_globals.CFG.use_3d_recon)),
            mask_per_frame_json="",
            use_source_bank=bool(payload.get("use_source_bank", roop_globals.CFG.use_source_bank)),
            use_frontalization=bool(payload.get("use_frontalization", roop_globals.CFG.use_frontalization)),
            frontalization_threshold=float(payload.get("frontalization_threshold", roop_globals.CFG.frontalization_threshold)),
            swap_model=payload.get("swap_model", roop_globals.CFG.swap_model),
            stabilize_face=bool(payload.get("stabilize_face", roop_globals.CFG.stabilize_face)),
            stabilize_method=payload.get("stabilize_method", roop_globals.CFG.stabilize_method),
            stabilize_min_cutoff=float(payload.get("stabilize_min_cutoff", roop_globals.CFG.stabilize_min_cutoff)),
            stabilize_beta=float(payload.get("stabilize_beta", roop_globals.CFG.stabilize_beta)),
            stabilize_enhancer=bool(payload.get("stabilize_enhancer", roop_globals.CFG.stabilize_enhancer)),
            stabilize_enhancer_strength=float(payload.get("stabilize_enhancer_strength", roop_globals.CFG.stabilize_enhancer_strength)),
            stabilize_mask=bool(payload.get("stabilize_mask", roop_globals.CFG.stabilize_mask)),
            stabilize_mask_strength=float(payload.get("stabilize_mask_strength", roop_globals.CFG.stabilize_mask_strength)),
            input_facesets=run_facesets)

        # ── AI upscale second pass (opt-in) ─────────────────────────────────
        # Upscale each finished output in place so the final result is a single
        # file per target (face swap + AI upscale baked in, audio preserved).
        # Skipped when the run was deliberately stopped.
        if getattr(roop_globals, "upscale_after_swap", False) and not _stop_requested["flag"]:
            # CRITICAL: free the swap's GPU memory FIRST. The swap (TensorRT +
            # detection pool + enhancer/mask) leaves ~12GB resident; if the
            # upscale sessions run on top of that they spill into shared system
            # RAM and crawl (26s/frame). release_resources() drops the ProcessMgr
            # + FaceAnalysis pool + empties the CUDA cache, freeing VRAM for the
            # upscale. The next swap re-initialises them (TRT engines are cached).
            # The printed "VRAM freed" line makes it visible whether it worked.
            def _free_gb():
                try:
                    import torch as _t
                    if _t.cuda.is_available():
                        return _t.cuda.mem_get_info(roop_globals.cuda_device_id)[0] / (1024 ** 3)
                except Exception:
                    pass
                return 0.0
            free_before = _free_gb()
            try:
                from roop.core import release_resources
                release_resources()
                import gc as _gc
                _gc.collect()
                import torch as _t
                if _t.cuda.is_available():
                    with _t.cuda.device(roop_globals.cuda_device_id):
                        _t.cuda.empty_cache()
                        _t.cuda.ipc_collect()
            except Exception:
                traceback.print_exc()
            free_after = _free_gb()
            print(f"[Upscale] freed swap VRAM before upscale: "
                  f"{free_before:.1f} → {free_after:.1f} GB free", flush=True)
            try:
                _run_post_swap_upscale(
                    _outputs_since(_pre_swap_outputs),
                    getattr(roop_globals, "upscale_model_after", "esrganx2"))
            except Exception:
                traceback.print_exc()

        # ── Frame interpolation pass (opt-in) ───────────────────────────────
        # Raises the output frame rate with motion-interpolated in-betweens
        # (RIFE, or ffmpeg minterpolate). Runs after the upscale pass so the
        # heavy AI upscaler only touches the original frame count.
        interp_mode = str(payload.get("interp_after_swap",
                                      getattr(roop_globals.CFG, "interp_after_swap", "off")) or "off")
        if interp_mode != "off" and not _stop_requested["flag"]:
            try:
                _run_post_swap_interp(_outputs_since(_pre_swap_outputs), interp_mode)
            except Exception:
                traceback.print_exc()

        _progress["progress"] = 1.0
        _progress["desc"] = "Done"
        _push_log("✓ Done", force=True)
        _record_run_history(payload, _outputs_since(_pre_swap_outputs))
        _record_last_output()
    except Exception as e:
        traceback.print_exc()
        _progress["error"] = str(e)
        _push_log("⚠ " + str(e), force=True)
    finally:
        roop_globals.pause = False
        # Safety net: normally end_processing() clears this, but if batch_process
        # raised before reaching it, clear here so a later terminal Ctrl-C doesn't
        # block in destroy() waiting on a batch that's no longer running.
        roop_globals.batch_active = False
        _progress["processing"] = False
        _progress["paused"] = False


# ── AI upscale / interpolation second pass ─────────────────────────────────
# Moved to post_swap.py — see that module's docstring. Imported back under
# the original names so every call site in this file is unchanged.
from post_swap import _snapshot_output_mtimes, _outputs_since, _classical_spec, _classical_image_apply, _run_post_swap_upscale, _run_post_swap_interp
import post_swap as _post_swap  # noqa: E402


def _record_last_output():
    out = roop_globals.output_path
    if not out or not os.path.isdir(out):
        return
    files = [os.path.join(out, f) for f in os.listdir(out)
             if not f.startswith(".") and os.path.isfile(os.path.join(out, f))]
    if not files:
        return
    latest = max(files, key=os.path.getmtime)
    kind = "video" if util.is_video(latest) else ("image" if util.is_image(latest) else "file")
    _last_output.update({"path": latest, "kind": kind})


@app.post("/api/stop")
def stop_swap():
    # Clear pause too so a stop while paused fully aborts (wait loop checks both).
    roop_globals.pause = False
    roop_globals.processing = False
    _stop_requested["flag"] = True
    _progress["paused"] = False
    _progress["desc"] = "Aborting…"
    return {"status": "stopping"}


@app.post("/api/pause")
def pause_swap():
    if not _progress["processing"]:
        # No active job — clicking Pause (e.g. the Pinokio sidebar button) while
        # idle is a harmless no-op, not an error. Returning 200 keeps the sidebar
        # script from surfacing a red failure on a misclick. The React run-bar
        # only shows Pause while processing, so it never relies on the 409.
        return {"status": "idle"}
    roop_globals.pause = True
    _progress["paused"] = True
    _progress["desc"] = "Paused"
    return {"status": "paused"}


@app.post("/api/resume")
def resume_swap():
    roop_globals.pause = False
    _progress["paused"] = False
    _progress["desc"] = "Resuming…"
    return {"status": "resumed"}


@app.get("/api/progress")
def get_progress():
    # The UI no longer streams live swapped frames into the preview box (it shows
    # a progress bar + elapsed/ETA instead), so we skip the per-poll JPEG encode
    # of the latest frame entirely. The fields are kept (empty) for shape
    # compatibility with any older client.
    # Mirror the current status line into the rolling terminal feed. Doing it here
    # (rather than at every _progress["desc"] = ... site) captures every stage's
    # output — swap, upscale, interpolate, combine — from one place.
    if _progress["processing"]:
        # Dedup the error against its own key — otherwise the desc push below
        # overwrites _log_state["last"] and the same error re-appends every poll.
        err = _progress.get("error")
        if err and err != _log_state.get("last_err"):
            _log_state["last_err"] = err
            _push_log("⚠ " + err, force=True)
        desc = _progress.get("desc", "")
        # The pinned line always mirrors the live status (including stages whose
        # desc carries no counter, e.g. "Analyzing faces", so it never goes
        # stale); the log keeps only the descs that are not per-frame counters.
        if desc:
            _log_state["status"] = desc
        _push_log(desc)
        # Announce each output part as it is finalized. Reported from here rather
        # than from the writer so segment_writer stays free of UI plumbing — and
        # the console is the only place a long run says what is already safe on
        # disk. Tagged with its own part so it lands in that tab.
        try:
            done_parts = [p for p in segment_writer.parts_snapshot() if p.get("done")]
        except Exception:
            done_parts = []
        seen = int(_log_state.get("parts_seen", 0) or 0)
        if len(done_parts) > seen:
            for p in done_parts[seen:]:
                _push_log(f"✓ part {p['index']} written · frames {p['first']}–{p['last']}"
                          f" · {p['bytes'] / 1048576:.0f} MB"
                          + (" (resumed)" if p.get("inherited") else ""),
                          force=True, part=p["index"])
            _log_state["parts_seen"] = len(done_parts)
        # Track the furthest "done / total" seen so the run can be summarised in
        # history. Max (not last) because the count restarts per stage; a stage
        # never has more frames than the source, so the peak total is the real
        # frame count and the peak done is a floor on frames actually processed.
        if _run_stats["start"]:
            m = _FRAME_RE.search(desc)
            if m:
                try:
                    done = int(m.group(1).replace(",", ""))
                    total = int(m.group(2).replace(",", ""))
                    if done > _run_stats["frames_done"]:
                        _run_stats["frames_done"] = done
                    if total > _run_stats["frames_total"]:
                        _run_stats["frames_total"] = total
                except ValueError:
                    pass
    try:
        parts = segment_writer.parts_snapshot()
    except Exception:
        parts = []
    # live_seq changes whenever the pipeline publishes a newer frame; the UI
    # uses it as /api/live_frame's cache key, so the image refetches exactly
    # when there is something new and never per poll. live_frame stays empty —
    # inlining a base64 still into every poll is what made this expensive.
    return {**_progress, "output": _last_output, "live_frame": "",
            "live_seq": live_preview.seq(),
            # Seconds remaining as the TERMINAL's progress bar is showing them.
            # None between stages (nothing is counting frames during encode/mux)
            # and during start-up, where the UI falls back to its own estimate.
            "eta_s": _procmgr_runtime.eta_seconds(),
            # Epoch seconds the run started, so elapsed/ETA survive a webview
            # reload (Pinokio reloads it on every tab switch). The UI used to
            # restart its own clock from zero there, which made a 40-minute run
            # read as "0s" and the ETA jump.
            "started_at": float(_run_stats.get("start") or 0.0),
            "log": list(_log_lines), "parts": parts,
            # The counter the console pins and rewrites in place instead of
            # scrolling — `desc` when it IS a counter, else the last one seen.
            "status_line": _log_state.get("status", "")}


@app.get("/api/live_frame")
def live_frame(seq: int = 0):
    """The newest processed frame, as a small JPEG.

    Already encoded by the pipeline (roop/live_preview.py), so this hands back
    bytes — no numpy, no lock held across an encode, and any number of pollers
    cost the same. `seq` is only a cache-buster: the UI puts the live_seq from
    /api/progress in the URL so the browser refetches exactly when a newer frame
    exists. 204 while nothing has been published (before the first frame, or
    with ROOP_LIVE_PREVIEW=0), which the UI reads as "show the still instead".
    """
    # Tells the pipeline someone is actually watching. Without a reader the
    # publish drops to a slow keep-alive cadence, so a run with this tab hidden
    # (or Pinokio parked on the Terminal) stops encoding frames for nobody.
    # Noted even on the 204 path — asking and getting nothing is still watching.
    live_preview.note_fetch()
    data, cur_seq, size = live_preview.snapshot()
    if not data:
        return Response(status_code=204)
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store",
                             "X-Live-Seq": str(cur_seq),
                             "X-Source-Size": f"{size[0]}x{size[1]}"})


@app.get("/api/output")
def list_output():
    out = getattr(roop_globals, "output_path", None)
    items = []
    if out and os.path.isdir(out):
        for f in sorted(os.listdir(out), reverse=True):
            full = os.path.join(out, f)
            if os.path.isfile(full) and not f.startswith("."):
                kind = "video" if util.is_video(full) else ("image" if util.is_image(full) else "file")
                items.append({"name": f, "kind": kind, "mtime": os.path.getmtime(full),
                              "size": os.path.getsize(full)})
    return {"output_path": out, "files": items[:50]}


@app.post("/api/output/delete")
def delete_output(payload: dict = Body(...)):
    filename = payload.get("name")
    out = getattr(roop_globals, "output_path", None)
    if not filename or not out or not os.path.isdir(out):
        return JSONResponse(status_code=400, content={"message": "invalid parameters"})
    filename = os.path.basename(filename)
    full_path = os.path.join(out, filename)
    if os.path.isfile(full_path):
        try:
            os.remove(full_path)
            global _last_output
            if _last_output.get("path") == full_path:
                _last_output.update({"path": "", "kind": ""})
            return {"status": "success"}
        except Exception as e:
            return JSONResponse(status_code=500, content={"message": f"failed to delete file: {e}"})
    return JSONResponse(status_code=404, content={"message": "file not found"})


@app.post("/api/reveal")
def reveal_output(payload: dict = Body(default={})):
    """Open the OS file manager at the output folder (optionally selecting a file)."""
    target = payload.get("path") or getattr(roop_globals, "output_path", None)
    if not target:
        return JSONResponse(status_code=404, content={"message": "no output folder"})
    target = os.path.abspath(target)
    is_file = os.path.isfile(target)
    folder = os.path.dirname(target) if is_file else target
    if not os.path.isdir(folder):
        return JSONResponse(status_code=404, content={"message": "folder not found"})
    try:
        if sys.platform.startswith("win"):
            if is_file:
                subprocess.Popen(["explorer", "/select,", target])
            else:
                os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", target] if is_file else ["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
    return {"status": "ok", "folder": folder}


def _open_shared(path: str):
    """Open *path* for reading in a way that does NOT lock it against move/delete.

    A normal open() on Windows omits FILE_SHARE_DELETE, so for the life of the
    handle the OS refuses to move/rename/delete the file ("the file is open in
    Python"). While a <video> element streams a finished output, that handle is
    alive — so the user can't move the result out of the output folder. We open
    via CreateFileW with all three share flags (READ|WRITE|DELETE) so Explorer can
    still move or delete the file even while we're streaming it. On non-Windows,
    POSIX already allows unlink/rename of open files, so a plain open() is fine.
    """
    if os.name != "nt":
        return open(path, "rb")
    import ctypes
    import msvcrt
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ, FILE_SHARE_WRITE, FILE_SHARE_DELETE = 0x1, 0x2, 0x4
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    CreateFileW = ctypes.windll.kernel32.CreateFileW
    CreateFileW.restype = ctypes.c_void_p
    CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                            ctypes.c_void_p]
    handle = CreateFileW(path, GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if not handle or handle == INVALID_HANDLE_VALUE:
        # Fall back to a normal open rather than failing the request outright.
        return open(path, "rb")
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    return os.fdopen(fd, "rb")


@app.get("/api/file")
def get_file(path: str, request: Request):
    """Serve an output/temp file by absolute path (guarded to known dirs).

    Streams the file (with HTTP Range support, so video seeking is instant) using
    a share-delete handle (_open_shared). This matches FileResponse's smooth
    seeking but, unlike FileResponse, never locks the output against move/delete —
    so the finished video can be moved out of the output folder while it's still
    showing in the player.
    """
    # Each root is a DIRECTORY OF MEDIA this endpoint is meant to hand out. It
    # is deliberately not "the app folder" and emphatically not its parent:
    # /api/file takes an arbitrary path from the query string, so every entry
    # here is readable over HTTP by anything that can reach the port — which
    # includes the network whenever the "public server (share)" setting is on.
    # Rooting it at the working directory's parent would publish the whole
    # project (config.yaml, .git, logs, models) and, depending on where the
    # process was launched from, neighbouring apps as well.
    #
    # `.pinokio-temp` is what needs reaching outside `app/`; it sits at the
    # project root. That one directory is named directly, resolved from this
    # file rather than from the working directory so it lands in the same place
    # no matter where the process was started.
    _app_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(_app_dir)
    roots = [
        API_TEMP,
        os.path.join(os.getcwd(), "temp"),
        os.path.join(os.getcwd(), ".pinokio-temp"),
        os.path.join(_app_dir, ".pinokio-temp"),
        os.path.join(_project_dir, ".pinokio-temp"),
        _faceset_library_dir(),
    ]
    out_dir = getattr(roop_globals, "output_path", "") or ""
    if out_dir:
        roots.append(out_dir)
    allowed = [os.path.normcase(os.path.abspath(r)) for r in roots]
    ap = os.path.abspath(path)
    ap_n = os.path.normcase(ap)

    def _within(child, parent):
        # commonpath (unlike startswith) can't be fooled by sibling dirs that
        # share a prefix, e.g. "output_evil" vs "output".
        try:
            return os.path.commonpath([child, parent]) == parent
        except ValueError:
            return False

    if not any(_within(ap_n, a) for a in allowed) or not os.path.isfile(ap):
        return JSONResponse(status_code=403, content={"message": "forbidden"})

    import mimetypes
    file_size = os.path.getsize(ap)
    media_type = mimetypes.guess_type(ap)[0] or "application/octet-stream"
    range_header = request.headers.get("range") or request.headers.get("Range")

    def _iter(start: int, length: int, chunk: int = 1024 * 1024):
        remaining = length
        f = _open_shared(ap)
        try:
            f.seek(start)
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data
        finally:
            f.close()

    if range_header and range_header.strip().lower().startswith("bytes="):
        spec = range_header.split("=", 1)[1].split(",", 1)[0].strip()
        start_s, _, end_s = spec.partition("-")
        try:
            start = int(start_s) if start_s else 0
        except ValueError:
            start = 0
        try:
            end = int(end_s) if end_s else file_size - 1
        except ValueError:
            end = file_size - 1
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        length = end - start + 1
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
        }
        return StreamingResponse(_iter(start, length), status_code=206,
                                 media_type=media_type, headers=headers)

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(file_size)}
    return StreamingResponse(_iter(0, file_size), media_type=media_type, headers=headers)


# ── Face Manager (faceset builder) ───────────────────────────────────────────
# Valid detector engines the Face Manager may extract with. SCRFD (buffalo_l's
# built-in det_10g) is the default and matches the swap pipeline; the others let
# the user harvest faces tricky footage where a different detector locks on.
_FACEMGR_DETECTORS = ["scrfd", "retinaface", "retinaface_r50", "yoloface", "yunet"]































# ── Extras: frame post-processors (AI upscale / colorize / stylize filters) ──
# These reuse the roop frame processors that the swap pipeline never wired in.
_FRAME_FILTERS = ["stylize", "detailenhance", "pencil", "cartoon", "C64"]
_FRAME_UPSCALERS = ["esrganx2", "esrganx4", "esrgan_anime_x4", "ultrasharp_x4", "lsdirx4",
                    "clear_reality_x4", "span_x4", "compact_x4", "nomos8k_x4"]
_FRAME_COLORIZERS = ["deoldify_artistic", "deoldify_stable"]














# ── Live camera (webcam → live swap → optional virtual camera) ───────────────
# Thin wrapper over roop/virtualcam.py (previously only reachable from the
# legacy Gradio UI). The capture/swap loop runs in its own thread and publishes
# each frame to ui_globals.ui_camera_frame; /api/livecam/frame serves that as a
# JPEG for the React panel's ~5fps preview. With stream_obs the swapped feed is
# also pushed to a system virtual camera (OBS etc.) via pyvirtualcam.





















# ── route modules ───────────────────────────────────────────────────────────
# Cohesive groups of endpoints split out of this file. Inclusion order is not
# significant: every /api route is a literal path with no path parameters, so
# none can shadow another.
import source_gallery as _source_gallery
import routes_faceset as _routes_faceset
import routes_facemgr as _routes_facemgr
app.include_router(_routes_faceset.router)
app.include_router(_routes_facemgr.router)
_source_gallery.API_TEMP = API_TEMP
_routes_faceset.API_TEMP = API_TEMP
_routes_facemgr.API_TEMP = API_TEMP
_routes_facemgr._FACEMGR_DETECTORS = _FACEMGR_DETECTORS
import api_media as _api_media
import routes_diagnostics as _routes_diagnostics
import routes_livecam as _routes_livecam
import routes_quality as _routes_quality
import routes_extras as _routes_extras
import routes_queue as _routes_queue
import routes_export as _routes_export
app.include_router(_routes_diagnostics.router)
app.include_router(_routes_livecam.router)
app.include_router(_routes_quality.router)
app.include_router(_routes_extras.router)
app.include_router(_routes_queue.router)
app.include_router(_routes_export.router)

# Shared objects the route modules read. All are mutated in place and never
# rebound here, so these bind one object rather than copying a value.
_api_media.API_TEMP = API_TEMP
_routes_diagnostics._progress = _progress
_routes_diagnostics.list_files_process = list_files_process
_routes_livecam._progress = _progress
_routes_quality._last_output = _last_output
_routes_extras._FRAME_COLORIZERS = _FRAME_COLORIZERS
_routes_extras._FRAME_FILTERS = _FRAME_FILTERS
_routes_extras._FRAME_UPSCALERS = _FRAME_UPSCALERS
# The queue runner drives the same single-run entry point the /api/swap handler
# does, so a queued job and a hand-started one are the identical code path.
# _run_swap and stop_swap are functions defined in this module (never rebound),
# so binding them here is a stable reference, not a copied value.
_routes_queue._progress = _progress
_routes_queue.list_files_process = list_files_process
_routes_queue._run_swap = _run_swap
_routes_queue._stop_current = stop_swap
_routes_queue._snapshot_outputs = _snapshot_output_mtimes
_routes_queue._outputs_since = _outputs_since
_routes_queue._benchmark_running = lambda: bool(_benchmark_state["running"])
_routes_queue.load()

# Helpers that left with their route groups but are still called by code that
# stayed here. Imported at the bottom rather than the top because these modules
# import back from this one; by the time any of these is *called* (all uses are
# inside request handlers or the run path) every module is fully initialised.
#
#   _make_frame_processor    -> the post_swap wiring just below
#   _gpu_name                -> _run_swap(), recording GPU in the run history
#   _faceset_library_dir     -> get_file(), resolving a served path
#   _frontal_crop_from_images -> source_refresh_thumbs()
from routes_extras import _make_frame_processor  # noqa: E402
from routes_diagnostics import _gpu_name  # noqa: E402
from routes_faceset import (  # noqa: E402
    _faceset_library_dir,
    _frontal_crop_from_images,
)

# ── post_swap shared-state wiring ────────────────────────────────────────────
# post_swap.py holds the upscale/interpolation machinery that used to live here.
# Two names it uses are owned by this module: the progress dict it writes into,
# and the frame-processor factory defined below its old position. Bind the SAME
# objects — never copies, since _progress is mutated in place and both modules
# must observe one dict — now that the whole module exists.
_post_swap._progress = _progress
_post_swap._make_frame_processor = _make_frame_processor

def run_api():
    try:
        port = int(os.environ.get("ROOP_API_PORT", 8001))
    except ValueError:
        port = 8001
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")

