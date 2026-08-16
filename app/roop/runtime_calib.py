"""Learned runtime estimation for video swaps.

The Face Swap tab shows an "EST. RUNTIME" before a run starts. A hardcoded
heuristic is a poor predictor because real speed depends heavily on the settings
combination (swap model, enhancer + size, detector engine/resolution, pixel
boost, thread count, GPU, TRT precision…). This module records the ACTUAL
wall-clock ms/frame after every completed video, keyed by a settings signature,
and serves that measurement back for the pre-run estimate — so the estimate gets
more accurate over time, per settings combo.

Design notes:
- One JSON store (app/runtime_calibration.json). Never fatal: every op is
  wrapped so a corrupt/locked file can't break processing or the estimate.
- The signature is built from the SAME payload the frontend sends to /api/swap
  and to /api/runtime_estimate, so the record-time and estimate-time keys match.
  The reader is tolerant of the couple of fields whose key differs between the
  swap payload ("enhancer") and the raw settings object ("selected_enhancer").
- ms/frame is stored as an exponential moving average (recent runs weighted
  more, so a new GPU/driver shifts the estimate) plus a sample count.
"""
import os
import json
import threading
import time

from roop.utilities import resolve_relative_path

_LOCK = threading.Lock()
_ALPHA = 0.35          # EMA weight for the newest run

# Bump when a change makes the PIPELINE materially faster or slower at settings
# the signature cannot see. Every stored ms/frame was measured against the old
# pipeline, so leaving them in place makes the pre-run estimate confidently
# wrong until the EMA drifts — at alpha 0.35 an entry with 89 samples behind it
# still misleads for several runs.
#
# 2 (2026-08-01): parallel stabilization stopped forcing the whole swap pass onto
# ONE thread, and the mask stopped being derived twice per face. The stored data
# shows the old bug plainly — 4 threads measured FASTER than 8 (58.6 vs 66.4
# ms/frame) because thread count was irrelevant to a serialised pass — so those
# entries describe a pipeline that no longer exists.
_VERSION = 2

# Perf-relevant settings. Each tuple is (canonical_name, [payload keys to try]).
# The enhancer name already encodes GPEN size ("GPEN 1024"), so no size field is
# needed. Order is irrelevant — the signature sorts by canonical name.
_SIG_FIELDS = [
    ("swap_model",        ["swap_model"]),
    ("enhancer",          ["enhancer", "selected_enhancer"]),
    ("detection",         ["detection", "face_detection_mode"]),
    ("det_size",          ["face_detector_size"]),
    ("detector",          ["detector_engine"]),
    ("swap_steps",        ["num_swap_steps"]),
    ("upscale",           ["upscale", "subsample_upscale"]),
    ("track",             ["track_identities"]),
    ("temporal",          ["temporal_detection"]),
    ("mask",              ["mask_engine"]),
    ("stab_face",         ["stabilize_face"]),
    ("stab_enh",          ["stabilize_enhancer"]),
    ("stab_mask",         ["stabilize_mask"]),
    # Expression restore and the post-swap AI upscaler are both whole GPU stages
    # (the restorer alone measured ~34 ms/face). Without them in the signature,
    # runs with and without them fold into ONE EMA bucket and the estimate drifts
    # toward a blend of two very different costs.
    ("expr",              ["expression_restore_strength"]),
    ("upscale_after",     ["upscale_after_swap"]),
]

# Merger post-ops (roop/procmgr_merger.py). Measured 0.6-7.5 ms each and 15.3 ms
# with all five on — half the expression restorer, which is in the table above
# for exactly this reason. Without them, a run with the whole chain enabled and
# one with it off fold into the SAME EMA bucket and the estimate settles on a
# blend of two pipelines.
#
# Only the five that COST something. `output_face_scale` is a change to the
# paste matrix with no measurable cost, so letting it move the signature would
# split buckets for nothing.
_MERGER_COST_KEYS = ("merger_hist_match", "merger_sharpen", "merger_motion_blur",
                     "merger_grain_match", "merger_degrade")


def _path():
    return resolve_relative_path('../runtime_calibration.json')


def _load():
    try:
        with open(_path(), 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "entries" in data:
            # The version was written but never read, so it could never do the one
            # job it exists for. A file from an older, slower pipeline is worse
            # than no file: the estimate is stated with confidence and is wrong.
            if int(data.get("version", 0)) == _VERSION:
                return data
    except Exception:
        pass
    return {"version": _VERSION, "entries": {}, "global_ms_per_frame": None,
            "global_samples": 0}


def _save(data):
    try:
        tmp = _path() + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, _path())
    except Exception:
        pass


def _norm(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def density_bucket(n):
    """Bucket average faces-per-frame into coarse classes so a single-face clip
    and a crowd scene calibrate separately without fragmenting the store."""
    try:
        n = float(n)
    except Exception:
        return "?"
    if n <= 0:
        return "0"
    if n < 1.5:
        return "1"
    if n < 2.5:
        return "2"
    if n < 4.5:
        return "3-4"
    return "5+"


def with_density(base_sig, bucket):
    """Append a face-density bucket to a base (settings+env) signature. Kept
    separate from signature_from_payload because the true density is only known
    at run completion, whereas the base signature is built at run start."""
    if not bucket:
        return base_sig
    return f"{base_sig}|density={bucket}"


def stats():
    """Aggregate store stats for the estimation panel."""
    try:
        with _LOCK:
            data = _load()
        return {"entries": len(data.get("entries", {})),
                "global_samples": data.get("global_samples", 0),
                "global_ms_per_frame": data.get("global_ms_per_frame")}
    except Exception:
        return {"entries": 0, "global_samples": 0, "global_ms_per_frame": None}


def signature_from_payload(payload, gpu="", threads="", precision=""):
    """Build a stable signature string from a swap/estimate payload plus the
    server-side environment (GPU, thread count, TRT precision) that the payload
    doesn't carry."""
    parts = []
    for name, keys in _SIG_FIELDS:
        val = None
        for k in keys:
            if k in payload and payload[k] is not None:
                val = payload[k]
                break
        parts.append(f"{name}={_norm(val)}")

    # Appended ONLY when something is on, and as a COUNT rather than five more
    # fields. Both halves of that matter:
    #   * omitting it when neutral leaves every signature already on disk
    #     byte-identical, since all of them were recorded before these existed
    #     and with them off — so no _VERSION bump and no lost calibration;
    #   * five continuous sliders as five fields would give almost every run its
    #     own bucket, and a bucket with one sample never learns anything.
    active = 0
    for key in _MERGER_COST_KEYS:
        try:
            if float(payload.get(key) or 0.0) != 0.0:
                active += 1
        except (TypeError, ValueError):
            pass
    if active:
        parts.append(f"merger={active}")

    parts.append(f"threads={_norm(threads)}")
    parts.append(f"precision={_norm(precision)}")
    parts.append(f"gpu={_norm(gpu)}")
    return "|".join(parts)


def record(signature, frames, elapsed_ms):
    """Fold one completed run into the store. Ignores tiny/degenerate runs whose
    ms/frame would be noise."""
    try:
        frames = int(frames)
        elapsed_ms = float(elapsed_ms)
        if not signature or frames < 24 or elapsed_ms < 1000.0:
            return
        mpf = elapsed_ms / frames
        with _LOCK:
            data = _load()
            e = data["entries"].get(signature)
            if e and e.get("samples", 0) > 0:
                e["ms_per_frame"] = (1 - _ALPHA) * e["ms_per_frame"] + _ALPHA * mpf
                e["samples"] = e.get("samples", 0) + 1
            else:
                e = {"ms_per_frame": mpf, "samples": 1}
            e["last_ms_per_frame"] = mpf
            e["updated"] = int(time.time())
            data["entries"][signature] = e
            # Global fallback for never-seen signatures.
            g = data.get("global_ms_per_frame")
            data["global_ms_per_frame"] = mpf if g is None else (1 - _ALPHA) * g + _ALPHA * mpf
            data["global_samples"] = data.get("global_samples", 0) + 1
            _save(data)
    except Exception:
        pass


def predict(signature):
    """Return {ms_per_frame, samples, source} for a signature, or None when there
    is no data at all. source is 'measured' (exact signature) or 'global'."""
    try:
        with _LOCK:
            data = _load()
        e = data["entries"].get(signature)
        if e and e.get("samples", 0) > 0:
            return {"ms_per_frame": e["ms_per_frame"], "samples": e["samples"],
                    "source": "measured"}
        g = data.get("global_ms_per_frame")
        if g is not None:
            return {"ms_per_frame": g, "samples": 0, "source": "global"}
    except Exception:
        pass
    return None
