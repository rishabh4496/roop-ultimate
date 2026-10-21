"""Durable project state for safe processing recovery.

The queue remembers what should run.  A project remembers what was actually
being rendered, including the immutable inputs and the output segments that
survived the last safe boundary.  Project files are deliberately small JSON
records; media and model files remain in their existing locations.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
import threading
import time
import uuid
from typing import Any, Callable, Mapping


# A project record is read-modify-written from several threads: the render worker (state, runtime,
# the Stage 1/2 records below) and the segment writer (a committed segment). Without one lock the
# later `save` silently discards the earlier one's change. Re-entrant so a helper may call another.
_RECORD_LOCK = threading.RLock()

PROJECT_SCHEMA_VERSION = 1
COMPATIBILITY = {
    "project_schema": PROJECT_SCHEMA_VERSION,
    "processing_contract": "segmented-video-v1",
}
PROJECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects")
STATES = ("PAUSED", "INTERRUPTED", "RECOVERABLE", "FAILED", "COMPLETED", "PROCESSING")

# Roughly 1.2 s of backoff in total: long enough to outlast an antivirus or
# indexer handle, short enough that a real permissions fault still surfaces
# promptly rather than stalling a render.
_REPLACE_ATTEMPTS = 8


def _json_default(value):
    """JSON fallback for the numpy that every detector fact arrives as.

    `tolist` is asked FIRST because a numpy ndarray has BOTH methods and only
    one of them is safe for it: `ndarray.item()` raises

        ValueError: can only convert an array of size 1 to a Python scalar

    for anything larger than a single element. Asking for `item` first
    therefore worked on the numpy SCALARS (age, gender, a score) and failed on
    every ARRAY -- bbox, kps, embedding, landmark_2d_106, landmark_3d_68 --
    i.e. on exactly the fields a project records when the user has captured
    target faces. `/api/swap` answered "cannot create a recoverable project:
    can only convert an array of size 1 to a Python scalar" and the run never
    started, while a swap with no captured target face was unaffected.

    Every project on disk reads `target_faces: 0` for this reason; the target
    faces a project exists to restore could never be written in the first
    place.

    `tolist` covers both shapes on its own -- a numpy scalar returns a Python
    scalar, a 0-d array returns its element, an array returns nested lists --
    so the `item` branch below now only serves objects that expose `item`
    WITHOUT `tolist`.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _atomic_write(path: str, value: Mapping[str, Any]) -> None:
    """Replace one JSON record only after its contents have reached disk.

    THE RENAME IS RETRIED, BECAUSE ON WINDOWS IT FAILS TRANSIENTLY.  `os.replace`
    raises PermissionError (WinError 5 / WinError 32) whenever any other process
    holds a handle to either file for even a moment -- Defender scanning the
    freshly written temporary, the Search indexer touching the destination, a
    backup agent.  Nothing in this application is holding them.

    Observed on the physical RTX 3060 host: a project's PROCESSING state update
    raised WinError 5, the exception escaped the render worker, and the run died
    while `_progress["processing"]` stayed true -- the UI showed a job
    generating forever with no error and no output.  A single unretried rename
    was therefore able to wedge the whole application.

    The retry window is short and bounded; a genuine, persistent failure (a
    read-only directory, a real permissions problem) still raises, so this
    cannot mask a broken projects directory.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=1, sort_keys=True, default=_json_default)
            fh.flush()
            os.fsync(fh.fileno())
        delay = 0.02
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.4)
    finally:
        try:
            os.remove(temporary)
        except OSError:
            pass


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str, precomputed_sha256: str = "", defer_hash: bool = False) -> dict:
    """Return an identity that detects replacement, edits, and truncation.

    `defer_hash` is for a caller on an HTTP request thread: SHA-256 of a multi-GB target costs
    seconds (0.63 ms/MB warm on the dev SSD: 1.3 s for 2 GB, the React client's deadline is 15 s
    and the stage budget for the response is 200 ms). The identity is returned with
    `sha256_pending: True` and size + mtime_ns - enough to catch a replaced or truncated file - and
    `complete_hashes` fills the digest in on the worker thread before the render starts."""
    absolute = os.path.abspath(os.path.expanduser(str(path)))
    stat = os.stat(absolute)
    identity = {
        "path": absolute,
        "name": os.path.basename(absolute),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if precomputed_sha256:
        identity["sha256"] = str(precomputed_sha256).strip()
    elif defer_hash:
        identity["sha256"] = ""
        identity["sha256_pending"] = True
    else:
        identity["sha256"] = file_sha256(absolute)
    return identity


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def project_path(project_id: str) -> str:
    safe = os.path.basename(str(project_id))
    if safe != str(project_id) or not safe:
        raise ValueError("invalid project id")
    return os.path.join(PROJECTS_DIR, safe + ".json")


_PROVIDER_ALIASES = {
    "CUDAExecutionProvider": "cuda",
    "TensorrtExecutionProvider": "tensorrt",
    "TensorRTExecutionProvider": "tensorrt",
    "CPUExecutionProvider": "cpu",
    "CoreMLExecutionProvider": "coreml",
}


def normalize_provider(value: Any) -> str:
    """Reduce anything ORT calls a provider to this project's short name.

    The shapes that actually reach here are not interchangeable and only one
    of them used to be handled:

        "tensorrt"                                  cfg.provider
        "TensorrtExecutionProvider"                 a bare EP name
        ("TensorrtExecutionProvider", {...opts})    an EP with options
        [("Tensorrt...", {...}), "CPUExecution..."] roop_globals.execution_providers

    The last one is what `api.py` passes as `effective_provider` on every real
    render, and a single unwrap of it yields the *tuple*, not the name — which
    `str()` then rendered as the whole `('TensorrtExecutionProvider', {...})`
    literal, engine-cache path and all. Validation recomputes the identity
    without an effective provider, gets the short "tensorrt", and the two can
    never be equal. Every project written under TensorRT was therefore reported
    RECOVERABLE while being permanently refused with "runtime provider differs
    from the checkpoint" — a checkpoint that could not be resumed by the very
    machine that wrote it, seconds earlier.

    So: unwrap until a string, then alias. `test_project_provider_identity.py`
    fails on the single-unwrap version.
    """
    for _ in range(4):
        if isinstance(value, (list, tuple)):
            value = value[0] if len(value) else ""
        else:
            break
    text = str(value or "")
    # Records written before the unwrap was fixed hold the stringified tuple.
    # Reading the EP name back out of it is lossless, so those projects become
    # resumable again without rewriting any file on disk — which matters,
    # because a migration that rewrites a checkpoint is a migration that can
    # corrupt one.
    if text[:1] in "([":
        match = re.match(r"[(\[]\s*['\"]([A-Za-z]+ExecutionProvider)['\"]", text)
        text = match.group(1) if match else text
    return _PROVIDER_ALIASES.get(text, text)


def runtime_identity(payload: Mapping[str, Any], cfg: Any, effective_provider: Any = None) -> dict:
    hardware = dict(getattr(cfg, "hardware", {}) or {})
    provider = normalize_provider(
        effective_provider or payload.get("provider") or getattr(cfg, "provider", ""))
    return {
        "provider": provider,
        "precision": str(payload.get("trt_precision") or getattr(cfg, "trt_precision", "")),
        "models": {
            key: payload.get(key, getattr(cfg, key, ""))
            for key in ("swap_model", "enhancer", "mask_engine", "mask_engine_2",
                        "detector_engine", "upscale_model_after", "interp_after_swap",
                        "recognizer")
        },
        "hardware": {
            "signature": str(getattr(cfg, "hardware_signature", "") or ""),
            "profile_key": hardware.get("hardware_profile_key"),
            "gpu": hardware.get("gpu") or hardware.get("gpu_name"),
            "vram_tier": hardware.get("vram_tier"),
            "vram_gb": hardware.get("vram_gb") or hardware.get("vram_total_gb"),
            "ram_gb": hardware.get("ram_gb") or hardware.get("ram_total_gb"),
        },
        "platform": platform.platform(),
    }


def new_project(*, job_id: str | None, name: str, payload: Mapping[str, Any],
                sources: list[dict], target: dict, frame_start: int,
                frame_end: int, output: dict, cfg: Any,
                target_faces: list[dict] | None = None,
                target_context: dict | None = None,
                app_version: str = "") -> dict:
    now = time.time()
    project_id = uuid.uuid4().hex[:16]
    payload_copy = json.loads(json.dumps(dict(payload), default=_json_default))
    record = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "id": project_id,
        "job_id": job_id,
        "name": name or target.get("name") or project_id,
        "state": "PROCESSING",
        "created_at": now,
        "updated_at": now,
        "application": {
            "compatibility": dict(COMPATIBILITY),
            "version": str(app_version or ""),
        },
        "inputs": {
            "sources": sources,
            "target": target,
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "target_faces": target_faces or [],
            "target_context": dict(target_context or {}),
        },
        "settings": {
            "payload": payload_copy,
            "fingerprint": fingerprint(payload_copy),
        },
        "runtime": runtime_identity(payload_copy, cfg),
        "output": output,
        "checkpoint": {
            "sequence": 0,
            "safe_frame": int(frame_start),
            "next_frame": int(frame_start),
            "segments": [],
            "manifest": "",
            "written_at": now,
        },
        "partial_output": {"files": [], "integrity": "unknown"},
        "error": "",
    }
    save(record)
    return record


def save(record: Mapping[str, Any]) -> dict:
    value = dict(record)
    value["updated_at"] = time.time()
    _atomic_write(project_path(value["id"]), value)
    return value


def load(project_id: str) -> dict:
    with open(project_path(project_id), encoding="utf-8") as fh:
        return json.load(fh)


def list_projects() -> list[dict]:
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    result = []
    for name in os.listdir(PROJECTS_DIR):
        if not name.endswith(".json") or name.startswith("."):
            continue
        try:
            result.append(load(name[:-5]))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    result.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
    return result


def update_state(project_id: str, state: str, error: str = "") -> dict | None:
    if state not in STATES:
        raise ValueError(f"unknown project state: {state}")
    with _RECORD_LOCK:
        try:
            record = load(project_id)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        record["state"] = state
        record["error"] = str(error or "")
        saved = save(record)
    if state == "COMPLETED":
        # A finished project never resumes, so its Stage 1/2 cache can only waste disk.
        discard_stage_cache(project_id)
    return saved


def update_runtime(project_id: str, runtime: Mapping[str, Any]) -> dict | None:
    with _RECORD_LOCK:
        try:
            record = load(project_id)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        record["runtime"] = dict(runtime)
        return save(record)


def update_checkpoint(project_id: str, *, safe_frame: int, next_frame: int,
                      segments: list[dict], manifest: str = "",
                      partial_files: list[dict] | None = None,
                      state: str | None = None) -> dict | None:
    # The identity of the manifest is computed BEFORE the lock: it hashes a file.
    manifest_identity = (file_identity(manifest) if manifest and os.path.isfile(manifest) else None)
    with _RECORD_LOCK:
        try:
            record = load(project_id)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        previous = record.get("checkpoint") or {}
        record["checkpoint"] = {
            "sequence": int(previous.get("sequence", 0) or 0) + 1,
            "safe_frame": int(safe_frame),
            "next_frame": int(next_frame),
            "segments": json.loads(json.dumps(segments, default=_json_default)),
            "manifest": os.path.abspath(manifest) if manifest else "",
            "manifest_identity": manifest_identity,
            "written_at": time.time(),
        }
        if partial_files is not None:
            record["partial_output"] = {
                "files": json.loads(json.dumps(partial_files, default=_json_default)),
                "integrity": "verified",
            }
        if state:
            record["state"] = state
        return save(record)


def complete_hashes(project_id: str) -> dict | None:
    """Fill in every `sha256_pending` digest of a project. Worker thread only (it reads whole files).

    A file that changed since the identity was taken (size or mtime_ns differs) is NOT hashed into the
    record: the record keeps the old size/mtime, so `validate` reports it as changed instead of
    blessing the new bytes."""
    try:
        record = load(project_id)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    inputs = record.get("inputs") or {}
    pending = [item for item in list(inputs.get("sources") or []) + [inputs.get("target") or {}]
               if item.get("sha256_pending")]
    if not pending:
        return record
    digests = {}                      # path -> digest, hashed OUTSIDE the lock (whole-file reads)
    for item in pending:
        path = item.get("path")
        try:
            current = os.stat(path)
            if (int(current.st_size) != item.get("size")
                    or int(current.st_mtime_ns) != item.get("mtime_ns")):
                continue
            digests[path] = file_sha256(path)
        except OSError:
            continue
    with _RECORD_LOCK:
        try:
            record = load(project_id)          # a segment commit may have saved while we hashed
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        inputs = record.get("inputs") or {}
        for item in list(inputs.get("sources") or []) + [inputs.get("target") or {}]:
            if item.get("sha256_pending") and item.get("path") in digests:
                item["sha256"] = digests[item["path"]]
                item.pop("sha256_pending", None)
        return save(record)


# ── Stage 1 / Stage 2 cache (Phase 6) ─────────────────────────────────────────────────────────────────
#
# WHAT A RESUME REDID. A project resumes from its committed segments (`checkpoint.segments`), but the
# two pre-passes that precede the swap pass - Stage 1 (the sparse presence scan) and Stage 2 (tracking
# and temporal detection) - were re-run on the remaining range every time: minutes of detector work
# whose answer the interrupted run already had. They are persisted here the moment each finishes:
#
#   stage1   small, in the project JSON: the active ranges (ABSOLUTE frames), the scan's stats, and the
#            hash of what they were computed from
#   stage2   the hand-off the swap pass consumes (`ProcessMgr._temporal_faces`, `_track_assignments`,
#            `_track_source_map`, shot cuts) in a SIDECAR file beside the record (`<id>.stage2.npz`,
#            written by `roop/stage_cache.py`), plus its digest and the serialized frame plan in the JSON
#   trajectory  the smoothed per-face alignment matrices and head pose of that same Stage 2, as one
#            memory-mappable array (`<id>.trajectory.npy`, written by `roop/trajectory_cache.py`). It is
#            DERIVED from stage2: re-recording stage1 or stage2 drops it, and it expires under the same key.
#
# The frame plan alone is NOT the hand-off (it holds no Face objects by design - see roop/stage2_plan.py),
# so persisting only the plan, as the brief for this stage said, could not skip Stage 2.
#
# VALID ONLY UNDER ITS KEY. Everything the two stages' answers depend on is folded into one digest:
# the target's SHA-256, the original trim, every source's SHA-256, the captured target faces, the
# settings fingerprint and the model identity. A resume under any other key re-runs the stages. The
# sidecar is additionally checked against its recorded size and SHA-256 before one byte of it is used.

STAGE_CACHE_VERSION = 1
STAGE_SIDECAR_SUFFIX = {"stage2": ".stage2.npz", "trajectory": ".trajectory.npy"}


# What each component of the basis is called when it is the one that changed (for the printed reason).
_BASIS_LABELS = {
    "version": "the cache format version",
    "target": "the target video (SHA-256 / size)",
    "trim": "the trim points",
    "sources": "a source file (SHA-256 / identity)",
    "target_faces": "the captured target faces",
    "target_context": "the target-to-source mapping",
    "settings": "the processing settings",
    "models": "the model configuration",
    "provider": "the execution provider",
    "precision": "the TensorRT precision",
    "live": "the application version or a ROOP_* environment variable",
}


def stage_cache_basis(record: Mapping[str, Any], live: Mapping[str, Any] | None = None) -> dict:
    """Everything the Stage 1 / Stage 2 results depend on, as small named digests. {} when an input digest is
    unknown.

    `live` is what the CURRENT process contributes and a record cannot know: the application version
    (the pre-passes are code) and the ROOP_* environment (their gap limit, flow, coast, track knobs). A
    cache written by other code or under other knobs is not the answer this run would compute.

    A target or source whose SHA-256 is still pending (the worker has not finalized the project yet)
    has no identity strong enough to key a cache on, so the basis is {} and nothing is stored or restored.
    """
    inputs = record.get("inputs") or {}
    target = inputs.get("target") or {}
    sources = list(inputs.get("sources") or [])
    if not target.get("sha256") or any(not s.get("sha256") for s in sources):
        return {}
    runtime = record.get("runtime") or {}
    return {
        "version": STAGE_CACHE_VERSION,
        "target": {"sha256": target.get("sha256"), "size": target.get("size")},
        "trim": [int(inputs.get("frame_start", 0) or 0), int(inputs.get("frame_end", 0) or 0)],
        "sources": [[s.get("sha256"), s.get("source_id", "")] for s in sources],
        "target_faces": fingerprint(inputs.get("target_faces") or []),
        "target_context": fingerprint({k: v for k, v in (inputs.get("target_context") or {}).items()
                                       if k in ("face_mapping", "target_person_source_mapping",
                                                "target_person_ids", "target_selected_source_id")}),
        "settings": (record.get("settings") or {}).get("fingerprint"),
        "models": runtime.get("models"),
        "provider": normalize_provider(runtime.get("provider")),
        "precision": runtime.get("precision"),
        "live": dict(live or {}),
    }


def stage_cache_key(record: Mapping[str, Any], live: Mapping[str, Any] | None = None) -> str:
    """The digest of `stage_cache_basis`; '' when an input digest is unknown."""
    basis = stage_cache_basis(record, live)
    return fingerprint(basis) if basis else ""


def basis_differences(stored: Mapping[str, Any] | None, current: Mapping[str, Any]) -> list[str]:
    """Which named components of the key changed, in words. [] when the stored basis is unknown."""
    if not stored:
        return []
    return [_BASIS_LABELS.get(name, name) for name in current if stored.get(name) != current.get(name)]


def stage_sidecar_path(project_id: str, stage: str) -> str:
    base = os.path.splitext(project_path(project_id))[0]
    return base + STAGE_SIDECAR_SUFFIX[stage]


def record_stage(project_id: str, stage: str, data: Mapping[str, Any],
                 live: Mapping[str, Any] | None = None) -> dict | None:
    """Store one stage's result under the project's current cache key.

    A stage already stored under a DIFFERENT key (the inputs changed between runs) is dropped first, so
    a record never mixes two inputs' results. Returns None when the project cannot be read or has no
    usable key (digests still pending): the caller then simply does not cache.
    """
    with _RECORD_LOCK:
        try:
            record = load(project_id)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        basis = stage_cache_basis(record, live)
        if not basis:
            return None
        key = fingerprint(basis)
        cache = record.get("stage_cache") or {}
        if cache.get("key") != key or cache.get("version") != STAGE_CACHE_VERSION:
            # `basis` is kept beside the key so a later refusal can name WHAT changed, not just that
            # something did. It holds digests and settings fingerprints only - no paths, no media.
            cache = {"version": STAGE_CACHE_VERSION, "key": key, "basis": basis}
            for name in STAGE_SIDECAR_SUFFIX:
                # The stage being recorded wrote its sidecar BEFORE calling here, under this key; every other
                # stage's file belongs to the key that just expired.
                if name != stage:
                    _remove_quietly(stage_sidecar_path(project_id, name))
        entry = json.loads(json.dumps(dict(data), default=_json_default))
        entry["written_at"] = time.time()
        cache[stage] = entry
        if stage == "stage1":
            cache.pop("stage2", None)       # Stage 2 was computed against the PREVIOUS Stage 1 ranges
            _remove_quietly(stage_sidecar_path(project_id, "stage2"))
        if stage in ("stage1", "stage2"):
            # The trajectory is derived from Stage 2's faces; a new Stage 2 (or the Stage 1 it was built on)
            # makes the old one describe different heads. Its writer records it again right after.
            cache.pop("trajectory", None)
            _remove_quietly(stage_sidecar_path(project_id, "trajectory"))
        record["stage_cache"] = cache
        return save(record)


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def discard_stage_cache(project_id: str) -> None:
    """Forget a project's Stage 1/2 cache: the record entry and every sidecar file."""
    with _RECORD_LOCK:
        for name in STAGE_SIDECAR_SUFFIX:
            _remove_quietly(stage_sidecar_path(project_id, name))
        try:
            record = load(project_id)
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if record.pop("stage_cache", None) is not None:
            save(record)


def verify_sidecar(project_id: str, stage: str, info: Mapping[str, Any]) -> list[str]:
    """Why the sidecar named by `info` must not be used (empty = it is the file that was written)."""
    path = stage_sidecar_path(project_id, stage)
    if not os.path.isfile(path):
        return [f"{stage} cache file is missing"]
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return [f"{stage} cache file cannot be read: {exc}"]
    if size != info.get("size"):
        return [f"{stage} cache file size differs from the record"]
    try:
        digest = file_sha256(path)
    except OSError as exc:
        return [f"{stage} cache file cannot be read: {exc}"]
    if digest != info.get("sha256"):
        return [f"{stage} cache file digest differs from the record"]
    return []


def usable_stage_cache(project_id: str, live: Mapping[str, Any] | None = None) -> dict:
    """What a resume may skip: ``{'stage1': dict|None, 'stage2': dict|None, 'reasons': [..]}``.

    `reasons` says, for each stage that is NOT usable, why - it is printed by the caller, because a
    cache that is silently ignored is indistinguishable from one that never existed. The sidecar is
    hashed here (a whole-file read): call from the worker, never from a request thread.
    """
    out = {"stage1": None, "stage2": None, "trajectory": None, "reasons": []}
    try:
        record = load(project_id)
    except (OSError, ValueError, json.JSONDecodeError):
        out["reasons"].append("project record could not be read")
        return out
    cache = record.get("stage_cache")
    if not cache:
        out["reasons"].append("no Stage 1/2 result was recorded for this project")
        return out
    if cache.get("version") != STAGE_CACHE_VERSION:
        out["reasons"].append("Stage 1/2 cache was written by another format version")
        return out
    basis = stage_cache_basis(record, live)
    if not basis:
        out["reasons"].append("input digests are not final, so the cache cannot be keyed")
        return out
    if cache.get("key") != fingerprint(basis):
        changed = basis_differences(cache.get("basis"), basis)
        out["reasons"].append("changed since the cache was written: " + (", ".join(changed) if changed else
                                                                         "inputs, trim, sources or settings"))
        return out
    if cache.get("stage1"):
        out["stage1"] = dict(cache["stage1"])
    stage2 = cache.get("stage2")
    if stage2:
        problems = verify_sidecar(project_id, "stage2", stage2.get("sidecar") or {})
        if problems:
            out["reasons"].extend(problems)
        else:
            out["stage2"] = dict(stage2)
    trajectory = cache.get("trajectory")
    if trajectory and out["stage2"] is not None:
        problems = verify_sidecar(project_id, "trajectory", trajectory.get("sidecar") or {})
        if problems:
            out["reasons"].extend(problems)
        else:
            out["trajectory"] = dict(trajectory)
    return out


def _same_file(want: Mapping[str, Any], reasons: list[str], label: str, quick: bool = False) -> None:
    """Compare a recorded file identity with the file on disk.

    `quick` is for a caller on an HTTP request thread: the SHA-256 of a multi-GB target or partial
    output is seconds of disk reads (the React client's deadline is 15 s; the budget for a start/resume
    response is 200 ms), so it compares the SIZE only. The digests are then re-checked by
    `finalize_inputs` on the worker before the render touches the files, so a same-size edit is still
    refused - just by the job, with the reason, instead of by the request."""
    path = want.get("path")
    if not path or not os.path.isfile(path):
        reasons.append(f"{label} is missing: {path or 'no path recorded'}")
        return
    try:
        if quick:
            stat = os.stat(path)
            actual = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
        else:
            actual = file_identity(path)
    except OSError as exc:
        reasons.append(f"{label} cannot be read: {exc}")
        return
    # A digest still pending (the process died between the response and the worker's first step) is
    # judged on size + mtime_ns: weaker than a digest, but it is all that was ever recorded.
    if want.get("sha256_pending"):
        keys = ("size", "mtime_ns")
    else:
        keys = ("size",) if quick else ("size", "sha256")
    for key in keys:
        if actual.get(key) != want.get(key):
            reasons.append(f"{label} changed ({key} mismatch)")
            break


def _input_files(record: Mapping[str, Any]):
    inputs = record.get("inputs") or {}
    for index, source in enumerate(inputs.get("sources") or []):
        yield f"source {index + 1}", source
    yield "target", inputs.get("target") or {}


def _partial_files(record: Mapping[str, Any]):
    manifest_identity = (record.get("checkpoint") or {}).get("manifest_identity")
    if manifest_identity:
        yield "checkpoint manifest", manifest_identity
    for item in (record.get("partial_output") or {}).get("files") or []:
        yield f"checkpoint output {item.get('name') or item.get('path')}", item


def _labelled_files(record: Mapping[str, Any]):
    yield from _input_files(record)
    yield from _partial_files(record)


def finalize_inputs(project_id: str) -> list[str]:
    """Worker-thread step before a render touches its files: fill pending digests, verify the rest.

    Pairs with `validate(..., quick=True)` on the request thread. Files whose digest was just computed
    are not hashed a second time. Returns the reasons a recorded file no longer matches (empty = fine)."""
    try:
        record = load(project_id)
    except (OSError, ValueError, json.JSONDecodeError):
        return [f"project {project_id} could not be read"]
    pending = {item.get("path") for _, item in _labelled_files(record) if item.get("sha256_pending")}
    if pending:
        complete_hashes(project_id)
    reasons: list[str] = []
    for label, item in _labelled_files(record):
        if item.get("sha256_pending") or item.get("path") in pending:
            continue
        _same_file(item, reasons, label)
    return reasons


def validate(record: Mapping[str, Any], cfg: Any, current_payload: Mapping[str, Any] | None = None,
             check_partial: bool = True, quick: bool = False) -> list[str]:
    reasons = []
    if record.get("schema_version") != PROJECT_SCHEMA_VERSION:
        reasons.append("project checkpoint schema is not supported by this application")
    compatibility = record.get("application", {}).get("compatibility") or {}
    for key, value in COMPATIBILITY.items():
        if compatibility.get(key) != value:
            reasons.append(f"incompatible checkpoint {key}")

    for label, item in _input_files(record):
        _same_file(item, reasons, label, quick=quick)

    saved_payload = (record.get("settings") or {}).get("payload") or {}
    payload = current_payload if current_payload is not None else saved_payload
    if fingerprint(payload) != (record.get("settings") or {}).get("fingerprint"):
        reasons.append("processing settings differ from the checkpoint")

    saved_runtime = record.get("runtime") or {}
    current_runtime = runtime_identity(payload, cfg)
    # The saved side goes through the same normalisation as the live side, so a
    # record written before normalize_provider was fixed compares on the EP name
    # it always meant rather than on the literal it was accidentally stored as.
    if normalize_provider(saved_runtime.get("provider")) != current_runtime.get("provider"):
        reasons.append("runtime provider differs from the checkpoint")
    if saved_runtime.get("precision") != current_runtime.get("precision"):
        reasons.append("runtime precision differs from the checkpoint")
    saved_models = saved_runtime.get("models") or {}
    for key, value in saved_models.items():
        if value != (current_runtime.get("models") or {}).get(key):
            reasons.append(f"model configuration '{key}' differs from the checkpoint")
    saved_hw = saved_runtime.get("hardware") or {}
    current_hw = current_runtime.get("hardware") or {}
    if saved_hw.get("signature") and saved_hw.get("signature") != current_hw.get("signature"):
        reasons.append("hardware profile differs from the checkpoint")
    if saved_runtime.get("platform") and saved_runtime.get("platform") != current_runtime.get("platform"):
        reasons.append("operating-system/runtime platform differs from the checkpoint")

    saved_output = record.get("output") or {}
    current_output = {
        "directory": os.path.abspath(str(getattr(cfg, "output_path", "") or
                                          saved_output.get("directory", ""))),
        "format": str(getattr(cfg, "output_video_format", "mp4") or "mp4"),
        "codec": str(getattr(cfg, "output_video_codec", "") or ""),
        "quality": getattr(cfg, "video_quality", None),
        "method": str(payload.get("output_method") or getattr(cfg, "output_method", "File")),
        "template": str(getattr(cfg, "output_template", "") or ""),
    }
    for key in ("directory", "format", "codec", "quality", "method", "template"):
        if saved_output.get(key) != current_output.get(key):
            reasons.append(f"output configuration '{key}' differs from the checkpoint")

    if check_partial:
        for label, item in _partial_files(record):
            _same_file(item, reasons, label, quick=quick)
    return reasons


def summarize(record: Mapping[str, Any], validation: list[str] | None = None) -> dict:
    checkpoint = record.get("checkpoint") or {}
    return {
        "id": record.get("id"),
        "job_id": record.get("job_id"),
        "name": record.get("name"),
        "state": record.get("state"),
        "updated_at": record.get("updated_at"),
        "safe_frame": checkpoint.get("safe_frame"),
        "next_frame": checkpoint.get("next_frame"),
        "segments": len(checkpoint.get("segments") or []),
        "recoverable": not validation,
        "validation_errors": list(validation or []),
        "error": record.get("error", ""),
        "project_file": os.path.splitext(project_path(str(record.get("id") or "")))[0] + ".roop"
        if record.get("id") else "",
    }


__all__ = [
    "COMPATIBILITY", "PROJECT_SCHEMA_VERSION", "PROJECTS_DIR", "STATES",
    "complete_hashes", "file_identity", "finalize_inputs", "file_sha256", "fingerprint", "list_projects", "load",
    "new_project", "normalize_provider", "project_path", "runtime_identity",
    "save", "summarize",
    "STAGE_CACHE_VERSION", "basis_differences", "discard_stage_cache", "record_stage", "stage_cache_basis",
    "stage_cache_key",
    "stage_sidecar_path", "usable_stage_cache", "verify_sidecar",
    "update_checkpoint", "update_runtime", "update_state", "validate",
]
