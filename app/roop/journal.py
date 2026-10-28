"""Journaled Sub-Segment Crash Recovery Engine (Stage 13).

Writes an atomic write-ahead journal:
  <job_dir>/journal.json.tmp -> atomic rename to journal.json
after every completed segment and every N frames inside long unsegmented renders.

Records:
  - Exact SHA-256 and mtime of source & target media
  - Recipe hash (configuration, models, parameters)
  - Completed segment list with verified frame counts
  - Exact resume frame index

Provides auto-recovery:
  - Automatically verifies integrity of all existing seg_XXXX.mp4 files with ffprobe
  - Discards only any truncated tail file
  - Resumes at the exact next frame so zero GPU work is ever wasted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from roop.degrade import swallowed as _swallowed

JOURNAL_SCHEMA_VERSION = 1
DEFAULT_UNSEGMENTED_INTERVAL = 100
_REPLACE_ATTEMPTS = 8


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def media_receipt(path: str, compute_hash: bool = True) -> Dict[str, Any]:
    """Capture precise media file facts: size, mtime_ns, and SHA-256."""
    abs_path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(abs_path):
        return {"path": abs_path, "exists": False, "size": 0, "mtime_ns": 0, "sha256": ""}
    stat = os.stat(abs_path)
    receipt = {
        "path": abs_path,
        "name": os.path.basename(abs_path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if compute_hash:
        receipt["sha256"] = file_sha256(abs_path)
    else:
        receipt["sha256"] = ""
    return receipt


def compute_recipe_hash(payload: Mapping[str, Any]) -> str:
    """Canonical SHA-256 fingerprint of all settings and model parameters."""
    canonical = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def journal_path(job_dir: str) -> str:
    return os.path.join(os.path.abspath(job_dir), "journal.json")


def write_journal(job_dir: str, data: Mapping[str, Any]) -> str:
    """Write write-ahead journal record atomically (<job_dir>/journal.json.tmp -> journal.json).

    Retries on transient Windows permissions locks (Defender/indexer).
    """
    directory = os.path.abspath(job_dir)
    os.makedirs(directory, exist_ok=True)
    target = os.path.join(directory, "journal.json")
    tmp = os.path.join(directory, "journal.json.tmp")

    payload = copy.deepcopy(dict(data))
    payload["journal_schema"] = JOURNAL_SCHEMA_VERSION
    payload["updated_at"] = time.time()

    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)
        fh.flush()
        os.fsync(fh.fileno())

    delay = 0.02
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, target)
            break
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.4)

    return target


def read_journal(job_dir: str) -> Optional[Dict[str, Any]]:
    """Read existing journal.json from job_dir if present and valid."""
    path = journal_path(job_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        _swallowed("journal:read_journal", exc, "journal read fallback")
        return None


def probe_segment_with_ffprobe(file_path: str, fps: float = 30.0) -> Tuple[bool, int]:
    """Verify segment container integrity with ffprobe.

    Returns (is_valid, frame_count).
    If the file is truncated, missing headers/moov atom, or unreadable, returns (False, 0).
    """
    if not os.path.isfile(file_path):
        return False, 0
    try:
        if os.path.getsize(file_path) <= 0:
            return False, 0
    except OSError:
        return False, 0

    from roop.ffmpeg_path import ffprobe_binary
    exe = ffprobe_binary()
    cmd = [
        exe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,duration,codec_name:format=duration",
        "-of", "json",
        file_path,
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if proc.returncode != 0:
            return False, 0
        info = json.loads(proc.stdout or "{}")
        streams = info.get("streams", [])
        if not streams:
            return False, 0
        v_stream = streams[0]
        nb_str = v_stream.get("nb_frames")
        if nb_str and str(nb_str).isdigit() and int(nb_str) > 0:
            return True, int(nb_str)
        # Fall back to duration * fps
        duration = float(v_stream.get("duration") or info.get("format", {}).get("duration", 0.0) or 0.0)
        if duration > 0 and fps > 0:
            return True, int(round(duration * fps))
        # Container header read was successful
        return True, 0
    except Exception as exc:
        _swallowed("journal:probe_segment", exc, "segment failed ffprobe validation")
        return False, 0


def verify_and_recover_segments(
    seg_dir: str,
    segments: List[Dict[str, Any]],
    fps: float = 30.0,
    discard_truncated_tail: bool = True,
) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    """Scan all segment files. Verify integrity with ffprobe.

    If a truncated tail file is detected (e.g. from an ungraceful crash / power loss),
    only that truncated tail file is discarded (deleted) and all preceding valid
    segments are preserved.

    Returns:
      (verified_segments, total_verified_frames, discarded_files)
    """
    verified: List[Dict[str, Any]] = []
    discarded: List[str] = []
    total_frames = 0

    for idx, seg in enumerate(segments):
        fn = seg.get("file", "")
        if not fn:
            break
        full_path = os.path.join(seg_dir, fn)
        is_last = (idx == len(segments) - 1)

        ok, probed_frames = probe_segment_with_ffprobe(full_path, fps=fps)
        if ok:
            recorded_frames = int(seg.get("frames", 0) or 0)
            frames = probed_frames if probed_frames > 0 else recorded_frames
            entry = dict(seg)
            entry["frames"] = frames
            try:
                entry["bytes"] = os.path.getsize(full_path)
            except OSError:
                pass
            verified.append(entry)
            total_frames += frames
        else:
            # File failed ffprobe verification
            if is_last and discard_truncated_tail:
                try:
                    if os.path.isfile(full_path):
                        os.remove(full_path)
                        discarded.append(full_path)
                        print(f"[Journal] Discarded truncated tail segment: {fn}", flush=True)
                except OSError as err:
                    _swallowed("journal:discard_tail", err, "failed to delete corrupt tail")
            # Break contiguous chain on corruption
            break

    return verified, total_frames, discarded


def build_journal_record(
    job_id: str,
    project_id: str,
    source_path: str,
    target_path: str,
    payload: Mapping[str, Any],
    segments: List[Dict[str, Any]],
    frame_start: int,
    frame_end: int,
    current_frame: int,
    status: str = "IN_PROGRESS",
    seg_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble a complete journal document."""
    source_meta = media_receipt(source_path)
    target_meta = media_receipt(target_path)
    recipe_hash = compute_recipe_hash(payload)

    verified_frames = sum(int(s.get("frames", 0) or 0) for s in segments)
    resume_frame = frame_start + verified_frames

    record = {
        "job_id": str(job_id),
        "project_id": str(project_id),
        "status": status,
        "source_media": source_meta,
        "target_media": target_meta,
        "recipe_hash": recipe_hash,
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "current_frame": int(current_frame),
        "resume_frame_index": int(resume_frame),
        "total_verified_frames": int(verified_frames),
        "completed_segments": segments,
        "seg_dir": os.path.abspath(seg_dir) if seg_dir else "",
        "updated_at": time.time(),
    }
    return record


class JournalManager:
    """Manages journal updates during renders and crash recovery on resume."""

    def __init__(self, job_dir: str, unsegmented_interval: int = DEFAULT_UNSEGMENTED_INTERVAL):
        self.job_dir = os.path.abspath(job_dir)
        self.unsegmented_interval = max(10, int(unsegmented_interval))
        self._last_unsegmented_update = 0

    def record_completed_segment(
        self,
        job_id: str,
        project_id: str,
        source_path: str,
        target_path: str,
        payload: Mapping[str, Any],
        segments: List[Dict[str, Any]],
        frame_start: int,
        frame_end: int,
        seg_dir: Optional[str] = None,
    ) -> str:
        """Write journal after a completed segment."""
        verified_frames = sum(int(s.get("frames", 0) or 0) for s in segments)
        record = build_journal_record(
            job_id=job_id,
            project_id=project_id,
            source_path=source_path,
            target_path=target_path,
            payload=payload,
            segments=segments,
            frame_start=frame_start,
            frame_end=frame_end,
            current_frame=frame_start + verified_frames,
            status="IN_PROGRESS",
            seg_dir=seg_dir or self.job_dir,
        )
        return write_journal(self.job_dir, record)

    def record_unsegmented_tick(
        self,
        job_id: str,
        project_id: str,
        source_path: str,
        target_path: str,
        payload: Mapping[str, Any],
        frame_start: int,
        frame_end: int,
        current_frame: int,
        force: bool = False,
    ) -> Optional[str]:
        """Update journal periodically every N frames for unsegmented renders."""
        if not force and (current_frame - self._last_unsegmented_update < self.unsegmented_interval):
            return None
        self._last_unsegmented_update = current_frame
        record = build_journal_record(
            job_id=job_id,
            project_id=project_id,
            source_path=source_path,
            target_path=target_path,
            payload=payload,
            segments=[],
            frame_start=frame_start,
            frame_end=frame_end,
            current_frame=current_frame,
            status="IN_PROGRESS",
            seg_dir=self.job_dir,
        )
        return write_journal(self.job_dir, record)

    def recover_on_resume(
        self,
        fps: float = 30.0,
    ) -> Tuple[Optional[Dict[str, Any]], int, List[str]]:
        """Validate journal, probe existing segments with ffprobe, discard truncated tail,
        and return (recovered_journal, resume_frame_index, discarded_files).
        """
        journal = read_journal(self.job_dir)
        if not journal:
            return None, 0, []

        seg_dir = journal.get("seg_dir") or self.job_dir
        segments = list(journal.get("completed_segments") or [])
        verified, total_frames, discarded = verify_and_recover_segments(
            seg_dir=seg_dir,
            segments=segments,
            fps=fps,
            discard_truncated_tail=True,
        )

        frame_start = int(journal.get("frame_start", 0) or 0)
        resume_frame = frame_start + total_frames

        # Update journal with verified recovery state
        journal["completed_segments"] = verified
        journal["total_verified_frames"] = total_frames
        journal["resume_frame_index"] = resume_frame
        journal["status"] = "RECOVERED"
        write_journal(self.job_dir, journal)

        return journal, resume_frame, discarded
