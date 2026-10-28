"""Stage 4 of a render: finalize the output file, verify it, and only then delete what produced it.

WHY. Until this module the end of a render was

    ffmpeg.restore_audio(temp, source, start, end, destination)       # return value ignored
    if os.path.isfile(destination): _remove_file_retry(temp)           # "exists" is the whole check

so a mux that died half way left a PARTIAL destination, `isfile` was true, the only complete copy of the
render (the temp video) was deleted, and the job was still marked COMPLETED. `os.path.isfile` is the same
class of instrument as the swap audit that counts intent: it reports a file, not a playable render.

WHAT THIS DOES, in order (`finalize_render`):

  1. probe the intermediate video (the REFERENCE: frame count, duration, colour tags),
  2. mux the source audio onto it with `util_ffmpeg.restore_audio` (an exact, audio-only cut for a trimmed
     render, a stream copy for both streams) or, for a silent render, move it into place,
  3. probe the DESTINATION once (ffprobe header read, no decode) and verify it against the reference:
       - the file exists, is non-empty and has a video stream with a non-zero duration,
       - its frame count equals the reference's (a mux must not drop frames) and, when the caller knows it,
         the frame count the trim implies,
       - the colour tags (matrix / primaries / transfer / range, bt709 or bt2020) are the reference's,
       - when the source had audio: audio is present, starts within one frame of the video and does not run
         past it by more than a frame (lip-sync cannot drift by a frame),
  4. ONLY IF every check passed: delete the intermediate and the helper files of this job.
     If any failed: keep the intermediate, remove the partial destination, say why.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

STAGE_4_FINALIZE = "STAGE_4_FINALIZE"

# ffprobe prints these for an untagged stream; they are "no information", not a colour space.
_UNSET = {"", "unknown", "unspecified", "reserved", "n/a", "na", "none"}
_COLOR_KEYS = ("color_space", "color_primaries", "color_transfer", "color_range")


def _probe_binary() -> str:
    from roop.ffmpeg_path import ffprobe_binary
    return ffprobe_binary() or "ffprobe"


def _fraction(text: Any) -> Optional[float]:
    try:
        num, _, den = str(text).partition("/")
        value = float(num) / float(den or 1)
        return value if value > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _float(value: Any) -> Optional[float]:
    try:
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def probe_container(path: str, timeout: float = 20.0) -> Optional[Dict[str, Any]]:
    """One ffprobe header read (no decode, no packet count). None when the file cannot be probed.

    The result is the raw ffprobe JSON reduced to what verification reads:
    ``{"size", "format_duration", "video": {...}|None, "audio": {...}|None}``."""
    cmd = [_probe_binary(), "-v", "error", "-of", "json",
           "-show_entries",
           "format=duration,size:stream=index,codec_type,codec_name,duration,nb_frames,avg_frame_rate,"
           "r_frame_rate,start_time,width,height," + ",".join(_COLOR_KEYS)]
    cmd.append(path)
    kwargs: Dict[str, Any] = {"capture_output": True, "text": True, "timeout": timeout, "check": False}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000          # CREATE_NO_WINDOW (see util_ffmpeg.run_ffmpeg)
    try:
        proc = subprocess.run(cmd, **kwargs)
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout or "{}")
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format") or {}

    def _reduce(s: Optional[dict]) -> Optional[dict]:
        if s is None:
            return None
        out = {"codec": s.get("codec_name"), "duration_s": _float(s.get("duration")),
               "start_s": _float(s.get("start_time")),
               "fps": _fraction(s.get("avg_frame_rate")) or _fraction(s.get("r_frame_rate")),
               "width": s.get("width"), "height": s.get("height")}
        nb = str(s.get("nb_frames") or "")
        out["frames"] = int(nb) if nb.isdigit() else None       # absent for mkv/webm: not a zero
        for key in _COLOR_KEYS:
            tag = str(s.get(key) or "").strip().lower()
            out[key] = None if tag in _UNSET else tag
        return out

    reduced_audio = _reduce(audio)
    if reduced_audio is not None and path.lower().endswith((".mp4", ".mov", ".m4v")):
        # ffprobe's audio duration is the MEDIA length: it ignores an edit list that skips the head of the audio (the
        # exact trim cut, util_ffmpeg._skip_audio_head) or ends it with the video. What a player presents is the edits.
        try:
            from roop import mp4_timing
            span = mp4_timing.edit_span(path, b"soun")
            if span is not None:
                reduced_audio["start_s"], reduced_audio["duration_s"] = span
        except Exception as exc:
            from roop.degrade import swallowed as _swallowed
            _swallowed("roop/finalize_output.py:edit_span", exc, "audio duration stays ffprobe's media length")
    return {"size": _float(fmt.get("size")), "format_duration": _float(fmt.get("duration")),
            "video": _reduce(video), "audio": reduced_audio}


def _frames_of(video: Dict[str, Any]) -> Optional[int]:
    """The header's frame count, else the duration at the stream's own rate (rounded)."""
    if video.get("frames") is not None:
        return int(video["frames"])
    d, fps = video.get("duration_s"), video.get("fps")
    return int(round(d * fps)) if d and fps else None


def color_tags(probe: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    video = (probe or {}).get("video") or {}
    return {k: video.get(k) for k in _COLOR_KEYS}


def verify_output(path: str, reference: Optional[Dict[str, Any]] = None,
                  expected_frames: Optional[int] = None, expect_audio: bool = False,
                  probe: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check `path` with one header probe. `ok` is False when any hard problem is listed.

    `reference` is the probe of the intermediate video the file was made from; passing it adds the checks
    that a mux must not change anything about the video (frame count, colour tags)."""
    problems: List[str] = []
    out: Dict[str, Any] = {"ok": False, "problems": problems, "probe_ms": None}
    if not os.path.isfile(path):
        problems.append("output file does not exist")
        return out
    size = os.path.getsize(path)
    if size <= 0:
        problems.append("output file is empty")
        return out
    t0 = time.perf_counter()
    probe = probe if probe is not None else probe_container(path)
    out["probe_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    if probe is None:
        problems.append("ffprobe cannot read the output container")
        return out
    video, audio = probe.get("video"), probe.get("audio")
    if video is None:
        problems.append("output has no video stream")
        return out
    duration = video.get("duration_s") or probe.get("format_duration")
    frames = _frames_of(video)
    fps = video.get("fps")
    out.update(bytes=int(size), video_codec=video.get("codec"), width=video.get("width"),
               height=video.get("height"), fps=round(fps, 4) if fps else None,
               video_duration_s=round(duration, 4) if duration else None, frames=frames,
               frames_source="header" if video.get("frames") is not None else "duration",
               audio_present=audio is not None, color=color_tags(probe))
    if not duration or duration <= 0:
        problems.append("video stream duration is zero")
    if frames is not None and frames <= 0:
        problems.append("video stream has no frames")
    # A header count is exact; a duration-derived one is good to a frame of rounding.
    slack = 0 if video.get("frames") is not None else 1
    if expected_frames is not None and frames is not None and abs(frames - expected_frames) > slack:
        problems.append(f"frame count {frames} != the {expected_frames} the trim implies")
    if reference is not None:
        ref_video = reference.get("video") or {}
        ref_frames = _frames_of(ref_video) if ref_video else None
        if ref_frames is not None and frames is not None and abs(frames - ref_frames) > slack:
            problems.append(f"the mux changed the frame count: {ref_frames} -> {frames}")
        ref_tags, tags = color_tags(reference), color_tags(probe)
        changed = {k: (ref_tags[k], tags[k]) for k in _COLOR_KEYS if ref_tags[k] != tags[k]}
        out["color_preserved"] = not changed
        # "Preserved" is vacuous when the intermediate carried no tags at all (None == None); say so.
        out["color_tagged"] = any(ref_tags.values())
        if changed:
            problems.append("colour tags changed by the mux: "
                            + ", ".join(f"{k} {a}->{b}" for k, (a, b) in changed.items()))
    if expect_audio:
        if audio is None:
            problems.append("the source has audio but the output has none")
        elif fps and duration:
            frame_s = 1.0 / fps
            a_start, v_start = audio.get("start_s") or 0.0, video.get("start_s") or 0.0
            offset = a_start - v_start
            a_end = (audio.get("duration_s") or 0.0) + a_start
            overrun = a_end - ((duration or 0.0) + v_start)
            out["audio_start_offset_ms"] = round(offset * 1000.0, 2)
            out["audio_overrun_ms"] = round(overrun * 1000.0, 2)
            if abs(offset) > frame_s:
                problems.append(f"audio starts {offset * 1000:.1f} ms from the video (> 1 frame)")
            if overrun > frame_s:
                problems.append(f"audio runs {overrun * 1000:.1f} ms past the video (> 1 frame)")
    out["ok"] = not problems
    return out


def _remove(path: str, attempts: int = 5, delay: float = 0.3) -> bool:
    for i in range(attempts):
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if i < attempts - 1:
                time.sleep(delay)
        except OSError:
            break
    return not os.path.exists(path)


def _helper_files(temp_video: str, destination: str) -> List[str]:
    """Scratch files a finished render of this job must not leave behind (never the segment parts: the
    segment writer owns those and keeps them on purpose after a Stop)."""
    stem = os.path.splitext(destination)[0]
    return [stem + ".__audio_cut.mka", temp_video + ".resume.json.tmp"]


def find_orphans(temp_video: str, destination: str) -> List[str]:
    """Files of THIS job still on disk after a completed render: the intermediate, the audio-cut scratch,
    the segment parts (`.<temp>.seg####`) and their resume manifest."""
    found = [p for p in _helper_files(temp_video, destination) if os.path.exists(p)]
    if os.path.exists(temp_video) and os.path.abspath(temp_video) != os.path.abspath(destination):
        found.append(temp_video)
    folder = os.path.dirname(temp_video) or "."
    base = os.path.basename(temp_video)
    found += glob.glob(os.path.join(glob.escape(folder), f".{glob.escape(base)}.seg*"))
    manifest = temp_video + ".resume.json"
    if os.path.exists(manifest):
        found.append(manifest)
    return sorted(set(found))


def finalize_render(temp_video: str, destination: str, *,
                    audio_source: Optional[str], trim_start: Optional[int], trim_end: Optional[int],
                    expected_frames: Optional[int] = None, skip_audio: bool = False,
                    stopped: bool = False, video_source: Optional[str] = None,
                    audio_offset_ms: float = 0.0,
                    mux: Optional[Callable[..., bool]] = None,
                    source_has_audio: Optional[Callable[[str], bool]] = None) -> Dict[str, Any]:
    """Mux (or move), verify, then delete. Returns the JSON-safe result described in the module docstring.

    `mux` and `source_has_audio` are injectable so a test can fail the mux on demand; production passes
    neither. `stopped` marks a partial render: its frame count is not compared to the trim (only to the
    intermediate it was made from) and segment parts are expected to remain for a resume."""
    t_all = time.perf_counter()
    result: Dict[str, Any] = {
        "stage": STAGE_4_FINALIZE, "ok": False, "output": destination,
        "expected_frames": None if stopped else expected_frames, "stopped": bool(stopped),
        "problems": [], "orphans": [], "temp_deleted": False, "wall_s": None, "mux_s": None,
    }
    if mux is None:
        from roop import util_ffmpeg
        mux = util_ffmpeg.restore_audio
    reference = probe_container(temp_video)
    if reference is None or reference.get("video") is None:
        result["problems"].append("the intermediate video cannot be probed; it is kept")
        result["wall_s"] = round(time.perf_counter() - t_all, 3)
        return result
    ref_video = reference["video"]
    result["reference_frames"] = _frames_of(ref_video)

    if not stopped and expected_frames is not None:
        ref_frames = result.get("reference_frames")
        if ref_frames is not None and ref_frames != expected_frames:
            result["problems"].append(f"frame count {ref_frames} != the {expected_frames} the trim implies")
            result["wall_s"] = round(time.perf_counter() - t_all, 3)
            result["kept_intermediate"] = temp_video if os.path.exists(temp_video) else None
            return result

    # A variable-frame-rate source: re-stamp the intermediate's samples at the source's own timestamps BEFORE the mux,
    # so the muxed file keeps every frame where the camera put it (the mux copies packets and their timestamps).
    vsrc = video_source or audio_source
    if vsrc and result.get("reference_frames"):
        from roop import vfr_timing
        result["vfr"] = vfr_timing.retime_intermediate(temp_video, vsrc, trim_start, int(result["reference_frames"]))

    expect_audio = False
    if not skip_audio and audio_source:
        if source_has_audio is None:
            from roop import utilities as _util
            source_has_audio = lambda p: _util.audio_sample_rate(p) is not None     # noqa: E731
        try:
            expect_audio = bool(source_has_audio(audio_source))
        except Exception as exc:
            from roop.degrade import swallowed as _swallowed
            _swallowed("roop/finalize_output.py:source_has_audio", exc,
                       "the source's audio could not be probed; audio presence is not verified")
            expect_audio = False
    result["expect_audio"] = expect_audio

    t_mux = time.perf_counter()
    mux_ok = True
    try:
        if skip_audio or not audio_source:
            shutil.move(temp_video, destination)
            moved = True
        else:
            moved = False
            # Stage 17: an override track (audio_source is not the video) or a slip needs the trim taken from the VIDEO's
            # timing, not from the audio file's; the historical call is unchanged when neither applies.
            if (video_source and audio_source != video_source) or abs(float(audio_offset_ms or 0.0)) > 1e-9:
                mux_ok = bool(mux(temp_video, audio_source, trim_start, trim_end, destination,
                                  timing_source=video_source or audio_source, audio_offset_ms=audio_offset_ms))
            else:
                mux_ok = bool(mux(temp_video, audio_source, trim_start, trim_end, destination))
    except OSError as exc:
        mux_ok = False
        result["problems"].append(f"could not write the output: {exc}")
        moved = not os.path.exists(temp_video)
    result["mux_s"] = round(time.perf_counter() - t_mux, 3)
    if not mux_ok:
        result["problems"].append("the audio mux failed")

    check = verify_output(destination, reference=reference,
                          expected_frames=None if stopped else expected_frames,
                          expect_audio=expect_audio)
    result["verify"] = {k: v for k, v in check.items() if k not in ("ok", "problems")}
    result["problems"] += check["problems"]
    result["ok"] = bool(mux_ok and check["ok"])

    if result["ok"]:
        result["temp_deleted"] = True if moved else _remove(temp_video)
        for path in _helper_files(temp_video, destination):
            _remove(path)
        if not stopped:
            result["orphans"] = find_orphans(temp_video, destination)
    else:
        # The intermediate is the only complete copy: keep it. Remove the partial/rejected destination so
        # the output folder does not hold a file that looks like a result. (Not after a MOVE: then the
        # destination IS the only copy and is left for the user to inspect.)
        if not moved and os.path.exists(destination):
            _remove(destination)
        for path in _helper_files(temp_video, destination):
            _remove(path)
        result["kept_intermediate"] = temp_video if os.path.exists(temp_video) else None
    result["wall_s"] = round(time.perf_counter() - t_all, 3)
    return result
