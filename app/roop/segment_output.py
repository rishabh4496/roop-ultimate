"""Named, numbered, retained segment outputs and their verified concat.

THE BUG THIS EXISTS FOR. A queue "Split into N segments" run (Batch Matrix `stageSegments`)
is N jobs on ONE target, each with its own `frame_start`/`frame_end`. Every one of them named
its final video from the output TEMPLATE alone (`{file}_{time}`, `utilities.replace_template`),
which knows nothing about the range: with a template that has no time token (`{file}`,
`{file}_swapped`) every segment landed on the same path and `restore_audio` /
`os.remove(existing)` replaced the previous one, so only the LAST segment survived; with the
default template two segments finishing inside the same second collided too, and even the
survivors carried no order to join them by.

THE RULES (all pure functions here, wired by `core.final_video_destination` and `routes_queue`):

* A segment job writes `{output_dir}/.segments/{group}/seg_{idx:04d}_{start}_{end}.mp4`.
  `{group}` is the queue's split id (the brief's "job id": one split render = one id, stamped by
  Batch Matrix on every slice), so two splits of two clips that share a file name, or two splits of
  one clip rendered with different settings, cannot meet in one folder. The earlier layout put
  `{stem}_segment{idx}_{start}_{end}.mp4` straight in the output folder, where exactly those two
  cases overwrote each other. Re-rendering the SAME slice of the SAME split overwrites only itself.
* The `.segments` folder survives `clear_output` (`api._clear_output_folder`): an ordinary job
  queued between two slices used to wipe the finished ones.
* Completed segment files are never deleted while a group is open, and are kept after the
  join: the joined file is an addition, not a replacement.
* Joining writes `{stem}_concat_list.txt` (concat-demuxer manifest) after checking that
  every entry exists, is non-empty and holds a decodable video stream, then stream-copies
  the video (GOP-aligned: each segment starts on its own IDR, because each was encoded as
  a separate file) into `{stem}_joined.mp4`.
* Audio: when the segments cover the whole source the source's own audio is muxed (one
  continuous track, no per-boundary priming gaps); otherwise the segments' own audio is
  carried, because it is already cut to each range.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from roop.ffmpeg_path import ffmpeg_binary, ffprobe_binary

SEGMENTS_DIRNAME = ".segments"
_SEG_RE = re.compile(r"^seg_(?P<idx>\d{4,})_(?P<start>\d+)_(?P<end>\d+)\.(?P<ext>[A-Za-z0-9]+)$")


@dataclass(frozen=True)
class SegmentSpec:
    """Where one segment of a split render belongs."""
    group: str          # shared by every segment of one split (the queue stamps it)
    index: int          # 1-based position inside the group
    count: int
    start_frame: int
    end_frame: int


def safe_stem(name: str) -> str:
    """A filesystem-safe stem: the target's own name minus extension and path separators."""
    stem = os.path.splitext(os.path.basename(str(name or "")))[0]
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
    return stem or "output"


def safe_group_id(group: str) -> str:
    """A folder name for a split id. Ids the queue stamps (`uuid`-like) pass through unchanged; anything
    with a separator or other unsafe character is sanitised AND suffixed with a short digest of the raw
    id, so `auto:a/b` and `auto:a_b` cannot land in one folder."""
    raw = str(group or "")
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    if clean and clean == raw:
        return clean
    digest = hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:8]
    return f"{clean[:48]}-{digest}" if clean else f"split-{digest}"


def segments_dir(output_dir: str, group: str) -> str:
    return os.path.join(output_dir, SEGMENTS_DIRNAME, safe_group_id(group))


def segment_filename(index: int, start_frame: int, end_frame: int, ext: str = "mp4") -> str:
    if index < 1 or start_frame < 0 or end_frame < start_frame:
        raise ValueError(f"bad segment {index} range {start_frame}-{end_frame}")
    return f"seg_{int(index):04d}_{int(start_frame)}_{int(end_frame)}.{ext.lstrip('.')}"


def segment_destination(output_dir: str, spec: SegmentSpec, ext: str = "mp4") -> str:
    """Where one slice is written. The folder is NOT created here (the caller that writes owns that)."""
    return os.path.join(segments_dir(output_dir, spec.group),
                        segment_filename(spec.index, spec.start_frame, spec.end_frame, ext))


def assert_safe_segment_destination(segment_dest: str, target_final_output: str) -> None:
    """Enforce that no code path reuses the final target output filename for intermediate segments."""
    if not segment_dest or not target_final_output:
        return
    if os.path.abspath(segment_dest) == os.path.abspath(target_final_output):
        raise ValueError(
            f"CRITICAL: Segment custody violation! Intermediate segment path reuses final output destination: {segment_dest}"
        )


def audit_segment_custody(path: str, expected_frames: Optional[int] = None,
                          expected_duration: Optional[float] = None) -> dict:
    """After each segment finishes encoding, run a fast custody check and print a terminal custody receipt.
    Receipt format:
    [CUSTODY OK] seg_0002.mp4 | Frames: 600/600 | Size: 34.8 MB | Hash/Path Verified
    """
    from roop import watchdog
    from roop.finalize_output import probe_container, _frames_of

    if not os.path.isfile(path):
        err = f"Segment file missing at: {path}"
        watchdog.broadcast_alert("[CUSTODY FAILED]", err, color_code="\033[91m", level="ERROR")
        return {"ok": False, "error": err, "path": path}

    size_b = os.path.getsize(path)
    if size_b <= 0:
        err = f"Segment file is 0 bytes: {path}"
        watchdog.broadcast_alert("[CUSTODY FAILED]", err, color_code="\033[91m", level="ERROR")
        return {"ok": False, "error": err, "path": path, "size_bytes": 0}

    size_mb = size_b / (1024 * 1024)

    # Frame count & duration probe
    probe = probe_container(path)
    actual_frames = None
    actual_dur = None
    if probe and probe.get("video"):
        actual_frames = _frames_of(probe["video"])
        actual_dur = probe["video"].get("duration_s")

    # Fast SHA-256 hash (first/last 64KB for speed on large video files, or whole file if <= 1MB)
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            if size_b <= 1024 * 1024:
                hasher.update(f.read())
            else:
                hasher.update(f.read(65536))
                f.seek(-65536, os.SEEK_END)
                hasher.update(f.read(65536))
    except Exception:
        pass
    digest = hasher.hexdigest()[:12]

    filename = os.path.basename(path)
    exp_f_str = f"{expected_frames}" if expected_frames is not None else (f"{actual_frames}" if actual_frames is not None else "--")
    act_f_str = f"{actual_frames}" if actual_frames is not None else "--"
    frames_receipt = f"{act_f_str}/{exp_f_str}"

    receipt_msg = f"{filename} | Frames: {frames_receipt} | Size: {size_mb:.1f} MB | Hash/Path Verified ({digest})"
    watchdog.broadcast_alert("[CUSTODY OK]", receipt_msg, color_code="\033[92m", level="INFO")

    return {
        "ok": True,
        "path": path,
        "filename": filename,
        "frames": actual_frames,
        "expected_frames": expected_frames,
        "duration_s": actual_dur,
        "size_bytes": size_b,
        "size_mb": round(size_mb, 2),
        "hash": digest,
    }


def parse_segment_filename(name: str) -> Optional[dict]:
    m = _SEG_RE.match(os.path.basename(name))
    if not m:
        return None
    return {"index": int(m["idx"]), "start": int(m["start"]), "end": int(m["end"]), "ext": m["ext"]}


def concat_list_path(output_dir: str, stem: str) -> str:
    return os.path.join(output_dir, f"{safe_stem(stem)}_concat_list.txt")


def joined_path(output_dir: str, stem: str, ext: str = "mp4") -> str:
    return os.path.join(output_dir, f"{safe_stem(stem)}_joined.{ext.lstrip('.')}")


def _run(cmd: List[str], timeout: int = 600) -> subprocess.CompletedProcess:
    kwargs = {"capture_output": True, "text": True, "timeout": timeout, "check": False}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000
    return subprocess.run(cmd, **kwargs)


def probe_has_video(path: str) -> bool:
    """True when ffprobe finds a video stream with a codec in `path`."""
    r = _run([ffprobe_binary(), "-v", "error", "-select_streams", "v:0", "-show_entries",
              "stream=codec_name", "-of", "json", path], timeout=120)
    if r.returncode:
        return False
    try:
        return bool(json.loads(r.stdout or "{}").get("streams"))
    except ValueError:
        return False


def verify_segments(paths: Sequence[str]) -> List[str]:
    """Reasons the segment files cannot be joined; empty means they can."""
    reasons: List[str] = []
    if not paths:
        return ["no segment files"]
    for p in paths:
        if not os.path.isfile(p):
            reasons.append(f"missing: {os.path.basename(p)}")
        elif os.path.getsize(p) <= 0:
            reasons.append(f"zero bytes: {os.path.basename(p)}")
        elif not probe_has_video(p):
            reasons.append(f"no valid video stream: {os.path.basename(p)}")
    return reasons


def write_concat_manifest(paths: Sequence[str], manifest: str) -> str:
    """Write the concat-demuxer list for `paths` in the order given, after verifying them."""
    reasons = verify_segments(paths)
    if reasons:
        raise RuntimeError("cannot join segments: " + "; ".join(reasons))
    tmp = manifest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for p in paths:
            fh.write("file '%s'\n" % os.path.abspath(p).replace("\\", "/").replace("'", r"'\''"))
    os.replace(tmp, manifest)
    return manifest


def concat_command(manifest: str, output: str, source_media: Optional[str] = None,
                   source_audio: bool = False, audio_codec: str = "aac") -> List[str]:
    """The stream-copy join. `source_audio` muxes the source's own track (brief's form);
    otherwise the segments' audio, already cut per range, is carried.

    `audio_codec` is `aac` (re-encode, always valid in mp4) or `copy` (the source's own audio
    stream untouched). `join_segments` picks `copy` for a source whose audio mp4 can hold as-is."""
    cmd = [ffmpeg_binary(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", manifest]
    if source_audio and source_media:
        cmd += ["-i", source_media, "-map", "0:v:0", "-map", "1:a?"]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a?"]
    # No `-shortest`: it trims the VIDEO to the shorter stream, and a source whose audio ends a few
    # frames early would silently lose those frames (it cost 4 of 30 in the test clip).
    return cmd + ["-c:v", "copy", "-c:a", audio_codec, "-movflags", "+faststart", output]


# Audio codecs an mp4 container can carry as-is. Anything else (opus, vorbis, pcm, dts...) is
# re-encoded to AAC rather than failing the join.
MP4_COPYABLE_AUDIO = frozenset({"aac", "mp3", "ac3", "eac3", "alac"})


def _ffprobe_json(args: List[str], timeout: int = 120) -> dict:
    r = _run([ffprobe_binary(), "-v", "error", *args], timeout=timeout)
    if r.returncode:
        return {}
    try:
        return json.loads(r.stdout or "{}")
    except ValueError:
        return {}


def probe_audio_codec(path: str) -> Optional[str]:
    streams = _ffprobe_json(["-select_streams", "a:0", "-show_entries", "stream=codec_name",
                             "-of", "json", path]).get("streams") or []
    return streams[0].get("codec_name") if streams else None


def audio_codec_for_join(source_media: Optional[str]) -> str:
    """`copy` when the source's audio can ride into an mp4 untouched, else `aac`."""
    return "copy" if source_media and probe_audio_codec(source_media) in MP4_COPYABLE_AUDIO else "aac"


def probe_video_duration(path: str) -> Optional[float]:
    """Seconds of the first video stream (the container's when the stream states none)."""
    info = _ffprobe_json(["-select_streams", "v:0", "-show_entries", "stream=duration:format=duration",
                          "-of", "json", path])
    for raw in ((info.get("streams") or [{}])[0].get("duration"), (info.get("format") or {}).get("duration")):
        try:
            v = float(raw)
            if v > 0:
                return v
        except (TypeError, ValueError):
            continue
    return None


def probe_fps(path: str) -> Optional[float]:
    streams = _ffprobe_json(["-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                             "-of", "json", path]).get("streams") or []
    try:
        num, _, den = str((streams[0] or {}).get("r_frame_rate", "")).partition("/")
        v = float(num) / float(den or 1)
        return v if v > 0 else None
    except (IndexError, ValueError, ZeroDivisionError):
        return None


def expected_join_duration(ranges: Iterable[tuple], fps: Optional[float],
                           source_media: Optional[str] = None, covers_source: bool = False) -> Optional[float]:
    """What the joined file's video should last: the SOURCE's own duration when the segments cover it
    whole, else the summed length of the ranges (inclusive 1-based frames) at the clip's frame rate."""
    if covers_source and source_media:
        d = probe_video_duration(source_media)
        if d:
            return d
    if not fps:
        return None
    frames = sum(int(e) - int(s) + 1 for s, e in ranges)
    return frames / float(fps) if frames > 0 else None


def verify_joined(joined: str, expected_s: Optional[float], fps: Optional[float],
                  tolerance_frames: float = 2.0) -> dict:
    """Does the joined file last as long as it should? `ok` is None (neither pass nor fail) when there
    is nothing to compare against: an unmeasured result must not read as either."""
    actual = probe_video_duration(joined) if os.path.isfile(joined) else None
    tol = max(0.1, tolerance_frames / float(fps)) if fps else 0.25
    out = {"expected_s": None if expected_s is None else round(expected_s, 3),
           "joined_s": None if actual is None else round(actual, 3),
           "tolerance_s": round(tol, 3), "ok": None, "delta_s": None}
    if actual is None:
        out["ok"] = False
        out["reason"] = "the joined file has no measurable video duration"
    elif expected_s is not None:
        out["delta_s"] = round(actual - expected_s, 3)
        out["ok"] = abs(actual - expected_s) <= tol
        if not out["ok"]:
            out["reason"] = (f"joined video is {actual:.2f}s, expected {expected_s:.2f}s "
                             f"(off by {actual - expected_s:+.2f}s, tolerance {tol:.2f}s)")
    return out


def join_segments(paths: Sequence[str], output: str, manifest: str,
                  source_media: Optional[str] = None, source_audio: bool = False) -> str:
    """Verify, write the manifest, join, and verify the result. Segments are left in place."""
    from roop import watchdog
    from roop.finalize_output import probe_container, _frames_of

    write_concat_manifest(paths, manifest)
    tmp = output + ".part.mp4"

    # Pre-concat verification: sum duration and frames of all slices
    sum_frames = 0
    sum_dur = 0.0
    for p in paths:
        probe_p = probe_container(p)
        if probe_p and probe_p.get("video"):
            f_cnt = _frames_of(probe_p["video"])
            if f_cnt:
                sum_frames += f_cnt
            dur = probe_p["video"].get("duration_s")
            if dur:
                sum_dur += dur

    try:
        codec = audio_codec_for_join(source_media) if (source_audio and source_media) else "aac"
        cmd = concat_command(manifest, tmp, source_media, source_audio, codec)
        cmd_str = " ".join(cmd)
        r = _run(cmd)

        if r.returncode or not os.path.isfile(tmp) or os.path.getsize(tmp) <= 0:
            stderr_tail = "\n".join((r.stderr or "").strip().splitlines()[-20:])
            err_box = [
                "",
                "\033[91m\033[1m╔══════════════════════════════════════════════════════════════════════════════════════════════╗\033[0m",
                "\033[91m\033[1m║                             STAGE 4 FFMPEG CONCATENATION FAILURE                             ║\033[0m",
                "\033[91m\033[1m╠══════════════════════════════════════════════════════════════════════════════════════════════╣\033[0m",
                f"\033[91m\033[1m║ Command  :\033[0m {cmd_str[:80]}...",
                f"\033[91m\033[1m║ Exit Code:\033[0m {r.returncode}",
                "\033[91m\033[1m╠──────────────────────────────────────────────────────────────────────────────────────────────╣\033[0m",
                "\033[91m\033[1m║ Last 20 lines of stderr:\033[0m",
                *[f"║   {line[:88]}" for line in stderr_tail.splitlines()],
                "\033[91m\033[1m╚══════════════════════════════════════════════════════════════════════════════════════════════╝\033[0m",
                "",
            ]
            try:
                from roop import terminal_dashboard
                terminal_dashboard.print_above(err_box)
            except Exception:
                for line in err_box:
                    print(line, flush=True)
            raise RuntimeError(f"FFmpeg concat failed ({r.returncode}):\n{stderr_tail}")

        if not probe_has_video(tmp):
            raise RuntimeError("joined file has no valid video stream")

        # Post-concat verification: verify joined video duration, frame count, and audio remux
        probe_joined = probe_container(tmp)
        if probe_joined and probe_joined.get("video"):
            joined_frames = _frames_of(probe_joined["video"])
            joined_dur = probe_joined["video"].get("duration_s")

            # Duration parity check
            if sum_dur > 0 and joined_dur is not None:
                if abs(joined_dur - sum_dur) > 0.5:
                    watchdog.broadcast_alert(
                        "[WARN]",
                        f"Joined video duration ({joined_dur:.2f}s) differs from sum of segments ({sum_dur:.2f}s).",
                        color_code="\033[93m",
                        level="WARNING"
                    )

            # Audio stream remux check
            if (source_audio and source_media) or any(probe_audio_codec(p) for p in paths):
                if not probe_joined.get("audio"):
                    watchdog.broadcast_alert(
                        "[WARN]",
                        "Joined video is missing audio stream after concatenation remux.",
                        color_code="\033[93m",
                        level="WARNING"
                    )

            joined_size_mb = os.path.getsize(tmp) / (1024 * 1024)
            joined_name = os.path.basename(output)
            j_f_str = f"{joined_frames}" if joined_frames is not None else "--"
            s_f_str = f"{sum_frames}" if sum_frames > 0 else j_f_str
            watchdog.broadcast_alert(
                "[CUSTODY OK]",
                f"{joined_name} | Frames: {j_f_str}/{s_f_str} | Size: {joined_size_mb:.1f} MB | Lossless Concat Verified",
                color_code="\033[92m",
                level="INFO"
            )

        os.replace(tmp, output)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return output


def covers_whole_source(ranges: Iterable[tuple], total_frames: int) -> bool:
    """True when the ranges, ordered, run from the first frame to the last without a gap.

    Ranges are inclusive frame numbers as `stageSegments` writes them (1-based)."""
    ordered = sorted((int(s), int(e)) for s, e in ranges)
    if not ordered or not total_frames:
        return False
    if ordered[0][0] > 1:
        return False
    for (_, prev_end), (nxt_start, _) in zip(ordered, ordered[1:]):
        if nxt_start > prev_end + 1:
            return False
    return ordered[-1][1] >= int(total_frames)
