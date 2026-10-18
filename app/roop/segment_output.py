"""Named, numbered, retained segment outputs and their verified concat.

THE BUG THIS EXISTS FOR. A queue "Split into N segments" run (Batch Matrix `stageSegments`)
is N jobs on ONE target, each with its own `frame_start`/`frame_end`. Every one of them named
its final video from the output TEMPLATE alone (`{file}_{time}`, `utilities.replace_template`),
which knows nothing about the range: with a template that has no time token (`{file}`,
`{file}_swapped`) every segment landed on the same path and `restore_audio` /
`os.remove(existing)` replaced the previous one, so only the LAST segment survived; with the
default template two segments finishing inside the same second collided too, and even the
survivors carried no order to join them by.

THE RULES (all pure functions here, wired by `core.batch_process_regular` and `routes_queue`):

* A segment job writes `{stem}_segment{idx:04d}_{start:06d}_{end:06d}.mp4`: unique per
  (group, index, range) and sortable. Re-rendering the SAME segment overwrites only itself.
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

import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from roop.ffmpeg_path import ffmpeg_binary, ffprobe_binary

_SEG_RE = re.compile(r"^(?P<stem>.+)_segment(?P<idx>\d{4})_(?P<start>\d{6})_(?P<end>\d{6})\.(?P<ext>[A-Za-z0-9]+)$")


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


def segment_filename(stem: str, index: int, start_frame: int, end_frame: int, ext: str = "mp4") -> str:
    if index < 1 or start_frame < 0 or end_frame < start_frame:
        raise ValueError(f"bad segment {index} range {start_frame}-{end_frame}")
    return (f"{safe_stem(stem)}_segment{int(index):04d}_{int(start_frame):06d}_"
            f"{int(end_frame):06d}.{ext.lstrip('.')}")


def segment_destination(output_dir: str, target_path: str, spec: SegmentSpec, ext: str = "mp4") -> str:
    return os.path.join(output_dir, segment_filename(target_path, spec.index, spec.start_frame,
                                                     spec.end_frame, ext))


def parse_segment_filename(name: str) -> Optional[dict]:
    m = _SEG_RE.match(os.path.basename(name))
    if not m:
        return None
    return {"stem": m["stem"], "index": int(m["idx"]), "start": int(m["start"]),
            "end": int(m["end"]), "ext": m["ext"]}


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
                   source_audio: bool = False) -> List[str]:
    """The stream-copy join. `source_audio` muxes the source's own track (brief's form);
    otherwise the segments' audio, already cut per range, is carried."""
    cmd = [ffmpeg_binary(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", manifest]
    if source_audio and source_media:
        cmd += ["-i", source_media, "-map", "0:v:0", "-map", "1:a?"]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a?"]
    return cmd + ["-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", output]


def join_segments(paths: Sequence[str], output: str, manifest: str,
                  source_media: Optional[str] = None, source_audio: bool = False) -> str:
    """Verify, write the manifest, join, and verify the result. Segments are left in place."""
    write_concat_manifest(paths, manifest)
    tmp = output + ".part.mp4"
    try:
        r = _run(concat_command(manifest, tmp, source_media, source_audio))
        if r.returncode or not os.path.isfile(tmp) or os.path.getsize(tmp) <= 0:
            raise RuntimeError(f"concat failed ({r.returncode}): {(r.stderr or '')[-600:]}")
        if not probe_has_video(tmp):
            raise RuntimeError("joined file has no valid video stream")
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
