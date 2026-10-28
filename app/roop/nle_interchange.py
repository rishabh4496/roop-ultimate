"""NLE interchange for Roop project timelines.

Exports a standards-oriented FCPXML sequence and CMX3600/Resolve EDLs.  The
exports are intentionally media-reference based: they never re-encode the
source. Face overlays are emitted as separate lanes/tracks and carry alpha
media references so a compositor can replace or grade them independently.
"""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping
from xml.etree import ElementTree as ET
from xml.dom import minidom


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fps(document: Mapping[str, Any], default: float = 30.0) -> float:
    timeline = document.get("timeline") or {}
    return max(1.0, _num(timeline.get("fps") or document.get("fps"), default))


def _frame_time(frames: int, fps: float) -> str:
    rate = Fraction(str(fps)).limit_denominator(100000)
    return f"{int(frames) * rate.denominator}/{rate.numerator}s"


def _timecode(frames: int, fps: float) -> str:
    rate = int(round(fps))
    frames = max(0, int(frames))
    hh, rem = divmod(frames, rate * 3600)
    mm, rem = divmod(rem, rate * 60)
    ss, ff = divmod(rem, rate)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def detect_scene_cuts(video_path: str, *, fps: float | None = None, use_pyscenedetect: bool = True) -> list[int]:
    """Return scene-cut frame numbers using the project's detector."""
    from roop.scene_detector import ContentAwareSceneDetector
    detector = ContentAwareSceneDetector(use_gpu=False, use_pyscenedetect=use_pyscenedetect)
    return sorted(int(frame) for frame in detector.scan_video(video_path))


def _cuts(document: Mapping[str, Any], scene_cuts: Iterable[int] | None) -> list[int]:
    if scene_cuts is not None:
        return sorted({int(x) for x in scene_cuts if int(x) > 0})
    return sorted({int(x) for x in ((document.get("timeline") or {}).get("scene_cuts") or []) if int(x) > 0})


def _segments(document: Mapping[str, Any], scene_cuts: Iterable[int] | None) -> list[tuple[int, int]]:
    timeline = document.get("timeline") or {}
    start = int(timeline.get("frame_start", 0) or 0)
    end = int(timeline.get("frame_end", 0) or 0)
    cuts = [c for c in _cuts(document, scene_cuts) if start < c < end]
    points = [start] + cuts + ([end] if end > start else [])
    if len(points) < 2:
        return [(start, max(start + 1, end))]
    return list(zip(points[:-1], points[1:]))


def _target_path(document: Mapping[str, Any], project_path: str) -> str:
    from roop.project_io import resolve_media
    return resolve_media(document.get("media", {}).get("target", {}), project_path)


def _face_segments(document: Mapping[str, Any]) -> list[dict]:
    return [dict(item) for item in ((document.get("timeline") or {}).get("face_segments") or [])]


def _face_asset(segment: Mapping[str, Any], document: Mapping[str, Any], project_path: str) -> str:
    from roop.project_io import resolve_media
    ref = segment.get("alpha_asset") or segment.get("asset") or {}
    if isinstance(ref, str):
        return os.path.abspath(ref if os.path.isabs(ref) else os.path.join(os.path.dirname(project_path), ref))
    return resolve_media(ref, project_path) if ref else ""


def export_fcpxml(document: Mapping[str, Any], project_path: str, output_path: str,
                  *, scene_cuts: Iterable[int] | None = None) -> str:
    """Write an FCPXML 1.10 sequence with markers and separate face lanes."""
    fps = _fps(document)
    segments = _segments(document, scene_cuts)
    target_path = _target_path(document, project_path)
    timeline = document.get("timeline") or {}
    duration = max(1, int(timeline.get("frame_end", 0) or 0) - int(timeline.get("frame_start", 0) or 0))

    fcpxml = ET.Element("fcpxml", version="1.10")
    resources = ET.SubElement(fcpxml, "resources")
    fmt = ET.SubElement(resources, "format", id="r1", name="Roop Timeline", frameDuration=_frame_time(1, fps), width="1920", height="1080")
    del fmt
    asset_id = "r2"
    ET.SubElement(resources, "asset", id=asset_id, name=os.path.basename(target_path), src=Path(target_path).as_uri(), start="0s", duration=_frame_time(duration, fps), hasVideo="1", hasAudio="1", format="r1")

    face_assets: dict[str, str] = {}
    next_id = 3
    for index, segment in enumerate(_face_segments(document)):
        path = _face_asset(segment, document, project_path)
        if not path:
            continue
        rid = f"r{next_id}"
        next_id += 1
        face_assets[str(index)] = rid
        ET.SubElement(resources, "asset", id=rid, name=os.path.basename(path), src=Path(path).as_uri(), start="0s", duration=_frame_time(max(1, int(segment.get("out", duration)) - int(segment.get("in", 0))), fps), hasVideo="1", hasAudio="0", format="r1")

    library = ET.SubElement(fcpxml, "library", location="file:///Roop/Projects")
    event = ET.SubElement(library, "event", name=str(document.get("name") or "Roop Project"))
    project = ET.SubElement(event, "project", name=str(document.get("name") or "Roop Project"))
    sequence = ET.SubElement(project, "sequence", format="r1", duration=_frame_time(duration, fps), tcStart="0s", tcFormat="NDF")
    spine = ET.SubElement(sequence, "spine")

    cursor = 0
    scene_clips = []
    for index, (start, end) in enumerate(segments):
        clip = ET.SubElement(spine, "asset-clip", name=f"Scene {index + 1}", ref=asset_id, offset=_frame_time(cursor, fps), start=_frame_time(start, fps), duration=_frame_time(max(1, end - start), fps), lane="0")
        for cut in _cuts(document, scene_cuts):
            if start == cut:
                ET.SubElement(clip, "marker", start="0s", duration=_frame_time(1, fps), value="Scene Cut")
        scene_clips.append((start, end, cursor, clip))
        cursor += end - start

    for index, segment in enumerate(_face_segments(document)):
        rid = face_assets.get(str(index))
        if not rid:
            continue
        start = int(segment.get("in", 0) or 0)
        end = int(segment.get("out", start + 1) or start + 1)
        parent = next((item for item in scene_clips if item[0] <= start < item[1]), None)
        if parent is None:
            parent = scene_clips[0] if scene_clips else (0, 0, 0, spine)
        parent_start, _, parent_offset, parent_element = parent
        clip = ET.SubElement(parent_element, "asset-clip", name=f"Face {segment.get('face_id', index + 1)}", ref=rid, offset=_frame_time(parent_offset + max(0, start - parent_start), fps), start="0s", duration=_frame_time(max(1, end - start), fps), lane=str(int(segment.get("track", index + 1) or index + 1)), enabled="1")
        ET.SubElement(clip, "adjust-transform", position="0 0")
        ET.SubElement(clip, "note", value="alpha=straight; codec=ProRes 4444 or PNG sequence; Roop face segment")

    xml_bytes = ET.tostring(fcpxml, encoding="utf-8")
    pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="UTF-8")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(pretty)
    return output_path


def export_resolve_edl(document: Mapping[str, Any], project_path: str, output_path: str,
                       *, scene_cuts: Iterable[int] | None = None) -> list[str]:
    """Write a Resolve-readable CMX3600 master EDL and one EDL per face lane."""
    fps = _fps(document)
    target = _target_path(document, project_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    segments = _segments(document, scene_cuts)
    lines = [f"TITLE: {document.get('name') or 'ROOP PROJECT'}", "FCM: NON-DROP FRAME"]
    for index, (start, end) in enumerate(segments, 1):
        lines.append(f"{index:03d}  AX       V     C        {_timecode(start, fps)} {_timecode(end, fps)} {_timecode(start, fps)} {_timecode(end, fps)}")
        lines.append(f"* FROM CLIP NAME: {os.path.basename(target)}")
        lines.append(f"* SCENE CUT: {_timecode(start, fps)}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    generated = [str(output)]
    for track, segment in enumerate(_face_segments(document), 1):
        path = _face_asset(segment, document, project_path)
        if not path:
            continue
        face_output = output.with_name(f"{output.stem}.face{track}{output.suffix}")
        start = int(segment.get("in", 0) or 0)
        end = int(segment.get("out", start + 1) or start + 1)
        face_lines = [f"TITLE: ROOP FACE TRACK {track}", "FCM: NON-DROP FRAME", f"001  AX       V     C        {_timecode(start, fps)} {_timecode(end, fps)} {_timecode(start, fps)} {_timecode(end, fps)}", f"* FROM CLIP NAME: {os.path.basename(path)}", "* ALPHA: STRAIGHT", "* CODEC: PRORES 4444 OR PNG IMAGE SEQUENCE"]
        face_output.write_text("\n".join(face_lines) + "\n", encoding="utf-8")
        generated.append(str(face_output))
    return generated




def frames_to_timecode(frames: int, fps: float = 30.0, drop_frame: bool = False) -> str:
    """Frame-accurate SMPTE timecode string for given frame index and framerate.
    Supports NDF (23.976, 24, 25, 29.97, 30, 59.94, 60) and DF (29.97, 59.94)."""
    frames = max(0, int(frames))
    if drop_frame and abs(fps - 29.97) < 0.1:
        # SMPTE 12M 29.97 Drop-Frame (drops 2 frames per minute except 10th minute)
        D = frames // 17982
        M = frames % 17982
        f_adj = frames + 18 * D + 2 * max(0, (M - 2) // 1798) if M >= 2 else frames + 18 * D
        ff = f_adj % 30
        ss = (f_adj // 30) % 60
        mm = (f_adj // 1800) % 60
        hh = f_adj // 108000
        return f"{hh:02d}:{mm:02d}:{ss:02d};{ff:02d}"
    elif drop_frame and abs(fps - 59.94) < 0.1:
        # SMPTE 12M 59.94 Drop-Frame (drops 4 frames per minute except 10th minute)
        D = frames // 35964
        M = frames % 35964
        f_adj = frames + 36 * D + 4 * max(0, (M - 4) // 3596) if M >= 4 else frames + 36 * D
        ff = f_adj % 60
        ss = (f_adj // 60) % 60
        mm = (f_adj // 3600) % 60
        hh = f_adj // 216000
        return f"{hh:02d}:{mm:02d}:{ss:02d};{ff:02d}"
    nom = int(round(fps)) if fps > 0 else 30
    nom = max(1, nom)
    ff = frames % nom
    ss = (frames // nom) % 60
    mm = (frames // (nom * 60)) % 60
    hh = frames // (nom * 3600)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def timecode_to_frames(timecode_str: str, fps: float = 30.0, drop_frame: bool = False) -> int:
    """Convert SMPTE timecode string back into an exact frame index."""
    cleaned = str(timecode_str).strip()
    is_df = drop_frame or (";" in cleaned)
    parts = cleaned.replace(";", ":").split(":")
    if len(parts) != 4:
        raise ValueError(f"Invalid timecode format: {timecode_str}")
    hh, mm, ss, ff = (int(p) for p in parts)
    if is_df and abs(fps - 29.97) < 0.1:
        total_minutes = hh * 60 + mm
        nominal = (hh * 3600 + mm * 60 + ss) * 30 + ff
        drop = 2 * (total_minutes - total_minutes // 10)
        return max(0, nominal - drop)
    elif is_df and abs(fps - 59.94) < 0.1:
        total_minutes = hh * 60 + mm
        nominal = (hh * 3600 + mm * 60 + ss) * 60 + ff
        drop = 4 * (total_minutes - total_minutes // 10)
        return max(0, nominal - drop)
    nom = int(round(fps)) if fps > 0 else 30
    nom = max(1, nom)
    return max(0, ((hh * 3600 + mm * 60 + ss) * nom) + ff)


def catalog_active_face_timecodes(
    video_path: str,
    *,
    fps: float | None = None,
    face_ranges: Iterable[Sequence[int]] | None = None,
    scene_cuts: Iterable[int] | None = None,
    speaker_cluster_id: str = "person_1",
    drop_frame: bool = False,
    min_run_length: int = 1,
) -> list[dict[str, Any]]:
    """Catalog active face timecode ranges [(tc_in, tc_out, speaker_cluster_id)] across a video.
    Leverages pre-detector face-free frame bypass and scene-cut detection."""
    from roop.timeline_tracks import merge_ranges
    rate = float(fps or 30.0)
    if rate <= 0:
        rate = 30.0

    cuts = set(int(c) for c in (scene_cuts or []))
    raw_ranges = list(face_ranges) if face_ranges is not None else [(0, 100)]

    split_ranges: list[tuple[int, int]] = []
    for start, end in merge_ranges(raw_ranges):
        curr_start = start
        for c in sorted(cuts):
            if curr_start < c < end:
                split_ranges.append((curr_start, c))
                curr_start = c
        if curr_start < end:
            split_ranges.append((curr_start, end))

    catalog = []
    for s, e in split_ranges:
        if (e - s) < min_run_length:
            continue
        catalog.append({
            "frame_in": s,
            "frame_out": e,
            "duration_frames": e - s,
            "tc_in": frames_to_timecode(s, rate, drop_frame),
            "tc_out": frames_to_timecode(e, rate, drop_frame),
            "speaker_cluster_id": speaker_cluster_id,
        })
    return catalog


def export_cmx3600_edl(
    document: Mapping[str, Any],
    output_path: str | None = None,
    *,
    fps: float | None = None,
    drop_frame: bool = False,
) -> str:
    """Generate a standard CMX3600 EDL where untouched source reel sits on Track V1
    and face-swapped segments sit on Track V2."""
    timeline = document.get("timeline") or {}
    rate = float(fps or timeline.get("fps") or document.get("fps") or 30.0)
    title = str(document.get("name") or "ROOP_ULTIMATE_TIMELINE")
    fcm = "DROP FRAME" if drop_frame else "NON-DROP FRAME"

    source_path = str(
        document.get("source_media")
        or (document.get("media", {}).get("target", {}).get("path")
            if isinstance(document.get("media", {}).get("target"), dict)
            else document.get("target_media") or "source_reel.mp4")
    )
    source_name = os.path.basename(source_path) or "SOURCE_REEL"
    swap_name = str(document.get("swap_media_name") or "face_swapped_v2.mov")

    total_frames = int(timeline.get("frame_end") or document.get("total_frames") or 1800)
    start_frame = int(timeline.get("frame_start") or 0)

    lines = [
        f"TITLE: {title}",
        f"FCM: {fcm}",
        "",
        f"001  AX       V     C        {frames_to_timecode(start_frame, rate, drop_frame)} {frames_to_timecode(total_frames, rate, drop_frame)} {frames_to_timecode(start_frame, rate, drop_frame)} {frames_to_timecode(total_frames, rate, drop_frame)}",
        f"* FROM CLIP NAME: {source_name}",
        "* TRACK: V1",
        "* COMMENT: Untouched source reel",
        "",
    ]

    segments = list(document.get("segments") or timeline.get("face_segments") or [])
    event_idx = 2
    for seg in segments:
        s_in = int(seg.get("in", seg.get("frame_in", 0)))
        s_out = int(seg.get("out", seg.get("frame_out", s_in + 1)))
        speaker = str(seg.get("speaker_cluster_id", seg.get("face_id", "person_1")))
        clip_name = str(seg.get("clip_name", swap_name))

        tc_in = frames_to_timecode(s_in, rate, drop_frame)
        tc_out = frames_to_timecode(s_out, rate, drop_frame)

        lines.append(f"{event_idx:03d}  SWAP     V     C        {tc_in} {tc_out} {tc_in} {tc_out}")
        lines.append(f"* FROM CLIP NAME: {clip_name}")
        lines.append("* TRACK: V2")
        lines.append(f"* SPEAKER: {speaker}")
        lines.append("")
        event_idx += 1

    edl_text = "\n".join(lines).strip() + "\n"
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(edl_text, encoding="utf-8")
    return edl_text


def parse_cmx3600_edl(edl_text: str, fps: float = 30.0, drop_frame: bool = False) -> list[dict[str, Any]]:
    """Zero-dependency parser for CMX3600 EDL cut lists."""
    events: list[dict[str, Any]] = []
    current_event: dict[str, Any] | None = None
    lines = edl_text.strip().splitlines()

    for line in lines:
        if line.startswith("FCM:"):
            if "DROP" in line and "NON-DROP" not in line:
                drop_frame = True
            break

    for line in lines:
        line_str = line.strip()
        if not line_str or line_str.startswith("TITLE:") or line_str.startswith("FCM:"):
            continue

        if line_str.startswith("*"):
            if current_event is not None:
                if line_str.startswith("* FROM CLIP NAME:"):
                    current_event["clip_name"] = line_str.replace("* FROM CLIP NAME:", "").strip()
                elif line_str.startswith("* TRACK:"):
                    current_event["track"] = line_str.replace("* TRACK:", "").strip()
                elif line_str.startswith("* SPEAKER:"):
                    current_event["speaker_cluster_id"] = line_str.replace("* SPEAKER:", "").strip()
            continue

        parts = line_str.split()
        if len(parts) >= 8 and parts[0].isdigit():
            try:
                ev_num = int(parts[0])
                reel = parts[1]
                track = parts[2]
                trans = parts[3]
                src_in_tc = parts[4]
                src_out_tc = parts[5]
                rec_in_tc = parts[6]
                rec_out_tc = parts[7]

                f_in = timecode_to_frames(rec_in_tc, fps, drop_frame)
                f_out = timecode_to_frames(rec_out_tc, fps, drop_frame)

                current_event = {
                    "event": ev_num,
                    "reel": reel,
                    "track": track,
                    "transition": trans,
                    "src_in": src_in_tc,
                    "src_out": src_out_tc,
                    "rec_in": rec_in_tc,
                    "rec_out": rec_out_tc,
                    "frame_in": f_in,
                    "frame_out": f_out,
                    "duration_frames": max(1, f_out - f_in),
                    "clip_name": reel,
                    "speaker_cluster_id": "person_1",
                }
                events.append(current_event)
            except Exception:
                continue

    return events


def export_otio_json(
    document: Mapping[str, Any],
    output_path: str | None = None,
    *,
    fps: float | None = None,
) -> str:
    """Generate compliant OpenTimelineIO schema JSON (Timeline.1) with Track V1 (source)
    and Track V2 (face-swapped segments and gaps)."""
    import json
    timeline = document.get("timeline") or {}
    rate = float(fps or timeline.get("fps") or document.get("fps") or 30.0)
    title = str(document.get("name") or "Roop Ultimate Timeline")

    total_frames = int(timeline.get("frame_end") or document.get("total_frames") or 1800)
    start_frame = int(timeline.get("frame_start") or 0)
    duration_frames = max(1, total_frames - start_frame)

    source_path = str(
        document.get("source_media")
        or (document.get("media", {}).get("target", {}).get("path")
            if isinstance(document.get("media", {}).get("target"), dict)
            else document.get("target_media") or "source_reel.mp4")
    )
    source_name = os.path.basename(source_path) or "Source Reel"
    swap_name = str(document.get("swap_media_name") or "face_swapped_v2.mov")

    v1_track = {
        "OTIO_SCHEMA": "Track.1",
        "metadata": {},
        "name": "V1_Source",
        "kind": "Video",
        "children": [
            {
                "OTIO_SCHEMA": "Clip.1",
                "metadata": {},
                "name": source_name,
                "source_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": {"OTIO_SCHEMA": "RationalTime.1", "value": start_frame, "rate": rate},
                    "duration": {"OTIO_SCHEMA": "RationalTime.1", "value": duration_frames, "rate": rate},
                },
                "media_reference": {
                    "OTIO_SCHEMA": "ExternalReference.1",
                    "target_url": Path(source_path).as_uri() if os.path.isabs(source_path) else source_path,
                },
            }
        ],
    }

    segments = list(document.get("segments") or timeline.get("face_segments") or [])
    segments.sort(key=lambda s: int(s.get("in", s.get("frame_in", 0))))

    v2_children = []
    cursor = start_frame
    for seg in segments:
        s_in = int(seg.get("in", seg.get("frame_in", 0)))
        s_out = int(seg.get("out", seg.get("frame_out", s_in + 1)))
        speaker = str(seg.get("speaker_cluster_id", seg.get("face_id", "person_1")))

        if s_in > cursor:
            gap_dur = s_in - cursor
            v2_children.append({
                "OTIO_SCHEMA": "Gap.1",
                "source_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": {"OTIO_SCHEMA": "RationalTime.1", "value": cursor, "rate": rate},
                    "duration": {"OTIO_SCHEMA": "RationalTime.1", "value": gap_dur, "rate": rate},
                },
            })

        clip_dur = max(1, s_out - s_in)
        v2_children.append({
            "OTIO_SCHEMA": "Clip.1",
            "name": f"FaceSwap_{speaker}_{s_in}_{s_out}",
            "source_range": {
                "OTIO_SCHEMA": "TimeRange.1",
                "start_time": {"OTIO_SCHEMA": "RationalTime.1", "value": s_in, "rate": rate},
                "duration": {"OTIO_SCHEMA": "RationalTime.1", "value": clip_dur, "rate": rate},
            },
            "metadata": {
                "roop": {
                    "track": "V2",
                    "speaker_cluster_id": speaker,
                    "face_id": seg.get("face_id", speaker),
                }
            },
            "media_reference": {
                "OTIO_SCHEMA": "ExternalReference.1",
                "target_url": swap_name,
            },
        })
        cursor = s_out

    if cursor < total_frames:
        gap_dur = total_frames - cursor
        v2_children.append({
            "OTIO_SCHEMA": "Gap.1",
            "source_range": {
                "OTIO_SCHEMA": "TimeRange.1",
                "start_time": {"OTIO_SCHEMA": "RationalTime.1", "value": cursor, "rate": rate},
                "duration": {"OTIO_SCHEMA": "RationalTime.1", "value": gap_dur, "rate": rate},
            },
        })

    v2_track = {
        "OTIO_SCHEMA": "Track.1",
        "metadata": {},
        "name": "V2_FaceSwap",
        "kind": "Video",
        "children": v2_children,
    }

    otio_doc = {
        "OTIO_SCHEMA": "Timeline.1",
        "metadata": {"roop_version": "2.1.0"},
        "name": title,
        "global_start_time": {
            "OTIO_SCHEMA": "RationalTime.1",
            "value": start_frame,
            "rate": rate,
        },
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "metadata": {},
            "name": "tracks",
            "children": [v1_track, v2_track],
        },
    }

    json_str = json.dumps(otio_doc, indent=2)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json_str, encoding="utf-8")
    return json_str


def parse_otio_json(otio_data: str | Mapping[str, Any], fps: float = 30.0) -> list[dict[str, Any]]:
    """Zero-dependency parser for OpenTimelineIO (OTIO) cut lists."""
    import json
    if isinstance(otio_data, str):
        try:
            doc = json.loads(otio_data)
        except Exception as e:
            raise ValueError(f"Invalid OTIO JSON: {e}")
    else:
        doc = dict(otio_data)

    rate = fps
    tracks = (doc.get("tracks") or {}).get("children") or []
    cuts = []

    for track in tracks:
        track_name = track.get("name", "")
        children = track.get("children") or []
        for child in children:
            schema = child.get("OTIO_SCHEMA", "")
            if "Clip" in schema:
                s_range = child.get("source_range") or {}
                start_obj = s_range.get("start_time") or {}
                dur_obj = s_range.get("duration") or {}

                c_rate = float(start_obj.get("rate") or dur_obj.get("rate") or rate)
                f_in = int(start_obj.get("value", 0))
                dur = int(dur_obj.get("value", 1))
                f_out = f_in + dur

                meta = (child.get("metadata") or {}).get("roop") or {}
                speaker = meta.get("speaker_cluster_id", child.get("name", "clip"))

                cuts.append({
                    "name": child.get("name", "clip"),
                    "track": track_name,
                    "frame_in": f_in,
                    "frame_out": f_out,
                    "duration_frames": dur,
                    "tc_in": frames_to_timecode(f_in, c_rate),
                    "tc_out": frames_to_timecode(f_out, c_rate),
                    "speaker_cluster_id": speaker,
                })

    return cuts


__all__ = ["detect_scene_cuts", "export_fcpxml", "export_resolve_edl", "frames_to_timecode", "timecode_to_frames", "catalog_active_face_timecodes", "export_cmx3600_edl", "parse_cmx3600_edl", "export_otio_json", "parse_otio_json"]
