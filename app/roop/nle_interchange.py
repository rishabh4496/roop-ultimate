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


__all__ = ["detect_scene_cuts", "export_fcpxml", "export_resolve_edl"]
