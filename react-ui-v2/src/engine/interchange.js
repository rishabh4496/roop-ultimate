// Cut-list export for an NLE (POST /api/timeline/export_edl | export_otio -> roop/nle_interchange.py).
//
// The document's face_segments are built from the presence lanes the engine already computed
// (/api/target/timeline_tracks: [{person_id, name, intervals: [[a, b], ...]}], 1-based inclusive
// frames), clipped to the target's In..Out. The exporter reads `in`/`out` as 0-based, `out`
// exclusive, so [a, b] becomes in = a - 1, out = b.
//
// 1.0 sent an empty face_segments list (a V1-only EDL) and read `edl_text` / `otio_json`, keys the
// engine never returns (it sends `edl` / `otio`), so its buttons never produced a file.

/** The export document for `target` (in/out 1-based inclusive) and the presence lanes. Pure. */
export function cutListDocument(target, presence, inOut) {
  const fps = Number(target?.fps) > 0 ? Number(target.fps) : 30;
  const lo = Math.max(1, inOut?.a || 1);
  const hi = Math.max(lo, inOut?.b || target?.frames || lo);
  const segments = [];
  (presence || []).forEach((p, rank) => {
    (p.intervals || []).forEach(([a, b]) => {
      const s = Math.max(lo, Number(a));
      const e = Math.min(hi, Number(b));
      if (!(e >= s)) return;
      segments.push({ in: s - 1, out: e, face_id: String(p.name || `Person ${rank + 1}`), speaker_cluster_id: String(p.person_id ?? rank), track: rank + 1 });
    });
  });
  segments.sort((x, y) => x.in - y.in || x.track - y.track);
  return {
    fps,
    document: {
      name: String(target?.name || 'roop_timeline').replace(/\.[^/.]+$/, ''),
      source_media: target?.filename || target?.path || target?.name || '',
      total_frames: Number(target?.frames) || hi,
      timeline: { fps, frame_start: lo - 1, frame_end: hi, face_segments: segments },
    },
  };
}

/** Save `text` as a download named `name`. */
export function downloadText(text, name, type = 'text/plain') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
