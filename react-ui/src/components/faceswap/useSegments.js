import { useCallback, useEffect, useState } from 'react';

// Named frame ranges on the current clip.
//
// A run has exactly ONE trim range, because the range lives on the target entry
// server-side (/api/target/set_frame), not in the swap payload. So "swap only
// the three shots this person is in" cannot be one run — it is one run per shot.
// This holds the ranges; the queue turns each into a job that sets the trim
// immediately before it dispatches (see _apply_segment in routes_queue.py), and
// /api/queue/join puts the finished pieces back together.
//
// Persisted per clip in localStorage, exactly like the chapter markers beside
// them: they are a convenience layered on the clip, not state the backend owns,
// and they should still be there tomorrow.

const key = (k) => `roop_segments_${k || 'default'}`;
const newId = () => Math.random().toString(36).slice(2, 10);

const norm = (list, maxFrames) => {
  const clean = (list || [])
    .filter((s) => s && Number.isFinite(s.start) && Number.isFinite(s.end))
    .map((s) => ({
      id: s.id || newId(),
      start: Math.max(1, Math.min(Math.round(s.start), maxFrames)),
      end: Math.max(1, Math.min(Math.round(s.end), maxFrames)),
    }))
    .map((s) => (s.start > s.end ? { ...s, start: s.end, end: s.start } : s));
  return clean.sort((a, b) => a.start - b.start || a.end - b.end);
};

export default function useSegments(targetKey, maxFrames = 1) {
  const [segments, setSegments] = useState([]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(key(targetKey));
      setSegments(norm(raw ? JSON.parse(raw) : [], maxFrames));
    } catch { setSegments([]); }
    // maxFrames is deliberately not a dependency: it arrives a tick after the
    // target changes, and re-reading on it would clamp the stored ranges against
    // the PREVIOUS clip's length for that tick and write the damage back.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetKey]);

  const persist = useCallback((list) => {
    const next = norm(list, maxFrames);
    setSegments(next);
    try {
      if (next.length) localStorage.setItem(key(targetKey), JSON.stringify(next));
      else localStorage.removeItem(key(targetKey));
    } catch { /* private mode / quota — segments are a convenience */ }
    return next;
  }, [targetKey, maxFrames]);

  return {
    segments,
    // Refuses a duplicate of a range that is already listed — queueing the same
    // shot twice renders it twice and joins it in twice.
    add: (start, end) => {
      const s = Math.min(start, end);
      const e = Math.max(start, end);
      if (segments.some((x) => x.start === s && x.end === e)) return null;
      return persist([...segments, { id: newId(), start: s, end: e }]);
    },
    remove: (id) => persist(segments.filter((s) => s.id !== id)),
    clear: () => persist([]),
    // Total frames the segments cover, for the estimate line. Overlaps are
    // counted once — they render once each but the user asked for the union.
    coveredFrames: () => {
      let total = 0;
      let cursor = 0;
      for (const s of segments) {
        const from = Math.max(s.start, cursor + 1);
        if (s.end >= from) {
          total += s.end - from + 1;
          cursor = s.end;
        }
      }
      return total;
    },
  };
}
