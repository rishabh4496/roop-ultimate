import { useCallback, useEffect, useMemo, useState } from 'react';

// The "Reference Only" marker on a person, for ONE target clip: { <target person id>: true }.
//
// It carries NO render behaviour: the person is excluded through the same face mapping as "Exclude (Never Swap)" (that is what
// makes the render skip them), and this only remembers that the exclusion was meant as "tracked, deliberately left alone"
// rather than "protect this face", so the canvas can draw the two differently. Kept in this browser per clip (localStorage can
// throw - private window, blocked site data - so every read and write is wrapped and the hook works in memory without it).
// A marker on a person who is not excluded means nothing (faceRoleChoices.choiceOf ignores it).

const KEY = (mediaId) => `roop_face_reference_tags:${mediaId ?? 'none'}`;

function load(mediaId) {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(KEY(mediaId)) || 'null');
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

export default function useReferenceTags(mediaId) {
  const [state, setState] = useState(() => ({ mediaId, tags: load(mediaId) }));
  useEffect(() => { setState((s) => (s.mediaId === mediaId ? s : { mediaId, tags: load(mediaId) })); }, [mediaId]);

  const setTag = useCallback((personId, on) => {
    if (personId === null || personId === undefined || personId === '') return;
    setState((s) => {
      const had = !!s.tags[personId];
      if (had === !!on) return s;
      const tags = { ...s.tags };
      if (on) tags[personId] = true; else delete tags[personId];
      try {
        if (Object.keys(tags).length) window.localStorage.setItem(KEY(s.mediaId), JSON.stringify(tags));
        else window.localStorage.removeItem(KEY(s.mediaId));
      } catch { /* storage unavailable: the marker lasts for this session */ }
      return { ...s, tags };
    });
  }, []);

  return useMemo(() => ({ tags: state.mediaId === mediaId ? state.tags : {}, setTag }), [state, mediaId, setTag]);
}
