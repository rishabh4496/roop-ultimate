import { useEffect, useMemo, useState, useCallback } from 'react';
import { useHistoryChannel } from '../../history/historyChannels';
import { restoreAutomationProjects } from './studioProjectModel';

export const STUDIO_PROJECT_KEY = 'roop_studio_projects_v1';
const CHANNEL = { scope: 'faceswap', coalesce: false, describe: () => 'Edited Studio project', equals: (a, b) => JSON.stringify(a) === JSON.stringify(b) };
const EMPTY = Object.freeze({ keyframes: [], bindings: [], intervals: [], strokes: [], mask: null, roto: [], pins: [], trim: null });

export function readStudioCheckpoint(storage) {
  for (const key of [STUDIO_PROJECT_KEY, `${STUDIO_PROJECT_KEY}_backup`]) {
    try {
      const record = JSON.parse(storage.getItem(key) || 'null');
      if (record?.version === 1 && record.projects && typeof record.projects === 'object' && !Array.isArray(record.projects)) return record.projects;
    } catch { /* try the last complete checkpoint */ }
  }
  return {};
}

export function writeStudioCheckpoint(storage, projects) {
  const next = JSON.stringify({ version: 1, projects });
  const prior = storage.getItem(STUDIO_PROJECT_KEY);
  // localStorage replaces one value atomically. A quota failure leaves the
  // last successful checkpoint intact and is surfaced by the hook.
  storage.setItem(STUDIO_PROJECT_KEY, next);
  if (prior) {
    try { storage.setItem(`${STUDIO_PROJECT_KEY}_backup`, prior); } catch { /* primary is durable */ }
  }
}

export default function useStudioProject(mediaKey, notify) {
  const [projects, setProjects] = useState(() => {
    try { return readStudioCheckpoint(window.localStorage); } catch { return {}; }
  });
  const project = useMemo(() => ({ ...EMPTY, ...(projects[mediaKey] || {}) }), [projects, mediaKey]);
  const update = useCallback((patch) => setProjects((all) => {
    const before = all[mediaKey] || EMPTY;
    const after = { ...before, ...patch };
    if (JSON.stringify(before) === JSON.stringify(after)) return all;
    return { ...all, [mediaKey]: after };
  }), [mediaKey]);
  // Brush/roto/pin owners already register their own history channels. Keep
  // their checkpoint copies out of this channel to avoid double undo steps.
  const automation = useMemo(() => Object.fromEntries(Object.entries(projects).filter(([, p]) =>
    p.keyframes?.length || p.bindings?.length || p.intervals?.length || p.strokes?.length || p.trim
  ).map(([key, p]) => [key, {
    keyframes: p.keyframes || [], bindings: p.bindings || [], intervals: p.intervals || [], strokes: p.strokes || [], trim: p.trim || null,
  }])), [projects]);
  useHistoryChannel('studio-projects', automation, (restored) => setProjects((all) => {
    return restoreAutomationProjects(all, restored);
  }), CHANNEL);
  useEffect(() => {
    const save = () => {
      try { writeStudioCheckpoint(window.localStorage, projects); }
      catch { notify?.('Studio checkpoint could not be saved: local storage is full or unavailable.', 'error'); }
    };
    const timer = setTimeout(save, 300);
    window.addEventListener('pagehide', save);
    return () => { clearTimeout(timer); window.removeEventListener('pagehide', save); };
  }, [projects, notify]);
  return useMemo(() => ({ project, update }), [project, update]);
}
