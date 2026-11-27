import React, { useState } from 'react';
import { API, fileUrl, postFiles, postJSON } from '../../api';
import { Button, Section, Select, Slider } from '../ui';
import { Icon } from '../../icons';

// ── Video to faceset (.fsz) ─────────────────────────────────────────────────
//
// Pull a faceset out of a video without leaving the Editor: load the clip,
// scrub to a frame where the person is well lit and facing the camera, cut the
// faces from it, repeat for a few frames, then build the `.fsz`.
//
// It drives the SAME backend workspace the Face Manager tab does
// (/api/facemgr/add, /cut, /build): one faceset being assembled, two ways in. So
// the faces cut here are the faces the Face Manager shows, where per-face
// quality scores and pruning live — the link at the bottom goes there rather than
// duplicating that grid.
const DETECTORS = [
  { value: 'scrfd', label: 'SCRFD (default)' },
  { value: 'retinaface_r50', label: 'RetinaFace R50' },
  { value: 'yoloface', label: 'YOLOFace' },
  { value: 'yunet', label: 'YuNet' },
];

export default function VideoFramesetExtractor({ notify, setTab }) {
  const [video, setVideo] = useState(null);       // { name, frames }
  const [frame, setFrame] = useState(1);
  const [detector, setDetector] = useState('scrfd');
  const [count, setCount] = useState(0);
  const [cuts, setCuts] = useState(0);
  const [built, setBuilt] = useState(null);
  const [busy, setBusy] = useState('');

  const run = async (what, fn) => {
    setBusy(what);
    try { await fn(); } catch (e) { notify(e.message || `${what} failed`, 'error'); } finally { setBusy(''); }
  };

  const load = (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    run('load', async () => {
      const res = await postFiles('/api/facemgr/add', [file], { detector, restore: 'false' }, { timeout: 0 });
      if (!res.video) throw new Error('That file has no video track to take frames from. Use the Face Manager tab for still images.');
      setVideo({ name: file.name, frames: Math.max(1, res.frames || 1) });
      setFrame(1);
      setCount((res.faces || []).length);
      setCuts(0);
      setBuilt(null);
    });
  };

  const cut = () => run('cut', async () => {
    const res = await postJSON('/api/facemgr/cut', { frame, detector, restore: false }, { timeout: 0 });
    setCount((res.faces || []).length);
    setCuts((n) => n + 1);
    setBuilt(null);
    notify(`Faces cut from frame ${frame}`);
  });

  const build = () => run('build', async () => {
    const res = await postJSON('/api/facemgr/build', {});
    setBuilt(res);
    notify('Faceset file created');
  });

  return (
    <div data-testid="video-faceset">
    <Section title="Video to faceset (.fsz)" icon={Icon.faces}>
      <label className="block cursor-pointer focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-[var(--accent)]">
        <div className="rounded-lg border-2 border-dashed border-white/15 px-4 py-4 text-center text-sm text-white/60 hover:border-[var(--accent)]/50">
          {busy === 'load' ? 'Reading the video…' : video ? `${video.name} · ${video.frames.toLocaleString()} frames (click to replace)` : 'Pick a video'}
        </div>
        <input type="file" accept="video/*" onChange={load} className="sr-only" />
      </label>

      <Select label="Face detector" value={detector} onChange={setDetector} options={DETECTORS} />

      {video && (
        <div className="space-y-2">
          <div className="overflow-hidden rounded-lg border border-white/10 bg-black/40">
            <img src={`${API}/api/facemgr/frame?frame=${frame}&t=${frame}`} alt={`Frame ${frame} of ${video.name}`} className="w-full" />
          </div>
          <Slider label="Frame" info={`${frame} / ${video.frames}`} min={1} max={video.frames} step={1} value={frame} defaultValue={1} onChange={setFrame} />
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" variant="secondary" onClick={cut} disabled={!!busy}>{busy === 'cut' ? 'Cutting…' : 'Cut faces from this frame'}</Button>
            <Button size="sm" variant="primary" onClick={build} disabled={!!busy || count === 0}>{busy === 'build' ? 'Building…' : 'Build .fsz'}</Button>
            <span className="text-mini text-muted">{count} face{count === 1 ? '' : 's'} so far{cuts ? `, ${cuts} cut${cuts === 1 ? '' : 's'}` : ''}</span>
          </div>
          {built?.path && (
            <a href={fileUrl(built.path)} download={built.name} className="inline-block text-sm text-accent underline">⬇ Download {built.name || 'faceset.fsz'}</a>
          )}
          {count > 0 && (
            <button type="button" onClick={() => setTab('facemgr')} className="block text-mini font-semibold text-muted underline hover:text-white">
              Review quality scores and prune in the Face Manager
            </button>
          )}
        </div>
      )}
    </Section>
    </div>
  );
}
