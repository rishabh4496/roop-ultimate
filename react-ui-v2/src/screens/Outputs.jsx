import React, { useEffect, useRef, useState } from 'react';
import { Trash2, ExternalLink, Film, Image as ImageIcon, RefreshCw, FolderOpen } from 'lucide-react';
import { getJSON, postJSON } from '@engine/api';
import { Panel, Marquee, SplitText, Meteors } from '../fx/effects';
import { notify } from '../ui/controls';
import QualityReport from '../ui/QualityReport';

const isVideo = (f) => f.kind === 'video' || /\.(mp4|mkv|webm|avi|mov)$/i.test(f.name || '');
const mb = (n) => `${((n || 0) / 1048576).toFixed(1)} MB`;
const when = (t) => (t ? new Date(t * 1000).toLocaleString() : '');

function Card({ f, onDelete }) {
  const vid = useRef(null);
  const video = isVideo(f);
  return (
    <article className="tile group flex flex-col" onPointerEnter={() => vid.current?.play().catch(() => {})}
      onPointerLeave={() => { if (vid.current) { vid.current.pause(); vid.current.currentTime = 0; } }}>
      <div className="relative aspect-video overflow-hidden bg-black">
        {video
          ? <video ref={vid} src={f.url} muted loop playsInline preload="metadata" className="h-full w-full object-cover" />
          : <img src={f.url} alt="" loading="lazy" className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105" />}
        <span className="absolute left-2 top-2 badge badge-dim bg-black/60">{video ? <Film size={11} /> : <ImageIcon size={11} />} {video ? 'video' : 'image'}</span>
        <div className="absolute right-2 top-2 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
          <a className="btn btn-sm btn-icon bg-black/70" href={f.url} target="_blank" rel="noreferrer" aria-label={`Open ${f.name}`}><ExternalLink size={13} /></a>
          <button type="button" className="btn btn-danger btn-sm btn-icon" aria-label={`Delete ${f.name}`} onClick={() => onDelete(f)}><Trash2 size={13} /></button>
        </div>
      </div>
      <div className="p-2.5">
        <div className="truncate text-[12.5px] font-semibold text-white" title={f.name}>{f.name}</div>
        <div className="font-mono text-[10.5px] text-[var(--fg-faint)]">{mb(f.size)} · {when(f.mtime)}</div>
        {f.absolute_path && <div className="mt-1.5"><QualityReport path={f.absolute_path} /></div>}
      </div>
    </article>
  );
}

export default function Outputs() {
  const [data, setData] = useState(null);
  const load = async () => {
    try { setData(await getJSON('/api/output')); } catch (e) { setData({ files: [], error: e.message }); }
  };
  useEffect(() => { load(); }, []);
  const files = (data?.files || []).slice().sort((a, b) => (b.mtime || 0) - (a.mtime || 0));
  const remove = async (f) => {
    if (!window.confirm(`Delete “${f.name}”? This removes the file on disk.`)) return;
    try { await postJSON('/api/output/delete', { name: f.name }); notify(`Deleted “${f.name}”`); load(); } catch (e) { notify(e.message, 'error'); }
  };
  return (
    <div className="mx-auto max-w-[1600px] space-y-3" data-testid="outputs">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight"><SplitText text="Outputs" className="gradient-text" /></h1>
          <p className="text-[13px] text-[var(--fg-dim)]">{files.length} file{files.length === 1 ? '' : 's'}{data?.output_path ? <> in <span className="font-mono">{data.output_path}</span></> : ''}. Hover a video to play it.</p>
        </div>
        <button type="button" className="btn ml-auto" onClick={load}><RefreshCw size={14} /> Refresh</button>
      </div>
      {files.length > 3 && (
        <Marquee seconds={Math.max(25, files.length * 4)}>
          {files.slice(0, 12).map((f) => (
            <a key={`m-${f.name}`} href={f.url} target="_blank" rel="noreferrer" className="tile block h-20 w-36 shrink-0 overflow-hidden">
              {isVideo(f) ? <video src={`${f.url}#t=0.5`} muted preload="metadata" className="h-full w-full object-cover" /> : <img src={f.url} alt="" className="h-full w-full object-cover" />}
            </a>
          ))}
        </Marquee>
      )}
      {data === null ? (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-4">{Array.from({ length: 8 }, (_, i) => <div key={i} className="tile aspect-video animate-pulse" />)}</div>
      ) : files.length === 0 ? (
        <Panel bodyClass="relative flex min-h-[280px] flex-col items-center justify-center gap-2 overflow-hidden text-center">
          <Meteors count={12} />
          <FolderOpen size={30} strokeWidth={1.5} className="relative text-[var(--accent-2)]" />
          <div className="relative text-[15px] font-semibold">No outputs yet</div>
          <div className="relative text-[12.5px] text-[var(--fg-dim)]">{data.error || 'Finished renders land here.'}</div>
        </Panel>
      ) : (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-4 3xl:grid-cols-5" data-testid="outputs-grid">
          {files.map((f) => <Card key={f.name} f={f} onDelete={remove} />)}
        </div>
      )}
    </div>
  );
}
