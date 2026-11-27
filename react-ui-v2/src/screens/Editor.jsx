import React, { useEffect, useRef, useState } from 'react';
import { Upload, Wand2, Crop, RotateCw, Music, Download, Film, Image as ImageIcon, Scissors, PackageCheck, Loader2 } from 'lucide-react';
import { getJSON, postFile, postFiles, postJSON, fileUrl } from '@engine/api';
import { Panel, SplitText } from '../fx/effects';
import { Select, Slider, notify } from '../ui/controls';

// ── Editor: media tools that run no swap ─────────────────────────────────────
//   Transform  - resize / rotate / crop / fps, and replace + slip the audio (/api/extras/apply)
//   Enhance    - the engine's frame post-processors, whatever /api/extras/frame_ops lists
//   Video -> faceset - cut faces from frames of a clip into a .fsz (/api/facemgr/add|cut|build)

const RESOLUTIONS = ['Original', '3840x', '2560x', '1920x', '1280x', '1024x', '640x'];
const ROTATIONS = ['None', '90° Clockwise', '90° Counter-Clockwise', '180°'];
const DEFAULTS = { resolution: 'Original', rotation: 'None', fps: 30, crop_left: 0, crop_right: 0, crop_top: 0, crop_bottom: 0, override_audio_path: '', audio_offset_ms: 0 };
const isVideoFile = (f) => !!f && (f.type?.startsWith('video') || /\.(mp4|mkv|mov|avi|webm|gif)$/i.test(f.name));

function Result({ res }) {
  if (!res?.path) return null;
  const url = fileUrl(res.path);
  return (
    <div className="mt-3 space-y-2" data-testid="editor-result">
      {res.kind === 'video'
        ? <video src={url} controls className="w-full rounded-xl border border-[var(--rule-strong)] bg-black" />
        : <img src={url} alt="Result" className="max-h-[420px] w-full rounded-xl border border-[var(--rule-strong)] bg-black object-contain" />}
      <a className="btn btn-sm" href={url} download><Download size={13} /> Download {res.name || ''}</a>
    </div>
  );
}

function Picker({ file, onFile, accept = 'video/*,image/*' }) {
  const input = useRef(null);
  const [over, setOver] = useState(false);
  const [url, setUrl] = useState('');
  useEffect(() => {
    if (!file) { setUrl(''); return undefined; }
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  return (
    <div>
      <button type="button" className={`dropzone w-full overflow-hidden ${file ? 'p-0' : 'h-44'} ${over ? 'over' : ''}`}
        onClick={() => input.current?.click()} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) onFile(f); }}>
        {file ? (isVideoFile(file)
          ? <video src={url} muted controls className="max-h-[300px] w-full bg-black" onClick={(e) => e.stopPropagation()} />
          : <img src={url} alt="" className="max-h-[300px] w-full bg-black object-contain" />)
          : <><Upload size={22} className="text-[var(--accent-2)]" /><span className="text-[13px] font-semibold">Drop a video or image</span><span className="text-[11.5px] text-[var(--fg-faint)]">or click to choose</span></>}
      </button>
      {file && <div className="mt-1.5 flex items-center gap-2 font-mono text-[11px] text-[var(--fg-dim)]">{isVideoFile(file) ? <Film size={12} /> : <ImageIcon size={12} />} {file.name} · {(file.size / 1048576).toFixed(1)} MB <button type="button" className="ml-auto text-[var(--accent-2)]" onClick={() => input.current?.click()}>change</button></div>}
      <input ref={input} type="file" accept={accept} className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ''; }} />
    </div>
  );
}

function Transform() {
  const [file, setFile] = useState(null);
  const [o, setO] = useState(DEFAULTS);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const [audioBusy, setAudioBusy] = useState(false);
  const audioInput = useRef(null);
  const set = (k, v) => setO((x) => ({ ...x, [k]: v }));
  const apply = async () => {
    if (!file) { notify('Pick a file first', 'warn'); return; }
    setBusy(true); setRes(null);
    try { setRes(await postFile('/api/extras/apply', file, o, { timeout: 0 })); notify('Transform done'); }
    catch (e) { notify(e.message, 'error'); } finally { setBusy(false); }
  };
  return (
    <Panel title="Transform" accent="resize · rotate · crop · audio" data-testid="editor-transform">
      <Picker file={file} onFile={(f) => { setFile(f); setRes(null); }} />
      <div className="mt-3 grid grid-cols-2 gap-2.5">
        <Select label="Resolution" value={o.resolution} options={RESOLUTIONS} onChange={(v) => set('resolution', v)} />
        <Select label="Rotation" value={o.rotation} options={ROTATIONS} onChange={(v) => set('rotation', v)} />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2">
        {[['crop_left', 'Crop left'], ['crop_right', 'Crop right'], ['crop_top', 'Crop top'], ['crop_bottom', 'Crop bottom']].map(([k, l]) => (
          <Slider key={k} label={l} value={o[k]} min={0} max={1000} step={2} format={(v) => `${v}px`} onChange={(v) => set(k, v)} />
        ))}
      </div>
      {isVideoFile(file) && (
        <div className="mt-3 space-y-2.5 rounded-xl border border-[var(--rule)] bg-black/30 p-2.5">
          <Slider label="Output fps" value={o.fps} min={1} max={120} step={1} onChange={(v) => set('fps', v)} />
          <div className="flex flex-wrap items-center gap-2">
            <Music size={14} className="text-[var(--accent-2)]" />
            <span className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-[var(--fg-dim)]">{o.override_audio_path || 'keeps the clip’s own audio'}</span>
            <button type="button" className="btn btn-sm" disabled={audioBusy} onClick={() => audioInput.current?.click()}>{audioBusy ? <Loader2 size={13} className="animate-spin" /> : <Music size={13} />} Replace audio</button>
            {o.override_audio_path && <button type="button" className="btn btn-ghost btn-sm" onClick={() => set('override_audio_path', '')}>Keep original</button>}
          </div>
          <Slider label="Audio slip" hint="+ = heard later" value={o.audio_offset_ms} min={-500} max={500} step={5} format={(v) => `${v > 0 ? '+' : ''}${v} ms`} onChange={(v) => set('audio_offset_ms', v)} />
          <input ref={audioInput} type="file" accept="audio/*,video/*" className="hidden" onChange={async (e) => {
            const f = e.target.files?.[0]; e.target.value = ''; if (!f) return;
            setAudioBusy(true);
            try { const r = await postFile('/api/lipsync/audio/add', f, undefined, { timeout: 0 }); set('override_audio_path', r.path); notify(`Audio track: ${f.name}`); }
            catch (err) { notify(err.message, 'error'); } finally { setAudioBusy(false); }
          }} />
        </div>
      )}
      <div className="mt-3 flex gap-2">
        <button type="button" className="btn btn-primary flex-1" onClick={apply} disabled={busy || !file} data-testid="editor-apply">{busy ? <Loader2 size={15} className="animate-spin" /> : <Crop size={15} />} Apply</button>
        <button type="button" className="btn btn-ghost" onClick={() => setO(DEFAULTS)}><RotateCw size={14} /> Reset</button>
      </div>
      <Result res={res} />
    </Panel>
  );
}

function Enhance() {
  const [ops, setOps] = useState(null);
  const [file, setFile] = useState(null);
  const [op, setOp] = useState('upscale');
  const [sub, setSub] = useState('');
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  useEffect(() => { getJSON('/api/extras/frame_ops').then(setOps).catch((e) => notify(e.message, 'error')); }, []);
  useEffect(() => { const list = ops?.[op] || []; if (!list.includes(sub)) setSub(list[0] || ''); }, [op, ops]); // eslint-disable-line react-hooks/exhaustive-deps
  const run = async () => {
    if (!file) { notify('Pick a file first', 'warn'); return; }
    setBusy(true); setRes(null);
    try { setRes(await postFile('/api/extras/enhance', file, { operation: op, subtype: sub }, { timeout: 0 })); notify('Done'); }
    catch (e) { notify(e.message, 'error'); } finally { setBusy(false); }
  };
  return (
    <Panel title="Enhance" accent="frame post-processing" data-testid="editor-enhance">
      <Picker file={file} onFile={(f) => { setFile(f); setRes(null); }} />
      <div className="mt-3 flex flex-wrap gap-1.5" role="radiogroup" aria-label="Operation">
        {(ops ? Object.keys(ops) : ['upscale', 'colorize', 'filter']).map((k) => (
          <button key={k} type="button" role="radio" aria-checked={op === k} className={`chip !h-7 ${op === k ? 'on' : ''}`} onClick={() => setOp(k)}>{k}</button>
        ))}
      </div>
      {(ops?.[op] || []).length > 0 && (
        <div className="mt-2 grid grid-cols-2 gap-1.5 sm:grid-cols-3">
          {ops[op].map((s) => (
            <button key={s} type="button" aria-pressed={sub === s} onClick={() => setSub(s)} className={`tile px-2.5 py-2 text-left text-[12px] font-medium ${sub === s ? 'selected' : ''}`}>{s}</button>
          ))}
        </div>
      )}
      <button type="button" className="btn btn-primary mt-3 w-full" onClick={run} disabled={busy || !file}>{busy ? <Loader2 size={15} className="animate-spin" /> : <Wand2 size={15} />} Run {op}{sub ? ` · ${sub}` : ''}</button>
      <Result res={res} />
    </Panel>
  );
}

function VideoToFaceset({ detectors }) {
  const [video, setVideo] = useState(null);
  const [frame, setFrame] = useState(1);
  const [count, setCount] = useState(0);
  const [detector, setDetector] = useState('scrfd');
  const [busy, setBusy] = useState('');
  const [built, setBuilt] = useState(null);
  const input = useRef(null);
  const call = async (what, fn) => { setBusy(what); try { await fn(); } catch (e) { notify(e.message, 'error'); } finally { setBusy(''); } };
  return (
    <Panel title="Video → faceset" accent=".fsz" data-testid="editor-faceset">
      <p className="text-[12px] text-[var(--fg-dim)]">Cut faces from frames of a clip, then pack them into a faceset you can load into the cast.</p>
      <div className="mt-2 grid grid-cols-2 gap-2.5">
        <Select label="Detector" value={detector} options={detectors?.length ? detectors : ['scrfd']} onChange={setDetector} />
        <div className="flex items-end"><button type="button" className="btn w-full" disabled={!!busy} onClick={() => input.current?.click()}><Upload size={14} /> {video ? 'Other clip' : 'Choose clip'}</button></div>
      </div>
      <input ref={input} type="file" accept="video/*" className="hidden" onChange={(e) => {
        const f = e.target.files?.[0]; e.target.value = ''; if (!f) return;
        call('add', async () => {
          const r = await postFiles('/api/facemgr/add', [f], { detector, restore: 'false' }, { timeout: 0 });
          if (!r.video) throw new Error('That file has no video track. Use the Face Manager for stills.');
          setVideo({ name: f.name, frames: Math.max(1, r.frames || 1) }); setFrame(1); setCount((r.faces || []).length); setBuilt(null);
        });
      }} />
      {video && (
        <div className="mt-3 space-y-2.5">
          <div className="font-mono text-[11.5px] text-[var(--fg-dim)]">{video.name} · {video.frames} frames · <b className="text-[var(--accent-2)]">{count}</b> faces cut</div>
          <Slider label="Frame" value={frame} min={1} max={video.frames} step={1} onChange={setFrame} />
          <div className="flex gap-2">
            <button type="button" className="btn flex-1" disabled={!!busy} onClick={() => call('cut', async () => { const r = await postJSON('/api/facemgr/cut', { frame, detector, restore: false }, { timeout: 0 }); setCount((r.faces || []).length); setBuilt(null); notify(`Faces cut from frame ${frame}`); })}>
              {busy === 'cut' ? <Loader2 size={14} className="animate-spin" /> : <Scissors size={14} />} Cut faces here
            </button>
            <button type="button" className="btn btn-primary flex-1" disabled={!!busy || !count} onClick={() => call('build', async () => { setBuilt(await postJSON('/api/facemgr/build', {})); notify('Faceset file created'); })}>
              {busy === 'build' ? <Loader2 size={14} className="animate-spin" /> : <PackageCheck size={14} />} Build .fsz
            </button>
          </div>
          {built?.path && <a className="btn btn-sm" href={fileUrl(built.path)} download><Download size={13} /> {built.name || 'faceset.fsz'}</a>}
        </div>
      )}
    </Panel>
  );
}

export default function Editor({ meta }) {
  return (
    <div className="mx-auto max-w-[1500px] space-y-3" data-testid="editor">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight"><SplitText text="Editor" className="gradient-text" /></h1>
        <p className="text-[13px] text-[var(--fg-dim)]">Media tools that run no swap. Results are written to the output folder.</p>
      </div>
      <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
        <Transform />
        <Enhance />
        <VideoToFaceset detectors={meta?.detector_engines} />
      </div>
    </div>
  );
}
