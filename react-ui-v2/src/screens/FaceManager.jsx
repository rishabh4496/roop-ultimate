import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Upload, Scissors, Trash2, Eraser, PackageCheck, Download, Loader2, Sparkles, FlaskConical, Save, Eye, ShieldOff, FolderInput } from 'lucide-react';
import { getJSON, postFile, postFiles, postJSON, fileUrl } from '@engine/api';
import { useStudio } from '../engine/studio';
import { Panel, SplitText, NumberTicker } from '../fx/effects';
import { Select, Slider, Switch, notify } from '../ui/controls';

// ── Face Manager: build facesets, see their pose coverage, blend identities ──
//   Builder    - images or a clip in, faces cut and scored, prune, build a .fsz  (/api/facemgr/*)
//   Pose radar - where the faceset's faces point; auto-prune blurry / duplicate faces
//   Morph Lab  - blend saved identities (slerp / linear), optional de-identification, preview, save

function Builder({ detectors, onChanged }) {
  const [faces, setFaces] = useState([]);
  const [scores, setScores] = useState([]);
  const [meta, setMeta] = useState([]);
  const [video, setVideo] = useState(null);
  const [frame, setFrame] = useState(1);
  const [detector, setDetector] = useState('scrfd');
  const [restore, setRestore] = useState(false);
  const [threshold, setThreshold] = useState(0.35);
  const [sel, setSel] = useState(0);
  const [busy, setBusy] = useState('');
  const [built, setBuilt] = useState(null);
  const add = useRef(null);
  const fsz = useRef(null);
  const apply = (r) => { setFaces(r.faces || []); setScores(r.scores || []); setMeta(r.meta || []); onChanged?.(); };
  const call = async (what, fn) => { setBusy(what); try { await fn(); } catch (e) { notify(e.message, 'error'); } finally { setBusy(''); } };

  return (
    <Panel title="Faceset builder" accent={`${faces.length} faces`} data-testid="facemgr-builder"
      actions={faces.length > 0 && <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Clear" title="Clear" onClick={() => call('clear', async () => { apply(await postJSON('/api/facemgr/clear', {})); setVideo(null); setBuilt(null); setSel(0); })}><Eraser size={14} /></button>}>
      <div className="grid grid-cols-2 gap-2.5">
        <Select label="Detector" value={detector} options={detectors?.length ? detectors : ['scrfd']} onChange={setDetector} />
        <div className="flex items-end"><Switch label="Restore faces" checked={restore} onChange={setRestore} /></div>
      </div>
      <div className="mt-2.5 flex flex-wrap gap-2">
        <button type="button" className="btn flex-1" disabled={!!busy} onClick={() => add.current?.click()}>{busy === 'add' ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />} Add images or a clip</button>
        <button type="button" className="btn" disabled={!!busy} onClick={() => fsz.current?.click()}><FolderInput size={14} /> Open .fsz</button>
      </div>
      <input ref={add} type="file" multiple accept="image/*,video/*" className="hidden" onChange={(e) => {
        const files = Array.from(e.target.files || []); e.target.value = ''; if (!files.length) return;
        call('add', async () => {
          const r = await postFiles('/api/facemgr/add', files, { detector, restore: String(restore) }, { timeout: 0 });
          apply(r); setBuilt(null);
          if (r.video) { setVideo({ name: files[0].name, frames: Math.max(1, r.frames || 1) }); setFrame(1); }
        });
      }} />
      <input ref={fsz} type="file" accept=".fsz" className="hidden" onChange={(e) => {
        const f = e.target.files?.[0]; e.target.value = ''; if (!f) return;
        call('fsz', async () => { apply(await postFile('/api/facemgr/faceset', f, undefined, { timeout: 0 })); notify('Faceset opened'); });
      }} />

      {video && (
        <div className="mt-3 rounded-xl border border-[var(--rule)] bg-black/30 p-2.5">
          <div className="mb-1 font-mono text-[11px] text-[var(--fg-dim)]">{video.name} · {video.frames} frames</div>
          <Slider label="Frame" value={frame} min={1} max={video.frames} step={1} onChange={setFrame} />
          <button type="button" className="btn btn-sm mt-1 w-full" disabled={!!busy} onClick={() => call('cut', async () => { apply(await postJSON('/api/facemgr/cut', { frame, detector, restore }, { timeout: 0 })); notify(`Faces cut from frame ${frame}`); })}>
            {busy === 'cut' ? <Loader2 size={13} className="animate-spin" /> : <Scissors size={13} />} Cut faces from frame {frame}
          </button>
        </div>
      )}

      <div className="mt-3 grid max-h-[300px] grid-cols-4 gap-1.5 overflow-y-auto sm:grid-cols-6" role="listbox" aria-label="Faces in the faceset">
        {faces.map((src, i) => {
          const s = scores[i];
          const m = meta[i] || {};
          const tone = typeof s !== 'number' ? 'var(--fg-faint)' : s >= 0.6 ? 'var(--ok)' : s >= threshold ? 'var(--warn)' : 'var(--danger)';
          return (
            <button key={i} type="button" role="option" aria-selected={i === sel} onClick={() => setSel(i)}
              className={`tile aspect-square ${i === sel ? 'selected' : ''}`} title={`${m.face_px ? `${m.face_px}px · ` : ''}${m.frontality != null ? `frontal ${(m.frontality * 100).toFixed(0)}% · ` : ''}${typeof s === 'number' ? `quality ${s.toFixed(2)}` : ''}`}>
              <img src={src} alt="" className="h-full w-full object-cover" />
              {typeof s === 'number' && <span className="absolute bottom-0.5 right-0.5 rounded bg-black/75 px-1 font-mono text-[9.5px]" style={{ color: tone }}>{s.toFixed(2)}</span>}
            </button>
          );
        })}
        {!faces.length && <div className="col-span-full py-6 text-center text-[12px] text-[var(--fg-faint)]">No faces yet - add photos of one person, or a clip and cut frames.</div>}
      </div>

      {faces.length > 0 && (
        <>
          <div className="mt-3 flex items-end gap-2">
            <div className="flex-1"><Slider label="Prune below quality" value={threshold} min={0} max={1} step={0.01} format={(v) => v.toFixed(2)} onChange={setThreshold} /></div>
            <button type="button" className="btn btn-sm" disabled={!!busy} onClick={() => call('prune', async () => { const r = await postJSON('/api/facemgr/prune', { threshold }); apply(r); notify(r.removed ? `Removed ${r.removed} low-quality face${r.removed === 1 ? '' : 's'}` : 'Nothing below the threshold'); })}>Prune</button>
          </div>
          <div className="mt-3 flex gap-2">
            <button type="button" className="btn btn-sm" disabled={!!busy} onClick={() => call('remove', async () => { const r = await postJSON('/api/facemgr/remove', { index: sel }); apply(r); setSel((x) => Math.max(0, Math.min(x, (r.faces || []).length - 1))); })}><Trash2 size={13} /> Remove selected</button>
            <button type="button" className="btn btn-primary btn-sm ml-auto" disabled={!!busy} onClick={() => call('build', async () => { setBuilt(await postJSON('/api/facemgr/build', {})); notify('Faceset file created'); })} data-testid="facemgr-build">
              {busy === 'build' ? <Loader2 size={13} className="animate-spin" /> : <PackageCheck size={13} />} Build .fsz
            </button>
          </div>
          {built?.path && (
            <div className="mt-2 flex flex-wrap gap-2">
              <a className="btn btn-sm" href={fileUrl(built.path)} download><Download size={13} /> {built.name || 'faceset.fsz'}</a>
            </div>
          )}
        </>
      )}
    </Panel>
  );
}

function PoseRadar({ refreshKey, onPruned }) {
  const [radar, setRadar] = useState(null);
  const [busy, setBusy] = useState(false);
  const load = useCallback(async () => { try { setRadar(await getJSON('/api/facemgr/radar')); } catch { setRadar(null); } }, []);
  useEffect(() => { load(); }, [load, refreshKey]);
  const pts = (radar?.faces || []).filter((f) => Number.isFinite(f.yaw) && Number.isFinite(f.pitch));
  const R = 110;
  const autoPrune = async () => {
    setBusy(true);
    try {
      const dry = await postJSON('/api/facemgr/auto_prune', { dry_run: true });
      const rm = dry.plan?.remove || [];
      const blurry = rm.filter((d) => d.reason === 'blurry').length;
      if (!rm.length) { notify('Nothing to prune - no blurry or duplicate faces', 'info'); return; }
      if (!window.confirm(`Remove ${rm.length} face(s): ${blurry} blurry, ${rm.length - blurry} duplicate?`)) return;
      const res = await postJSON('/api/facemgr/auto_prune', {});
      notify(`Removed ${res.removed} face${res.removed === 1 ? '' : 's'}`);
      onPruned?.(); load();
    } catch (e) { notify(e.message, 'error'); } finally { setBusy(false); }
  };
  return (
    <Panel title="Pose radar" accent={`${pts.length} faces`} data-testid="facemgr-radar"
      actions={<button type="button" className="btn btn-sm" disabled={busy || !pts.length} onClick={autoPrune}><Sparkles size={13} /> Auto-prune</button>}>
      <div className="flex flex-col items-center">
        <svg width={R * 2 + 20} height={R * 2 + 20} viewBox={`${-R - 10} ${-R - 10} ${R * 2 + 20} ${R * 2 + 20}`} role="img" aria-label="Face poses: yaw left-right, pitch up-down">
          {[1, 0.66, 0.33].map((k) => <circle key={k} r={R * k} fill="none" stroke="rgba(255,255,255,.1)" strokeDasharray="3 4" />)}
          <line x1={-R} x2={R} y1={0} y2={0} stroke="rgba(255,255,255,.12)" /><line y1={-R} y2={R} x1={0} x2={0} stroke="rgba(255,255,255,.12)" />
          <text x={R - 2} y={-4} textAnchor="end" fontSize="9" fill="var(--fg-faint)">right</text>
          <text x={-R + 2} y={-4} fontSize="9" fill="var(--fg-faint)">left</text>
          <text x={4} y={-R + 10} fontSize="9" fill="var(--fg-faint)">up</text>
          {pts.map((f, i) => {
            const x = Math.max(-1, Math.min(1, f.yaw / 90)) * R;
            const y = Math.max(-1, Math.min(1, -f.pitch / 60)) * R;
            const sharp = Number.isFinite(f.sharpness) ? Math.min(1, f.sharpness / 200) : 0.6;
            return <circle key={i} cx={x} cy={y} r={4} fill="var(--accent-2)" opacity={0.35 + sharp * 0.65} style={{ filter: 'drop-shadow(0 0 4px var(--accent-2))' }}><title>{`face ${f.index ?? i} · yaw ${f.yaw.toFixed(0)}° · pitch ${f.pitch.toFixed(0)}°`}</title></circle>;
          })}
        </svg>
        {radar?.missing?.length > 0 && <div className="mt-1 text-center text-[12px] text-[var(--warn)]">No faces looking {radar.missing.map((m) => String(m).replace(/_/g, ' ')).join(', ')} - swaps at those angles fall back to the nearest face.</div>}
        {!pts.length && <div className="text-[12px] text-[var(--fg-faint)]">Add faces in the builder to see their pose coverage.</div>}
      </div>
    </Panel>
  );
}

function MorphLab() {
  const [cands, setCands] = useState(null);
  const [picked, setPicked] = useState([]);
  const [weights, setWeights] = useState({});
  const [mode, setMode] = useState('slerp');
  const [deid, setDeid] = useState({ enabled: false, references: [], max_cosine: 0.3 });
  const [summary, setSummary] = useState(null);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState('');
  const load = async () => { try { setCands((await getJSON('/api/facemgr/morph/candidates')).candidates || []); } catch (e) { notify(e.message, 'error'); setCands([]); } };
  useEffect(() => { load(); }, []);
  const body = useMemo(() => ({
    components: picked.map((id) => ({ id, weight: weights[id] ?? 1 })),
    mode,
    deidentify: deid.enabled && deid.references.length ? deid : { enabled: false, references: [] },
  }), [picked, weights, mode, deid]);
  const usable = picked.length >= 2;
  const req = useRef(0);
  useEffect(() => {
    if (!usable) { setSummary(null); return undefined; }
    const id = ++req.current;
    const t = setTimeout(async () => {
      try { const r = await postJSON('/api/facemgr/morph/blend', body); if (req.current === id) setSummary(r); }
      catch (e) { if (req.current === id) setSummary({ error: e.message }); }
    }, 250);
    return () => clearTimeout(t);
  }, [JSON.stringify(body), usable]); // eslint-disable-line react-hooks/exhaustive-deps
  const toggle = (id) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
  const byId = Object.fromEntries((cands || []).map((c) => [c.id, c]));

  return (
    <Panel title="Morph Lab" accent="identity blending" beam={usable} data-testid="facemgr-morph">
      <p className="text-[12px] text-[var(--fg-dim)]">Pick two or more identities and weigh them. Slerp keeps the blend on the identity sphere; linear averages.</p>
      <div className="mt-2 grid max-h-[220px] grid-cols-3 gap-1.5 overflow-y-auto sm:grid-cols-4">
        {(cands || []).map((c) => (
          <button key={c.id} type="button" aria-pressed={picked.includes(c.id)} disabled={c.available === false} title={c.reason || c.label}
            onClick={() => toggle(c.id)} className={`tile flex flex-col items-center gap-1 p-1.5 ${picked.includes(c.id) ? 'selected' : ''} ${c.available === false ? 'opacity-40' : ''}`}>
            {c.thumb ? <img src={c.thumb} alt="" className="h-12 w-12 rounded-full object-cover" /> : <span className="grid h-12 w-12 place-items-center rounded-full bg-white/5"><FlaskConical size={16} /></span>}
            <span className="w-full truncate text-center text-[10.5px]">{c.label}</span>
            <span className="font-mono text-[9.5px] text-[var(--fg-faint)]">{c.kind}{c.faces ? ` · ${c.faces}` : ''}</span>
          </button>
        ))}
        {cands && !cands.length && <div className="col-span-full py-4 text-center text-[12px] text-[var(--fg-faint)]">No identities yet: add cast faces or save facesets.</div>}
      </div>
      {picked.length > 0 && (
        <div className="mt-3 space-y-2">
          {picked.map((id) => (
            <Slider key={id} label={byId[id]?.label || id} value={weights[id] ?? 1} min={0} max={1} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => setWeights((w) => ({ ...w, [id]: v }))} />
          ))}
          <div className="flex gap-1.5" role="radiogroup" aria-label="Blend mode">
            {[['slerp', 'Slerp'], ['linear', 'Linear']].map(([m, l]) => <button key={m} type="button" role="radio" aria-checked={mode === m} className={`chip !h-7 ${mode === m ? 'on' : ''}`} onClick={() => setMode(m)}>{l}</button>)}
          </div>
          <div className="rounded-xl border border-[var(--rule)] bg-black/30 p-2">
            <Switch label="De-identify" hint="Keep the blend away from chosen identities" checked={deid.enabled} onChange={(v) => setDeid((d) => ({ ...d, enabled: v }))} />
            {deid.enabled && (
              <>
                <div className="mt-1 flex flex-wrap gap-1">
                  {(cands || []).map((c) => <button key={c.id} type="button" aria-pressed={deid.references.includes(c.id)} className={`chip ${deid.references.includes(c.id) ? 'on' : ''}`}
                    onClick={() => setDeid((d) => ({ ...d, references: d.references.includes(c.id) ? d.references.filter((x) => x !== c.id) : [...d.references, c.id] }))}><ShieldOff size={11} /> {c.label}</button>)}
                </div>
                <div className="mt-2"><Slider label="Max similarity to them" value={deid.max_cosine} min={0} max={1} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => setDeid((d) => ({ ...d, max_cosine: v }))} /></div>
              </>
            )}
          </div>
        </div>
      )}
      {summary && (
        <div className="mt-3 rounded-xl border border-[var(--rule)] bg-black/40 p-2.5 text-[12px]" data-testid="morph-summary">
          {summary.error ? <span className="text-[var(--danger)]">{summary.error}</span> : (
            <>
              <div className="mb-1 font-mono text-[10.5px] uppercase tracking-[0.14em] text-[var(--fg-faint)]">similarity of the blend to each input</div>
              {(summary.cosine_to_each || []).map((c, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="w-28 truncate">{byId[picked[i]]?.label || picked[i]}</span>
                  <div className="bar flex-1"><i style={{ width: `${Math.max(0, c) * 100}%` }} /></div>
                  <span className="w-12 text-right font-mono tnum"><NumberTicker value={c} decimals={3} /></span>
                </div>
              ))}
              {summary.warning && <div className="mt-1 text-[var(--warn)]">{summary.warning}</div>}
            </>
          )}
        </div>
      )}
      <div className="mt-3 flex flex-wrap items-end gap-2">
        <button type="button" className="btn btn-sm" disabled={!usable || !!busy} onClick={async () => {
          setBusy('preview');
          try { await postJSON('/api/facemgr/morph/preview', body, { timeout: 0 }); notify('Blend set as the live identity - open the Studio to see it', 'info'); useStudio.getState().requestPreview(); }
          catch (e) { notify(e.message, 'error'); } finally { setBusy(''); }
        }}><Eye size={13} /> Preview in Studio</button>
        <input className="input !h-8 flex-1 text-[12px]" placeholder="blend name" value={name} onChange={(e) => setName(e.target.value)} aria-label="Blend name" />
        <button type="button" className="btn btn-primary btn-sm" disabled={!usable || !name.trim() || !!busy} onClick={async () => {
          setBusy('save');
          try { const r = await postJSON('/api/facemgr/morph/save', { ...body, name: name.trim() }); notify(`Saved “${r.saved}.fsz” to the faceset library`); setName(''); load(); }
          catch (e) { notify(e.message, 'error'); } finally { setBusy(''); }
        }}><Save size={13} /> Save</button>
      </div>
    </Panel>
  );
}

export default function FaceManager({ meta }) {
  const [key, setKey] = useState(0);
  const bump = () => setKey((k) => k + 1);
  return (
    <div className="mx-auto max-w-[1500px] space-y-3" data-testid="facemgr">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight"><SplitText text="Face Manager" className="gradient-text" /></h1>
        <p className="text-[13px] text-[var(--fg-dim)]">Build facesets, check their pose coverage, and blend identities.</p>
      </div>
      <div className="grid gap-3 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <Builder detectors={meta?.detector_engines} onChanged={bump} />
        <div className="flex flex-col gap-3">
          <PoseRadar refreshKey={key} onPruned={bump} />
          <MorphLab />
        </div>
      </div>
    </div>
  );
}
