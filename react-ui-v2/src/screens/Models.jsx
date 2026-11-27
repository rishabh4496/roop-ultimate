import React, { useEffect, useState } from 'react';
import { Cpu, ShieldCheck, ShieldAlert, Download, Wand2, Layers, Crosshair, Sparkles, ScanFace, Palette, Maximize, Zap, Loader2 } from 'lucide-react';
import { getJSON } from '@engine/api';
import { swapperLabel } from '@engine/components/faceswap/pipelineRules.js';
import { useStudio } from '../engine/studio';
import { Panel, SplitText, NumberTicker } from '../fx/effects';
import { notify } from '../ui/controls';

// ── Models: every model family the engine offers, and what is on disk ──────
// The lists are /api/meta's (so a model the engine adds appears here without a UI change); each
// card sets the setting that selects it. Integrity is /api/models/integrity: every manifest file,
// verified by hash, downloading, missing or corrupt.

const FAMILIES = [
  { key: 'swap_model', list: 'swap_models', title: 'Swapper', icon: Wand2, label: swapperLabel, hint: 'Who the face becomes. realswap pairs hyperswap and hififace per face pose.' },
  { key: 'selected_enhancer', list: 'enhancers', title: 'Restorer', icon: Sparkles, hint: 'Face restoration after the swap.' },
  { key: 'mask_engine', list: 'mask_engines', title: 'Mask', icon: Layers, hint: 'What protects hair, hands and occluders.' },
  { key: 'mask_engine_2', list: 'mask_engines', title: 'Second mask', icon: Layers, hint: 'Combined with the first.' },
  { key: 'detector_engine', list: 'detector_engines', title: 'Detector', icon: Crosshair, hint: 'Finds the faces.' },
  { key: 'recognizer', list: 'recognizers', title: 'Recognizer', icon: ScanFace, hint: 'Decides who is who.' },
  { key: 'restoration_mode', list: 'restoration_modes', title: 'Restoration mode', icon: Sparkles },
  { key: 'subsample_upscale', list: 'upscale', title: 'Swap resolution', icon: Maximize },
  { key: 'color_transfer_mode', list: 'color_transfer_modes', title: 'Colour match', icon: Palette },
  { key: 'sam2_model_size', list: 'sam2_model_sizes', title: 'SAM 2 size', icon: Layers },
];

const STATUS = {
  ok: ['badge-ok', 'verified'], verifying: ['badge-accent', 'verifying'], downloading: ['badge-accent', 'downloading'],
  pending: ['badge-dim', 'waiting'], missing: ['badge-warn', 'missing'], corrupt: ['badge-danger', 'failed check'],
};

export default function Models() {
  const meta = useStudio((s) => s.meta) || {};
  const settings = useStudio((s) => s.settings) || {};
  const set = useStudio((s) => s.setSetting);
  const [integrity, setIntegrity] = useState(null);
  useEffect(() => {
    let alive = true;
    let timer = null;
    const poll = async () => {
      try {
        const snap = await getJSON('/api/models/integrity', { timeout: 6000 });
        if (!alive) return;
        setIntegrity(snap);
        if (snap.busy) timer = setTimeout(poll, 800);
      } catch { if (alive) setIntegrity({ phase: 'unavailable', files: [] }); }
    };
    poll();
    return () => { alive = false; clearTimeout(timer); };
  }, []);
  const files = integrity?.files || [];
  const okCount = files.filter((f) => f.status === 'ok').length;

  return (
    <div className="mx-auto max-w-[1500px] space-y-3" data-testid="models">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight"><SplitText text="Models" className="gradient-text" /></h1>
          <p className="text-[13px] text-[var(--fg-dim)]">Every model family the engine ships. Pick one to use it; files download on first use.</p>
        </div>
        <div className="ml-auto flex flex-wrap gap-2">
          <span className="chip"><Cpu size={12} /> provider <b>{meta.active_provider || settings.provider || '—'}</b></span>
          <span className="chip"><Zap size={12} /> TensorRT <b>{meta.tensorrt_active ? 'active' : meta.tensorrt_available ? 'available' : 'off'}</b></span>
          {meta.degradation_reason && <span className="badge badge-warn">{meta.degradation_reason}</span>}
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
        {FAMILIES.filter((f) => Array.isArray(meta[f.list]) && meta[f.list].length).map((f) => (
          <Panel key={f.key} title={f.title} accent={(f.label || String)(settings[f.key] ?? '—')} data-testid={`family-${f.key}`}>
            {f.hint && <p className="mb-2 text-[12px] text-[var(--fg-faint)]">{f.hint}</p>}
            <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3">
              {meta[f.list].map((m) => {
                const on = String(settings[f.key]) === String(m);
                return (
                  <button key={m} type="button" aria-pressed={on} onClick={() => { set(f.key, m); notify(`${f.title}: ${(f.label || String)(m)}`, 'info'); }}
                    className={`tile flex items-center gap-2 px-2.5 py-2 text-left ${on ? 'selected beam' : ''}`}>
                    <f.icon size={14} className={on ? 'text-[var(--accent)]' : 'text-[var(--fg-faint)]'} />
                    <span className="min-w-0 truncate text-[12px] font-medium" title={String(m)}>{(f.label || String)(m)}</span>
                  </button>
                );
              })}
            </div>
          </Panel>
        ))}
        <Panel title="Execution" accent="restart to apply provider">
          <div className="grid grid-cols-2 gap-2">
            {(meta.providers || []).map((pv) => (
              <button key={pv} type="button" aria-pressed={settings.provider === pv} onClick={() => { set('provider', pv); notify(`Provider set to ${pv}; restart the app to build sessions on it`, 'info'); }}
                className={`tile px-3 py-2.5 text-left ${settings.provider === pv ? 'selected beam' : ''}`}>
                <div className="text-[13px] font-semibold uppercase">{pv}</div>
                <div className="font-mono text-[10.5px] text-[var(--fg-faint)]">{meta.active_provider === pv ? 'active now' : (meta.available_providers || []).includes(pv) ? 'available' : 'not available'}</div>
              </button>
            ))}
          </div>
          {Array.isArray(meta.trt_precisions) && (
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="text-[12px] text-[var(--fg-dim)]">TensorRT precision</span>
              {meta.trt_precisions.map((pr) => <button key={pr} type="button" className={`chip ${settings.trt_precision === pr ? 'on' : ''}`} onClick={() => set('trt_precision', pr)}>{pr}</button>)}
            </div>
          )}
        </Panel>
      </div>

      <Panel title="On disk" accent={integrity ? `${okCount}/${files.length} verified` : 'checking'} data-testid="integrity"
        actions={integrity?.busy ? <Loader2 size={15} className="animate-spin text-[var(--accent-2)]" /> : files.length && okCount === files.length ? <ShieldCheck size={16} className="text-[var(--ok)]" /> : <ShieldAlert size={16} className="text-[var(--warn)]" />}>
        {integrity?.message && <div className="mb-2 text-[12.5px] text-[var(--fg-dim)]">{integrity.message}</div>}
        {files.length > 0 && (
          <div className="grid grid-cols-1 gap-1.5 md:grid-cols-2 xl:grid-cols-3">
            {files.map((f) => {
              const [tone, word] = STATUS[f.status] || ['badge-dim', f.status];
              const pct = f.bytes_total ? (f.bytes_done / f.bytes_total) : null;
              return (
                <div key={f.file} className="rounded-xl border border-[var(--rule)] bg-black/40 px-3 py-2">
                  <div className="flex items-center gap-2">
                    <Download size={13} className="text-[var(--fg-faint)]" />
                    <span className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-white" title={f.file}>{f.file}</span>
                    <span className={`badge ${tone}`}>{word}</span>
                  </div>
                  <div className="mt-1 flex justify-between font-mono text-[10.5px] text-[var(--fg-faint)]">
                    <span>{f.family}{f.required ? ' · required' : ''}</span>
                    <span>{f.bytes_total ? <><NumberTicker value={f.bytes_total / 1048576} decimals={0} /> MB</> : ''}</span>
                  </div>
                  {f.status === 'downloading' && pct != null && <div className="bar mt-1.5"><i style={{ width: `${pct * 100}%` }} /></div>}
                  {f.error && <div className="mt-1 truncate text-[11px] text-[var(--danger)]" title={f.error}>{f.error}</div>}
                </div>
              );
            })}
          </div>
        )}
      </Panel>
    </div>
  );
}
