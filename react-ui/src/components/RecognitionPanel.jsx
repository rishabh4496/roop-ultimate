import React, { useCallback, useEffect, useState } from 'react';
import { getJSON, postJSON } from '../api';
import { Button, Select } from './ui';

// Embedding backend for the identity API: which recogniser and which provider, what this machine
// is best at, and applying it. Data comes from /api/recognition/models and /current, applying is
// POST /api/recognition/set (routes_recognition.py -> roop/ui_recognition.py). Applying may
// download the model and build a TensorRT engine, so it can take minutes; a failed apply keeps
// the old engine and saves nothing.

const specLine = (m) => [
  `input ${m.input}`,
  m.color_space,
  `${m.dim}-d`,
  m.quality_score ? 'quality score' : null,
  m.downloaded ? 'downloaded' : 'downloads on first use',
  m.same_file_as?.length ? `same file as ${m.same_file_as.join(', ')}` : null,
].filter(Boolean).join(' · ');

export default function RecognitionPanel({ notify, onApplied }) {
  const [data, setData] = useState(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState('default');
  const [provider, setProvider] = useState('app');
  const [busy, setBusy] = useState(false);
  const [last, setLast] = useState(null);

  const load = useCallback(async () => {
    try {
      const [list, cur] = await Promise.all([
        getJSON('/api/recognition/models', { timeout: 20000 }),
        getJSON('/api/recognition/current', { timeout: 20000 }),
      ]);
      setModels(list.models);
      setData(cur);
      setModel(cur.selection.model);
      setProvider(cur.selection.provider);
    } catch (e) {
      notify?.(`Embedding backend: ${e.message || e}`, 'error');
    }
  }, [notify]);

  useEffect(() => { load(); }, [load]);

  if (!data) return <p className="text-xs text-white/40">Reading the recognition backend…</p>;

  const { advice, providers, active } = data;
  const spec = models.find((m) => m.key === model);
  const chosen = providers.find((p) => p.value === provider);
  const unavailable = chosen && !chosen.available;
  const dirty = model !== data.selection.model || provider !== data.selection.provider;
  const hint = advice.model_hint && advice.model_hint !== model
    ? models.find((m) => m.key === advice.model_hint) : null;

  const apply = async () => {
    setBusy(true);
    try {
      const res = await postJSON('/api/recognition/set', { model_name: model, provider }, { timeout: 15 * 60 * 1000 });
      setLast(res);
      setData(res.after);
      setModels((await getJSON('/api/recognition/models', { timeout: 20000 })).models);   // downloaded flags
      onApplied?.({ model: res.model, provider: res.provider });
      notify?.(res.message, res.degraded ? 'warning' : 'success');
    } catch (e) {
      notify?.(e.message || String(e), 'error');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-3 text-xs">
      <p className="text-white/50">
        The <b>Recognition model</b> control above decides who is who in renders. This one picks the backbone
        for the identity embedding API (Face Bank style pooling and tracking helpers).
      </p>

      <div className="rounded-md border border-white/10 bg-white/[0.03] p-2 flex flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="px-2 py-0.5 rounded-md border border-white/20 text-nano font-semibold text-white/80">{advice.label}</span>
          <span className="text-white/70">{advice.device_name || 'No GPU detected'}</span>
          {advice.vram_gb > 0 && <span className="text-white/40">{advice.vram_gb} GB</span>}
          {advice.architecture && <span className="text-white/40">{advice.architecture} (SM {advice.compute_capability})</span>}
        </div>
        <p className="text-white/60">Suggested: <b className="text-white/80">{providers.find((p) => p.value === advice.provider)?.label}</b> — {advice.strategy}</p>
        <p className="text-nano text-white/40">{advice.reason}</p>
        <div className="flex flex-wrap gap-2 pt-1">
          <Button size="sm" variant="secondary" onClick={() => setProvider(advice.provider)} disabled={busy || provider === advice.provider}>
            Use suggested provider
          </Button>
          {hint && (
            <Button size="sm" variant="secondary" onClick={() => setModel(hint.key)} disabled={busy}>
              Use suggested model ({hint.display_name})
            </Button>
          )}
        </div>
      </div>

      <Select
        label="Embedding API model"
        info="Backbone used by the embedding API. A different model is a different identity metric: embeddings from two models are never comparable."
        value={model}
        onChange={setModel}
        options={models.map((m) => ({ value: m.key, label: m.display_name }))}
        disabled={busy}
      />
      {spec && (
        <div className="-mt-2 flex flex-col gap-0.5">
          <p className="text-white/60">{spec.description}</p>
          <p className="text-nano text-white/45">{specLine(spec)}</p>
        </div>
      )}

      <Select
        label="Hardware acceleration"
        info="Fallback chain is TensorRT, CUDA, DirectML / CoreML, CPU. The result is verified after a real inference, so a provider that silently failed to bind is reported, not assumed."
        value={provider}
        onChange={setProvider}
        options={providers.map((p) => ({ value: p.value, label: p.available ? p.label : `${p.label} (unavailable)` }))}
        disabled={busy}
      />
      {unavailable && <p className="-mt-2 text-nano text-amber-300">{chosen.reason}. The engine would fall back down the chain.</p>}

      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" onClick={apply} disabled={busy || !dirty}>
          {busy ? 'Loading… (may download / build engines)' : 'Apply'}
        </Button>
        <Button size="sm" variant="secondary" onClick={load} disabled={busy}>Refresh</Button>
        {dirty && !busy && <span className="text-nano text-white/40">Unsaved change</span>}
      </div>

      {active ? (
        <p className="text-white/60">
          Loaded: <b className="text-white/80">{active.model}</b> on{' '}
          <b className={active.degraded ? 'text-amber-300' : 'text-emerald-300'}>
            {(active.active_providers?.[0] || '').replace('ExecutionProvider', '') || 'CPU'}
          </b>
          {active.degraded && ' — a GPU provider was requested but the session is CPU-only'}
        </p>
      ) : (
        <p className="text-white/40">Nothing loaded yet: the model loads when first used or when you press Apply.</p>
      )}
      {last?.degraded && last.active.fallback_log?.length > 0 && (
        <ul className="list-disc pl-4 text-nano text-amber-300/80">
          {last.active.fallback_log.map((l) => <li key={l}>{l}</li>)}
        </ul>
      )}
      <p className="text-nano text-white/40">{data.scope}</p>
    </div>
  );
}
