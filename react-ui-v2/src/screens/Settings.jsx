import React, { useEffect, useMemo, useState } from 'react';
import { Search, RotateCcw, SlidersHorizontal, HardDrive, Cpu, Loader2, Gauge } from 'lucide-react';
import { getJSON } from '@engine/api';
import schema from '@schema';
import { useStudio } from '../engine/studio';
import { Panel, SplitText } from '../fx/effects';
import { Switch, notify } from '../ui/controls';
import { StoragePanel, SystemPanels } from './System';
import TuningPanels from './Tuning';

// ── Settings: every setting the engine exposes, rendered from its own schema ──
// app/settings.schema.json is generated from app/settings.py (the single source for a setting), so a
// setting added there appears here with its label, section and type. Choices come from /api/meta.
// Saving is the store's debounced partial patch to /api/settings.

const CHOICES = {
  swap_model: 'swap_models', selected_enhancer: 'enhancers', mask_engine: 'mask_engines', mask_engine_2: 'mask_engines',
  detector_engine: 'detector_engines', recognizer: 'recognizers', restoration_mode: 'restoration_modes',
  color_transfer_mode: 'color_transfer_modes', face_detection_mode: 'face_detection_modes', subsample_upscale: 'upscale',
  output_method: 'output_methods', provider: 'providers', trt_precision: 'trt_precisions', no_face_action: 'no_face_actions',
  sam2_model_size: 'sam2_model_sizes', process_priority: 'priorities',
  // The settings are output_*; `video_codec`, `video_format`, `image_format` and `encoder_preset` do not
  // exist, so these four rendered as free-text boxes where a typo was saved as the codec.
  output_video_codec: 'video_codecs', output_video_format: 'video_formats', output_image_format: 'image_formats',
  perf_encoder_preset: 'encoder_presets',
};
// A `tristate` ROOP_* flag takes auto / on / off (settings.apply_env: on -> 1, off -> 0, else untouched).
const TRISTATE = ['auto', 'on', 'off'];
const choicesFor = (item, meta) => {
  if (CHOICES[item.key] && Array.isArray(meta?.[CHOICES[item.key]])) return meta[CHOICES[item.key]];
  if (item.type === 'str' && (item.env?.kind === 'tristate' || item.env?.kind === 'tristate_on')) return TRISTATE;
  return null;
};

export const UI_SETTINGS = (() => {
  const all = schema.settings || {};
  const order = schema.ui_order || [];
  return order.filter((k) => all[k]?.ui).map((k) => ({ key: k, ...all[k], label: all[k].ui.label, section: all[k].ui.section }));
})();

function Control({ item, value, meta, onChange }) {
  const id = `set-${item.key}`;
  const choices = choicesFor(item, meta);
  if (item.type === 'bool') return <Switch label={item.label} checked={!!value} onChange={onChange} />;
  const label = <label className="field-label" htmlFor={id}>{item.label}</label>;
  if (choices) {
    return (
      <div>{label}
        <select id={id} className="select" value={value ?? ''} onChange={(e) => onChange(e.target.value)}>
          {!choices.map(String).includes(String(value ?? '')) && <option value={value ?? ''}>{String(value ?? '—')}</option>}
          {choices.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>
    );
  }
  if (item.type === 'float' || item.type === 'int') {
    const isUnit = item.type === 'float' && typeof item.default === 'number' && item.default >= 0 && item.default <= 1 && (value == null || (value >= 0 && value <= 1));
    if (isUnit) {
      const v = Number(value ?? item.default ?? 0);
      return (
        <div>
          <div className="flex items-baseline justify-between">{label}<span className="font-mono text-[11.5px] text-[var(--accent-2)] tnum">{v.toFixed(2)}</span></div>
          <input id={id} type="range" className="range" min={0} max={1} step={0.01} value={v} style={{ '--fill': `${v * 100}%` }} onChange={(e) => onChange(Number(e.target.value))} />
        </div>
      );
    }
    return (
      <div>{label}
        <input id={id} type="number" className="input tnum" step={item.type === 'int' ? 1 : 0.01} value={value ?? ''}
          onChange={(e) => { const n = e.target.value === '' ? null : Number(e.target.value); if (n === null || Number.isFinite(n)) onChange(item.type === 'int' && n !== null ? Math.round(n) : n); }} />
      </div>
    );
  }
  if (item.type === 'list' || item.type === 'dict') {
    return <div>{label}<div className="rounded-lg border border-[var(--rule)] bg-black/40 px-2 py-1.5 font-mono text-[11px] text-[var(--fg-dim)]">{JSON.stringify(value ?? item.default ?? null)}</div></div>;
  }
  return <div>{label}<input id={id} className="input" value={value ?? ''} onChange={(e) => onChange(e.target.value)} /></div>;
}

// Reset every setting this screen shows to what a fresh install on THIS machine would start with
// (GET /api/settings/defaults loads a throwaway Settings from a missing file; some defaults, such as
// max_threads, are derived from the hardware). One confirmed patch; nothing else is touched.
async function resetAll(settings, setSettings) {
  let defaults;
  try { defaults = await getJSON('/api/settings/defaults'); } catch (e) { notify(e.message, 'error'); return; }
  const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
  const patch = {};
  UI_SETTINGS.forEach((i) => { if (Object.prototype.hasOwnProperty.call(defaults || {}, i.key) && !same(settings[i.key], defaults[i.key])) patch[i.key] = defaults[i.key]; });
  const n = Object.keys(patch).length;
  if (!n) { notify('Every setting is already at its default', 'info'); return; }
  const names = UI_SETTINGS.filter((i) => i.key in patch).slice(0, 12).map((i) => `  ${i.label}`).join('\n');
  const more = n > 12 ? `\n  ...and ${n - 12} more` : '';
  if (!window.confirm(`Reset ${n} setting${n === 1 ? '' : 's'} to the defaults for this machine?\n\n${names}${more}`)) return;
  setSettings(patch);
  notify(`Reset ${n} setting${n === 1 ? '' : 's'} to default`);
}

function EngineSettings() {
  const settings = useStudio((s) => s.settings) || {};
  const meta = useStudio((s) => s.meta);
  const set = useStudio((s) => s.setSetting);
  const setMany = useStudio((s) => s.setSettings);
  const [q, setQ] = useState('');
  const [section, setSection] = useState('All');
  const [resetting, setResetting] = useState(false);
  const sections = useMemo(() => ['All', ...Array.from(new Set(UI_SETTINGS.map((i) => i.section)))], []);
  const items = UI_SETTINGS.filter((i) => (section === 'All' || i.section === section)
    && (!q || `${i.label} ${i.key} ${i.section}`.toLowerCase().includes(q.toLowerCase())));
  const grouped = items.reduce((acc, i) => { (acc[i.section] ||= []).push(i); return acc; }, {});

  return (
    <div className="space-y-3" data-testid="settings-engine">
      <div className="flex flex-wrap items-end gap-3">
        <p className="text-[13px] text-[var(--fg-dim)]">{UI_SETTINGS.length} engine settings, from the engine's own schema. Changes save as you make them.</p>
        <label className="relative ml-auto">
          <span className="sr-only">Search settings</span>
          <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--fg-faint)]" />
          <input className="input !h-9 w-72 pl-8" placeholder="Search settings (e.g. thread, nvenc)" value={q} onChange={(e) => setQ(e.target.value)} data-testid="settings-search" />
        </label>
        <button type="button" className="btn !h-9" disabled={resetting} data-testid="settings-reset-all"
          onClick={async () => { setResetting(true); try { await resetAll(settings, setMany); } finally { setResetting(false); } }}>
          {resetting ? <Loader2 size={14} className="animate-spin" /> : <RotateCcw size={14} />} Reset all
        </button>
      </div>
      <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Sections">
        {sections.map((s) => (
          <button key={s} type="button" role="tab" aria-selected={section === s} className={`chip !h-7 ${section === s ? 'on' : ''}`} onClick={() => setSection(s)}>{s}</button>
        ))}
      </div>
      <div className="columns-1 gap-3 lg:columns-2 2xl:columns-3">
        {Object.entries(grouped).map(([sec, list]) => (
          <Panel key={sec} title={sec} accent={`${list.length}`} className="mb-3 break-inside-avoid">
            <div className="space-y-3">
              {list.map((item) => (
                <div key={item.key} className="group relative">
                  <Control item={item} value={settings[item.key]} meta={meta} onChange={(v) => set(item.key, v)} />
                  {item.default !== undefined && item.type !== 'bool' && settings[item.key] !== item.default && (
                    <button type="button" className="absolute right-0 top-0 opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100 text-[var(--fg-faint)] hover:text-white"
                      title={`Reset to default (${JSON.stringify(item.default)})`} aria-label={`Reset ${item.label}`} onClick={() => set(item.key, item.default)}>
                      <RotateCcw size={12} />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </Panel>
        ))}
      </div>
      {!items.length && <div className="py-10 text-center text-[var(--fg-dim)]">No setting matches “{q}”.</div>}
    </div>
  );
}

const TABS = [
  { id: 'engine', label: 'Engine settings', icon: SlidersHorizontal },
  { id: 'storage', label: 'Storage', icon: HardDrive },
  { id: 'tuning', label: 'Tuning', icon: Gauge },
  { id: 'system', label: 'System', icon: Cpu },
];

export default function Settings() {
  const tabFromHash = () => {
    const t = String(window.location.hash || '').split('?')[1]?.match(/tab=(\w+)/)?.[1];
    return TABS.some((x) => x.id === t) ? t : null;
  };
  const [tab, setTab] = useState(() => tabFromHash() || 'engine');
  // A link (or the command palette) to #/settings?tab=x while Settings is already open.
  useEffect(() => {
    const onHash = () => { const t = tabFromHash(); if (t) setTab(t); };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  return (
    <div className="mx-auto max-w-[1500px] space-y-3" data-testid="settings">
      <div className="flex flex-wrap items-end gap-3">
        <h1 className="text-2xl font-semibold tracking-tight"><SplitText text="Settings" className="gradient-text" /></h1>
        <div className="ml-auto flex gap-1.5" role="tablist" aria-label="Settings pages">
          {TABS.map((t) => (
            <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} className={`chip !h-8 ${tab === t.id ? 'on' : ''}`}
              onClick={() => setTab(t.id)} data-testid={`settings-tab-${t.id}`}><t.icon size={13} /> {t.label}</button>
          ))}
        </div>
      </div>
      {tab === 'engine' && <EngineSettings />}
      {tab === 'storage' && <StoragePanel />}
      {tab === 'tuning' && <TuningPanels />}
      {tab === 'system' && <SystemPanels />}
    </div>
  );
}
