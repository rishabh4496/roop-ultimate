import React, { useMemo, useState } from 'react';
import { Search, RotateCcw } from 'lucide-react';
import schema from '@schema';
import { useStudio } from '../engine/studio';
import { Panel, SplitText } from '../fx/effects';
import { Switch } from '../ui/controls';

// ── Settings: every setting the engine exposes, rendered from its own schema ──
// app/settings.schema.json is generated from app/settings.py (the single source for a setting), so a
// setting added there appears here with its label, section and type. Choices come from /api/meta.
// Saving is the store's debounced partial patch to /api/settings.

const CHOICES = {
  swap_model: 'swap_models', selected_enhancer: 'enhancers', mask_engine: 'mask_engines', mask_engine_2: 'mask_engines',
  detector_engine: 'detector_engines', recognizer: 'recognizers', restoration_mode: 'restoration_modes',
  color_transfer_mode: 'color_transfer_modes', face_detection_mode: 'face_detection_modes', subsample_upscale: 'upscale',
  output_method: 'output_methods', provider: 'providers', trt_precision: 'trt_precisions', no_face_action: 'no_face_actions',
  sam2_model_size: 'sam2_model_sizes', video_codec: 'video_codecs', video_format: 'video_formats', image_format: 'image_formats',
  encoder_preset: 'encoder_presets', process_priority: 'priorities',
};

export const UI_SETTINGS = (() => {
  const all = schema.settings || {};
  const order = schema.ui_order || [];
  return order.filter((k) => all[k]?.ui).map((k) => ({ key: k, ...all[k], label: all[k].ui.label, section: all[k].ui.section }));
})();

function Control({ item, value, meta, onChange }) {
  const id = `set-${item.key}`;
  const choices = CHOICES[item.key] && Array.isArray(meta?.[CHOICES[item.key]]) ? meta[CHOICES[item.key]] : null;
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

export default function Settings() {
  const settings = useStudio((s) => s.settings) || {};
  const meta = useStudio((s) => s.meta);
  const set = useStudio((s) => s.setSetting);
  const [q, setQ] = useState('');
  const [section, setSection] = useState('All');
  const sections = useMemo(() => ['All', ...Array.from(new Set(UI_SETTINGS.map((i) => i.section)))], []);
  const items = UI_SETTINGS.filter((i) => (section === 'All' || i.section === section)
    && (!q || `${i.label} ${i.key} ${i.section}`.toLowerCase().includes(q.toLowerCase())));
  const grouped = items.reduce((acc, i) => { (acc[i.section] ||= []).push(i); return acc; }, {});

  return (
    <div className="mx-auto max-w-[1500px] space-y-3" data-testid="settings">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight"><SplitText text="Settings" className="gradient-text" /></h1>
          <p className="text-[13px] text-[var(--fg-dim)]">{UI_SETTINGS.length} engine settings, from the engine's own schema. Changes save as you make them.</p>
        </div>
        <label className="relative ml-auto">
          <span className="sr-only">Search settings</span>
          <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--fg-faint)]" />
          <input className="input !h-9 w-72 pl-8" placeholder="Search settings (e.g. thread, nvenc)" value={q} onChange={(e) => setQ(e.target.value)} data-testid="settings-search" />
        </label>
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
