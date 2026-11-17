import React from 'react';
import { Select, Slider, Toggle } from '../ui';
import { STUDIO_CONTROLS, studioSetting, studioValue } from './studioControls';

export default function StudioParameters({ values, onChange }) {
  return <details className="rounded-xl border border-white/10 p-3" data-testid="studio-parameters">
    <summary className="cursor-pointer font-semibold">Studio parameters</summary>
    <div className="mt-3 space-y-3">
      <Slider label="UltraMax maximum batch" min={1} max={8} step={1} value={Number(values.ultramax_batch_size || 1)} onChange={(v) => onChange('ultramax_batch_size', v)} />
      {STUDIO_CONTROLS.map((control) => {
      const current = studioValue(values, control);
      const change = (v) => onChange(...studioSetting(control, v));
      if (control.toggle) return <Toggle key={control.wire} label={control.label} checked={!!current} onChange={change} />;
      if (control.choices) return <Select key={control.wire} label={control.label} value={String(current)} options={control.choices.map((v) => ({ value: String(v), label: String(v) }))} onChange={(v) => change(control.wire === 'pixel_boost' ? Number(v) : v)} />;
      return <Slider key={control.wire} label={control.label} min={control.min} max={control.max} step={control.step || 0.01} value={Number(current)} onChange={change} />;
    })}</div>
  </details>;
}
