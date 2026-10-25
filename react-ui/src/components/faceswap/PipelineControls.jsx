import React from 'react';
import { Select, Slider, Toggle } from '../ui';
import {
  BYPASS_SWAPPERS, STRIDE_MAX, STRIDE_MIN, clampStride, restorerModeOf, restorerNote, restorerOptions, restorerPatch,
  strideText, swapperOptions,
} from './pipelineRules';
import { SMART_SKIP_INFO, STRIDE_INFO, SWAPPER_INFO } from './pipelineControlsInfo';

// The four controls that decide how the 3-stage pipeline spends its GPU, in one place:
//
//   Swapper model      Stage 3 swap network (and so the resolution the restorer can add to)
//   Restorer mode      Off / Auto-Bypass >=256px, Laplacian Split RF++, Full RF++  (three real settings, one choice)
//   Pre-pass stride    K = 1..8: Stage 2 runs the detector on every Kth frame and carries the rest by optical flow
//   Temporal Smart-Skip  Stage 3 reuses the previous composite for faces that did not move
//
// Used by #/faceswap and #/settings. Both tabs edit the same settings object, so a change in one is the other's
// value. The built-in Select / Slider / Toggle are the ones the rest of both tabs use, so the 12px body text, the
// 24px range hit area and the keyboard behaviour are inherited, not re-implemented. There is deliberately no run
// button here: starting a render stays the one control in the Face Swap run bar.
//
// `bind` is optional and keyed by setting name: Settings passes its `bind(key)` result for each key, and only the three
// marker fields are taken from it (`settingKey`, `modified`, `onReset`) so its "changed from default" markers and
// command-palette jumps keep working. Its `value` / `onChange` are NOT used: this component owns what each control
// reads and writes (the restorer mode is not any one setting). Face Swap passes nothing.
export default function PipelineControls({ values, models, onChange, onChangeMany, bind = {}, className = '' }) {
  const p = values || {};
  const marks = (key) => {
    const b = bind[key];
    return b ? { settingKey: b.settingKey, modified: b.modified, onReset: b.onReset } : {};
  };
  const mode = restorerModeOf(p);
  const stride = clampStride(p.prepass_detect_interval ?? 1);
  const swapperNote = BYPASS_SWAPPERS.test(String(p.swap_model ?? ''))
    ? null
    : (p.auto_bypass_restorer_native_swap ? 'This swapper is below 256 px native, so the restorer bypass will not fire for it.' : null);

  return (
    <div className={`space-y-4 ${className}`} data-testid="pipeline-controls">
      <Select
        label="Swapper model"
        info={SWAPPER_INFO}
        value={p.swap_model}
        onChange={(v) => onChange('swap_model', v)}
        options={swapperOptions(models, p.swap_model)}
        {...marks('swap_model')}
      />
      {swapperNote && <p className="-mt-2 text-note text-amber-300/90">{swapperNote}</p>}

      {/* The mode is one control over three settings; the bypass one is also addressable by its own name, so the command
          palette entry for "Auto-Bypass Restorer" lands here. */}
      <div data-setting={bind.auto_bypass_restorer_native_swap?.settingKey}>
        <Select
          label="Restorer mode"
          info="Off / Auto-Bypass: faces ≤320 px tall on Alpha 256 or HyperSwap skip the neural restorer for a fast DoG sharpen, because the native swap already holds more detail than the face footprint can keep (up to 3x Stage 3 throughput); everything else still gets the selected restorer. Laplacian Split RF++: RestoreFormer++ recombined by frequency, so the swap keeps its own tone and shape and only the restorer's fine detail is added (Restore Ultra). Full RF++: RestoreFormer++ replaces the swapped crop. Any other restorer (UltraMax, GPEN, CodeFormer...) is picked in the enhancer list on Face Swap and shows here as Custom."
          value={mode}
          onChange={(v) => {
            const patch = restorerPatch(v);
            if (Object.keys(patch).length) onChangeMany(patch);
          }}
          options={restorerOptions(p)}
          {...marks('selected_enhancer')}
        />
        <p className="mt-1.5 text-note text-muted" data-testid="restorer-note">{restorerNote(p)}</p>
      </div>

      <div>
        <Slider
          label="Pre-pass optical-flow stride (K)"
          info={`${STRIDE_INFO}${stride > 1 ? ' CURRENTLY ABOVE 1.' : ''}`}
          min={STRIDE_MIN}
          max={STRIDE_MAX}
          step={1}
          value={stride}
          onChange={(v) => onChange('prepass_detect_interval', clampStride(v))}
          {...marks('prepass_detect_interval')}
        />
        <p className="mt-1 text-note text-muted" data-testid="stride-note">K = {stride}: {strideText(stride)}.</p>
      </div>

      <Toggle
        label="Temporal Smart-Skip (Lossless Static Reuse)"
        info={SMART_SKIP_INFO}
        checked={!!p.temporal_smart_skip}
        onChange={(v) => onChange('temporal_smart_skip', v)}
        {...marks('temporal_smart_skip')}
      />
    </div>
  );
}
