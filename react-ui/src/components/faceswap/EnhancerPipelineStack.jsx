import React, { useId, useState } from 'react';
import { Select, Slider } from '../ui';
import { Icon } from '../../icons';
import { FACESWAP_DEFAULTS } from './defaults';
import { swapperOptions, swapperLabel } from './pipelineRules';
import { AI_UPSCALE_MODELS } from './upscaleModels';
import {
  latencyFor, stageState, stageSummary, togglePatch, MIX_UNAVAILABLE, isPercentMix, num,
} from './effectRackModel';

// ── The effect rack ─────────────────────────────────────────────────────────────────────────────────────────────────
//
// The post-swap chain as a rack of modules, in the order the engine runs it:
//
//   1 Swapper Core  ->  2 Face Restorer  ->  3 Texture & Grain Closer  ->  4 LAB Colour & Skin-Tone Match
//                   ->  5 Occlusion Mask (a second masking engine)  ->  6 Frame Upscaling (after the swap pass)
//
// Each module card has
//   * a power switch that BYPASSES it. The switch remembers what it replaced (the restorer, the grain amount, the colour mode,
//     the occlusion engine) and puts it back, so A/B-ing a stage is one click each way and loses nothing;
//   * a wet/dry MIX bar (0-100%) wherever the render has a setting that blends the stage over its input (see effectRackModel.js:
//     the restorer, the grain closer, the colour match). Where it has none - a mask, an upscaler - the bar is shown disabled at
//     100% and says why, instead of a control bound to nothing;
//   * a latency badge (Low / Med / High VRAM) whose tooltip says whether the level was measured or estimated;
//   * a collapsible body with the module's model choice and its advanced parameters.
//
// The cards are NOT reorderable: the engine's order is fixed, and a drag handle would rearrange the cards and not the render.
//
// Every control is an existing setting; nothing here invents a key. Reusable: Face Swap mounts it with `slots` carrying the
// model-specific controls it owns (UltraMax detail, CodeFormer fidelity, tile diffusion, ...); the Extras tab mounts it bare.
// Both write the same live settings.

// What each power switch last held, and which cards the user folded, per page load. Module scope on purpose: the rack unmounts
// whenever its tab does, and "turn the restorer back on" should still bring back the one you had.
const remembered = { restorer: null, grain: null, color: null, occlusion: null };
// The slot-heavy bodies start folded (the mix bar and the power switch are always visible): a rack with every body open adds
// ~65 tab stops to Face Swap, past e2e/tab-stops.spec.js's ceiling. Once the user opens or folds a card it stays that way.
const DEFAULT_FOLDED = { restorer: true, grain: true, color: true, occlusion: true, upscale: true };
const folded = {};

function PowerGlyph() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      <path d="M8 1.8v5.2" />
      <path d="M4.6 3.9a5.4 5.4 0 1 0 6.8 0" />
    </svg>
  );
}

function LatencyBadge({ stage, settings }) {
  const l = latencyFor(stage, settings);
  const title = `${l.label === 'Core' ? 'Always runs' : `GPU cost: ${l.label}`} (${l.basis === 'measured' ? 'measured' : 'estimated'}). ${l.note || ''}`.trim();
  return (
    <span data-testid="rack-latency" data-level={l.level} data-basis={l.basis} title={title}
      className={`rack-latency rack-latency--${l.level}`}>
      {l.label}{l.basis === 'estimate' && l.level !== 'core' ? <span aria-hidden="true">~</span> : null}
    </span>
  );
}

function Module({ id, index, title, summary, enabled, locked, lockedHint, onToggle, latency, mix, children }) {
  const [open, setOpen] = useState(() => (id in folded ? !folded[id] : !DEFAULT_FOLDED[id]));
  const bodyId = useId();
  const fold = () => setOpen((v) => { folded[id] = v; return !v; });   // v is the OLD state: open -> folded[id] = true
  return (
    <li data-testid={`rack-module-${id}`} data-enabled={enabled ? 'true' : 'false'} className={`rack-card ${enabled ? 'is-on' : 'is-bypassed'}`}>
      <div className="rack-card__head">
        <span aria-hidden="true" className="rack-card__index">{index}</span>
        <button type="button" onClick={fold} aria-expanded={open} aria-controls={bodyId} className="rack-card__title">
          <span className="rack-card__name">{title}</span>
          <span className="rack-card__summary">{summary}</span>
        </button>
        {latency}
        <button
          type="button"
          role="switch"
          aria-checked={!!enabled}
          aria-label={`${title} enabled`}
          title={locked ? lockedHint : (enabled ? 'Bypass this stage (click again to bring it back)' : 'Stage is bypassed — click to bring it back')}
          disabled={locked}
          onClick={() => !locked && onToggle(!enabled)}
          className="rack-power"
        >
          <PowerGlyph />
        </button>
        <Icon.expand size={11} aria-hidden="true" className={`rack-card__chevron ${open ? 'is-open' : ''}`} />
      </div>
      {mix && <div className="rack-card__mix">{mix}</div>}
      {open && <div id={bodyId} className="rack-card__body">{children}</div>}
    </li>
  );
}

const Connector = () => (
  <li aria-hidden="true" className="flex justify-center py-0.5 text-muted/60">
    <Icon.down size={13} />
  </li>
);

export default function EnhancerPipelineStack({
  settings, onChange, meta, defaults = FACESWAP_DEFAULTS, slots = {}, upscaleModels = AI_UPSCALE_MODELS,
}) {
  const p = settings || {};
  const d = defaults;

  const swapper = p.swap_model || d.swap_model;
  const enhancer = p.selected_enhancer ?? d.selected_enhancer;
  const restorerOn = stageState('restorer', p, d).enabled;
  const enhancers = (meta?.enhancers || []).filter((e) => e !== 'None');
  if (restorerOn && !enhancers.includes(enhancer)) enhancers.unshift(enhancer);
  const colorMode = p.color_transfer_mode || d.color_transfer_mode;
  const colorOn = stageState('color', p, d).enabled;
  const colorModes = (meta?.color_transfer_modes || ['none', 'rct', 'lct', 'mkl', 'idt']).filter((m) => m !== 'none');
  const occ = p.mask_engine_2 ?? d.mask_engine_2 ?? 'None';
  const occOn = stageState('occlusion', p, d).enabled;
  const maskEngines = (meta?.mask_engines || []).filter((e) => e !== 'None');
  if (occOn && !maskEngines.includes(occ)) maskEngines.unshift(occ);

  const toggle = (stage) => (on) => {
    const patch = togglePatch(stage, on, p, d, remembered);
    if (patch) onChange(patch[0], patch[1]);
  };
  const state = (stage) => stageState(stage, p, d);
  const common = (stage, index, title) => ({
    id: stage, index, title,
    summary: stageSummary(stage, p, d, swapperLabel),
    enabled: state(stage).enabled,
    onToggle: toggle(stage),
    latency: <LatencyBadge stage={stage} settings={p} />,
  });

  // The wet/dry bar of a module: the real setting where there is one, a disabled 100% bar that says why where there is not.
  const mixBar = (stage, label, info, extra = {}) => {
    const st = state(stage);
    if (!st.mixKey) {
      return (
        <Slider label="Mix" min={0} max={1} step={0.01} value={1} percent disabled onChange={() => {}}
          disabledNote={MIX_UNAVAILABLE[stage]} testId={`rack-mix-${stage}`} />
      );
    }
    return (
      <Slider
        label={label} info={info} min={0} max={1} step={isPercentMix(stage) ? 0.01 : 0.05} percent
        value={st.mix} defaultValue={st.mixDefault}
        onChange={(v) => onChange(st.mixKey, v)}
        testId={`rack-mix-${stage}`}
        {...extra}
      />
    );
  };

  return (
    <ol aria-label="Enhancer pipeline" className="m-0 list-none space-y-0 p-0">
      <Module
        {...common('swapper', 1, 'Swapper Core')}
        locked
        lockedHint="The swapper is the pipeline's core and cannot be turned off"
        mix={mixBar('swapper')}
      >
        <Select
          label="Swap model"
          info="The face-swap network. Changing it takes effect on the next preview or render."
          value={swapper}
          onChange={(v) => onChange('swap_model', v)}
          options={swapperOptions(meta?.swap_models, swapper)}
        />
        <Slider
          label="Swapping steps"
          info="more = more likeness"
          min={1} max={5} step={1}
          value={num(p.num_swap_steps, d.num_swap_steps)}
          defaultValue={d.num_swap_steps}
          onChange={(v) => onChange('num_swap_steps', v)}
        />
        {slots.swapper}
      </Module>

      <Connector />

      <Module
        {...common('restorer', 2, 'Face Restorer')}
        mix={mixBar('restorer', 'Original/Enhanced blend',
          "How much of the restorer's output is mixed back over the swap. 100% = the restorer's face, 0% = the swap as it came out.")}
      >
        <Select
          label="Restorer model"
          info="Cost, measured on an RTX 4070: GPEN-256 ~3 ms, GPEN-512 ~20 ms, RestoreFormer++ ~23 ms, CodeFormer ~24 ms per face. Which looks better on your footage is yours to judge."
          value={restorerOn ? enhancer : ''}
          onChange={(v) => onChange('selected_enhancer', v)}
          options={[...(restorerOn ? [] : [{ value: '', label: 'Choose a restorer…' }]), ...enhancers]}
        />
        {slots.restorer}
      </Module>

      <Connector />

      <Module
        {...common('grain', 3, 'Texture & Grain Closer')}
        mix={mixBar('grain', 'Grain match',
          "Adds noise matched to the original footage's own measured noise floor, so the face is not cleaner than everything around it. 0% = off; try 50–100% on noisy footage.")}
      >
        {slots.texture || <p className="text-mini text-muted">The mix above is the whole control: it sets how much matched noise is added.</p>}
      </Module>

      <Connector />

      <Module
        {...common('color', 4, 'LAB Color & Skin-Tone Match')}
        mix={mixBar('color', 'Histogram match',
          "Matches the shape of the face's per-channel histogram to the original crop, which catches a face that is subtly flatter or more contrasted than the plate. 0% = off.")}
      >
        <Select
          label="Colour transfer"
          info="RCT = per-channel (fast). LCT = corrects hue casts. MKL = fullest linear match. IDT = matches the full colour distribution, the quality ceiling and much the dearest."
          value={colorOn ? colorMode : ''}
          onChange={(v) => onChange('color_transfer_mode', v)}
          options={[...(colorOn ? [] : [{ value: '', label: 'Choose a method…' }]), ...colorModes.map((m) => ({ value: m, label: m.toUpperCase() }))]}
        />
        {slots.color}
      </Module>

      <Connector />

      <Module
        {...common('occlusion', 5, 'Occlusion Mask')}
        mix={mixBar('occlusion')}
      >
        <Select
          label="Occlusion engine"
          info="A second, independent masking engine run after the first (the Masking section's engine): hands, microphones and other objects in front of the face. They compose as a union. The power switch sets it to None and back."
          value={occOn ? occ : ''}
          onChange={(v) => onChange('mask_engine_2', v)}
          options={[...(occOn ? [] : [{ value: '', label: 'Choose an engine…' }]), ...maskEngines]}
        />
        {slots.occlusion}
      </Module>

      <Connector />

      <Module
        {...common('upscale', 6, 'Frame Upscaling')}
        mix={mixBar('upscale')}
      >
        <Select
          label="AI upscale model"
          info="Runs as the final step of the swap pass: each frame is swapped and enhanced first, then upscaled, producing one output file. Resampling filters (Lanczos, FSR, Spline, Sinc) cost almost nothing; the ×4 networks are the heaviest stage in the chain."
          value={p.upscale_model_after || d.upscale_model_after}
          onChange={(v) => onChange('upscale_model_after', v)}
          options={upscaleModels}
        />
        {slots.upscale}
      </Module>
    </ol>
  );
}
