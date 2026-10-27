import React, { useId, useState } from 'react';
import { Select, Slider } from '../ui';
import { Icon } from '../../icons';
import { FACESWAP_DEFAULTS } from './defaults';
import { swapperOptions, swapperLabel } from './pipelineRules';
import MiniSwitch from './MiniSwitch';

// ── The post-swap chain, as four stages ─────────────────────────────────────
//
//   Swapper Core  ->  Face Restorer  ->  Texture & Grain Closer  ->  LAB Color & Skin-Tone Match
//
// Each stage is one card: an enable switch, the model it runs (where it has
// one), and the ONE slider that matters most, with everything else behind
// "Options". Every control is an existing setting; nothing here invents a key.
//
//   stage              switch means                 model              primary slider
//   Swapper Core       always on (it IS the swap)   swap_model         num_swap_steps
//   Face Restorer      selected_enhancer != None    selected_enhancer  blend_ratio
//   Texture & Grain    merger_grain_match > 0       (none: it is a    merger_grain_match
//                                                    DSP pass)
//   LAB Colour         color_transfer_mode != none  color_transfer_mode merger_hist_match
//
// A switch that is turned off remembers what it replaced (the previous restorer,
// the previous grain level, the previous colour mode) and puts it back when
// turned on again, instead of snapping to a default.
//
// Reusable: Face Swap mounts it with `slots` carrying the model-specific
// controls it owns (UltraMax detail, CodeFormer fidelity, tile diffusion, ...);
// the Editor tab mounts it bare. Both write the same live settings.
//
// Every slider takes `defaultValue`, so double-clicking it resets it (see Slider
// in ui.jsx), and Shift+drag / typing a value work on all of them.

const num = (v, d) => {
  const x = Number(v);
  return Number.isFinite(x) ? x : d;
};

// What each switch last held, per page load. Module scope on purpose: the stack
// unmounts whenever its tab does, and "turn the restorer back on" should still
// bring back the one you had.
const remembered = { restorer: null, grain: null, color: null };

function Stage({ index, title, summary, enabled, locked, lockedHint, onToggle, dim, children, options }) {
  const [open, setOpen] = useState(false);
  const bodyId = useId();
  return (
    <li className="relative">
      <div className="rounded-xl border border-white/[0.08] bg-white/[0.025]">
        <div className="flex items-center gap-2.5 px-3 py-2.5">
          <span aria-hidden="true" className="grid h-5 w-5 shrink-0 place-items-center rounded-md border border-white/10 bg-black/30 text-micro font-bold text-muted">{index}</span>
          <div className="min-w-0 flex-1">
            <div className="truncate text-compact font-semibold text-white/90">{title}</div>
            <div className="truncate text-mini text-muted">{summary}</div>
          </div>
          <MiniSwitch
            label={`${title} enabled`}
            checked={enabled}
            disabled={locked}
            title={locked ? lockedHint : undefined}
            onChange={onToggle}
          />
        </div>

        <div className={`space-y-3 px-3 pb-3 ${dim ? 'opacity-60' : ''}`}>
          {children}
          {options && (
            <div>
              <button
                type="button"
                onClick={() => setOpen((v) => !v)}
                aria-expanded={open}
                aria-controls={bodyId}
                className="flex items-center gap-1.5 text-mini font-semibold text-white/60 transition-colors hover:text-white"
              >
                <Icon.expand size={11} className={`transition-transform duration-200 ${open ? 'rotate-90' : ''}`} />
                Options
              </button>
              {open && <div id={bodyId} className="mt-2.5 space-y-3 border-t border-white/[0.06] pt-2.5">{options}</div>}
            </div>
          )}
        </div>
      </div>
    </li>
  );
}

const Connector = () => (
  <li aria-hidden="true" className="flex justify-center py-0.5 text-muted/60">
    <Icon.down size={13} />
  </li>
);

export default function EnhancerPipelineStack({
  settings, onChange, meta, defaults = FACESWAP_DEFAULTS, slots = {},
}) {
  const p = settings || {};
  const d = defaults;

  // ── 1 Swapper
  const swapper = p.swap_model || d.swap_model;

  // ── 2 Restorer
  const enhancer = p.selected_enhancer ?? d.selected_enhancer;
  const restorerOn = enhancer && enhancer !== 'None';
  const enhancers = (meta?.enhancers || []).filter((e) => e !== 'None');
  if (restorerOn && !enhancers.includes(enhancer)) enhancers.unshift(enhancer);

  // ── 3 Texture
  const grain = num(p.merger_grain_match, d.merger_grain_match);

  // ── 4 Colour
  const colorMode = p.color_transfer_mode || d.color_transfer_mode;
  const colorOn = colorMode !== 'none';
  const colorModes = (meta?.color_transfer_modes || ['none', 'rct', 'lct', 'mkl', 'idt']).filter((m) => m !== 'none');

  return (
    <ol aria-label="Enhancer pipeline" className="m-0 list-none space-y-0 p-0">
      <Stage
        index={1}
        title="Swapper Core"
        summary={swapperLabel(swapper)}
        enabled
        locked
        lockedHint="The swapper is the pipeline's core and cannot be turned off"
        onToggle={() => {}}
        options={slots.swapper}
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
      </Stage>

      <Connector />

      <Stage
        index={2}
        title="Face Restorer"
        summary={restorerOn ? enhancer : 'Off'}
        enabled={!!restorerOn}
        dim={!restorerOn}
        onToggle={(on) => {
          if (on) onChange('selected_enhancer', remembered.restorer || d.selected_enhancer || 'UltraMax');
          else { remembered.restorer = enhancer; onChange('selected_enhancer', 'None'); }
        }}
        options={slots.restorer}
      >
        <Select
          label="Restorer model"
          info="Cost, measured on an RTX 4070: GPEN-256 ~3 ms, GPEN-512 ~20 ms, RestoreFormer++ ~23 ms, CodeFormer ~24 ms per face. Which looks better on your footage is yours to judge."
          value={restorerOn ? enhancer : ''}
          onChange={(v) => onChange('selected_enhancer', v)}
          options={[...(restorerOn ? [] : [{ value: '', label: 'Choose a restorer…' }]), ...enhancers]}
        />
        <Slider
          label="Original/Enhanced blend"
          info="How much of the restorer's output is mixed back over the swap. 1 = the restorer's face, 0 = the swap as it came out."
          min={0} max={1} step={0.01}
          value={num(p.blend_ratio, d.blend_ratio)}
          defaultValue={d.blend_ratio}
          onChange={(v) => onChange('blend_ratio', v)}
        />
      </Stage>

      <Connector />

      <Stage
        index={3}
        title="Texture & Grain Closer"
        summary={grain > 0 ? `Grain match ${grain.toFixed(2)}` : 'Off'}
        enabled={grain > 0}
        dim={!(grain > 0)}
        onToggle={(on) => {
          if (on) onChange('merger_grain_match', remembered.grain || d.merger_grain_match || 0.45);
          else { remembered.grain = grain; onChange('merger_grain_match', 0); }
        }}
        options={slots.texture}
      >
        <Slider
          label="Grain match"
          info="Adds noise matched to the original footage's own measured noise floor, so the face is not cleaner than everything around it. 0 = off; try 0.5–1.0 on noisy footage."
          min={0} max={1} step={0.05}
          value={grain}
          defaultValue={d.merger_grain_match}
          onChange={(v) => onChange('merger_grain_match', v)}
        />
      </Stage>

      <Connector />

      <Stage
        index={4}
        title="LAB Color & Skin-Tone Match"
        summary={colorOn ? `${String(colorMode).toUpperCase()} · histogram ${num(p.merger_hist_match, d.merger_hist_match).toFixed(2)}` : 'Off'}
        enabled={colorOn}
        dim={!colorOn}
        onToggle={(on) => {
          if (on) onChange('color_transfer_mode', remembered.color || d.color_transfer_mode || 'lct');
          else { remembered.color = colorMode; onChange('color_transfer_mode', 'none'); }
        }}
        options={slots.color}
      >
        <Select
          label="Colour transfer"
          info="RCT = per-channel (fast). LCT = corrects hue casts. MKL = fullest linear match. IDT = matches the full colour distribution, the quality ceiling and much the dearest."
          value={colorOn ? colorMode : ''}
          onChange={(v) => onChange('color_transfer_mode', v)}
          options={[...(colorOn ? [] : [{ value: '', label: 'Choose a method…' }]), ...colorModes.map((m) => ({ value: m, label: m.toUpperCase() }))]}
        />
        <Slider
          label="Histogram match"
          info="Matches the shape of the face's per-channel histogram to the original crop, which catches a face that is subtly flatter or more contrasted than the plate. 0 = off."
          min={0} max={1} step={0.05}
          value={num(p.merger_hist_match, d.merger_hist_match)}
          defaultValue={d.merger_hist_match}
          onChange={(v) => onChange('merger_hist_match', v)}
        />
      </Stage>
    </ol>
  );
}
