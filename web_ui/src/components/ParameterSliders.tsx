import type { ReactNode } from "react";
import {
  ENHANCER_MODELS,
  MASK_TYPES,
  PIXEL_BOOSTS,
  PROVIDERS,
  SWAPPER_MODELS,
  type ExecutionProvider,
  type MaskType,
  type Preset,
  type RenderParams,
} from "../lib/types";

export interface ParameterSlidersProps {
  params: RenderParams;
  /** Hardware profile presets from GET /api/options (with measured fps). */
  presets?: Record<string, Preset>;
  onChange: (patch: Partial<RenderParams>) => void;
  /** model name -> reason it is unavailable */
  unavailable: Record<string, string>;
  providers: Record<ExecutionProvider, boolean>;
  running: boolean;
  canStart: boolean;
  onStart: () => void;
  onStop: () => void;
  onPreview: () => void;
  previewBusy?: boolean;
}

const BOOST_LABELS: Record<RenderParams["pixel_boost"], string> = {
  none: "none",
  "256x256": "256",
  "512x512": "512",
  "1024x1024": "1024",
};

/** True when every parameter the preset sets has the preset's value. */
export function matchesPreset(params: RenderParams, preset: Preset): boolean {
  return Object.entries(preset.params).every(([key, value]) => {
    const current = params[key as keyof RenderParams];
    return Array.isArray(value)
      ? Array.isArray(current) && [...current].sort().join() === [...value].sort().join()
      : current === value;
  });
}

function PresetButtons({
  presets,
  params,
  disabled,
  onApply,
}: {
  presets: Record<string, Preset>;
  params: RenderParams;
  disabled: boolean;
  onApply: (patch: Partial<RenderParams>) => void;
}) {
  return (
    <div role="group" aria-label="Hardware profile presets" className="grid grid-cols-3 gap-2">
      {Object.entries(presets).map(([key, preset]) => {
        const active = matchesPreset(params, preset);
        return (
          <button
            key={key}
            type="button"
            aria-pressed={active}
            disabled={disabled}
            onClick={() => onApply(preset.params)}
            className={`flex flex-col items-start rounded border px-2 py-1.5 text-left ${
              active ? "border-sky-400 bg-sky-950/60" : "border-zinc-700 hover:bg-zinc-800"
            } disabled:opacity-40`}
          >
            <span className="text-xs font-semibold">{preset.label}</span>
            <span className="text-[10px] text-zinc-400" data-testid={`preset-fps-${key}`}>
              {preset.measured_fps == null ? "not measured yet" : `~${preset.measured_fps.toFixed(0)} fps measured`}
            </span>
          </button>
        );
      })}
    </div>
  );
}

const MASK_LABELS: Record<MaskType, string> = {
  box: "Box (feathered)",
  occlusion: "Occlusion (XSeg)",
  region: "Face regions (BiSeNet)",
};

const PADDING_SIDES = [
  ["mask_padding_top", "Top"],
  ["mask_padding_bottom", "Bottom"],
  ["mask_padding_left", "Left"],
  ["mask_padding_right", "Right"],
] as const;

function Field({ label, htmlFor, children }: { label: string; htmlFor: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={htmlFor} className="text-xs uppercase tracking-wide text-zinc-400">
        {label}
      </label>
      {children}
    </div>
  );
}

const selectClass =
  "w-full min-w-0 rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm focus:border-sky-500 focus:outline-none disabled:opacity-50";

function Slider(props: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  disabled?: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs text-zinc-400">
        <label htmlFor={props.id}>{props.label}</label>
        <span aria-live="polite">{props.format(props.value)}</span>
      </div>
      <input
        id={props.id}
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        disabled={props.disabled}
        onChange={(e) => props.onChange(Number(e.target.value))}
        className="accent-sky-500 disabled:opacity-40"
      />
    </div>
  );
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

/**
 * Render parameters: presets, models, Pixel Boost, masks, blends and the
 * fine controls, plus Preview / Start / Stop. Everything is disabled while a
 * render runs (the job's parameters are fixed once it starts).
 */
export function ParameterSliders(props: ParameterSlidersProps) {
  const { params, onChange, running } = props;
  const locked = running;

  const toggleMask = (mask: MaskType, on: boolean) => {
    const next = on
      ? Array.from(new Set([...params.mask_types, mask]))
      : params.mask_types.filter((m) => m !== mask);
    onChange({ mask_types: MASK_TYPES.filter((m) => next.includes(m)) });
  };

  return (
    <section aria-label="Pipeline controls" className="flex flex-col gap-4">
      {props.presets && Object.keys(props.presets).length > 0 && (
        <PresetButtons presets={props.presets} params={params} disabled={locked} onApply={onChange} />
      )}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Field label="Swapper" htmlFor="swapper">
          <select
            id="swapper"
            className={selectClass}
            value={params.swapper_model}
            disabled={locked}
            onChange={(e) => onChange({ swapper_model: e.target.value as RenderParams["swapper_model"] })}
          >
            {SWAPPER_MODELS.map((m) => (
              <option key={m} value={m} disabled={m in props.unavailable} title={props.unavailable[m]}>
                {m}
                {m in props.unavailable ? " (unavailable)" : ""}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Pixel boost" htmlFor="pixel-boost">
          <select
            id="pixel-boost"
            className={selectClass}
            value={params.pixel_boost}
            disabled={locked}
            onChange={(e) => onChange({ pixel_boost: e.target.value as RenderParams["pixel_boost"] })}
          >
            {PIXEL_BOOSTS.map((b) => (
              <option key={b} value={b}>
                {BOOST_LABELS[b]}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Enhancer" htmlFor="enhancer">
          <select
            id="enhancer"
            className={selectClass}
            value={params.enhancer_model}
            disabled={locked}
            onChange={(e) => onChange({ enhancer_model: e.target.value as RenderParams["enhancer_model"] })}
          >
            {ENHANCER_MODELS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Execution provider" htmlFor="provider">
          <select
            id="provider"
            className={selectClass}
            value={params.execution_provider}
            disabled={locked}
            onChange={(e) => onChange({ execution_provider: e.target.value as ExecutionProvider })}
          >
            {PROVIDERS.map((p) => (
              <option key={p} value={p} disabled={!props.providers[p]}>
                {p}
                {props.providers[p] ? "" : " (not in this onnxruntime)"}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <Slider
        id="enhancer-blend"
        label="Enhancer blend"
        value={params.enhancer_blend}
        min={0}
        max={100}
        step={1}
        format={(v) => `${v}%`}
        disabled={locked || params.enhancer_model === "none"}
        onChange={(v) => onChange({ enhancer_blend: v })}
      />

      <fieldset className="flex flex-col gap-2" disabled={locked}>
        <legend className="mb-1 text-xs uppercase tracking-wide text-zinc-400">Masks</legend>
        {MASK_TYPES.map((mask) => (
          <label key={mask} className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={params.mask_types.includes(mask)}
              onChange={(e) => toggleMask(mask, e.target.checked)}
              className="accent-sky-500"
            />
            {MASK_LABELS[mask]}
          </label>
        ))}
      </fieldset>

      <div className="grid grid-cols-2 gap-3">
        {PADDING_SIDES.map(([key, label]) => (
          <Slider
            key={key}
            id={key}
            label={`Padding ${label.toLowerCase()}`}
            value={params[key]}
            min={0}
            max={0.5}
            step={0.01}
            format={pct}
            disabled={locked || !params.mask_types.includes("box")}
            onChange={(v) => onChange({ [key]: v })}
          />
        ))}
      </div>
      <Slider
        id="detection-stride"
        label="Detection stride (render)"
        value={params.detection_stride}
        min={1}
        max={8}
        step={1}
        format={(v) => (v === 1 ? "every frame" : `every ${v} frames`)}
        disabled={locked}
        onChange={(v) => onChange({ detection_stride: v })}
      />
      <Slider
        id="mask-blur"
        label="Gaussian feathering"
        value={params.mask_blur}
        min={0}
        max={1}
        step={0.01}
        format={pct}
        disabled={locked || !params.mask_types.includes("box")}
        onChange={(v) => onChange({ mask_blur: v })}
      />

      <div className="flex flex-wrap gap-2 pt-1">
        <button
          type="button"
          onClick={props.onPreview}
          disabled={locked || !props.canStart || props.previewBusy}
          className="rounded border border-zinc-600 px-3 py-1.5 text-sm hover:bg-zinc-800 disabled:opacity-40"
        >
          {props.previewBusy ? "Rendering preview…" : "Preview frame"}
        </button>
        {running ? (
          <button
            type="button"
            onClick={props.onStop}
            className="rounded bg-red-700 px-4 py-1.5 text-sm font-semibold hover:bg-red-600"
          >
            Stop render
          </button>
        ) : (
          <button
            type="button"
            onClick={props.onStart}
            disabled={!props.canStart}
            className="rounded bg-sky-600 px-4 py-1.5 text-sm font-semibold hover:bg-sky-500 disabled:opacity-40"
          >
            Start render
          </button>
        )}
      </div>
    </section>
  );
}
