import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TRACKER_SLIDERS, TRACKER_DEFAULT_VALUES, TRACKER_GROUPS } from './trackerConfig';
import { Icon } from '../../icons';

const STORAGE_KEY = 'roop_user_slider_presets';
const GROUPS_KEY = 'roop_slider_tracker_collapsed';

const num = (v, fallback) => (typeof v === 'number' && !isNaN(v) ? v : fallback);

// Slider steps are 0.01/0.05, so a value that has been through parseFloat or a
// JSON round trip can sit a float ULP off the table it is compared against.
// Both users of that fact need the same tolerance: a preset would never
// register as "active", and the per-slider "modified" ring would light on a
// slider nobody touched.
const isOff = (a, b) => Math.abs(a - b) >= 1e-6;

const valuesMatch = (params, values) =>
  TRACKER_SLIDERS.every((s) =>
    !isOff(num(params[s.key], s.defaultVal), num(values[s.key], s.defaultVal)));

// Custom presets come from localStorage, which is user-writable and survives
// across versions — anything malformed in there would otherwise crash the whole
// Face Swap tab on the spread below.
const sanitizeCustomPresets = (raw) => {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((p) => p && typeof p.name === 'string' && p.values && typeof p.values === 'object')
    .map((p, i) => ({
      name: p.name,
      values: Object.fromEntries(
        TRACKER_SLIDERS.map((s) => [s.key, num(p.values[s.key], s.defaultVal)])),
      isCustom: true,
      id: String(p.id ?? `legacy-${i}`),
    }));
};

// Spread over the defaults rather than respelling every key: a preset that
// omitted one would leave that slider untouched on apply while `valuesMatch`
// still compared it, so the pill you just clicked would read "Custom". Listing
// only what a preset actually CHANGES also makes the recipe readable.
const preset = (name, overrides) =>
  ({ name, values: { ...TRACKER_DEFAULT_VALUES, ...overrides } });

// ── What the merger values in each preset follow from ─────────────────────
// Three facts decide these; they are not five variations on "some grain".
//
// 1. GRAIN is the one that appears everywhere but Default. Its amount tracks
//    how hard the preset is already pushing, because the cleaner a stage
//    leaves the face the further it has drifted from the plate's noise floor.
//
// 2. HISTOGRAM MATCH PULLS THE FACE TOWARD THE TARGET. That makes it a
//    trade-off, not a quality dial: it is the highest in the presets whose job
//    is to make the face belong to the shot, and deliberately near zero in
//    Strong Likeness, where dragging tonality back toward the target is
//    working against the entire point of the preset.
//
// 3. SHARPEN'S SIGN IS THE PRESET'S TEXTURE. Cinematic and Natural Soft go
//    negative because film and "soft" are not video-crisp; Strong Likeness
//    goes positive to hold the source's detail. Ultra Realism sits at 0 — its
//    job is to match the plate, and matching is not a look.
//
// FACE SIZE IS ZERO IN EVERY PRESET, on purpose. It depends on how this
// source's face compares to this target's, which no preset can know. A number
// there would be a guess wearing a preset's name.

const BUILTIN_PRESETS = [
  { name: 'Default', values: TRACKER_DEFAULT_VALUES },

  // Match the plate as exactly as possible. Grain is highest here because
  // "too clean" is the tell this preset exists to beat; motion blur is on
  // because a real face smears with the camera; sharpen stays neutral.
  preset('Ultra Realism', {
    blend_ratio: 0.85,
    detail_transfer_strength: 0.4,
    expression_restore_strength: 0.8,
    face_mask_blend: 25,
    num_swap_steps: 2,
    jaw_reshape_strength: 0.4,
    stabilize_enhancer_strength: 0.6,
    stabilize_mask_strength: 0.6,
    merger_grain_match: 0.75,
    merger_hist_match: 0.45,
    merger_motion_blur: 0.35,
    merger_sharpen: 0,
    merger_degrade: 0,
  }),

  // A film look, which is a specific set of artefacts: 180-degree shutter
  // (the most motion blur of any preset), grain, and an image that is NOT
  // video-sharp — hence the negative sharpen and a little degrade.
  preset('Cinematic', {
    blend_ratio: 0.9,
    detail_transfer_strength: 0.25,
    expression_restore_strength: 1.2,
    face_mask_blend: 30,
    num_swap_steps: 2,
    stabilize_enhancer_strength: 0.5,
    stabilize_mask_strength: 0.5,
    merger_grain_match: 0.6,
    merger_motion_blur: 0.55,
    merger_sharpen: -0.15,
    merger_degrade: 0.2,
    merger_hist_match: 0.35,
  }),

  // Minimal intervention: every core slider is the lowest of any preset, so
  // the merger half matches. Just enough grain not to read as pasted, and
  // nothing that restyles the face.
  preset('Subtle Touchup', {
    blend_ratio: 0.6,
    detail_transfer_strength: 0.15,
    expression_restore_strength: 0.4,
    face_mask_blend: 15,
    jaw_reshape_strength: 0.2,
    stabilize_enhancer_strength: 0.3,
    stabilize_mask_strength: 0.3,
    merger_grain_match: 0.25,
    merger_hist_match: 0.15,
  }),

  // Maximum source identity — blend 1.0, three swap passes, jaw 0.75. So the
  // merger half must not undo that: histogram match is near zero because it
  // pulls tonality back toward the target, and motion blur and degrade are
  // off because both throw away the likeness detail the preset just paid
  // three passes for. Sharpen goes positive to hold it.
  preset('Strong Likeness', {
    blend_ratio: 1.0,
    detail_transfer_strength: 0.5,
    expression_restore_strength: 0.9,
    num_swap_steps: 3,
    jaw_reshape_strength: 0.75,
    stabilize_enhancer_strength: 0.7,
    stabilize_mask_strength: 0.7,
    merger_grain_match: 0.35,
    merger_hist_match: 0.1,
    merger_sharpen: 0.2,
    merger_motion_blur: 0,
    merger_degrade: 0,
  }),

  // "Soft" is the instruction: the widest mask feather of any preset (40px),
  // so the merger half is the only one with a strongly negative sharpen, plus
  // a touch of degrade. Histogram match is high because "natural" here means
  // sitting in the scene rather than standing out of it.
  preset('Natural Soft', {
    blend_ratio: 0.75,
    detail_transfer_strength: 0.2,
    expression_restore_strength: 0.6,
    face_mask_blend: 40,
    jaw_reshape_strength: 0.3,
    merger_grain_match: 0.4,
    merger_sharpen: -0.35,
    merger_hist_match: 0.4,
    merger_degrade: 0.15,
    merger_motion_blur: 0.2,
  }),
];

// ── One slider ────────────────────────────────────────────────────────────
// Two rows, not four. The old card spent a whole row on a "min / def: x / max"
// footer; min and max now sit inline beside the track and the default is a pip
// ON it, which is where you were looking anyway. The value badge absorbed the
// footer's other job — click it to reset. That is ~48px per slider instead of
// ~92px, which is what makes fourteen of them fit without the bar swallowing
// the workspace.
function TrackerSlider({ slider: s, value, enabled, onSetParam }) {
  const span = s.max - s.min;
  const pct = (v) => Math.max(0, Math.min(100, ((v - s.min) / span) * 100));
  const percent = pct(value);
  const modified = isOff(value, s.defaultVal);
  // A pip is only information when the default sits somewhere you would not
  // guess. At either end of the track it marks the track's own edge, half of
  // it hanging outside the rail — so it is dropped there.
  const showPip = isOff(s.defaultVal, s.min) && isOff(s.defaultVal, s.max);

  return (
    <div
      className={`group/card rounded-lg bg-black/40 border px-2.5 py-2 transition-all duration-200 ${
        modified && enabled
          ? 'border-[var(--accent)]/40 bg-black/60'
          : 'border-white/5 hover:border-white/15'
      }`}
    >
      <div className="flex items-center justify-between gap-1.5">
        <span
          className="text-mini font-semibold text-white/75 truncate group-hover/card:text-white transition-colors"
          title={s.info}
        >
          {s.label}
        </span>
        <button
          type="button"
          onClick={() => enabled && onSetParam && onSetParam(s.key, s.defaultVal)}
          disabled={!enabled || !modified}
          title={modified ? `Reset to ${s.format(s.defaultVal)}` : 'At default'}
          aria-label={modified
            ? `${s.label} is ${s.format(value)}. Reset to ${s.format(s.defaultVal)}`
            : `${s.label} is ${s.format(value)}, the default`}
          className={`shrink-0 text-mini font-mono font-bold tabular-nums px-1.5 py-0.5 rounded border transition-colors ${
            modified && enabled
              ? 'text-[var(--accent)] bg-[var(--accent)]/10 border-[var(--accent)]/30 hover:bg-[var(--accent)]/20 cursor-pointer'
              : 'text-white/45 bg-white/[0.03] border-white/10 cursor-default'
          }`}
        >
          {s.format(value)}
        </button>
      </div>

      <div className="flex items-center gap-1.5 mt-1.5">
        <span className="text-nano font-mono text-white/45 tabular-nums shrink-0">{s.min}</span>
        <div className="relative flex-1 flex items-center">
          <input
            type="range"
            min={s.min}
            max={s.max}
            step={s.step}
            value={value}
            disabled={!enabled}
            aria-label={s.label}
            onChange={(e) => onSetParam && onSetParam(s.key, parseFloat(e.target.value))}
            className="w-full h-1 rounded-lg appearance-none bg-white/10 cursor-pointer accent-[var(--accent)] focus:outline-none focus-visible:ring-1 focus-visible:ring-[var(--accent)] disabled:cursor-not-allowed"
            style={{
              background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${percent}%, rgba(255,255,255,0.1) ${percent}%, rgba(255,255,255,0.1) 100%)`,
            }}
          />
          {/* Where the slider ships. Not a control — the value badge resets. */}
          {showPip && (
            <span
              aria-hidden="true"
              className="pointer-events-none absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-2 w-px bg-white/40"
              style={{ left: `${pct(s.defaultVal)}%` }}
            />
          )}
        </div>
        <span className="text-nano font-mono text-white/45 tabular-nums shrink-0">{s.max}</span>
      </div>
    </div>
  );
}

export default function SliderTrackerBar({
  params = {},
  onSetParam,
  sliderEffectEnabled = true,
  onToggleSliderEffect,
  onResetSliders,
  onRefreshPreview,
}) {
  const [expanded, setExpanded] = useState(true);
  // Per-section folds, remembered across reloads. Stored as the COLLAPSED list
  // so a section added later starts open — storing the open ones would hide
  // every new group from anyone with an existing saved value.
  const [collapsedGroups, setCollapsedGroups] = useState(() => {
    try {
      const raw = JSON.parse(localStorage.getItem(GROUPS_KEY) || '[]');
      return Array.isArray(raw) ? raw.filter((n) => typeof n === 'string') : [];
    } catch {
      return [];
    }
  });
  const [customPresets, setCustomPresets] = useState([]);
  const [isSaving, setIsSaving] = useState(false);
  const [newPresetName, setNewPresetName] = useState('');

  // Load custom presets from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) setCustomPresets(sanitizeCustomPresets(JSON.parse(saved)));
    } catch {
      /* ignore */
    }
  }, []);

  // Save custom presets to localStorage
  const saveCustomPresetsToStorage = (updated) => {
    setCustomPresets(updated);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch {
      /* ignore */
    }
  };

  const toggleGroup = (name) => {
    setCollapsedGroups((prev) => {
      const next = prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name];
      try {
        localStorage.setItem(GROUPS_KEY, JSON.stringify(next));
      } catch {
        /* ignore */
      }
      return next;
    });
  };

  const allPresets = useMemo(() => [...BUILTIN_PRESETS, ...customPresets], [customPresets]);

  // DERIVED, never stored. As state it went stale in four ways: it started at
  // 'Default' regardless of the loaded config, it did not follow edits made in
  // the main settings panel, clicking every "def:" pip left it reading 'Custom',
  // and "↺ Reset" and preset-delete had to remember to correct it by hand.
  // Reading it off `params` means the highlight is always the truth.
  const activePreset = useMemo(() => {
    const hit = allPresets.find((pr) => valuesMatch(params, pr.values));
    return hit ? hit.name : 'Custom';
  }, [params, allPresets]);

  const applyPreset = (pObj) => {
    Object.entries(pObj.values).forEach(([k, v]) => {
      if (onSetParam) onSetParam(k, v);
    });
    if (onRefreshPreview) onRefreshPreview();
  };

  const handleSaveCurrentPreset = () => {
    // Strip any ★ the user typed (or pasted from an existing pill's label) so
    // re-saving "★ mine" cannot produce "★ ★ mine".
    const name = newPresetName.replace(/^[★\s]+/, '').trim();
    if (!name) return;

    const values = Object.fromEntries(
      TRACKER_SLIDERS.map((s) => [s.key, num(params[s.key], s.defaultVal)]));

    const newPreset = {
      name: `★ ${name}`,
      values,
      isCustom: true,
      // Date.now() alone collides when two presets are saved inside the same
      // millisecond, which would make one delete remove both.
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    };

    saveCustomPresetsToStorage(
      [...customPresets.filter((p) => p.name !== newPreset.name), newPreset]);
    setNewPresetName('');
    setIsSaving(false);
  };

  const handleDeleteCustomPreset = (presetId, e) => {
    e.stopPropagation();
    saveCustomPresetsToStorage(customPresets.filter((p) => p.id !== presetId));
  };

  return (
    <div className="w-full rounded-2xl glass-panel p-4 mb-4 border border-white/10 shadow-2xl transition-all duration-300 relative overflow-hidden group">
      {/* Subtle top glow line */}
      <div
        className={`absolute top-0 inset-x-0 h-0.5 transition-colors duration-500 ${
          sliderEffectEnabled
            ? 'bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent opacity-80'
            : 'bg-white/10 opacity-30'
        }`}
      />

      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 select-none">
        {/* Left title & active status badge */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-extrabold uppercase tracking-[0.16em] text-white/90">
              Slider Tracker
            </span>
          </div>

          {/* Toggle status indicator badge */}
          <button
            type="button"
            onClick={onToggleSliderEffect}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-micro font-bold tracking-wider uppercase border transition-all duration-200 cursor-pointer ${
              sliderEffectEnabled
                ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.25)]'
                : 'bg-amber-500/15 border-amber-500/40 text-amber-300'
            }`}
            title="Click to toggle slider effects ON or OFF"
          >
            <span
              className={`h-2 w-2 rounded-full ${
                sliderEffectEnabled
                  ? 'bg-emerald-400 animate-pulse shadow-[0_0_6px_#10b981]'
                  : 'bg-amber-400'
              }`}
            />
            {sliderEffectEnabled ? 'Slider Effect: ON' : 'Slider Effect: BYPASSED'}
          </button>
        </div>

        {/* Right action controls */}
        <div className="flex items-center gap-2">
          {/* Master Slider Effect Switch */}
          <button
            type="button"
            onClick={onToggleSliderEffect}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all duration-200 cursor-pointer ${
              sliderEffectEnabled
                ? 'bg-[var(--accent)]/20 border-[var(--accent)]/50 text-white shadow-[0_0_14px_var(--accent-glow)]'
                : 'bg-white/[0.04] border-white/10 text-white/50 hover:border-white/20 hover:text-white/80'
            }`}
          >
            <div
              className={`w-7 h-4 rounded-full p-0.5 transition-colors duration-200 ${
                sliderEffectEnabled ? 'bg-[var(--accent)]' : 'bg-white/20'
              }`}
            >
              <div
                className={`w-3 h-3 rounded-full bg-white transition-transform duration-200 ${
                  sliderEffectEnabled ? 'translate-x-3' : 'translate-x-0'
                }`}
              />
            </div>
            <span>Effect {sliderEffectEnabled ? 'ON' : 'OFF'}</span>
          </button>

          {/* Reset sliders button */}
          {onResetSliders && (
            <button
              type="button"
              onClick={onResetSliders}
              className="px-2.5 py-1.5 rounded-xl text-mini font-semibold bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 text-white/60 hover:text-white transition-colors"
              title="Reset all trackers to default values"
            >
              ↺ Reset
            </button>
          )}

          {/* Refresh preview button */}
          {onRefreshPreview && (
            <button
              type="button"
              onClick={onRefreshPreview}
              className="px-2.5 py-1.5 rounded-xl text-mini font-semibold bg-[var(--accent)]/10 hover:bg-[var(--accent)]/20 border border-[var(--accent)]/30 text-[var(--accent)] transition-colors"
              title="Force re-render preview with current slider tracker settings"
            >
              Refresh Preview
            </button>
          )}

          {/* Collapse/Expand button */}
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="px-2.5 py-1.5 rounded-xl text-mini font-semibold bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 text-white/70 transition-colors flex items-center gap-1"
          >
            <span>{expanded ? 'Collapse' : 'Expand'}</span>
            <svg
              className={`w-3 h-3 transition-transform duration-200 ${
                expanded ? 'rotate-180' : 'rotate-0'
              }`}
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>
        </div>
      </div>

      {/* Preset Profiles Selector & Save Strip */}
      {expanded && (
        <div className="flex flex-wrap items-center justify-between gap-2 mt-3 pt-3 border-t border-white/5 select-none">
          {/* Left preset pills list */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-micro font-bold uppercase tracking-wider text-white/45 mr-1">
              Presets:
            </span>
            {activePreset === 'Custom' && (
              <span
                className="px-2 py-1 rounded-lg text-micro font-bold border border-dashed border-white/25 text-white/60 bg-white/[0.03]"
                title="Current slider values do not match any saved preset"
              >
                Custom
              </span>
            )}
            {allPresets.map((p) => {
              const isSel = activePreset === p.name;
              return (
                // The delete control is a SIBLING of the preset button, not a
                // child of it: a disabled <button> swallows clicks on its
                // descendants, so nesting it made "×" dead whenever the effect
                // was bypassed — and a button inside a button is invalid markup.
                <div
                  key={p.id || p.name}
                  className={`relative inline-flex items-center rounded-lg border transition-all duration-200 ${
                    isSel && sliderEffectEnabled
                      ? 'bg-[var(--accent)] text-white border-white/20 shadow-[0_0_10px_var(--accent-glow)]'
                      : 'bg-white/[0.03] border-white/10 text-white/70 hover:border-white/20 hover:text-white'
                  } ${sliderEffectEnabled ? '' : 'opacity-35'}`}
                >
                  <button
                    type="button"
                    onClick={() => applyPreset(p)}
                    disabled={!sliderEffectEnabled}
                    className="px-2.5 py-1 text-micro font-bold disabled:cursor-not-allowed"
                  >
                    {p.name}
                  </button>
                  {p.isCustom && (
                    <button
                      type="button"
                      onClick={(e) => handleDeleteCustomPreset(p.id, e)}
                      className="mr-1.5 px-1 rounded bg-black/40 hover:text-red-300 transition-colors"
                      title="Delete custom preset"
                      aria-label={`Delete custom preset ${p.name}`}
                    >
                      <Icon.close size={11} />
                    </button>
                  )}
                </div>
              );
            })}
          </div>

          {/* Right Save Preset Button / Inline Input */}
          <div className="flex items-center gap-1.5">
            {isSaving ? (
              <div className="flex items-center gap-1.5 bg-black/60 p-1 rounded-xl border border-[var(--accent)]/40 shadow-lg">
                <input
                  type="text"
                  placeholder="Preset Name..."
                  value={newPresetName}
                  onChange={(e) => setNewPresetName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSaveCurrentPreset()}
                  className="px-2 py-1 rounded-lg bg-black/80 text-white text-mini border border-white/10 focus:outline-none focus:border-[var(--accent)]"
                  autoFocus
                />
                <button
                  type="button"
                  onClick={handleSaveCurrentPreset}
                  disabled={!newPresetName.trim()}
                  className="px-2.5 py-1 rounded-lg text-micro font-bold bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Save
                </button>
                <button
                  type="button"
                  onClick={() => setIsSaving(false)}
                  className="px-2 py-1 rounded-lg text-micro font-bold bg-white/10 text-white/60 hover:text-white transition-colors"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setIsSaving(true)}
                disabled={!sliderEffectEnabled}
                className="px-2.5 py-1 rounded-lg text-micro font-bold bg-[var(--accent)]/15 hover:bg-[var(--accent)]/25 border border-[var(--accent)]/40 text-white transition-all shadow-[0_0_10px_var(--accent-glow)] flex items-center gap-1 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                title="Save current slider values as a custom preset"
              >
                <span>Save Preset</span>
              </button>
            )}
          </div>
        </div>
      )}

      {/* Grid of slider controls */}
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div
              className={`mt-3 pt-3 border-t border-white/5 transition-opacity duration-300 ${
                sliderEffectEnabled ? 'opacity-100' : 'opacity-55 grayscale-[25%]'
              }`}
            >
              {TRACKER_GROUPS.map((g) => {
                const collapsed = collapsedGroups.includes(g.name);
                const activeCount = g.sliders.filter(
                  (s) => isOff(num(params[s.key], s.defaultVal), s.defaultVal)).length;

                return (
                  <section key={g.name} className="mt-2 first:mt-0">
                    <button
                      type="button"
                      onClick={() => toggleGroup(g.name)}
                      aria-expanded={!collapsed}
                      className="w-full flex items-center gap-2 py-1 text-left group/sec"
                    >
                      <svg
                        className={`w-2.5 h-2.5 shrink-0 text-white/40 transition-transform duration-200 ${
                          collapsed ? '-rotate-90' : 'rotate-0'
                        }`}
                        viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"
                      >
                        <polyline points="6 9 12 15 18 9" />
                      </svg>
                      <span className="text-micro font-bold uppercase tracking-[0.14em] text-white/45 group-hover/sec:text-white/70 transition-colors">
                        {g.name}
                      </span>
                      {/* Survives collapsing: a folded section must still say
                          that something inside it is doing work, or a knob can
                          be left on with nothing on screen admitting it. */}
                      {activeCount > 0 && (
                        <span className="px-1.5 py-px rounded-full text-nano font-bold tabular-nums bg-[var(--accent)]/15 text-[var(--accent)] border border-[var(--accent)]/30">
                          {activeCount} active
                        </span>
                      )}
                      <span className="flex-1 h-px bg-white/5" />
                      <span className="text-nano font-mono text-white/45 tabular-nums">
                        {g.sliders.length}
                      </span>
                    </button>

                    {/* The grid stops at 4 columns until the ultrawide
                        breakpoints. The bar sits between the two side panels,
                        so its container is well under the viewport width — six
                        columns at 2xl left about 105px for the label, which
                        truncates "Original / Enhanced Blend" to nothing useful.
                        3xl/4xl are this project's own ultrawide steps
                        (1920/2560px), added for exactly this kind of layout. */}
                    {!collapsed && (
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 3xl:grid-cols-5 4xl:grid-cols-6 gap-2 mt-1.5">
                        {g.sliders.map((s) => (
                          <TrackerSlider
                            key={s.key}
                            slider={s}
                            value={num(params[s.key], s.defaultVal)}
                            enabled={sliderEffectEnabled}
                            onSetParam={onSetParam}
                          />
                        ))}
                      </div>
                    )}
                  </section>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
