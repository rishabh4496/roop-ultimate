import React, { useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence, spring } from '../motion';
import { Icon } from '../icons';
import { allThemes, applyThemeToDom } from '../themes';
import {
  RECIPE_LIMITS, normalizeRecipe, deriveThemeVars, contrastRatio, luminance,
} from '../themeVars';

// ── Theme Studio ──────────────────────────────────────────────────────────
// Authors a custom theme from the six decisions that actually matter, and
// previews it live on the real page rather than in a swatch — the whole point
// of a theme is how the app feels wearing it, and a 12px chip cannot tell you
// that. Cancelling restores whatever was applied when the studio opened.

const HUE_PRESETS = [
  '#E94560', '#f472a0', '#f59e0b', '#a3e635', '#22c55e',
  '#10b981', '#00bbf9', '#38bdf8', '#6366f1', '#9b5de5', '#d4d4d8',
];

// A labelled row that keeps the studio's controls visually identical to the
// rest of Settings without dragging in the Field/Slider primitives, which
// render their own label block and would double up here.
const Row = ({ label, hint, children }) => (
  <div className="flex items-center justify-between gap-4 py-1.5">
    <div className="min-w-0">
      <div className="text-compact font-semibold text-white/80">{label}</div>
      {hint && <div className="text-nano text-white/45 mt-0.5 leading-snug">{hint}</div>}
    </div>
    <div className="shrink-0 flex items-center gap-2">{children}</div>
  </div>
);

// Native colour input, restyled. `type=color` gives us the OS picker (eyedropper
// included on Windows/macOS) for free — a hand-rolled HSV square would be more
// code and less capable.
const ColorWell = ({ value, onChange, label }) => (
  <label className="relative h-8 w-14 rounded-lg overflow-hidden border border-white/15 cursor-pointer block shrink-0 hover:border-white/35 transition-colors">
    <span className="absolute inset-0" style={{ background: value }} />
    <input
      type="color"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      aria-label={label}
      className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
    />
  </label>
);

const NumberSlider = ({ value, onChange, min, max, step, format }) => (
  <>
    <input
      type="range"
      min={min} max={max} step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-36 accent-[var(--accent)] h-1.5"
    />
    <span className="w-10 text-right text-mini font-semibold tabular-nums text-white/55">
      {format(value)}
    </span>
  </>
);

// Saving/deleting is reported by the caller (it owns the settings write), so
// this component takes no `notify`.
export default function ThemeStudio({ open, onClose, initial, customThemes, onSave, onDelete }) {
  const [recipe, setRecipe] = useState(() => normalizeRecipe(initial));
  // The theme that was live when the studio opened, so Cancel is a true undo.
  const restoreRef = useRef(null);
  const editingName = initial?.name;

  useEffect(() => {
    if (!open) return undefined;
    setRecipe(normalizeRecipe(initial));
    restoreRef.current = null;
    return undefined;
  }, [open, initial]);

  // Live preview: paint the recipe onto the document while the studio is open.
  // Restoring on close (rather than on every keystroke) keeps this to one DOM
  // write per edit and makes Cancel exact.
  useEffect(() => {
    if (!open) return undefined;
    if (!restoreRef.current) {
      const root = document.documentElement;
      restoreRef.current = {
        classes: Array.from(root.classList).filter((c) => c.startsWith('theme-')),
        mode: root.getAttribute('data-theme-mode'),
        inline: root.getAttribute('style') || '',
      };
    }
    applyThemeToDom(recipe);
    return undefined;
  }, [open, recipe]);

  const restore = () => {
    const snap = restoreRef.current;
    restoreRef.current = null;
    if (!snap) return;
    const root = document.documentElement;
    const { body } = document;
    // Re-apply the exact pre-open state. Going through the DOM rather than
    // re-deriving from settings avoids a flash when the caller is about to
    // re-render with a new theme anyway.
    Array.from(root.classList).filter((c) => c.startsWith('theme-')).forEach((c) => {
      root.classList.remove(c); body.classList.remove(c);
    });
    root.setAttribute('style', snap.inline);
    snap.classes.forEach((c) => { root.classList.add(c); body.classList.add(c); });
    if (snap.mode) root.setAttribute('data-theme-mode', snap.mode);
  };

  const set = (k, v) => setRecipe((r) => ({ ...r, [k]: v }));

  // Name collisions matter: `name` is the id a theme is selected by, so two
  // themes sharing one would make the second unreachable.
  const nameTaken = useMemo(() => {
    const n = (recipe.name || '').trim().toLowerCase();
    if (!n) return false;
    return allThemes(customThemes)
      .filter((t) => t.name !== editingName)
      .some((t) => t.name.toLowerCase() === n);
  }, [recipe.name, customThemes, editingName]);

  // The recipe's declared mode drives the light-theme ink correction. If the
  // user picks a pale background but leaves mode on dark, every label turns
  // white-on-white — so say so instead of shipping an unreadable theme.
  const modeMismatch = useMemo(() => {
    const l = luminance(recipe.bg);
    if (recipe.mode === 'dark' && l > 0.35) return 'That background is light, but the mode is Dark — text will be hard to read. Switch to Light.';
    if (recipe.mode === 'light' && l < 0.12) return 'That background is dark, but the mode is Light — text will be hard to read. Switch to Dark.';
    return '';
  }, [recipe.bg, recipe.mode]);

  // Accent-on-background legibility, reported plainly rather than enforced —
  // the accent is used for fills and hairlines as well as text.
  const accentContrast = useMemo(
    () => Math.round(contrastRatio(recipe.accent, recipe.bg) * 10) / 10,
    [recipe.accent, recipe.bg],
  );

  const canSave = (recipe.name || '').trim().length > 0 && !nameTaken;

  const save = () => {
    if (!canSave) return;
    restoreRef.current = null;   // keep the preview — it is what we just saved
    onSave({ ...recipe, name: recipe.name.trim() }, editingName);
  };

  const cancel = () => { restore(); onClose(); };

  if (!open) return null;

  const vars = deriveThemeVars(recipe);

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-[80] flex items-center justify-center p-4"
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      >
        <div
          className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          onClick={cancel}
          aria-hidden
        />
        <motion.div
          role="dialog"
          aria-modal="true"
          aria-label="Theme Studio"
          initial={{ opacity: 0, y: 24, scale: 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 12, scale: 0.98 }}
          transition={spring.snappy}
          onKeyDown={(e) => { if (e.key === 'Escape') { e.stopPropagation(); cancel(); } }}
          className="relative w-full max-w-lg max-h-[88vh] overflow-y-auto rounded-2xl glass-panel p-6 shadow-2xl"
        >
          <div className="flex items-start justify-between gap-3 mb-5">
            <div>
              <h2 className="text-lead font-bold text-white/95 flex items-center gap-2">
                <Icon.theme size={16} className="text-[var(--accent)]" />
                Theme Studio
              </h2>
              <p className="text-mini text-white/45 mt-1">
                Six choices; everything else is derived. The page behind this dialog is the preview.
              </p>
            </div>
            <button
              type="button" onClick={cancel} aria-label="Close theme studio"
              className="h-8 w-8 grid place-items-center rounded-lg text-white/45 hover:text-white hover:bg-white/10 transition-colors shrink-0"
            ><Icon.close size={15} /></button>
          </div>

          <div className="space-y-1 divide-y divide-white/[0.06]">
            <Row label="Name">
              <input
                type="text"
                value={recipe.name}
                onChange={(e) => set('name', e.target.value)}
                aria-label="Theme name"
                aria-invalid={nameTaken || undefined}
                className={`w-44 px-3 py-1.5 rounded-lg glass-input text-white text-compact focus:outline-none ${nameTaken ? 'border-red-500/60' : ''}`}
              />
            </Row>

            <Row label="Mode" hint="Sets how text and hairlines are derived.">
              <div className="flex rounded-lg bg-white/[0.05] border border-white/10 p-0.5">
                {['dark', 'light'].map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => set('mode', m)}
                    aria-pressed={recipe.mode === m}
                    className={`px-3 py-1 rounded-md text-mini font-semibold capitalize transition-colors ${
                      recipe.mode === m ? 'bg-[var(--accent)] text-white' : 'text-white/50 hover:text-white/85'
                    }`}
                  >{m}</button>
                ))}
              </div>
            </Row>

            <Row label="Accent" hint={`Contrast against the background: ${accentContrast}:1`}>
              <ColorWell value={recipe.accent} onChange={(v) => set('accent', v)} label="Accent colour" />
            </Row>

            <div className="flex flex-wrap gap-1.5 py-2">
              {HUE_PRESETS.map((h) => (
                <button
                  key={h}
                  type="button"
                  onClick={() => set('accent', h)}
                  aria-label={`Use accent ${h}`}
                  title={h}
                  className={`h-6 w-6 rounded-full border transition-transform hover:scale-110 ${
                    recipe.accent.toLowerCase() === h.toLowerCase() ? 'border-white/80 scale-110' : 'border-white/15'
                  }`}
                  style={{ background: h }}
                />
              ))}
            </div>

            <Row label="Background" hint="Every neutral in the theme is mixed from this.">
              <ColorWell value={recipe.bg} onChange={(v) => set('bg', v)} label="Background colour" />
            </Row>

            <Row label="Surface" hint="Panel fill opacity — the glass strength.">
              <NumberSlider
                value={recipe.surface} onChange={(v) => set('surface', v)}
                {...RECIPE_LIMITS.surface}
                format={(v) => `${Math.round(v * 100)}%`}
              />
            </Row>

            <Row label="Hairline" hint="Border strength on panels and inputs.">
              <NumberSlider
                value={recipe.border} onChange={(v) => set('border', v)}
                {...RECIPE_LIMITS.border}
                format={(v) => `${Math.round(v * 100)}%`}
              />
            </Row>

            <Row label="Corner radius">
              <NumberSlider
                value={recipe.radius} onChange={(v) => set('radius', v)}
                {...RECIPE_LIMITS.radius}
                format={(v) => `${v}px`}
              />
            </Row>
          </div>

          {/* A compact read-out of what the recipe expanded into, so the
              derivation is inspectable rather than magic. */}
          <div className="mt-4 flex flex-wrap gap-1.5">
            {['--card-bg', '--text-main', '--text-muted', '--border-color', '--accent-hover'].map((k) => (
              <span key={k} className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-white/[0.04] border border-white/10">
                <span className="h-3 w-3 rounded-sm border border-white/20" style={{ background: vars[k] }} />
                <span className="text-nano font-mono text-white/45">{k.replace('--', '')}</span>
              </span>
            ))}
          </div>

          {(nameTaken || modeMismatch) && (
            <div role="alert" className="mt-4 space-y-2">
              {nameTaken && (
                <p className="text-mini text-red-300 flex items-start gap-1.5">
                  <Icon.warning size={13} className="mt-px shrink-0" />
                  A theme called “{recipe.name.trim()}” already exists. Names select the theme, so they must be unique.
                </p>
              )}
              {modeMismatch && (
                <p className="text-mini text-amber-300/90 flex items-start gap-1.5">
                  <Icon.warning size={13} className="mt-px shrink-0" />
                  {modeMismatch}
                </p>
              )}
            </div>
          )}

          <div className="mt-6 flex items-center justify-between gap-2">
            {editingName ? (
              <button
                type="button"
                onClick={() => { restore(); onDelete(editingName); }}
                className="px-3 py-2 rounded-lg text-mini font-semibold text-red-300 bg-red-500/10 border border-red-500/25 hover:bg-red-500/20 transition-colors"
              >Delete</button>
            ) : <span />}
            <div className="flex items-center gap-2">
              <button
                type="button" onClick={cancel}
                className="px-4 py-2 rounded-lg text-mini font-semibold text-white/65 hover:text-white bg-white/[0.05] border border-white/10 hover:bg-white/10 transition-colors"
              >Cancel</button>
              <button
                type="button" onClick={save} disabled={!canSave}
                className="px-4 py-2 rounded-lg text-mini font-bold text-white bg-[var(--accent)] hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
              >{editingName ? 'Save changes' : 'Create theme'}</button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
