import React, { useState, useEffect, useRef } from 'react';
import { PERSON_COLORS } from './constants';
import { motion, AnimatePresence, fadeUp, spring, useTilt, TiltGlare } from '../motion';
import { Icon } from '../icons';

// Panels rise in on mount (cinematic reveal) and lift on hover.
//
// ── Elevation ─────────────────────────────────────────────────────────────
// Tilt + cursor glare used to be the DEFAULT for every Card, which meant every
// panel in the app — thirty-odd form sections, dialogs, list rows — tilted
// under the cursor. An effect applied to everything stops reading as emphasis
// and starts reading as noise; it also fought the controls inside form panels,
// which is why the one place that cared had already opted out by hand.
//
// So elevation is now explicit, and there are only three kinds of surface:
//
//   panel (default) — still. Hairline border, quiet hover, resting specular.
//                     This is what a form or a list belongs in.
//   hero            — panel plus the mouse-follow tilt and cursor-tracking
//                     accent glare. For DISPLAY surfaces (media tiles), where
//                     depth is the point and there is nothing to click through.
//   flat            — no hover response at all, for cards nested in something
//                     that already responds.
//
// `tilt` / `glare` remain as explicit overrides so a call site can dissent from
// its elevation without inventing a fourth level. Both collapse to nothing
// under reduced-motion, and the tilt runs on motion values, so hovering never
// re-renders the card. Pass animate={false} to opt out of the entrance.
const ELEVATION = {
  panel: { tilt: false, glare: false, hover: true },
  hero: { tilt: true, glare: true, hover: true },
  flat: { tilt: false, glare: false, hover: false },
};
export const Card = ({
  children, className = '', animate = true, hover, elevation = 'panel',
  tilt, tiltMax = 3.5, glare,
  onMouseMove, onMouseLeave, ...rest
}) => {
  const t = useTilt({ max: tiltMax });
  // A lookup rather than a chain of comparisons, for the same reason Button's
  // sizes are one: an unknown key is then a single well-defined fallback, and
  // test_ui_primitive_props.py can check every call site against these names.
  const e = ELEVATION[elevation] || ELEVATION.panel;
  // An explicit prop wins over the elevation's own answer.
  const wantTilt = tilt ?? e.tilt;
  const wantGlare = glare ?? e.glare;
  const wantHover = hover ?? e.hover;
  const tiltOn = wantTilt && !t.reduce;
  const glareOn = wantGlare && !t.reduce;
  // The pointer handlers are needed whenever either is on, since the glare
  // also tracks the cursor.
  const active = tiltOn || glareOn;
  return (
    <motion.div
      ref={active ? t.ref : undefined}
      onMouseMove={active ? (e) => { t.onMouseMove(e); onMouseMove?.(e); } : onMouseMove}
      onMouseLeave={active ? (e) => { t.onMouseLeave(e); onMouseLeave?.(e); } : onMouseLeave}
      style={tiltOn ? t.style : undefined}
      className={`rounded-2xl glass-panel ${wantHover ? 'card-lift' : ''} ${active ? 'group/spot relative' : ''} ${className}`}
      variants={fadeUp}
      initial={animate ? 'hidden' : false}
      animate={animate ? 'show' : false}
      {...rest}
    >
      {children}
      {glareOn && <TiltGlare glareBg={t.glareBg} group="spot" />}
    </motion.div>
  );
};

const MOTION_ICON_SIZES = {
  sm: { container: 'h-6 w-6 rounded-lg', iconSize: 12 },
  md: { container: 'h-8 w-8 rounded-xl', iconSize: 16 },
  lg: { container: 'h-10 w-10 rounded-xl', iconSize: 20 },
  xl: { container: 'h-12 w-12 rounded-2xl', iconSize: 24 },
};

const MOTION_ICON_VARIANTS = {
  accent: 'bg-[var(--accent)]/15 border-[var(--accent)]/30 text-[var(--accent)] shadow-[0_0_12px_rgba(233,69,96,0.25)]',
  emerald: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.25)]',
  amber: 'bg-amber-500/15 border-amber-500/30 text-amber-400 shadow-[0_0_12px_rgba(251,191,36,0.25)]',
  purple: 'bg-purple-500/15 border-purple-500/30 text-purple-400 shadow-[0_0_12px_rgba(168,85,247,0.25)]',
  cyan: 'bg-cyan-500/15 border-cyan-500/30 text-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.25)]',
  subtle: 'bg-white/[0.05] border-white/10 text-white/70 hover:text-white hover:bg-white/10',
};

export const MotionIcon = ({
  icon: IconComp,
  size = 'md',
  variant = 'accent',
  animate = false,
  spin = false,
  className = '',
  iconClassName = '',
}) => {
  if (!IconComp) return null;
  const sz = MOTION_ICON_SIZES[size] || MOTION_ICON_SIZES.md;
  const varStyle = MOTION_ICON_VARIANTS[variant] || MOTION_ICON_VARIANTS.accent;

  return (
    <motion.span
      whileHover={{ scale: 1.1, rotate: [0, -4, 4, 0] }}
      whileTap={{ scale: 0.92 }}
      transition={spring.snappy}
      className={`inline-grid place-items-center border backdrop-blur-md transition-all shrink-0 ${sz.container} ${varStyle} ${className}`}
    >
      <IconComp
        size={sz.iconSize}
        className={`${spin || animate === 'spin' ? 'animate-spin' : animate === 'pulse' ? 'animate-pulse' : ''} ${iconClassName}`}
      />
    </motion.span>
  );
};

export const Section = ({ title, icon, iconVariant = 'accent', action, children, className = '', collapsible = false, defaultOpen = true, tilt, glare, hover, elevation }) => {
  const [open, setOpen] = useState(defaultOpen);
  const showBody = !collapsible || open;
  const marker = icon ? (
    <MotionIcon icon={icon} size="sm" variant={iconVariant} />
  ) : (
    <span className="h-3.5 w-[3px] rounded-full bg-[var(--accent)]/80 shrink-0" />
  );
  const label = 'text-mini font-semibold uppercase tracking-[0.14em] text-white/45';
  return (
    <Card className={`p-5 ${className}`} tilt={tilt} glare={glare} hover={hover} elevation={elevation}>
      {title && (
        <div className={`flex items-center justify-between gap-3 ${showBody ? 'mb-4' : 'mb-0'}`}>
          {collapsible ? (
            <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
                    className="flex items-center gap-2 min-w-0 group/sec">
              {marker}
              <span className={`${label} group-hover/sec:text-white/60 transition-colors`}>{title}</span>
              <svg className={`w-3 h-3 text-white/30 transition-transform duration-200 ${open ? 'rotate-90' : ''}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M9 6l6 6-6 6" /></svg>
            </button>
          ) : (
            <h3 className={`${label} flex items-center gap-2`}>{marker}{title}</h3>
          )}
          {action}
        </div>
      )}
      {collapsible ? (
        <AnimatePresence initial={false}>
          {showBody && (
            <motion.div
              key="section-body"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ ...spring.smooth, opacity: { duration: 0.2 } }}
              style={{ overflow: 'hidden' }}
            >
              <div className="space-y-4">{children}</div>
            </motion.div>
          )}
        </AnimatePresence>
      ) : (
        showBody && <div className="space-y-4">{children}</div>
      )}
    </Card>
  );
};

// Shared "?" tooltip badge. shrink-0 so it never wraps to its own line, and
// stays pinned to the top of a multi-line label.
//
// A real <button>, not a decorative span: these carry the only explanation of
// what a setting does, and on a hover-only span that text was unreachable by
// keyboard and invisible to a screen reader. Now it is tabbable, opens on focus
// as well as hover, and the text itself is the accessible name — so it is read
// out rather than announced as an unlabelled "?".
export const InfoBadge = ({ info }) => (
  <span className="relative group inline-flex items-center shrink-0 mt-0.5">
    <button
      type="button"
      aria-label={typeof info === 'string' ? info : 'More information'}
      onClick={(e) => e.preventDefault()}
      className="text-micro text-white/45 hover:text-white/60 focus-visible:text-white/60 cursor-help bg-white/5 rounded-full w-4.5 h-4.5 flex items-center justify-center font-bold apple-transition"
    >
      ?
    </button>
    <span
      role="tooltip"
      className="absolute bottom-full right-0 mb-2 hidden group-hover:block group-focus-within:block tooltip-content z-50 w-max max-w-xs p-3 rounded-xl bg-black/95 backdrop-blur-lg border border-white/10 shadow-2xl text-xs text-white/70 whitespace-normal leading-relaxed pointer-events-none text-left"
    >
      {info}
    </span>
  </span>
);

// Marks a control whose value differs from what a fresh install would have, and
// offers a one-click revert. Both props are optional, so every existing call
// site is unaffected — only Settings, which knows the defaults, passes them.
//
// Not a plain icon button: `title` alone is invisible to a screen reader and
// slow to a sighted user, so the accessible name names the SETTING being
// reverted rather than saying "reset" thirty times in a row.
export const ResetMark = ({ label, onReset }) => (
  <button
    type="button"
    onClick={(e) => { e.preventDefault(); onReset(); }}
    title={`Changed from default — click to reset${label ? ` “${label}”` : ''}`}
    aria-label={label ? `Reset ${label} to default` : 'Reset to default'}
    className="shrink-0 mt-0.5 inline-flex items-center gap-1 px-1.5 h-4.5 rounded-full bg-[var(--accent)]/12 border border-[var(--accent)]/30 text-[var(--accent)] hover:bg-[var(--accent)]/25 apple-transition group/reset"
  >
    <span className="h-1 w-1 rounded-full bg-[var(--accent)] group-hover/reset:hidden" />
    <Icon.reset size={9} className="hidden group-hover/reset:block" />
    <span className="text-nano font-bold tracking-wide">MOD</span>
  </button>
);

// `settingKey` is stamped onto the DOM as `data-setting` so the command palette
// can scroll to and highlight a specific control by name. It is only supplied
// by the Settings panel's binding helper; everywhere else it is undefined and
// React omits the attribute entirely.
export const Field = ({ label, info, children, modified, onReset, settingKey }) => (
  <label className="block" data-setting={settingKey}>
    <div className="flex items-start justify-between gap-2 mb-1.5">
      <span className="text-xs font-medium text-white/70 leading-snug min-w-0">{label}</span>
      <span className="flex items-start gap-1.5 shrink-0">
        {modified && onReset && <ResetMark label={label} onReset={onReset} />}
        {info && <InfoBadge info={info} />}
      </span>
    </div>
    {children}
  </label>
);

export const Select = ({ label, info, value, onChange, options = [], modified, onReset, settingKey }) => (
  <Field label={label} info={info} modified={modified} onReset={onReset} settingKey={settingKey}>
    <select
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value)}
      className="w-full px-3 py-2.5 rounded-xl glass-input text-white text-compact focus:outline-none cursor-pointer"
    >
      {options.map((o) => {
        // Accept plain strings (value === label) or {value, label} objects.
        const val = typeof o === 'object' && o !== null ? o.value : o;
        const lbl = typeof o === 'object' && o !== null ? o.label : o;
        return <option key={val} value={val} className="bg-[#121420]">{lbl}</option>;
      })}
    </select>
  </Field>
);

// ── Slider ─────────────────────────────────────────────────────────────────
// The thumb is driven by LOCAL state and only COMMITTED upward on a trailing
// debounce (plus immediately on release), instead of calling `onChange` on
// every input event.
//
// Why: a range input fires `onChange` on every pointer move — 60+ a second
// during a drag — and in this app `onChange` is `set(key, value)`, which lands
// in App's `settings` state. That is a prop of every screen, so one drag
// re-rendered the entire tree sixty times a second: FaceSwap (3,300 lines of
// JSX) plus InteractivePreview, Timeline and SliderTrackerBar under it, none of
// which are memoised. That is the drag lag.
//
// NOTHING VISIBLE IS LOST by committing late. The value readout and the thumb
// track `local`, so they are exactly as responsive as before. The preview is
// already debounced 350 ms downstream (see FaceSwap's previewKey effect) and
// the backend settings POST is debounced in App, so neither ever saw the
// intermediate values anyway — they were only ever paid for in re-renders.
//
// COMMIT_MS is a safety net for the paths that have no release event: keyboard
// arrows on a focused slider, and assistive tech. Release handlers commit
// immediately so a drag still feels instant to everything downstream.
const SLIDER_COMMIT_MS = 90;

export const Slider = ({ label, info, value, onChange, min = 0, max = 1, step = 0.01, modified, onReset, settingKey }) => {
  const [local, setLocal] = useState(value ?? 0);
  const timerRef = useRef(null);
  const pendingRef = useRef(null);
  // `onChange` is an inline arrow at nearly every call site, so it is a new
  // function each render; reading it through a ref keeps the commit effect from
  // being torn down and rebuilt on every one of them.
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  // Follow the prop when it moves for a reason that is not this drag — a preset
  // being loaded, a Reset, a profile applied. `pendingRef` is the guard: while a
  // commit of ours is in flight the incoming prop is still the OLD value, and
  // adopting it would snap the thumb back under the user's finger.
  useEffect(() => {
    if (pendingRef.current === null) setLocal(value ?? 0);
    else if (pendingRef.current === value) pendingRef.current = null;
  }, [value]);

  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current); }, []);

  const commit = (v) => {
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    pendingRef.current = v;
    onChangeRef.current(v);
  };

  const handleInput = (e) => {
    const v = parseFloat(e.target.value);
    setLocal(v);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => { timerRef.current = null; commit(v); }, SLIDER_COMMIT_MS);
  };

  // Release: commit now rather than waiting out the debounce. `blur` is in the
  // list so a value changed by keyboard and then tabbed away from is never left
  // uncommitted.
  const flush = () => { if (timerRef.current) commit(local); };

  return (
    <Field label={label} info={info} modified={modified} onReset={onReset} settingKey={settingKey}>
      <div className="flex items-center gap-3">
        {/* The native range thumb uses `accent-[var(--accent)]`, not the literal
            #E94560 it had before — that pinned it to Obsidian's crimson in all 38
            themes. */}
        <input
          type="range"
          min={min} max={max} step={step}
          value={local}
          onChange={handleInput}
          onPointerUp={flush}
          onPointerCancel={flush}
          onKeyUp={flush}
          onBlur={flush}
          className="flex-1 accent-[var(--accent)] h-1.5 apple-transition"
        />
        <span className="w-12 text-right text-xs font-semibold tabular-nums text-white/60">{Number(local ?? 0).toFixed(step < 1 ? 2 : 0)}</span>
      </div>
    </Field>
  );
};

// The checkbox is `sr-only`, NOT `hidden`. Tailwind's `hidden` is display:none,
// which takes an element out of the tab order entirely — and since the visible
// switch is a div, that left every toggle in the app (most of Settings, most of
// the Face Swap panel) impossible to reach or operate from the keyboard at all.
// sr-only keeps it invisible but focusable, and `peer` carries its focus state
// out to the switch so the ring lands on the thing the eye is looking at.
export const Toggle = ({ label, info, checked, onChange, modified, onReset, settingKey }) => (
  <label data-setting={settingKey} className="flex items-start justify-between gap-3 w-full text-left cursor-pointer group/toggle select-none">
    <input
      type="checkbox"
      className="sr-only peer"
      checked={!!checked}
      onChange={(e) => onChange(e.target.checked)}
    />
    <span className="flex items-start gap-1.5 min-w-0 flex-1">
      <span className="text-compact font-semibold tracking-wide leading-snug text-white/80 group-hover/toggle:text-white transition-colors">{label}</span>
      {modified && onReset && <ResetMark label={label} onReset={onReset} />}
      {info && <InfoBadge info={info} />}
    </span>
    <motion.span
      aria-hidden
      whileTap={{ scale: 0.9 }}
      transition={spring.snappy}
      className={`relative block shrink-0 mt-0.5 w-10 h-[22px] rounded-full transition-colors duration-200 border peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-[var(--accent)] ${checked ? 'bg-[var(--accent)] border-[var(--accent)]' : 'bg-white/[0.06] border-white/10 group-hover/toggle:border-white/20'}`}
    >
      <motion.span
        className="absolute top-[2px] left-[2px] w-[16px] h-[16px] rounded-full bg-white shadow-[0_1px_3px_rgba(0,0,0,0.5)]"
        animate={{ x: checked ? 18 : 0 }}
        transition={spring.bouncy}
      />
    </motion.span>
  </label>
);

export const TextInput = ({ label, info, value, onChange, placeholder, type = 'text', modified, onReset, settingKey }) => (
  <Field label={label} info={info} modified={modified} onReset={onReset} settingKey={settingKey}>
    <input
      type={type}
      value={value ?? ''}
      placeholder={placeholder}
      onChange={(e) => onChange(type === 'number' ? Number(e.target.value) : e.target.value)}
      className="w-full px-3 py-2.5 rounded-xl glass-input text-white text-compact focus:outline-none placeholder:text-white/25"
    />
  </Field>
);

export const Button = ({ children, onClick, variant = 'primary', disabled, className = '', size = 'md' }) => {
  const variants = {
    // `.fill-accent` rather than the utilities spelled out, so there is ONE
    // definition of "accent-filled button" in the app — this primitive and the
    // run bar's Start button were drifting apart, and the ink is the reason it
    // matters: `text-white` here measured 2.03:1 on Catppuccin's mauve accent and
    // 3.83:1 on the DEFAULT crimson, both under AA. The class computes the ink
    // from the accent's own lightness (see index.css).
    //
    // The accent glow is gone with it. It was
    // `shadow-[0_2px_10px_var(--accent-glow)]` growing to `0_4px_18px` on hover
    // — coloured glow-on-hover, which is the decoration this UI was deliberately
    // cleaned of, on the one control that appears on every screen. Depth now
    // comes from a neutral elevation shadow; the hover/tap SPRING is untouched,
    // because that is the approved motion, not decoration.
    primary: 'fill-accent shadow-[0_2px_10px_rgba(0,0,0,0.35)] hover:shadow-[0_4px_16px_rgba(0,0,0,0.45)]',
    secondary: 'bg-white/[0.05] hover:bg-white/[0.09] text-white/85 hover:text-white border border-white/10 hover:border-white/20',
    // Was red-500/red-300 literals; --danger is the same concept with a token,
    // and it carries a light-mode value the literals could not.
    stop: 'state-danger hover:brightness-110',
    ghost: 'bg-transparent hover:bg-white/[0.06] text-white/55 hover:text-white',
  };
  const sizes = {
    // `xs` was missing while two call sites asked for it. An unknown key made
    // `sizes[size]` undefined, which template-interpolated the literal string
    // "undefined" into className — so those buttons rendered with no padding,
    // no font size and square corners. The `||` fallbacks below stop the next
    // typo doing the same thing silently; test_ui_primitive_props.py fails the
    // build on an unknown value rather than letting it degrade quietly.
    xs: 'px-2.5 py-1 text-micro tracking-wide rounded-lg',
    sm: 'px-3 py-1.5 text-mini tracking-wide rounded-lg',
    md: 'px-4 py-2.5 text-compact tracking-wide rounded-xl',
    lg: 'px-6 py-3.5 text-sm tracking-wide rounded-xl',
  };
  return (
    <motion.button
      type="button"
      onClick={onClick}
      disabled={disabled}
      whileHover={disabled ? undefined : { y: -2, scale: 1.03 }}
      whileTap={disabled ? undefined : { scale: 0.95, y: 0 }}
      transition={spring.snappy}
      className={`font-semibold text-center transition-colors duration-200 disabled:opacity-40 disabled:cursor-not-allowed ${variants[variant] || variants.primary} ${sizes[size] || sizes.md} ${className}`}
    >
      {children}
    </motion.button>
  );
};

// Gallery of face thumbnails with selection + move/remove controls.
export const FaceGallery = ({ title, faces, selected, onSelect, onRemove, empty, groups, vertical = false, info = [], draggable = false, large = false }) => {
  const personCount = groups && groups.length ? new Set(groups).size : 0;
  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-white/40">{title}</span>
        {faces.length > 0 && (
          <span className="px-1.5 py-0.5 rounded-full bg-white/10 text-micro text-white/60 tabular-nums">
            {faces.length}{personCount > 1 ? ` · ${personCount} people` : ''}
          </span>
        )}
      </div>
      {faces.length === 0 ? (
        <div className="h-24 flex items-center justify-center rounded-lg border border-dashed border-white/10 text-xs text-white/30">
          {empty || 'None yet'}
        </div>
      ) : (
        <div className={vertical ? "flex flex-col gap-2" : (large ? "grid grid-cols-2 3xl:grid-cols-3 gap-2.5" : "grid grid-cols-4 3xl:grid-cols-5 4xl:grid-cols-6 gap-2")}>
          {faces.map((src, i) => {
            const person = groups && i < groups.length ? groups[i] : null;
            const color = person != null ? PERSON_COLORS[person % PERSON_COLORS.length] : null;
            const itemInfo = info && info[i];
            const hasMultiFaces = itemInfo && itemInfo.count > 1;
            return (
              <motion.div
                key={i}
                layout
                initial={{ opacity: 0, scale: 0.7 }}
                animate={{ opacity: 1, scale: 1, transition: { ...spring.snappy, delay: Math.min(i * 0.025, 0.35) } }}
                whileHover={vertical ? { x: 3 } : { y: -4, scale: 1.04, zIndex: 5 }}
                whileTap={{ scale: 0.95 }}
                transition={spring.snappy}
                draggable={draggable}
                onDragStart={draggable ? (e) => {
                  e.dataTransfer.setData('text/roop-source', String(i));
                  e.dataTransfer.effectAllowed = 'link';
                } : undefined}
                title={draggable ? `Face ${i + 1} — drag onto a person to assign` : undefined}
                className={`group relative ${vertical ? 'flex items-center gap-3 p-2' : 'aspect-square'} rounded-xl overflow-hidden border-2 transition-colors duration-200 cursor-pointer ${draggable ? 'active:cursor-grabbing' : ''} ${selected === i ? (vertical ? 'bg-white/5' : 'ring-2 ring-[var(--accent)]/30') : 'hover:border-white/30'}`}
                style={{ borderColor: selected === i ? (color || 'var(--accent)') : (color ? `${color}66` : 'transparent') }}
                onClick={() => onSelect(i)}
                // Selecting a face is the primary action of this whole panel and
                // it lived on a bare div: no tab stop, no role, no key handling.
                role="button"
                tabIndex={0}
                aria-pressed={selected === i}
                aria-label={person != null ? `Person ${person + 1}, face ${i + 1}` : `Face ${i + 1}`}
                // Only when the tile ITSELF has focus. The Remove button is a
                // descendant, so its Enter/Space bubble up here: unguarded, this
                // handler selected the face instead of removing it (Space —
                // preventDefault suppresses the button's own activation) or as
                // well as removing it (Enter). stopPropagation on the button's
                // onClick cannot help; that is a different event.
                onKeyDown={(e) => {
                  if (e.target !== e.currentTarget) return;
                  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect?.(i); }
                }}
              >
                {vertical ? (
                  <>
                    {/* An empty src resolves to the page URL and paints a broken-image
                        icon, so a face whose crop could not be built gets a plain
                        placeholder tile — it still occupies its slot, because the
                        gallery is positionally tied to the backend's faceset list. */}
                    {src ? (
                      <img src={src} alt={`face ${i}`} className="w-10 h-10 rounded-lg object-cover shrink-0" />
                    ) : (
                      <div className="w-10 h-10 rounded-lg shrink-0 grid place-items-center bg-white/[0.04] text-white/25 text-xs">?</div>
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-white/90">
                        {person != null ? `Person ${person + 1}` : `Face ${i + 1}`}
                      </div>
                    </div>
                    {onRemove && (
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); onRemove(i); }}
                        title="Remove this face"
                        aria-label={`Remove face ${i + 1}`}
                        className="h-7 w-7 shrink-0 rounded-full bg-black/40 text-white/60 hover:bg-[var(--accent-hover)] hover:text-white hover:scale-110 active:scale-90 flex items-center justify-center"
                      ><Icon.close size={13} /></button>
                    )}
                  </>
                ) : (
                  <>
                    {src ? (
                      <img src={src} alt={`face ${i}`} className="w-full h-full object-cover" />
                    ) : (
                      <div className="w-full h-full grid place-items-center bg-white/[0.04] text-white/25 text-lg">?</div>
                    )}
                    {hasMultiFaces && (
                      <span className="absolute top-1 left-1 px-1 py-0.5 rounded bg-black/75 backdrop-blur text-nano font-bold text-[var(--accent)] border border-[var(--accent)]/30 shadow-md pointer-events-none select-none">
                        {itemInfo.count}F
                      </span>
                    )}
                    {person != null && (
                      <span className="absolute bottom-1 left-1 px-1.5 rounded-md text-nano font-bold leading-tight text-white shadow-sm"
                        style={{ backgroundColor: color }}>P{person + 1}</span>
                    )}
                    {onRemove && (
                      <button
                        type="button"
                        title="Remove this face"
                        aria-label={`Remove face ${i + 1}`}
                        onClick={(e) => { e.stopPropagation(); onRemove(i); }}
                        // focus-visible as well as group-hover: this is a real
                        // tab stop, and opacity-0 alone left a keyboard user
                        // focused on a control they could not see.
                        className="absolute top-1 right-1 h-6 w-6 rounded-full bg-black/70 text-white/80 leading-none opacity-0 group-hover:opacity-100 focus-visible:opacity-100 hover:bg-[var(--accent-hover)] hover:scale-110 active:scale-90 transition-all duration-300 flex items-center justify-center"
                      ><Icon.close size={12} /></button>
                    )}
                  </>
                )}
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// Smoothly tweens a displayed number toward `value` (eased). Great for %/fps/ETA.
export const AnimatedNumber = ({ value, duration = 500, decimals = 0, suffix = '', className = '' }) => {
  const [disp, setDisp] = useState(value || 0);
  const fromRef = useRef(value || 0);
  const rafRef = useRef();
  useEffect(() => {
    const from = fromRef.current;
    const to = value || 0;
    if (from === to) return;
    const start = performance.now();
    const tick = (t) => {
      const p = Math.min(1, (t - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisp(from + (to - from) * eased);
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
      else fromRef.current = to;
    };
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value, duration]);
  return <span className={className}>{Number(disp).toFixed(decimals)}{suffix}</span>;
};

// Shimmering placeholder block for loading states.
export const Skeleton = ({ className = '' }) => (
  <div className={`rounded-lg bg-white/5 relative overflow-hidden ${className}`}>
    <div className="absolute inset-0 -translate-x-full animate-[skeleton_1.4s_infinite] bg-gradient-to-r from-transparent via-white/10 to-transparent" />
  </div>
);

// One-shot celebratory confetti burst. Renders nothing when inactive.
export const Confetti = ({ active }) => {
  if (!active) return null;
  const colors = ['#E94560', '#3DA5D9', '#52B788', '#E9C46A', '#9B5DE5', '#F4A261', '#00BBF9'];
  const pieces = Array.from({ length: 90 });
  return (
    <div className="fixed inset-0 z-[70] pointer-events-none overflow-hidden">
      {pieces.map((_, i) => {
        const left = Math.random() * 100;
        const delay = Math.random() * 0.5;
        const dur = 1.8 + Math.random() * 1.4;
        const size = 6 + Math.random() * 6;
        const color = colors[i % colors.length];
        const rot = Math.random() * 360;
        return (
          <span
            key={i}
            style={{
              position: 'absolute', top: '-5%', left: `${left}%`,
              width: size, height: size * 0.5, background: color,
              transform: `rotate(${rot}deg)`,
              animation: `confetti-fall ${dur}s cubic-bezier(0.4,0,0.6,1) ${delay}s forwards`,
              borderRadius: 1,
            }}
          />
        );
      })}
    </div>
  );
};

// One family, one optical weight, and the tint comes from the theme rather than
// from the glyph. The three emoji these replaced each came from a different
// part of the vendor's artwork, at different sizes and saturations, so they
// never read as three states of one thing.
const toastIcon = (type) => (type === 'error' ? Icon.error : type === 'info' ? Icon.info : Icon.success);
const toastColor = (type) => (type === 'error' ? 'text-red-400' : type === 'info' ? 'text-blue-300' : 'text-green-400');

// Stacked toasts. Multiple can be visible at once (a batch run or several errors
// no longer clobber each other), newest on top. Dismissable by click. The stack
// is not the accessibility channel — App renders a separate aria-live region so
// screen readers are announced regardless of the visual stack.
export const Toasts = ({ toasts = [], onDismiss }) => (
  <div className="fixed bottom-6 right-6 z-50 flex flex-col-reverse items-end gap-2 pointer-events-none">
    <AnimatePresence initial={false}>
      {toasts.map((t) => (
        <motion.div
          key={t.id}
          layout
          initial={{ opacity: 0, y: 24, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, x: 24, scale: 0.95 }}
          transition={spring.bouncy}
          onClick={() => onDismiss?.(t.id)}
          // Dismissal was click-only on a div. The message itself is announced
          // through App's aria-live region either way, so this is about being
          // able to clear a stack of them without a mouse.
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onDismiss?.(t.id); }
          }}
          title="Dismiss"
          aria-label={`Dismiss notification: ${t.message}`}
          className="pointer-events-auto cursor-pointer px-4 py-3 rounded-xl shadow-2xl bg-[#0E0F15]/95 backdrop-blur-xl border border-white/10 flex items-center gap-3 min-w-[250px] max-w-sm"
        >
          {(() => { const I = toastIcon(t.type); return <I size={18} className={toastColor(t.type)} />; })()}
          <span className={`text-sm font-medium ${toastColor(t.type)}`}>{t.message}</span>
        </motion.div>
      ))}
    </AnimatePresence>
  </div>
);

// Legacy single-toast (kept for any caller still passing one object).
export const Toast = ({ toast }) => (
  <Toasts toasts={toast ? [{ ...toast, id: 'single' }] : []} />
);
