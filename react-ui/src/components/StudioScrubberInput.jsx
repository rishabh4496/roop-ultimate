import React, { useEffect, useRef, useState } from 'react';
import { settleTypedValue } from '../numeric/mathExpr';
import { decimalsFor } from '../numeric/scrubMath';
import { useScrub } from '../numeric/useScrub';
import { Field, FocusedInput } from './ui';
import ExpressionPreview from './ExpressionPreview';
import { fillExtent, scrubberDecimals, STEP_FINE } from './scrubberModel';

// ── The studio scrubber ─────────────────────────────────────────────────────────────────────────────────────────────
//
// The control every numeric parameter in the app uses (`Slider` in ui.jsx renders it, so every call site got it at once).
// It behaves like the value fields of a node / layer inspector rather than a web form:
//
//   drag the LABEL or the BAR      scrub the value from where it is. The bar is NOT a track you aim at: pressing it does not
//                                  jump the value, only travel changes it, so a press-and-release never moves a parameter by
//                                  accident. Shift x10, Alt x0.1 (numeric/scrubMath.js; Alt snaps to 0.001, an integer
//                                  control keeps its step - a fraction of a thread is not a value).
//   double-click the bar           type a value; the field takes arithmetic (`512/2`, `+10%` - numeric/mathExpr.js).
//   right-click the bar            reset to the default
//   click the cyan dot             reset to the default (shown only when the value is not the default)
//   Backspace / Delete, Enter      the keyboard routes to the same two things; Arrow keys step, Shift+Arrow is the fine step
//
// WHY A HIDDEN NATIVE RANGE IS STILL HERE. The bar is a mouse affordance. The keyboard, a screen reader, form restore and
// the e2e suite all talk to an <input type="range"> (one tab stop, aria value, Arrow/Home/End/PageUp handling that no hand
// rolled slider matches), so the real input stays - full size, invisible, and `pointer-events: none` so a press reaches the
// bar instead of the browser's own absolute-position thumb jump. It is the ONLY tab stop of the control.
//
// The fill is `--accent-cyan` (obsidian-vfx-tokens.css). A range that crosses zero fills outward from zero, not from the left
// edge, so a signed control (erode / dilate) reads as a signed one.
//
// Commits are debounced while scrubbing for the reason Slider's always were: `onChange` is `set(key, value)` into App's
// settings state, and one drag would otherwise re-render the tree at pointer-move rate. The bar and readout follow `local`,
// so they are exactly as responsive; the parent hears on the trailing edge and at once on release.

const COMMIT_MS = 90;
const clampNum = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const snapTo = (v, step) => Number((Math.round(v / step) * step).toFixed(6));

export function NumberEditor({ initial, current, min, max, step, label, onDone, onDraft }) {
  const [draft, setDraftState] = useState(initial);
  // The parent draws the live preview pill (this editor sits inside a clipped bar), so it hears every keystroke.
  const setDraft = (text) => { setDraftState(text); onDraft?.(text); };
  const [error, setError] = useState('');
  const doneRef = useRef(false);
  // `commit` is true for Enter and for leaving the field. Enter on something that does not parse keeps the editor open
  // (the user is looking at it and can fix it); leaving it just restores the old value.
  const finish = (commit, viaEnter = false) => {
    if (doneRef.current) return;
    if (!commit) { doneRef.current = true; onDone(null); return; }
    const r = settleTypedValue(draft, { current, min, max, step });
    if (!r.ok) {
      if (viaEnter) { setError(r.error); return; }
      doneRef.current = true;
      onDone(null);
      return;
    }
    doneRef.current = true;
    onDone(r.value);
  };
  return (
    <>
      <FocusedInput
        select
        type="text"
        inputMode="text"
        autoComplete="off"
        spellCheck={false}
        aria-label={`${label} value`}
        aria-invalid={error ? true : undefined}
        title={error || 'Type a number or an expression: 512/2, *2, /1.5, +16, -10%, min(v*2, 64), max(v/2, 8)'}
        value={draft}
        onChange={(e) => { setDraft(e.target.value); setError(''); }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') { e.preventDefault(); finish(true, true); }
          else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); finish(false); }
        }}
        onBlur={() => finish(true)}
        className={`studio-scrub__editor ${error ? 'is-invalid' : ''}`}
      />
      {error && <span role="alert" className="sr-only">{error}</span>}
    </>
  );
}

// The cyan dot: this control is not at its default. A mouse affordance like the bar (tabindex -1, aria-hidden): the range
// input already announces its own value, Backspace/Delete on it already resets, and a second tab stop per modified control is
// exactly what e2e/tab-stops.spec.js exists to refuse.
function ModifiedDot({ label, defaultText, onReset }) {
  return (
    <button
      type="button"
      tabIndex={-1}
      aria-hidden="true"
      data-testid="modified-dot"
      onClick={(e) => { e.preventDefault(); onReset(); }}
      title={`${label ? `${label}: ` : ''}changed from the default (${defaultText}) — click to reset`}
      className="shrink-0 inline-flex items-center justify-center h-4 w-4 rounded-full group/dot cursor-pointer"
    >
      <span className="h-1.5 w-1.5 rounded-full bg-[var(--mod-dot,#22d3ee)] shadow-[0_0_6px_var(--mod-dot,#22d3ee)] group-hover/dot:scale-150 apple-transition" />
    </button>
  );
}

export default function StudioScrubberInput({
  label, info, value, onChange, min = 0, max = 1, step = 0.01,
  modified, onReset, settingKey, defaultValue, unit, disabled = false, testId,
  // percent: the value is stored 0..1 and shown, scrubbed-through and typed as 0..100 % (a wet/dry mix reads as a percentage)
  percent = false, disabledNote,
}) {
  const [local, setLocal] = useState(value ?? 0);
  const [editing, setEditing] = useState(false);
  const [draftText, setDraftText] = useState('');             // what is typed while editing, for the live preview pill
  const timerRef = useRef(null);
  const nativeRef = useRef(null);
  const pendingRef = useRef(null);
  const localRef = useRef(local);
  localRef.current = local;
  // `onChange` is an inline arrow at nearly every call site, so it is a new function each render; reading it through a ref
  // keeps the commit effect from being torn down and rebuilt on every one of them.
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  // Follow the prop when it moves for a reason that is not this drag - a preset being loaded, a Reset, a profile applied.
  // `pendingRef` is the guard: while a commit of ours is in flight the incoming prop is still the OLD value, and adopting it
  // would snap the bar back under the user's pointer.
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
  // Local first, parent on the trailing debounce (or at once with `now`).
  const setValue = (v, now = false) => {
    setLocal(v);
    if (timerRef.current) clearTimeout(timerRef.current);
    if (now) commit(v);
    else timerRef.current = setTimeout(() => { timerRef.current = null; commit(v); }, COMMIT_MS);
  };
  // Release: commit now rather than waiting out the debounce. `blur` is wired too, so a value changed by keyboard and then
  // tabbed away from is never left uncommitted.
  const flush = () => { if (timerRef.current) commit(localRef.current); };

  const scrub = useScrub({
    getValue: () => localRef.current,
    min, max, step,
    disabled,
    onScrub: (v) => { if (v !== localRef.current) setValue(v); },
    onEnd: flush,
  });

  const hasDefault = defaultValue !== undefined && defaultValue !== null;
  const canReset = hasDefault || (modified && !!onReset);
  const reset = () => {
    if (hasDefault) {
      if (localRef.current === defaultValue && !timerRef.current && Number(value) === defaultValue) return;
      setLocal(defaultValue);
      commit(defaultValue);
    } else if (modified && onReset) {
      onReset();
    }
  };

  const fineStep = step >= 1 ? step : Math.min(step, STEP_FINE);
  const onKeyDown = (e) => {
    if (e.key === 'Enter') { e.preventDefault(); setEditing(true); return; }
    if ((e.key === 'Backspace' || e.key === 'Delete') && canReset) { e.preventDefault(); reset(); return; }
    if (e.shiftKey && e.key.startsWith('Arrow')) {
      const dir = e.key === 'ArrowRight' || e.key === 'ArrowUp' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowDown' ? -1 : 0;
      if (!dir) return;
      e.preventDefault();
      setValue(clampNum(snapTo(localRef.current + dir * fineStep, fineStep), min, max));
    }
  };

  // Up to three decimals when the value needs them (an Alt-scrub lands on 0.001), never fewer than the step implies.
  const scale = percent ? 100 : 1;
  const baseDecimals = percent ? 0 : scrubberDecimals(step);
  const decimals = decimalsFor((local ?? 0) * scale, baseDecimals);
  const nameText = typeof label === 'string' ? label : 'Value';
  const shown = (Number(local ?? 0) * scale).toFixed(decimals);
  const suffix = percent ? '%' : (unit || '');
  const { from, to, pct, pctDefault } = fillExtent(Number(local ?? 0), min, max, hasDefault ? Number(defaultValue) : null);

  const hint = disabledNote || [
    'Drag to scrub · Shift ×10 · Alt ×0.1',
    'Double-click to type (512/2, *2, +16, -10%, min(), max())',
    canReset ? 'Right-click to reset' : null,
  ].filter(Boolean).join(' · ');
  // The Settings panel marks a changed control with its own MOD pill (it knows the backend defaults); a caller that gives a
  // `defaultValue` instead gets the dot. Never both.
  const showDot = hasDefault && !(modified && onReset) && Math.abs(Number(local ?? 0) - Number(defaultValue)) > 1e-6;
  const labelProps = { ...(disabled ? {} : scrub.bind), className: `${disabled ? '' : 'cursor-ew-resize'} select-none`, title: 'Drag to scrub · Shift ×10 · Alt ×0.1' };

  return (
    <Field
      label={label}
      info={info}
      modified={modified}
      onReset={onReset}
      settingKey={settingKey}
      labelProps={labelProps}
      badge={showDot ? <ModifiedDot label={nameText} defaultText={`${(Number(defaultValue) * scale).toFixed(baseDecimals)}${suffix}`} onReset={reset} /> : null}
    >
      <div className={`studio-scrub ${scrub.scrubbing ? 'is-scrubbing' : ''} ${disabled ? 'is-disabled' : ''}`} data-testid={testId || 'scrubber'}>
        {/* Stage 15: `Formula → Result`, live, before Enter. Outside the bar because the bar clips its children. */}
        {editing && (
          <ExpressionPreview text={draftText} current={Number(local ?? 0) * scale} min={min * scale} max={max * scale} step={step * scale} />
        )}
        <input
          ref={nativeRef}
          type="range"
          className="studio-scrub__native"
          min={min} max={max} step={step}
          value={local}
          disabled={disabled}
          title={hint}
          onChange={(e) => setValue(parseFloat(e.target.value))}
          onKeyDown={onKeyDown}
          onKeyUp={flush}
          onBlur={flush}
        />
        <div
          role="presentation"   // a mouse affordance: its keyboard twin is the hidden range input above
          className="studio-scrub__bar"
          data-testid="scrubber-bar"
          title={hint}
          {...(disabled || editing ? {} : scrub.bind)}
          // The bar sits inside the Field's <label>, so a click would be forwarded to the hidden range input, and Chrome sets a
          // range from the position of a click it is handed: the value would jump to wherever the bar was pressed. Swallow
          // the forwarding and give the keyboard its focus directly.
          onClick={(e) => { if (e.target.closest('.studio-scrub__editor')) return; e.preventDefault(); if (!editing) nativeRef.current?.focus({ preventScroll: true }); }}
          onDoubleClick={disabled ? undefined : () => { if (!editing) setEditing(true); }}
          onContextMenu={canReset && !disabled ? (e) => { e.preventDefault(); reset(); } : undefined}
        >
          <span className="studio-scrub__fill" style={{ left: `${from * 100}%`, width: `${Math.max(0, to - from) * 100}%` }} aria-hidden="true" />
          <span className="studio-scrub__edge" style={{ left: `${pct * 100}%` }} aria-hidden="true" />
          {pctDefault !== null && <span className="studio-scrub__default" style={{ left: `${pctDefault * 100}%` }} aria-hidden="true" />}
          {editing ? (
            <NumberEditor
              initial={shown}
              current={Number(local ?? 0) * scale}
              min={min * scale}
              max={max * scale}
              step={step * scale}
              label={nameText}
              onDraft={setDraftText}
              onDone={(typed) => {
                setEditing(false);
                setDraftText('');
                const v = typed === null ? null : Number((typed / scale).toFixed(6));
                if (v !== null && v !== localRef.current) setValue(v, true);
              }}
            />
          ) : (
            <span className="studio-scrub__value tabular-nums" aria-hidden="true" data-testid="scrubber-value">{shown}{suffix}</span>
          )}
        </div>
      </div>
    </Field>
  );
}
