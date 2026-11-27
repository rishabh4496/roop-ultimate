import React, { useEffect, useRef, useState } from 'react';
import { settleTypedValue } from '../numeric/mathExpr';
import { clampNum, decimalsFor, snapStep, snapTo } from '../numeric/scrubMath';
import { useScrub } from '../numeric/useScrub';

// A number field with the workstation's pro ergonomics:
//
//   * drag the LABEL, or the field itself while it is not being edited, to scrub the value
//       (Shift = x10, Alt = x0.1 - see numeric/scrubMath.js)
//   * click it to type, and type arithmetic: `512/2`, `24*1.5`, `+10%`, `(3+4)*2`
//       (numeric/mathExpr.js - a hand-written parser, no eval)
//   * Up / Down arrows step by `step`; Shift x10, Alt the finest grid the control allows
//   * Enter or leaving the field commits; Escape puts the old value back
//   * an expression that does not parse is NOT silently dropped: Enter keeps the field open and says why
//
// It replaces the `<input type="number">` + draft-state pairs that each inspector hand-rolled. Those could not take
// an expression (a number input refuses letters and symbols), and the browser's spinner steps by `step` only.
//
// Commits are debounced while scrubbing for the same reason Slider's are (ui.jsx): `onCommit` lands in app state, and a
// drag would otherwise re-render the whole tree at pointer-move rate.

const COMMIT_MS = 90;

const fmt = (v, step) => {
  const base = step >= 1 ? 0 : String(step).split('.')[1]?.length || 2;
  return Number(v).toFixed(decimalsFor(v, Math.min(base, 4)));
};

export default function NumericInput({
  label, value, onCommit, min = -Infinity, max = Infinity, step = 1, unit,
  ariaLabel, testId, disabled = false,
  className = '', inputClassName = '', labelClassName = '',
}) {
  const [draft, setDraftState] = useState(null);     // null = idle, showing `value`
  const draftRef = useRef(null);                      // the source of truth for settle(): state would be stale in a blur that follows Escape
  const [error, setError] = useState('');
  const [live, setLive] = useState(null);             // the value while a scrub is in flight
  const inputRef = useRef(null);
  const timer = useRef(null);
  const liveRef = useRef(null);
  const commitRef = useRef(onCommit);
  commitRef.current = onCommit;

  const setDraft = (d) => { draftRef.current = d; setDraftState(d); };
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  const commitNow = (v) => {
    if (timer.current) { clearTimeout(timer.current); timer.current = null; }
    commitRef.current?.(v);
  };

  const scrub = useScrub({
    getValue: () => (Number.isFinite(liveRef.current) ? liveRef.current : value),
    min: Number.isFinite(min) ? min : value - 1000,
    max: Number.isFinite(max) ? max : value + 1000,
    step,
    disabled: disabled || draft !== null,
    onScrub: (v) => {
      liveRef.current = v;
      setLive(v);
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => { timer.current = null; commitRef.current?.(v); }, COMMIT_MS);
    },
    onEnd: () => {
      if (Number.isFinite(liveRef.current)) commitNow(liveRef.current);
      liveRef.current = null;
      setLive(null);
    },
  });

  const open = () => {
    if (disabled || draftRef.current !== null) return;
    setError('');
    setDraft(fmt(value, step));
    requestAnimationFrame(() => inputRef.current?.select());
  };

  /** Evaluate what is in the field. Returns false only when the field must stay open (Enter on a bad expression). */
  const settle = (viaEnter) => {
    const text = draftRef.current;
    if (text === null) return true;
    const r = settleTypedValue(text, { current: value, min, max, step });
    if (!r.ok) {
      if (viaEnter) { setError(r.error); return false; }      // keep it open so the mistake can be fixed
      setDraft(null); setError('');                            // leaving the field puts the old value back
      return true;
    }
    setDraft(null); setError('');
    if (r.value !== value) commitNow(r.value);
    return true;
  };

  const onKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (settle(true)) e.currentTarget.blur();
      return;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      setDraft(null); setError('');          // draftRef is null before the blur below, so onBlur settles nothing
      e.currentTarget.blur();
      return;
    }
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault();
      const dir = e.key === 'ArrowUp' ? 1 : -1;
      const grid = snapStep(step, e);
      const delta = e.altKey && !e.shiftKey ? grid : e.shiftKey && !e.altKey ? step * 10 : step;
      const typed = settleTypedValue(draftRef.current ?? '', { current: value, min, max, step });
      const base = typed.ok ? typed.value : value;
      setDraft(fmt(clampNum(snapTo(base + dir * delta, grid), min, max), grid));
      setError('');
    }
  };

  const idle = draft === null;
  const labelCls = `select-none ${disabled ? '' : 'cursor-ew-resize'} ${labelClassName}`;
  const shown = live ?? value;

  return (
    <label className={`inline-flex items-center gap-1 ${className}`}>
      {label !== undefined && label !== null && (
        <span
          {...(disabled ? {} : scrub.bind)}
          className={labelCls}
          title={disabled ? undefined : 'Drag to scrub · Shift ×10 · Alt ×0.1'}
        >{label}</span>
      )}
      <input
        ref={inputRef}
        type="text"
        inputMode="text"
        autoComplete="off"
        spellCheck={false}
        disabled={disabled}
        data-testid={testId}
        aria-label={ariaLabel}
        aria-invalid={error ? true : undefined}
        title={error || (idle && !disabled ? 'Click to type a value or an expression like 512/2 · drag to scrub' : undefined)}
        value={idle ? fmt(shown, step) : draft}
        onFocus={open}
        onChange={(e) => { setDraft(e.target.value); setError(''); }}
        onBlur={() => settle(false)}
        onKeyDown={idle ? undefined : onKeyDown}
        // Idle, the field is a scrub handle: a press must NOT focus it (that would open the editor under the drag).
        // A click that never became a drag focuses it in onClick instead.
        {...(idle ? scrub.bind : {})}
        onMouseDown={idle && !disabled ? (e) => e.preventDefault() : undefined}
        onClick={idle && !disabled ? () => { if (!scrub.scrubbing) inputRef.current?.focus(); } : undefined}
        className={`${inputClassName} ${idle && !disabled ? 'cursor-ew-resize' : ''} ${error ? '!border-red-400' : ''}`}
      />
      {unit}
      {error && <span role="alert" className="sr-only">{error}</span>}
    </label>
  );
}
