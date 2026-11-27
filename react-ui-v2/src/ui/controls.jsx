// React UI 2.0's form primitives and toasts. Plain, accessible elements styled by app.css.
import React, { useId } from 'react';
import { create } from 'zustand';
import { AnimatePresence, motion } from 'framer-motion';
import { CheckCircle2, AlertTriangle, XCircle, Info, X } from 'lucide-react';
import { termLog, levelOfToast } from '../fx/termFeed';

export function Field({ label, hint, children, htmlFor, className = '' }) {
  return (
    <div className={className}>
      {label && <label className="field-label" htmlFor={htmlFor}>{label}</label>}
      {children}
      {hint && <div className="field-hint mt-1">{hint}</div>}
    </div>
  );
}

/** options: [value] | [{value,label}] | [{group, options:[...]}] */
export function Select({ label, hint, value, options = [], onChange, className = '', ...rest }) {
  const id = useId();
  const opt = (o) => (typeof o === 'object' ? o : { value: o, label: String(o) });
  const list = options.length && options.some((o) => o && o.group) ? options : null;
  const known = (list ? list.flatMap((g) => g.options) : options).map(opt).some((o) => String(o.value) === String(value ?? ''));
  return (
    <Field label={label} hint={hint} htmlFor={id} className={className}>
      <select id={id} className="select" value={value ?? ''} onChange={(e) => onChange(e.target.value)} {...rest}>
        {!known && value != null && value !== '' && <option value={value}>{String(value)}</option>}
        {list
          ? list.map((g) => (
            <optgroup key={g.group} label={g.group}>
              {g.options.map(opt).map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </optgroup>
          ))
          : options.map(opt).map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </Field>
  );
}

export function Slider({ label, value, min = 0, max = 1, step = 0.01, onChange, format, hint }) {
  const id = useId();
  const v = Number.isFinite(Number(value)) ? Number(value) : min;
  const fill = `${((v - min) / (max - min || 1)) * 100}%`;
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label className="field-label" htmlFor={id}>{label}</label>
        <span className="font-mono text-[11.5px] text-[var(--accent-2)] tnum">{format ? format(v) : v}</span>
      </div>
      <input id={id} type="range" className="range" min={min} max={max} step={step} value={v}
        style={{ '--fill': fill }} onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <div className="field-hint">{hint}</div>}
    </div>
  );
}

export function Switch({ label, hint, checked, onChange }) {
  return (
    <button type="button" role="switch" aria-checked={!!checked} onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between gap-3 rounded-lg px-1 py-1.5 text-left hover:bg-white/[0.03]">
      <span className="min-w-0">
        <span className="block text-[13px] font-medium text-white">{label}</span>
        {hint && <span className="block text-[11.5px] text-[var(--fg-faint)]">{hint}</span>}
      </span>
      <span className={`switch ${checked ? 'on' : ''}`} aria-hidden="true"><i /></span>
    </button>
  );
}

export function Kbd({ children }) {
  return <kbd className="rounded-md border border-[var(--rule-strong)] bg-white/[0.04] px-1.5 py-0.5 font-mono text-[10.5px] text-[var(--fg-dim)]">{children}</kbd>;
}

// ── Toasts ──────────────────────────────────────────────────────────────────
export const useToasts = create((set) => ({
  toasts: [],
  push: (t) => set((s) => ({ toasts: [...s.toasts, t].slice(-4) })),
  drop: (id) => set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) })),
}));

/** notify(message, type) - type: success | info | warn | error. Every toast is also a terminal line. */
export function notify(message, type = 'success') {
  const id = `${Date.now()}-${Math.random()}`;
  useToasts.getState().push({ id, message: String(message), type });
  termLog(message, levelOfToast(type), 'toast');
  setTimeout(() => useToasts.getState().drop(id), type === 'error' ? 7000 : 4200);
}

const ICON = { success: CheckCircle2, info: Info, warn: AlertTriangle, warning: AlertTriangle, error: XCircle };
const TONE = { success: 'var(--ok)', info: 'var(--accent-2)', warn: 'var(--warn)', warning: 'var(--warn)', error: 'var(--danger)' };

export function Toasts() {
  const toasts = useToasts((s) => s.toasts);
  const drop = useToasts((s) => s.drop);
  return (
    <div className="pointer-events-none fixed right-4 top-16 z-[100] flex flex-col items-end gap-2" role="status" aria-live="polite">
      <AnimatePresence>
        {toasts.map((t) => {
          const Ico = ICON[t.type] || Info;
          return (
            <motion.div key={t.id} layout
              initial={{ opacity: 0, x: 40, scale: 0.96 }} animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 30, scale: 0.96, transition: { duration: 0.18 } }}
              className="toast pointer-events-auto">
              <Ico size={16} style={{ color: TONE[t.type] }} className="mt-0.5 shrink-0" />
              <span className="min-w-0 flex-1 selectable break-words">{t.message}</span>
              <button type="button" className="btn btn-ghost btn-sm btn-icon -mr-1 -mt-1" aria-label="Dismiss" onClick={() => drop(t.id)}><X size={14} /></button>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
