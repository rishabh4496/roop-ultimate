import React from 'react';
import { OVERRIDE_FIELDS, canBypass } from './overrides';
import { clamp } from './viewMath';

// The editor for the SELECTED scene override. Each field is opt-in: an unticked field inherits the
// clip's settings, so an override only ever states what is different about this range.

const inputCls = 'rounded-md border border-[var(--border-color)] bg-[var(--input-bg)] px-1.5 py-0.5 text-mini font-mono tabular-nums text-[var(--text-main)] outline-none focus:border-[var(--accent)]';

function NumberBox({ label, value, min, max, onCommit }) {
  const [draft, setDraft] = React.useState(null);
  const commit = () => {
    if (draft !== null) {
      const n = parseInt(draft, 10);
      if (Number.isFinite(n) && n !== value) onCommit(clamp(n, min, max));
    }
    setDraft(null);
  };
  return (
    <label className="inline-flex items-center gap-1 text-nano uppercase tracking-wider text-[var(--text-muted)]">
      {label}
      <input
        type="number" min={min} max={max} value={draft ?? value}
        aria-label={label}
        onFocus={(e) => { setDraft(String(value)); e.target.select(); }}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); if (e.key === 'Escape') { setDraft(null); e.currentTarget.blur(); } }}
        className={`${inputCls} w-[8ch] text-right`}
      />
    </label>
  );
}

export default function OverrideInspector({
  override, maxFrames, swapModels = [], detectionMode, baseValues = {},
  onChange, onRemove, onJump, onUseAsRange,
}) {
  if (!override) return null;
  const bypassOk = canBypass(detectionMode);
  const setParam = (id, value) => {
    const params = { ...override.params };
    if (value === undefined) delete params[id]; else params[id] = value;
    onChange({ params });
  };

  return (
    <div
      data-testid="override-inspector"
      className="rounded-xl border border-[var(--border-color)] bg-[var(--card-bg)] px-3 py-2.5 space-y-2.5"
    >
      <div className="flex flex-wrap items-center gap-2">
        <input
          type="text" value={override.label} aria-label="Override name"
          onChange={(e) => onChange({ label: e.target.value })}
          className={`${inputCls} w-40 font-sans`}
        />
        <NumberBox label="From" value={override.start} min={1} max={override.end} onCommit={(n) => onChange({ start: n })} />
        <NumberBox label="To" value={override.end} min={override.start} max={maxFrames} onCommit={(n) => onChange({ end: n })} />
        <span className="font-mono text-note tabular-nums text-[var(--text-muted)]">{(override.end - override.start + 1).toLocaleString()} f</span>
        <span className="flex-1" />
        <button type="button" onClick={onJump} className="rounded-md px-2 py-1 text-mini font-semibold text-[var(--text-muted)] hover:bg-white/[0.06] hover:text-[var(--text-main)]">Go to</button>
        <button type="button" onClick={onUseAsRange} title="Make this range the In/Out work area"
                className="rounded-md px-2 py-1 text-mini font-semibold text-[var(--text-muted)] hover:bg-white/[0.06] hover:text-[var(--text-main)]">Use as In/Out</button>
        <button type="button" onClick={onRemove} className="rounded-md px-2 py-1 text-mini font-semibold text-red-400 hover:bg-red-500/10">Remove</button>
      </div>

      <div className="grid gap-x-5 gap-y-2 sm:grid-cols-2">
        {OVERRIDE_FIELDS.map((f) => {
          const on = override.params[f.id] !== undefined;
          const disabled = override.bypass;
          const inherited = baseValues[f.id];
          return (
            <div key={f.id} className={`flex items-center gap-2 ${disabled ? 'opacity-40' : ''}`} title={f.hint}>
              <input
                type="checkbox" checked={on} disabled={disabled}
                aria-label={`Override ${f.label}`}
                onChange={(e) => setParam(f.id, e.target.checked ? (f.kind === 'range' ? (inherited ?? f.base) : (inherited ?? swapModels[0] ?? '')) : undefined)}
                className="accent-[var(--accent)]"
              />
              <span className="w-32 shrink-0 text-mini text-[var(--text-muted)]">{f.label}</span>
              {f.kind === 'range' ? (
                <>
                  <input
                    type="range" min={f.min} max={f.max} step={f.step}
                    disabled={!on || disabled}
                    value={on ? override.params[f.id] : (inherited ?? f.base)}
                    aria-label={f.label}
                    onChange={(e) => setParam(f.id, Number(e.target.value))}
                    className="h-1 min-w-0 flex-1 accent-[var(--accent)]"
                  />
                  <span className="w-11 shrink-0 text-right font-mono text-note tabular-nums text-[var(--text-main)]">
                    {on ? override.params[f.id] : <span className="text-[var(--text-muted)]">{inherited ?? f.base}</span>}
                  </span>
                </>
              ) : (
                <select
                  disabled={!on || disabled}
                  value={on ? override.params[f.id] : ''}
                  aria-label={f.label}
                  onChange={(e) => setParam(f.id, e.target.value)}
                  className={`${inputCls} min-w-0 flex-1 font-sans`}
                >
                  {!on && <option value="">{inherited || 'Clip default'}</option>}
                  {swapModels.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              )}
            </div>
          );
        })}
      </div>

      <label className={`flex items-center gap-2 text-mini ${bypassOk ? 'text-[var(--text-main)]' : 'text-[var(--text-muted)]'}`}
             title={bypassOk ? 'Render this range with no swap' : 'Bypass needs a targeted detection mode (Selected face): in "all faces" modes there is no match threshold to close.'}>
        <input
          type="checkbox" checked={override.bypass} disabled={!bypassOk && !override.bypass}
          aria-label="Bypass swap for this range"
          onChange={(e) => onChange({ bypass: e.target.checked })}
          className="accent-[var(--accent)]"
        />
        Bypass swap for this range
        {!bypassOk && <span className="text-note text-[var(--text-muted)]">- needs Selected-face detection mode</span>}
      </label>
    </div>
  );
}
