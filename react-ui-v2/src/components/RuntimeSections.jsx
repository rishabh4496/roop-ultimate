import React from 'react';
import { Section } from './ui';
import { Icon } from '../icons';
import { RUNTIME_SECTIONS, isObj, rowsOf } from './runtimeRows';

// ── The runtime report: all fourteen sections ───────────────────────────────
//
// /api/progress carries `runtime`: the backend's own snapshot of what it is
// doing, as fourteen named sections (roop/runtime_state.SECTION_NAMES). The
// Processing tab reads a few of them into its HUD; this renders every one, in
// the backend's order, so "what is the engine actually using right now" has one
// complete answer outside a run too.
//
// It renders what the backend SENDS. A section the backend has not reported is
// shown as exactly that ("not reported") rather than being filled in or hidden:
// a diagnostics page that quietly omits a section would be reporting success
// about something it has not looked at.
const TONE = (status) => {
  const s = String(status || '').toUpperCase();
  if (/ERROR|FAIL|BLOCK|DEGRADED/.test(s)) return 'text-red-300 border-red-400/30 bg-red-400/10';
  if (/WARN|UNKNOWN|NOT/.test(s)) return 'text-amber-300 border-amber-400/30 bg-amber-400/10';
  return 'text-emerald-300 border-emerald-400/30 bg-emerald-400/10';
};

export default function RuntimeSections({ runtime }) {
  const sections = isObj(runtime?.sections) ? runtime.sections : {};
  const reported = RUNTIME_SECTIONS.filter((n) => isObj(sections[n])).length;

  return (
    <Section title="Runtime diagnostics" icon={Icon.terminal}>
      <p className="-mt-1 text-mini text-muted">
        {reported} of {RUNTIME_SECTIONS.length} sections reported by the engine. Read straight from the backend's runtime snapshot; nothing here is estimated.
      </p>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {RUNTIME_SECTIONS.map((name) => {
          const sec = sections[name];
          const rows = rowsOf(sec?.values);
          return (
            <section key={name} aria-label={`${name} diagnostics`} data-runtime-section={name} className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <h3 className="m-0 text-mini font-semibold uppercase tracking-[0.14em] text-muted">{name}</h3>
                <span className={`rounded-full border px-2 py-0.5 text-micro font-bold uppercase tracking-wide ${sec ? TONE(sec.status) : 'border-white/10 bg-white/[0.04] text-muted'}`}>
                  {sec ? (sec.status || 'REPORTED') : 'Not reported'}
                </span>
              </div>
              {rows.length ? (
                <dl className="m-0 space-y-1">
                  {rows.map(([label, value]) => (
                    <div key={label} className="flex items-baseline justify-between gap-3 text-mini">
                      <dt className="min-w-0 truncate text-muted">{label}</dt>
                      <dd className="m-0 min-w-0 max-w-[60%] truncate text-right font-mono font-semibold text-white/85" title={value}>{value}</dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <p className="m-0 text-mini text-muted">{sec ? 'No values in this section.' : 'The engine has not sent this section yet.'}</p>
              )}
            </section>
          );
        })}
      </div>
    </Section>
  );
}
