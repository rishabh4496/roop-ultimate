import React, { useId, useState } from 'react';
import { Icon } from '../../icons';

// ── Inspector accordions ────────────────────────────────────────────────────
//
// The Face Swap inspector was one 100-control column. It is now a few named
// groups (Detection, Masking, Restoration & Color, Temporal, ...) that open and
// close, and remember which are open across a reload.
//
// A closed group UNMOUNTS its body, which is what makes the column dense: ~25
// controls on screen instead of ~100, and ~60 fewer tab stops
// (e2e/tab-stops.spec.js). The cost is that nothing inside a closed group can be
// reached by selector, so anything a test or a command needs to reach must live
// in a group that is open by default.

const KEY = 'roop_inspector_open';

const readAll = () => {
  try {
    const v = JSON.parse(window.localStorage.getItem(KEY) || '{}');
    return v && typeof v === 'object' ? v : {};
  } catch {
    return {};
  }
};

const remember = (id, open) => {
  try {
    window.localStorage.setItem(KEY, JSON.stringify({ ...readAll(), [id]: open }));
  } catch { /* storage unavailable: the state lasts for this session */ }
};

function useOpen(id, defaultOpen) {
  const [open, setOpen] = useState(() => {
    const saved = id ? readAll()[id] : undefined;
    return typeof saved === 'boolean' ? saved : defaultOpen;
  });
  const set = (next) => {
    setOpen(next);
    if (id) remember(id, next);
  };
  return [open, set];
}

/** A top-level inspector group: a titled card whose body opens and closes. */
export default function InspectorGroup({ id, title, icon: Ico, defaultOpen = false, badge, children }) {
  const [open, setOpen] = useOpen(id, defaultOpen);
  const bodyId = useId();
  return (
    <section className="contain-panel rounded-xl border border-white/[0.08] bg-white/[0.02]">
      <h2 className="m-0">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          aria-controls={bodyId}
          className="group/ig flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left"
        >
          {Ico && <Ico size={14} className="text-accent" />}
          <span className="text-mini font-semibold uppercase tracking-[0.14em] text-muted transition-colors group-hover/ig:text-white/70">{title}</span>
          {badge && <span className="ml-1 rounded-full border border-white/10 bg-white/[0.06] px-1.5 py-0.5 text-micro font-bold tracking-wide text-white/70">{badge}</span>}
          <Icon.expand size={12} className={`ml-auto text-muted transition-transform duration-200 ${open ? 'rotate-90' : ''}`} />
        </button>
      </h2>
      {open && <div id={bodyId} className="space-y-3 px-3 pb-3">{children}</div>}
    </section>
  );
}

/**
 * A quieter disclosure for a sub-block inside a group ("Advanced Detector
 * Tuning"). Same unmount-when-closed rule, no card of its own.
 */
export function Disclosure({ id, title, defaultOpen = false, children }) {
  const [open, setOpen] = useOpen(id, defaultOpen);
  const bodyId = useId();
  return (
    <div className="rounded-lg border border-white/[0.06] bg-black/20">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-controls={bodyId}
        className="flex w-full items-center gap-1.5 rounded-lg px-2.5 py-2 text-left text-mini font-semibold text-white/70 transition-colors hover:text-white"
      >
        <Icon.expand size={11} className={`text-muted transition-transform duration-200 ${open ? 'rotate-90' : ''}`} />
        {title}
      </button>
      {open && <div id={bodyId} className="space-y-3 px-2.5 pb-2.5">{children}</div>}
    </div>
  );
}
