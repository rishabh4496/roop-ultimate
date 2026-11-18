import React, { useEffect, useMemo, useState } from 'react';
import { KEY_ROWS, KEY_WIDTH, bindingsByKey, boundCount, keyState, tokenOfEvent } from '../../shortcuts/keyboardModel';
import { formatCombo } from '../../shortcuts/combo';

// The `?` overlay's keyboard view: a drawn keyboard where every key a workstation shortcut uses is lit, and the
// keys you press flash as you press them.
//
// Lit keys read the live keymap (remaps included). With a search in the overlay, only the keys whose actions match
// stay lit, so "compare" lights C and the keys around it. Hovering or focusing a key names what it does.
// The flash is a press-feedback cue only: it changes nothing, and it is off for users who prefer reduced motion.

const FLASH_MS = 240;

export default function VirtualKeyboard({ overrides = {}, query = '' }) {
  const byKey = useMemo(() => bindingsByKey(overrides), [overrides]);
  const [pressed, setPressed] = useState({});
  const q = query.trim().toLowerCase();

  // Flash the key under each press. Capture phase, so the flash shows even if a handler stops propagation.
  useEffect(() => {
    const timers = new Set();
    const flash = (token, on) => {
      setPressed((p) => {
        if (on) return { ...p, [token]: true };
        const next = { ...p };
        delete next[token];
        return next;
      });
    };
    const onDown = (e) => {
      const token = tokenOfEvent(e);
      if (!token || !KEY_ROWS.flat().includes(token)) return;
      flash(token, true);
      const t = setTimeout(() => flash(token, false), FLASH_MS);
      timers.add(t);
    };
    window.addEventListener('keydown', onDown, true);
    return () => {
      window.removeEventListener('keydown', onDown, true);
      timers.forEach((t) => clearTimeout(t));
    };
  }, []);

  const litFor = (token) => {
    const st = keyState(token, byKey);
    if (!st.bound) return false;
    if (!q) return true;
    return st.entries.some((e) => e.actions.some((a) => `${a.label} ${e.combo} ${formatCombo(e.combo)}`.toLowerCase().includes(q)));
  };

  const lit = useMemo(() => KEY_ROWS.flat().filter((t) => litFor(t)).length, [byKey, q]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-3" data-testid="virtual-keyboard">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted">
        <span>
          <strong className="text-white/85 tabular-nums">{lit}</strong> keys lit
          {q ? ' match your search' : ` of ${boundCount(byKey)} that drive a workstation action`}
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm bg-[var(--accent)]/70" /> bound</span>
          <span className="inline-flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm bg-white/10" /> free</span>
        </span>
      </div>

      <div className="rounded-xl border border-white/10 bg-black/30 p-3 sm:p-4 overflow-x-auto" role="group" aria-label="Keyboard">
        <div className="mx-auto flex min-w-[40rem] flex-col gap-1.5">
          {KEY_ROWS.map((row, r) => (
            <div key={r} className="flex justify-center gap-1.5">
              {row.map((token) => {
                const st = keyState(token, byKey);
                const on = litFor(token);
                const width = KEY_WIDTH[token] || 1;
                const isPressed = !!pressed[token];
                return (
                  <button
                    key={token}
                    type="button"
                    tabIndex={-1}
                    data-key={token}
                    data-bound={st.bound ? 'yes' : 'no'}
                    data-lit={on ? 'yes' : 'no'}
                    title={st.title}
                    aria-label={`${st.label}: ${st.bound ? `${st.actionCount} action${st.actionCount === 1 ? '' : 's'}` : 'not bound'}`}
                    style={{ flexBasis: `${width * 2.6}rem`, flexGrow: 0, flexShrink: 0 }}
                    className={`h-10 rounded-md border px-1 font-mono text-nano font-bold tabular-nums select-none transition-[background-color,border-color,box-shadow,transform] duration-150 ease-out motion-reduce:transition-none ${
                      on
                        ? 'border-[var(--accent)]/60 bg-[var(--accent)]/25 text-white shadow-[0_0_12px_var(--accent-glow)]'
                        : st.bound
                          ? 'border-white/15 bg-white/[0.06] text-white/45'
                          : 'border-white/[0.06] bg-white/[0.02] text-white/25'
                    } ${isPressed ? 'scale-[0.94] motion-reduce:scale-100 bg-[var(--accent)]/45 border-[var(--accent)] text-white' : ''}`}
                  >
                    {st.label}
                    {st.bound && <span className="ml-0.5 align-super text-[0.6rem] opacity-70">{st.actionCount}</span>}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      <ul className="m-0 grid list-none grid-cols-1 gap-1.5 p-0 text-xs md:grid-cols-2" aria-label="Lit keys">
        {KEY_ROWS.flat().filter((t) => litFor(t)).map((token) => {
          const st = keyState(token, byKey);
          return (
            <li key={token} className="rounded-lg border border-white/10 bg-white/[0.02] px-2.5 py-1.5" data-lit-key={token}>
              <span className="font-mono font-bold text-accent">{st.label}</span>
              <span className="ml-2 text-white/70">
                {st.entries.flatMap((e) => e.actions.map((a) => `${a.label} (${formatCombo(e.combo)})`)).join(' · ')}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
