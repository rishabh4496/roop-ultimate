import React from 'react';

// A compact on/off switch for a card header (the enhancer stack's stages). The
// full-size labelled one is `Toggle` in ui.jsx; this carries no visible label
// of its own, so `label` is mandatory and becomes the accessible name.
export default function MiniSwitch({ label, checked, onChange, disabled, title }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={!!checked}
      aria-label={label}
      title={title}
      disabled={disabled}
      onClick={() => !disabled && onChange(!checked)}
      className={`relative h-[18px] w-8 shrink-0 rounded-full border transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)] ${
        checked ? 'border-[var(--accent)] bg-[var(--accent)]' : 'border-white/15 bg-white/[0.06] hover:border-white/30'
      } ${disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}
    >
      <span
        aria-hidden="true"
        className={`absolute left-[2px] top-[2px] h-3 w-3 rounded-full bg-white shadow transition-transform duration-150 ${checked ? 'translate-x-[14px]' : ''}`}
      />
    </button>
  );
}
