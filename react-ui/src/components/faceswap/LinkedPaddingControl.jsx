import React, { useState } from 'react';
import { Slider } from '../ui';
import { Icon } from '../../icons';

// ── Mask padding: one slider, or four ───────────────────────────────────────
//
// mask_top / mask_bottom / mask_left / mask_right grow the face mask beyond the
// landmark hull on each side. Nearly everyone wants the same value on all four,
// so the default is ONE slider that writes all of them, and a link button that
// splits it into the four when the sides really do differ (a hat brim, a hand).
//
// Whether it starts linked is read from the values: four equal sides (the
// shipped 0/0/0/0 included) start linked; four that differ start unlinked, so
// opening the panel never silently flattens a deliberate asymmetric mask.
// While linked, the slider shows the LARGEST of the four and says so if they
// differ; moving it sets all four, which is the only thing "linked" can mean.
const SIDES = [
  ['top', 'mask_top', 'Top'],
  ['bottom', 'mask_bottom', 'Bottom'],
  ['left', 'mask_left', 'Left'],
  ['right', 'mask_right', 'Right'],
];

const n = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);

export default function LinkedPaddingControl({ settings, onChangeMany, defaults = {} }) {
  const values = SIDES.map(([, key]) => n(settings?.[key]));
  const allEqual = values.every((v) => v === values[0]);
  const [linked, setLinked] = useState(allEqual);

  const setAll = (v) => onChangeMany(Object.fromEntries(SIDES.map(([, key]) => [key, v])));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-white/70">Mask padding</span>
        <button
          type="button"
          onClick={() => setLinked((v) => !v)}
          aria-pressed={linked}
          aria-label={linked ? 'Unlink top, bottom, left and right padding' : 'Link top, bottom, left and right padding'}
          title={linked ? 'Linked: one value for all four sides. Click to unlink.' : 'Unlinked: each side on its own. Click to link.'}
          className={`grid h-7 w-7 place-items-center rounded-lg border transition-colors ${
            linked ? 'border-[var(--accent)]/40 bg-[var(--accent)]/15 text-accent' : 'border-white/10 bg-white/[0.04] text-white/60 hover:text-white'
          }`}
        >
          {linked ? <Icon.link size={14} /> : <Icon.unlink size={14} />}
        </button>
      </div>

      {linked ? (
        <>
          <Slider
            label="Padding (all sides)"
            info="Grows the face mask beyond the landmark hull by this much on every side. Use the link button to set the sides separately."
            min={0}
            max={2}
            step={0.01}
            value={Math.max(...values)}
            defaultValue={n(defaults.mask_top)}
            onChange={setAll}
          />
          {!allEqual && (
            <p className="text-mini leading-snug text-amber-300/80">The four sides differ right now. Moving this sets all four to the same value.</p>
          )}
        </>
      ) : (
        SIDES.map(([side, key, label]) => (
          <Slider
            key={side}
            label={`Padding ${label.toLowerCase()}`}
            min={0}
            max={2}
            step={0.01}
            value={n(settings?.[key])}
            defaultValue={n(defaults[key])}
            onChange={(v) => onChangeMany({ [key]: v })}
          />
        ))
      )}
    </div>
  );
}
