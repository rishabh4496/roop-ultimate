import React from 'react';
import { Disclosure } from './InspectorGroup';

/**
 * Which parsed parts of the face the Face Parser mask counts as "swap this".
 *
 * The engine has always produced all 19 CelebAMask-HQ classes and always
 * thrown 10 of them away against a hardcoded list — so hair, glasses, ears and
 * neck were permanently excluded, with no way to say otherwise. That default
 * is right most of the time, which is exactly why it went unnoticed: the cases
 * it is wrong for are the awkward ones (a fringe over the forehead you DO want
 * swapped, heavy glasses whose rims the model traces too tightly, a jawline
 * that needs a little neck to blend).
 *
 * Grouped rather than exposed as 19 classes. Nobody wants to decide about
 * `l_brow` and `r_brow` separately, and a mask with one brow in and one out
 * would be a bug in every case.
 *
 * The regions are a PILL GRID: one toggle per region, two one-click sets, and
 * the per-region grow sliders folded away until someone wants them.
 *
 * GROW is the half that makes this useful rather than merely present. The
 * model's boundaries are tight and correct, and a swap that stops exactly at
 * the parsed hairline still shows a seam — a couple of pixels of growth on the
 * skin is usually what closes it.
 */

// Order is the order they are drawn. Face-inward parts first (the ones on by
// default), then the surrounding parts, so the panel reads as "the face" and
// then "how much around it".
//
// There is no separate "Lips" region: the engine groups the inner mouth and
// both lips as `mouth` (classes 11-13, Mask_FaceParser.PARSER_REGIONS), so the
// pill is named for both. A `lips` pill would be a control the engine ignores,
// which test_parser_regions.py exists to prevent.
const REGION_LABELS = [
  ['skin', 'Skin'],
  ['brows', 'Brows'],
  ['eyes', 'Eyes'],
  ['nose', 'Nose'],
  ['mouth', 'Mouth & lips'],
  ['glasses', 'Glasses'],
  ['ears', 'Ears'],
  ['neck', 'Neck'],
  ['hair', 'Hair'],
  ['hat', 'Hat'],
  ['cloth', 'Clothing'],
];

const REGION_HINTS = {
  skin: 'The face itself. Excluding it makes the mask almost nothing — this is on in every sane configuration.',
  brows: 'Eyebrow shape carries a surprising amount of identity, so this is normally in.',
  eyes: 'The eye openings. Leave in unless you are also using Restore original eyes, which puts the plate back over the top anyway.',
  nose: 'Normally in.',
  mouth: 'Inner mouth and both lips. Leave in unless Restore original mouth is doing that job instead.',
  glasses: 'OFF by default, which keeps the target\'s real glasses. Turn ON only if the source person wears the same frames, or the swap will paint over them.',
  ears: 'Includes earrings. Off by default — ears sit at the silhouette where the alignment is least reliable.',
  neck: 'Includes necklaces. A little neck can help a jawline blend on a low camera angle; too much and the swap runs down the throat.',
  hair: 'OFF by default, and the one that changes the look most. On, the swap covers a fringe hanging over the forehead instead of stopping at it — which is right when the hair is thin or blurred, and very wrong when it is not.',
  hat: 'Almost always leave off; a hat belongs to the plate.',
  cloth: 'Collars and shoulders. There is rarely a reason for this.',
};

const DEFAULT_ON = ['skin', 'brows', 'eyes', 'nose', 'mouth'];
const ALL = REGION_LABELS.map(([key]) => key);

export default function ParserRegions({ regions, grow, onChange }) {
  const on = Array.isArray(regions) && regions.length ? regions : DEFAULT_ON;
  const g = grow && typeof grow === 'object' ? grow : {};

  const toggle = (key) => {
    const next = on.includes(key) ? on.filter((k) => k !== key) : [...on, key];
    // An empty set would produce an empty mask and a completely un-swapped
    // face, which reads as "the app broke" rather than "I turned everything
    // off" — so the last region cannot be removed.
    if (!next.length) return;
    onChange({ regions: next, grow: g });
  };

  const setGrow = (key, px) => onChange({ regions: on, grow: { ...g, [key]: px } });

  const sameSet = (a, b) => a.length === b.length && b.every((k) => a.includes(k));
  const noGrow = !Object.values(g).some((v) => Number(v) > 0);
  const isDefault = sameSet(on, DEFAULT_ON) && noGrow;
  const isAll = sameSet(on, ALL);

  const chip = 'rounded-md border px-2 py-1 text-nano font-bold uppercase tracking-wide transition-colors';
  const growing = REGION_LABELS.filter(([key]) => on.includes(key));

  return (
    <div className="rounded-xl bg-black/25 border border-white/[0.07] p-3 space-y-2.5">
      <div className="flex items-center justify-between gap-2">
        <span className="text-micro font-semibold uppercase tracking-[0.14em] text-muted">
          Parsed regions to swap
        </span>
        <div className="flex gap-1.5">
          <button
            type="button"
            onClick={() => onChange({ regions: ALL, grow: g })}
            aria-pressed={isAll}
            title="Swap every parsed region, accessories and clothing included"
            className={`${chip} ${isAll ? 'border-[var(--accent)]/40 bg-[var(--accent)]/15 text-white' : 'border-white/10 bg-white/[0.04] text-muted hover:text-white'}`}
          >
            Select all
          </button>
          <button
            type="button"
            onClick={() => onChange({ regions: DEFAULT_ON, grow: {} })}
            aria-pressed={isDefault}
            title="Occlusion only: the inner face is swapped; glasses, hair, hat, ears, neck and clothing stay as the original and occlude the swap. This is the mask the engine has always produced."
            className={`${chip} ${isDefault ? 'border-[var(--accent)]/40 bg-[var(--accent)]/15 text-white' : 'border-white/10 bg-white/[0.04] text-muted hover:text-white'}`}
          >
            Occlusion only
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Regions to swap">
        {REGION_LABELS.map(([key, label]) => {
          const active = on.includes(key);
          return (
            <button
              key={key}
              type="button"
              onClick={() => toggle(key)}
              aria-pressed={active}
              title={REGION_HINTS[key]}
              className={`rounded-full border px-2.5 py-1 text-mini font-semibold transition-all duration-150 ${
                active
                  ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-white'
                  : 'bg-white/[0.02] border-white/10 text-muted hover:text-white/75 hover:border-white/20'
              }`}
            >
              {label}
            </button>
          );
        })}
      </div>

      <Disclosure id="inspector.parser-grow" title={`Edge growth${noGrow ? '' : ' (active)'}`}>
        <div className="space-y-1">
          {growing.map(([key, label]) => {
            const px = Number(g[key] || 0);
            return (
              <div key={key} className="flex items-center gap-2">
                <span className="w-[88px] shrink-0 truncate text-mini text-white/70">{label}</span>
                <input
                  type="range"
                  min={0}
                  max={30}
                  step={1}
                  value={px}
                  onChange={(e) => setGrow(key, Number(e.target.value))}
                  onDoubleClick={() => setGrow(key, 0)}
                  aria-label={`${label} grow, pixels`}
                  className="flex-1 min-w-0 accent-[var(--accent)]"
                  title={`Grow ${label.toLowerCase()} outward by ${px}px before it joins the mask. Double-click to reset.`}
                />
                <span className={`w-8 shrink-0 text-right text-mini font-mono tabular-nums ${px > 0 ? 'text-accent' : 'text-muted'}`}>{px}</span>
              </div>
            );
          })}
        </div>
        <p className="text-mini text-muted leading-relaxed">
          Grow is in pixels of the 512² parse, applied to each region separately
          before they are combined — so growing the mouth does not also push the
          outer edge of the face outward.
        </p>
      </Disclosure>
    </div>
  );
}
