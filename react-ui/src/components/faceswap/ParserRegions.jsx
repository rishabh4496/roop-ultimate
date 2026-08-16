import React from 'react';

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
 * GROW is the half that makes this useful rather than merely present. The
 * model's boundaries are tight and correct, and a swap that stops exactly at
 * the parsed hairline still shows a seam — a couple of pixels of growth on the
 * skin is usually what closes it.
 */

// Order is the order they are drawn. Face-inward parts first (the ones on by
// default), then the surrounding parts, so the panel reads as "the face" and
// then "how much around it".
const REGION_LABELS = [
  ['skin', 'Skin'],
  ['brows', 'Eyebrows'],
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

  const isDefault =
    on.length === DEFAULT_ON.length && DEFAULT_ON.every((k) => on.includes(k))
    && !Object.values(g).some((v) => Number(v) > 0);

  return (
    <div className="rounded-xl bg-black/25 border border-white/[0.07] p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45">
          Parsed regions to swap
        </span>
        {!isDefault && (
          <button
            type="button"
            onClick={() => onChange({ regions: DEFAULT_ON, grow: {} })}
            className="px-2 py-0.5 rounded-md text-nano font-bold text-white/45 hover:text-[var(--accent)] bg-white/[0.04] border border-white/10 hover:border-[var(--accent)]/30 transition-colors"
            title="Back to the inner face with no growth — the mask this engine has always produced"
          >
            Reset
          </button>
        )}
      </div>

      <div className="space-y-1">
        {REGION_LABELS.map(([key, label]) => {
          const active = on.includes(key);
          const px = Number(g[key] || 0);
          return (
            <div key={key} className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => toggle(key)}
                aria-pressed={active}
                title={REGION_HINTS[key]}
                className={`w-[104px] shrink-0 px-2 py-1 rounded-lg text-mini font-semibold text-left border transition-all duration-150 ${
                  active
                    ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-white'
                    : 'bg-white/[0.02] border-white/10 text-white/45 hover:text-white/75 hover:border-white/20'
                }`}
              >
                {label}
              </button>
              {/* Grow is only meaningful for a region that is in the mask. */}
              <input
                type="range"
                min={0}
                max={30}
                step={1}
                value={px}
                disabled={!active}
                onChange={(e) => setGrow(key, Number(e.target.value))}
                aria-label={`${label} grow, pixels`}
                className="flex-1 min-w-0 accent-[var(--accent)] disabled:opacity-25"
                title={active
                  ? `Grow ${label.toLowerCase()} outward by ${px}px before it joins the mask`
                  : `${label} is excluded from the mask, so there is nothing to grow`}
              />
              <span className={`w-8 shrink-0 text-right text-micro font-mono tabular-nums ${
                active && px > 0 ? 'text-[var(--accent)]' : 'text-white/45'
              }`}>
                {active ? `${px}` : '—'}
              </span>
            </div>
          );
        })}
      </div>

      <p className="text-micro text-white/45 leading-relaxed">
        Grow is in pixels of the 512² parse, applied to each region separately
        before they are combined — so growing the mouth does not also push the
        outer edge of the face outward.
      </p>
    </div>
  );
}
