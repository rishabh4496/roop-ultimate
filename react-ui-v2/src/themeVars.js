// ── Custom theme derivation ───────────────────────────────────────────────
// The 38 built-in themes are hand-written blocks of CSS variables in
// index.css. That is fine for a curated set someone tuned by eye, but it is not
// something a user can author: getting a theme right means setting ~16 related
// variables consistently, and most of them are not independent choices at all —
// `--accent-hover` is just the accent lifted, `--text-muted` is the ink at 55%,
// the scrollbar follows the accent, and so on.
//
// So a custom theme is stored as a short RECIPE — mode, accent, background,
// surface/border strength, radius — and this module expands it into the same
// variable set the built-in blocks declare by hand. Six decisions instead of
// sixteen, and the derived relationships stay correct by construction.
//
// Everything here is plain sRGB arithmetic on hex strings. It deliberately
// avoids a colour library: the whole job is mixing toward black/white and
// emitting rgba(), and a dependency for that would outweigh the code.

// ── Colour primitives ─────────────────────────────────────────────────────

// Accepts '#rgb' and '#rrggbb'. Returns [r,g,b] 0-255, or null when the string
// is not a colour we can parse — callers fall back rather than emitting
// `rgba(NaN,…)`, which would silently blank out a surface.
export function hexToRgb(hex) {
  if (typeof hex !== 'string') return null;
  let h = hex.trim().replace(/^#/, '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  if (!/^[0-9a-fA-F]{6}$/.test(h)) return null;
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

const clamp255 = (n) => Math.max(0, Math.min(255, Math.round(n)));

export function rgbToHex([r, g, b]) {
  return `#${[r, g, b].map((c) => clamp255(c).toString(16).padStart(2, '0')).join('')}`;
}

// Linear mix in sRGB. t=0 keeps `a`, t=1 lands on `b`.
export function mix(a, b, t) {
  const A = hexToRgb(a) || [0, 0, 0];
  const B = hexToRgb(b) || [0, 0, 0];
  return rgbToHex([
    A[0] + (B[0] - A[0]) * t,
    A[1] + (B[1] - A[1]) * t,
    A[2] + (B[2] - A[2]) * t,
  ]);
}

export const lighten = (hex, t) => mix(hex, '#ffffff', t);
export const darken = (hex, t) => mix(hex, '#000000', t);

// `alpha('#E94560', 0.14)` -> 'rgba(233, 69, 96, 0.14)'.
export function alpha(hex, a) {
  const c = hexToRgb(hex) || [0, 0, 0];
  // Trim float noise so the emitted CSS stays readable in devtools.
  const av = Math.round(Math.max(0, Math.min(1, a)) * 1000) / 1000;
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${av})`;
}

// WCAG relative luminance — used to decide whether a chosen background is
// really light or really dark, so the studio can warn when the recipe's `mode`
// disagrees with the colour the user actually picked.
export function luminance(hex) {
  const c = hexToRgb(hex) || [0, 0, 0];
  const f = c.map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2];
}

export function contrastRatio(a, b) {
  const la = luminance(a);
  const lb = luminance(b);
  const hi = Math.max(la, lb);
  const lo = Math.min(la, lb);
  return (hi + 0.05) / (lo + 0.05);
}

// ── The recipe ────────────────────────────────────────────────────────────

// A custom theme as stored in CFG.custom_themes. `name` doubles as the id, so
// it must be unique across presets and customs alike (the studio enforces it).
export const DEFAULT_RECIPE = {
  name: 'My Theme',
  custom: true,
  mode: 'dark',
  accent: '#E94560',
  bg: '#0B0C12',
  surface: 0.55,  // card fill opacity — the "glass" strength
  border: 0.07,   // hairline strength
  radius: 16,     // --radius-lg, in px
};

export const RECIPE_LIMITS = {
  surface: { min: 0.15, max: 0.95, step: 0.01 },
  border: { min: 0.02, max: 0.3, step: 0.005 },
  radius: { min: 0, max: 28, step: 1 },
};

export const normalizeRecipe = (t) => ({ ...DEFAULT_RECIPE, ...(t || {}), custom: true });

// ── Semantic status ───────────────────────────────────────────────────────
// The one part of a theme that is NOT derived from the recipe. "Warning" has to
// look like a warning in every theme, so these do not follow the accent — a
// theme-tinted danger colour on Obsidian would be crimson, i.e. identical to
// its accent, and the distinction the colour exists to make would vanish.
//
// Only `mode` matters, and only for contrast: the dark set is tuned for ink on a
// dark surface, and on a light page the same hues carry ~1.9:1 as text, so the
// light set is the 700-weight equivalents (all above 4.5:1 on white).
//
// These MUST be emitted rather than left to the CSS. A custom theme's variables
// are written as INLINE STYLE, which outranks the `[data-theme-mode="light"]`
// block in index.css — so omitting them would pin a light custom theme to the
// dark values and put 1.9:1 status text on a white card. Same trap as
// `--input-bg-focus` below, and `test_ui_light_themes` guards both.
// Keep in step with the `--ok/--warn/--danger/--info` values in index.css.
export const STATUS_COLORS = {
  dark: { '--ok': '#34D399', '--warn': '#FBBF24', '--danger': '#F87171', '--info': '#60A5FA' },
  light: { '--ok': '#047857', '--warn': '#B45309', '--danger': '#B91C1C', '--info': '#1D4ED8' },
};

// ── Derivation ────────────────────────────────────────────────────────────

/**
 * Expand a recipe into the CSS custom properties a theme must define.
 *
 * Returns a plain object of `--var` -> value, ready to write onto an element's
 * inline style. The key set intentionally mirrors what the hand-written blocks
 * in index.css declare, so a custom theme is indistinguishable from a preset
 * once applied — including `--input-bg-focus`, which only the light presets set
 * and which a light custom theme needs for exactly the same reason (see the
 * `.glass-input:focus` note in index.css).
 */
export function deriveThemeVars(recipe) {
  const t = normalizeRecipe(recipe);
  const light = t.mode === 'light';
  // Guard against an unparseable colour reaching the arithmetic below.
  const accent = hexToRgb(t.accent) ? t.accent : DEFAULT_RECIPE.accent;
  const bg = hexToRgb(t.bg) ? t.bg : (light ? '#e5e9f0' : DEFAULT_RECIPE.bg);

  // `ink` is the colour that sits ON the page; every translucent hairline,
  // scrim and scrollbar is a wash of it. Deriving the neutrals from the
  // background rather than from pure white/black is what keeps a tinted theme
  // (espresso, forest) from looking like grey furniture on a coloured floor.
  const ink = light ? darken(bg, 0.86) : lighten(bg, 0.94);
  const inkHex = ink;

  // The card sits slightly off the page: lifted on dark, recessed on light.
  // Both directions read as "a surface", where a single rule would make one of
  // the two modes look flat.
  const cardTint = light ? lighten(bg, 0.75) : lighten(bg, 0.06);
  const inputTint = light ? lighten(bg, 0.55) : darken(bg, 0.45);

  const surface = Number(t.surface) || DEFAULT_RECIPE.surface;
  const border = Number(t.border) || DEFAULT_RECIPE.border;
  const radius = Number.isFinite(Number(t.radius)) ? Number(t.radius) : DEFAULT_RECIPE.radius;

  // A light theme needs a darker hover (the accent is already bright against
  // white); a dark theme needs a lighter one. Same 12% either way.
  const accentHover = light ? darken(accent, 0.12) : lighten(accent, 0.14);

  return {
    '--bg-gradient': [
      `radial-gradient(at 18% -12%, ${alpha(accent, light ? 0.1 : 0.06)} 0px, transparent 42%)`,
      `radial-gradient(at 100% 108%, ${alpha(accentHover, light ? 0.08 : 0.04)} 0px, transparent 45%)`,
      `linear-gradient(162deg, ${light ? lighten(bg, 0.5) : darken(bg, 0.35)}, ${bg} 55%, ${light ? bg : darken(bg, 0.2)})`,
    ].join(', '),
    '--card-bg': alpha(cardTint, surface),
    '--surface-2': alpha(inkHex, light ? 0.05 : 0.035),
    '--text-main': ink,
    '--text-muted': alpha(inkHex, 0.55),
    '--accent': accent,
    '--accent-hover': accentHover,
    '--accent-glow': alpha(accent, light ? 0.18 : 0.28),
    '--border-color': alpha(inkHex, border),
    // The hover border is the resting one roughly doubled, floored so a very
    // faint hairline still has a visible hover state.
    '--border-strong': alpha(inkHex, Math.min(0.4, Math.max(border * 1.9, border + 0.05))),
    '--input-bg': alpha(inputTint, light ? 0.5 : 0.55),
    '--input-focus': alpha(accent, light ? 0.12 : 0.16),
    '--scrollbar-thumb': alpha(inkHex, light ? 0.14 : 0.13),
    '--scrollbar-thumb-hover': alpha(inkHex, light ? 0.3 : 0.3),
    '--shadow-card': light
      ? '0 1px 2px rgba(0, 0, 0, 0.06), 0 10px 30px rgba(0, 0, 0, 0.07)'
      : '0 1px 2px rgba(0, 0, 0, 0.28), 0 10px 30px rgba(0, 0, 0, 0.22)',
    '--radius-lg': `${radius}px`,
    // The panel's top-edge highlight. White catches light on a dark page; on a
    // light one the panel is already the brighter surface, so the same read
    // comes from a faint inner shade instead. See the note in index.css.
    '--specular': light ? 'rgba(0, 0, 0, 0.035)' : alpha(lighten(bg, 1), 0.05),
    // Semantic status — mode-picked, not accent-derived. See STATUS_COLORS.
    ...STATUS_COLORS[light ? 'light' : 'dark'],
    // Light themes only: see the `.glass-input:focus` comment in index.css.
    // Emitting it unconditionally on dark would darken-then-lighten inputs on
    // focus, so it is scoped the same way the preset blocks scope it.
    ...(light ? { '--input-bg-focus': alpha(lighten(bg, 0.85), 0.92) } : {}),
  };
}

// Swatch colours for the gallery card, so a custom theme previews exactly like
// a preset entry (which carries `accent` / `bg` directly).
export const recipeSwatch = (recipe) => {
  const t = normalizeRecipe(recipe);
  return { accent: t.accent, bg: t.bg, mode: t.mode };
};
