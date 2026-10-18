// PostCSS plugin: a per-theme FLOOR under every translucent text colour.
//
// Tailwind compiles `text-white/60` (and every variant of it: `hover:`, `focus-visible:`,
// `group-hover/x:`, `placeholder:`) to
//     color: color-mix(in oklab, var(--color-white) 60%, transparent)
// and index.css repoints `--color-white` at the theme's ink, so the class means "the page ink at
// 60%". How quiet that can be depends on the theme: 60% of white on the near-black default is
// fine, 60% of a dark ink on a pale page, or of white on a mid-tone teal, is under WCAG AA. The
// call sites cannot know (there are ~1,000 of them and the themes are user-extensible), so the
// theme states a floor instead:
//     --ink-floor: 86%      /* no text-white/N renders fainter than the ink at 86% */
// and this plugin rewrites every such rule to
//     color-mix(in oklab, var(--color-white) max(60%, var(--ink-floor, 0%)), transparent)
//
// The same goes for a palette hue with an opacity suffix (`text-emerald-300/80`):
//     color-mix(in oklab, var(--color-emerald-300) 80%, transparent)
// becomes
//     color-mix(in oklab,
//       oklch(from var(--color-emerald-300) clamp(var(--hue-min, 0), l, var(--hue-max, 1)) c h)
//       max(80%, var(--ink-floor, 0%)), transparent)
// so the lightness window index.css gives plain `text-emerald-300` (sunk on a light page, lifted on
// a lighter dark preset) also reaches the translucent forms, which are different class names.
//
// `--ink-floor`, `--hue-min` and `--hue-max` are unset in the default theme, so it renders as before.
// It runs after Tailwind so it sees the generated rules, including the `@supports` branch. Only
// `color` declarations are touched: borders and fills that use the same palette keep their alpha.
const HUES = 'red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose';
const WHITE = /color-mix\(in oklab, var\(--color-white\) (\d+(?:\.\d+)?)%, transparent\)/g;
const HUE = new RegExp(`color-mix\\(in oklab, (var\\(--color-(?:${HUES})-(?:200|300|400)\\)) (\\d+(?:\\.\\d+)?)%, transparent\\)`, 'g');
const EXACT = new RegExp(`^var\\(--color-(?:${HUES})-(?:200|300|400)\\)$`);
const FLOOR = (n) => `max(${n}%, var(--ink-floor, 0%))`;

const plugin = () => ({
  postcssPlugin: 'postcss-ink-floor',
  OnceExit(root) {
    root.walkDecls('color', (decl) => {
      // `text-red-400`, `!text-orange-400`, `hover:text-sky-300`: the plain palette colour.
      if (EXACT.test(decl.value.trim())) {
        decl.value = `oklch(from ${decl.value.trim()} clamp(var(--hue-min, 0), l, var(--hue-max, 1)) c h)`;
        return;
      }
      decl.value = decl.value
        .replace(WHITE, (_m, n) => `color-mix(in oklab, var(--color-white) ${FLOOR(n)}, transparent)`)
        .replace(HUE, (_m, v, n) => `color-mix(in oklab, oklch(from ${v} clamp(var(--hue-min, 0), l, var(--hue-max, 1)) c h) ${FLOOR(n)}, transparent)`);
    });
  },
});
plugin.postcss = true;
export default plugin;
