// WCAG contrast with COMPUTED backgrounds, for the text axe cannot decide.
//
// axe-core gives up ("background color could not be determined due to a
// background gradient") on anything painted over this app's gradient body or its
// translucent glass panels -- ~1,000 nodes across the 9 tabs sit in `incomplete`,
// neither passing nor failing. Leaving them there is not a pass, so this measures
// them: for every visible text node it resolves the foreground colour (alpha and
// ancestor opacity included) and the background by compositing the ancestor
// chain bottom-up from the first opaque layer. A gradient layer is evaluated at
// EACH of its colour stops and the worst contrast wins -- a conservative bound,
// never an average.
//
// `measureContrast` is passed to page.evaluate, so it is one self-contained
// function with no imports and no closure over the module.
//
// Known approximations, all on the lenient side of exactly zero for this UI:
//   * backdrop-filter blur and box-shadow are ignored (blur does not change the
//     mean colour under flat text);
//   * the two fixed 4%-alpha ambient glows are ignored;
//   * text over an <img>/<canvas>/<video> is reported as `unmeasurable`, not
//     passed -- its backdrop is a picture.
export function measureContrast(opts = {}) {
  const AA_NORMAL = 4.5;
  const AA_LARGE = 3;

  const canvas = document.createElement('canvas');
  canvas.width = 1; canvas.height = 1;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  // Any CSS colour (oklch, color(), color-mix result, named) -> sRGB rgba.
  const toRgba = (css) => {
    ctx.clearRect(0, 0, 1, 1);
    ctx.fillStyle = '#000';
    ctx.fillStyle = css;
    ctx.fillRect(0, 0, 1, 1);
    const d = ctx.getImageData(0, 0, 1, 1).data;
    return { r: d[0], g: d[1], b: d[2], a: d[3] / 255 };
  };
  const over = (top, under) => {
    const a = top.a + under.a * (1 - top.a);
    if (a === 0) return { r: 0, g: 0, b: 0, a: 0 };
    const mix = (t, u) => (t * top.a + u * under.a * (1 - top.a)) / a;
    return { r: mix(top.r, under.r), g: mix(top.g, under.g), b: mix(top.b, under.b), a };
  };
  const lum = (c) => {
    const f = (v) => { const s = v / 255; return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b);
  };
  const ratio = (a, b) => {
    const [hi, lo] = lum(a) >= lum(b) ? [lum(a), lum(b)] : [lum(b), lum(a)];
    return (hi + 0.05) / (lo + 0.05);
  };

  // Split on top-level commas (not inside parentheses).
  const splitTop = (s) => {
    let depth = 0; let cur = ''; const out = [];
    for (const ch of s) {
      if (ch === '(') depth += 1;
      if (ch === ')') depth -= 1;
      if (ch === ',' && depth === 0) { out.push(cur.trim()); cur = ''; } else cur += ch;
    }
    if (cur.trim()) out.push(cur.trim());
    return out;
  };

  // Colour stops of ONE computed gradient, paren-balanced (a stop may itself be
  // `color(srgb 1 1 1 / 0.5)`). Positions and angles are dropped.
  const gradientStops = (image) => {
    const stops = [];
    const inner = image.slice(image.indexOf('(') + 1, image.lastIndexOf(')'));
    const args = splitTop(inner);
    for (const arg of args) {
      // colour token = leading function call / hex / keyword
      const m = /^(#[0-9a-f]{3,8}|[a-z-]+\((?:[^()]|\([^()]*\))*\)|[a-z]+)/i.exec(arg);
      if (!m) continue;
      const c = toRgba(m[1]);
      // `to right`, `circle`, `at 20% 0%` and the like parse as black via the
      // fillStyle fallback: keep only tokens that are genuinely colours.
      const probe = new Option().style; probe.color = m[1];
      if (probe.color) stops.push(c);
    }
    return stops;
  };

  // Candidate backgrounds behind `el`: one per colour stop of the gradient that
  // sets the page's base tone, or a single composite when there is none. Layers
  // are collected innermost/topmost first, each node's images above its colour.
  const backgrounds = (el) => {
    const layers = [];
    let unmeasurable = false;
    for (let node = el; node && node.nodeType === 1; node = node.parentElement) {
      const cs = getComputedStyle(node);
      let nodeOpaque = false;
      const img = cs.backgroundImage;
      if (img && img !== 'none') {
        for (const piece of splitTop(img)) {
          if (/gradient\(/.test(piece)) {
            const stops = gradientStops(piece);
            if (stops.length) {
              const opaqueGradient = stops.every((s) => s.a >= 0.999);
              layers.push({ stops, opaqueGradient });
              if (opaqueGradient) nodeOpaque = true;
            }
          } else if (/url\(|image-set\(/.test(piece)) {
            unmeasurable = true;      // a picture; `none` layers are just absent
          }
        }
      }
      const bg = toRgba(cs.backgroundColor);
      if (bg.a > 0) layers.push({ color: bg });
      if (bg.a >= 0.999) nodeOpaque = true;
      if (nodeOpaque) break;
    }
    // Page canvas default is white; reaching here with no opaque layer means
    // html/body were transparent.
    const base = { r: 255, g: 255, b: 255, a: 1 };
    // The dimension we vary: the first fully opaque gradient (the base tone),
    // else the first gradient of any kind.
    let dim = layers.findIndex((l) => l.stops && l.opaqueGradient);
    if (dim === -1) dim = layers.findIndex((l) => l.stops);
    // A translucent glow gradient is taken at its strongest stop (<= ~10% alpha
    // in this UI, so the choice moves the result by well under 0.1:1).
    const strongest = (stops) => stops.reduce((m, s) => (s.a > m.a ? s : m), stops[0]);
    const variants = dim === -1 ? [null] : layers[dim].stops.map((s, i) => i);
    return {
      unmeasurable,
      list: variants.map((vi) => {
        let acc = base;
        for (let i = layers.length - 1; i >= 0; i--) {
          const L = layers[i];
          const c = L.stops ? (i === dim ? L.stops[vi] : strongest(L.stops)) : L.color;
          acc = over(c, acc);
        }
        return acc;
      }),
    };
  };

  const effectiveOpacity = (el) => {
    let o = 1;
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
      o *= parseFloat(getComputedStyle(n).opacity);
    }
    return o;
  };

  const hidden = (el) => {
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
      const cs = getComputedStyle(n);
      if (cs.display === 'none' || cs.visibility === 'hidden' || n.hasAttribute('inert')) return true;
    }
    const r = el.getBoundingClientRect();
    return r.width < 1 || r.height < 1;
  };

  const describe = (el) => {
    const bits = [el.tagName.toLowerCase()];
    if (el.id) bits.push(`#${el.id}`);
    const cls = (typeof el.className === 'string' ? el.className : '').split(/\s+/).filter(Boolean).slice(0, 4);
    if (cls.length) bits.push(`.${cls.join('.')}`);
    return bits.join('');
  };

  const results = { checked: 0, fails: [], unmeasurable: [] };
  const seen = new Set();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let t = walker.nextNode(); t; t = walker.nextNode()) {
    const text = t.nodeValue.replace(/\s+/g, ' ').trim();
    if (!text) continue;
    const el = t.parentElement;
    if (!el || seen.has(el)) continue;
    if (/^(script|style|noscript)$/i.test(el.tagName)) continue;
    if (hidden(el)) continue;
    // WCAG 1.4.3 exempts inactive controls, and "pure decoration" -- which here
    // means aria-hidden: a separator glyph announced to nobody. Content that
    // conveys anything must not hide behind this.
    if (el.closest('[disabled],[aria-disabled="true"],[aria-hidden="true"]')) continue;
    seen.add(el);

    const cs = getComputedStyle(el);
    const fg0 = toRgba(cs.color);
    const op = effectiveOpacity(el);
    // Invisible until hover/focus (opacity 0): there is nothing on screen to
    // measure, and the focus/hover state that reveals it is a different state.
    if (op < 0.05) continue;
    const bgs = backgrounds(el);
    // Text sitting on a picture cannot be judged from styles.
    if (bgs.unmeasurable) { results.unmeasurable.push({ target: describe(el), text: text.slice(0, 40) }); continue; }

    const px = parseFloat(cs.fontSize);
    const bold = parseInt(cs.fontWeight, 10) >= 700;
    const large = px >= 24 || (px >= 18.66 && bold);
    const need = large ? AA_LARGE : AA_NORMAL;

    let worst = Infinity; let worstBg = null; let worstFg = null;
    for (const bg of bgs.list) {
      const fg = over({ ...fg0, a: fg0.a * op }, bg);
      const r = ratio(fg, bg);
      if (r < worst) { worst = r; worstBg = bg; worstFg = fg; }
    }
    results.checked += 1;
    if (worst < need - 0.005) {
      const hex = (c) => `#${[c.r, c.g, c.b].map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')}`;
      results.fails.push({
        target: describe(el), text: text.slice(0, 40), ratio: Number(worst.toFixed(2)), need,
        fg: hex(worstFg), bg: hex(worstBg), size: px, bold,
        cls: (typeof el.className === 'string' ? el.className : ''),
      });
    }
  }
  void opts;
  return results;
}
