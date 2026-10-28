// Canvas drawing for the timeline's lanes. Each function paints ONE lane from plain data and a
// view window; none of them reads React state, and every loop is bounded by what is on screen
// (pixels, visible intervals, visible frames), never by the length of the clip. That is what keeps
// a 50,000-frame clip as cheap to redraw as a 100-frame one.

import {
  FRAME_CELL_DETAIL_PX, FRAME_CELL_MIN_PX, clamp, fmtFrameLabel, labelFor, normalizeView, rulerTicks, viewSpan, xAtFrame,
} from './viewMath';
import { audioSourceTime, visibleIntervals } from './laneModel';
import { describeOverride } from './overrides';

export const LANE_H = { ruler: 26, cache: 4, film: 78, over: 26, audio: 34 };
export const SCENE_ROW_H = 13;
export const SCENE_CUT_H = 16;
export const MAX_PRESENCE_ROWS = 6;
export const IDENTITY_ROW_H = 9;
export const MAX_IDENTITY_ROWS = 4;
export const identityLaneHeight = (rows) => clamp(rows, 1, MAX_IDENTITY_ROWS) * IDENTITY_ROW_H + 2;
export const IDENTITY_AMBER = '#f59e0b';
export const sceneLaneHeight = (rows) => SCENE_CUT_H + clamp(rows, 1, MAX_PRESENCE_ROWS) * SCENE_ROW_H + 3;

export const STATE_COLORS = {
  rendered: '#22c55e',     // green  — a render of these settings covers it
  reuse: '#22d3ee',        // cyan   — motion-compensated smart-skip reuse
  bypass: '#a855f7',       // purple — face-free, passed through
  dirty: '#f59e0b',        // amber  — never rendered / stale
};
export const STATE_LABELS = {
  rendered: 'Rendered', reuse: 'Smart-skip reuse', bypass: 'Face-free bypass', dirty: 'Dirty / unrendered',
};
const CUT_COLOR = '#fb7a3c';

const FONT = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

/** Theme colours the canvas needs, read from the live CSS variables (canvas cannot use var()). */
export function readTheme(el) {
  const cs = getComputedStyle(el);
  const v = (name, fallback) => cs.getPropertyValue(name).trim() || fallback;
  return {
    text: v('--text-main', '#f4f5f7'),
    muted: v('--text-muted', 'rgba(244,245,247,0.55)'),
    border: v('--border-color', 'rgba(255,255,255,0.07)'),
    bg: v('--input-bg', 'rgba(8,9,13,0.55)'),
    surface: v('--surface-2', 'rgba(255,255,255,0.035)'),
    accent: v('--accent', '#E94560'),
  };
}

const roundRect = (ctx, x, y, w, h, r) => {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
};

/** Text clipped to a width with an ellipsis; returns without drawing when nothing fits. */
function fitText(ctx, text, maxW) {
  if (maxW < 10) return '';
  if (ctx.measureText(text).width <= maxW) return text;
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ctx.measureText(`${text.slice(0, mid)}…`).width <= maxW) lo = mid; else hi = mid - 1;
  }
  return lo > 0 ? `${text.slice(0, lo)}…` : '';
}

// ── identity confidence ─────────────────────────────────────────────────────

/**
 * One thin row per person. Each SAMPLED detection is a cell covering half the gap to its neighbours:
 * green when the match is comfortably inside the threshold, dimmer when it is fair, AMBER when it sits
 * near the threshold or barely ahead of the next person. Amber cells get a bright top edge so they stay
 * visible when the strip is zoomed out far enough that a cell is a single pixel.
 */
export function drawIdentityConfidence(ctx, { w, h, theme }, { view, maxFrames, rows, emptyNote }) {
  const v = normalizeView(view, maxFrames);
  ctx.font = FONT;
  ctx.fillStyle = theme.surface;
  ctx.fillRect(0, 0, w, h);
  const shown = rows.slice(0, MAX_IDENTITY_ROWS);
  if (!shown.length) {
    if (emptyNote) { ctx.fillStyle = theme.muted; ctx.fillText(emptyNote, 8, IDENTITY_ROW_H); }
    return;
  }
  shown.forEach((row, r) => {
    const y = 1 + r * IDENTITY_ROW_H;
    const ph = IDENTITY_ROW_H - 2;
    ctx.fillStyle = r % 2 ? 'rgba(255,255,255,0.02)' : 'rgba(255,255,255,0.04)';
    ctx.fillRect(0, y - 1, w, IDENTITY_ROW_H);
    const cells = row.cells;
    // cells are sorted and non-overlapping: binary-search the first visible one, stop at the last.
    let lo = 0;
    let hi = cells.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cells[mid].end < v.start) lo = mid + 1; else hi = mid; }
    for (let i = lo; i < cells.length && cells[i].start <= v.end; i++) {
      const c = cells[i];
      const x0 = xAtFrame(c.start - 0.5, w, v);
      const x1 = xAtFrame(c.end + 0.5, w, v);
      const px0 = Math.max(0, x0);
      const pw = Math.max(1.5, Math.min(w, x1) - px0);
      if (c.amber) {
        ctx.fillStyle = IDENTITY_AMBER;
        ctx.globalAlpha = 0.95;
        ctx.fillRect(px0, y, pw, ph);
        ctx.globalAlpha = 1;
        ctx.fillStyle = '#fff4d6';
        ctx.fillRect(px0, y, pw, 1.5);
      } else {
        ctx.fillStyle = row.color;
        // strength reads as opacity: a fair match is visibly fainter than a strong one
        ctx.globalAlpha = c.tone === 'ok' ? 0.85 : 0.45;
        ctx.fillRect(px0, y, pw, ph);
        ctx.globalAlpha = 1;
      }
    }
    const label = fitText(ctx, row.name, 70);
    if (label) {
      const lw = ctx.measureText(label).width + 8;
      ctx.fillStyle = 'rgba(0,0,0,0.62)';
      roundRect(ctx, 2, y, lw, ph, 3);
      ctx.fill();
      ctx.fillStyle = row.color;
      ctx.fillText(label, 6, y + ph - 1.5);
    }
  });
}

// ── ruler ───────────────────────────────────────────────────────────────────

export function drawRuler(ctx, { w, h, theme }, { view, maxFrames, fps }) {
  const v = normalizeView(view, maxFrames);
  const ppf = w / viewSpan(v);
  const ticks = rulerTicks(v, maxFrames, fps, w);
  ctx.font = FONT;
  ctx.textBaseline = 'alphabetic';

  ctx.fillStyle = theme.border;
  ctx.fillRect(0, h - 1, w, 1);

  // Every frame gets a hairline once frames are wide enough to tell apart.
  if (ppf >= 6) {
    ctx.fillStyle = theme.muted;
    ctx.globalAlpha = 0.28;
    for (let f = Math.ceil(v.start); f <= v.end; f++) ctx.fillRect(Math.round(xAtFrame(f, w, v)), h - 4, 1, 3);
    ctx.globalAlpha = 1;
  } else {
    ctx.fillStyle = theme.muted;
    ctx.globalAlpha = 0.3;
    for (const f of ticks.minor) ctx.fillRect(Math.round(xAtFrame(f, w, v)), h - 5, 1, 4);
    ctx.globalAlpha = 1;
  }

  for (const f of ticks.major) {
    const x = Math.round(xAtFrame(f, w, v));
    ctx.fillStyle = theme.muted;
    ctx.globalAlpha = 0.6;
    ctx.fillRect(x, h - 9, 1, 8);
    ctx.globalAlpha = 1;
    const text = labelFor(ticks.mode, f, fps, maxFrames);
    const tw = ctx.measureText(text).width;
    // Labels at the edges would hang off the lane; anchor those to the edge instead of centring.
    const tx = clamp(x - tw / 2, 1, w - tw - 1);
    ctx.fillStyle = theme.muted;
    ctx.fillText(text, tx, h - 12);
  }
  return ticks;
}

// ── render / cache strip ────────────────────────────────────────────────────

export function drawCacheStrip(ctx, { w, h, theme }, { view, maxFrames, spans }) {
  const v = normalizeView(view, maxFrames);
  ctx.fillStyle = theme.surface;
  ctx.fillRect(0, 0, w, h);
  // `spans` tile the clip in order, so the first one on screen is found by bisection and the walk
  // stops at the first one past the right edge: cost follows what is visible, not the clip.
  let a = 0;
  let b = spans.length;
  while (a < b) {
    const mid = (a + b) >> 1;
    if (spans[mid].end < v.start) a = mid + 1; else b = mid;
  }
  for (let i = a; i < spans.length && spans[i].start <= v.end; i++) {
    const s = spans[i];
    const x0 = Math.max(0, Math.round(xAtFrame(s.start - 0.5, w, v)));
    const x1 = Math.min(w, Math.round(xAtFrame(s.end + 0.5, w, v)));
    ctx.globalAlpha = s.state === 'dirty' ? 0.55 : 0.95;
    ctx.fillStyle = STATE_COLORS[s.state];
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
  }
  ctx.globalAlpha = 1;
}

// ── filmstrip + waveform ────────────────────────────────────────────────────

/** Max of peaks[] over the time span [a, b] of a dataset {t0, t1, peaks}. 0..1. */
function peakOver(ds, a, b) {
  const n = ds.peaks.length;
  const dur = ds.t1 - ds.t0;
  if (!n || dur <= 0 || b < ds.t0 || a > ds.t1) return -1;
  const i0 = clamp(Math.floor(((a - ds.t0) / dur) * n), 0, n - 1);
  const i1 = clamp(Math.ceil(((b - ds.t0) / dur) * n), i0, n - 1);
  let m = 0;
  for (let i = i0; i <= i1; i++) if (ds.peaks[i] > m) m = ds.peaks[i];
  return m / 255;
}

export function drawFilmstrip(ctx, { w, h, theme }, { view, maxFrames, fps, slots, tileFor, waveCoarse, waveFine, thumbsHidden }) {
  const v = normalizeView(view, maxFrames);
  const ppf = w / viewSpan(v);
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, w, h);

  // Stills. A slot with no still of its own borrows the nearest one within the slot's reach, so a
  // zoom never flashes empty cells while the intermediate frames are still being fetched.
  if (!thumbsHidden) {
    for (const s of slots) {
      if (s.dup) continue;
      const reach = Math.max(1, Math.round((s.w / Math.max(ppf, 1e-6)) * 1.5));
      const bmp = tileFor(s.frame, reach);
      if (!bmp) {
        ctx.fillStyle = theme.surface;
        ctx.fillRect(Math.round(s.x), 0, Math.max(1, Math.round(s.w) - 1), h);
        continue;
      }
      // object-fit: cover — crop the still to the slot's aspect instead of squashing it.
      const slotAspect = s.w / h;
      const srcAspect = bmp.width / bmp.height;
      let sx = 0;
      let sy = 0;
      let sw = bmp.width;
      let sh = bmp.height;
      if (srcAspect > slotAspect) { sw = bmp.height * slotAspect; sx = (bmp.width - sw) / 2; }
      else { sh = bmp.width / slotAspect; sy = (bmp.height - sh) / 2; }
      ctx.drawImage(bmp, sx, sy, sw, sh, Math.round(s.x), 0, Math.ceil(s.w), h);
    }
  }

  // Scrim: keeps the waveform and frame numbers legible over any still.
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, 'rgba(0,0,0,0.25)');
  g.addColorStop(0.55, 'rgba(0,0,0,0.10)');
  g.addColorStop(1, 'rgba(0,0,0,0.72)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);

  // Audio amplitude, mirrored about a baseline near the bottom.
  const sets = [waveFine, waveCoarse].filter(Boolean);
  if (sets.length) {
    const rate = fps || 25;
    const base = h - 22;
    const amp = 18;
    ctx.fillStyle = 'rgba(125, 211, 252, 0.62)';
    for (let x = 0; x < w; x++) {
      const fa = v.start + (x / w) * viewSpan(v) - 0.5;
      const fb = v.start + ((x + 1) / w) * viewSpan(v) - 0.5;
      const ta = Math.max(0, (fa - 1) / rate);
      const tb = Math.max(ta, (fb - 1) / rate);
      let p = -1;
      for (const ds of sets) { p = peakOver(ds, ta, tb); if (p >= 0) break; }
      if (p < 0) continue;
      const half = Math.max(0.5, p * amp);
      ctx.fillRect(x, base - half, 1, half * 2);
    }
  }

  // Frame-by-frame view: cell boundaries and exact frame numbers.
  if (ppf >= FRAME_CELL_DETAIL_PX) {
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let f = Math.ceil(v.start); f <= v.end + 1; f++) {
      const x = Math.round(xAtFrame(f - 0.5, w, v)) + 0.5;
      ctx.moveTo(x, h - 14);
      ctx.lineTo(x, h);
    }
    ctx.stroke();
  }
  if (ppf >= FRAME_CELL_MIN_PX) {
    ctx.font = FONT;
    ctx.fillStyle = 'rgba(255,255,255,0.88)';
    ctx.textBaseline = 'alphabetic';
    for (let f = Math.ceil(v.start); f <= v.end; f++) {
      const text = fmtFrameLabel(f, maxFrames);
      const x = xAtFrame(f, w, v) - ctx.measureText(text).width / 2;
      ctx.fillText(text, Math.round(x), h - 3);
    }
  }
}

// ── audio track ────────────────────────────────────────────

/**
 * The delivered audio as its own row: the bound external track, or the target's own audio, drawn where it will PLAY.
 * `waves` are datasets {t0, t1, peaks} (finest first); `slipSeconds` moves them right (+) or left (-) of the video, and
 * the stretch of video the audio does not cover after the slip is left empty with a hatch, so a slip that exposes silence
 * is visible before a render rather than after it.
 */
export function drawAudioTrack(ctx, { w, h, theme }, { view, maxFrames, fps, waves, slipSeconds = 0, active = true }) {
  const v = normalizeView(view, maxFrames);
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, w, h);
  const sets = (waves || []).filter(Boolean);
  const rate = fps || 25;
  const span = viewSpan(v);
  const mid = h / 2;
  const total = sets.reduce((m, ds) => Math.max(m, ds.t1 || 0), 0);
  ctx.fillStyle = active ? 'rgba(167, 139, 250, 0.78)' : 'rgba(167, 139, 250, 0.35)';
  const hatch = [];
  for (let x = 0; x < w; x++) {
    const fa = v.start + (x / w) * span - 0.5;
    const fb = v.start + ((x + 1) / w) * span - 0.5;
    const ta = Math.max(0, (fa - 1) / rate);
    const tb = Math.max(ta, (fb - 1) / rate);
    const sa = audioSourceTime(ta, slipSeconds);
    const sb = audioSourceTime(tb, slipSeconds);
    if (sb < 0 || (total > 0 && sa > total)) { hatch.push(x); continue; }
    let p = -1;
    for (const ds of sets) { p = peakOver(ds, Math.max(0, sa), Math.max(0, sb)); if (p >= 0) break; }
    if (p < 0) continue;
    const half = Math.max(0.5, p * (h / 2 - 3));
    ctx.fillRect(x, mid - half, 1, half * 2);
  }
  if (hatch.length) {
    ctx.fillStyle = 'rgba(255,255,255,0.10)';
    for (const x of hatch) if (x % 4 === 0) ctx.fillRect(x, 3, 1, h - 6);
  }
  ctx.fillStyle = 'rgba(255,255,255,0.12)';
  ctx.fillRect(0, Math.round(mid), w, 1);
}

// ── scene cuts + presence ───────────────────────────────────────────────────

export function drawScenes(ctx, { w, h, theme }, { view, maxFrames, cuts, presence, selectedPersonId, selectedInterval, emptyNote }) {
  const v = normalizeView(view, maxFrames);
  const ppf = w / viewSpan(v);
  ctx.font = FONT;
  ctx.fillStyle = theme.surface;
  ctx.fillRect(0, 0, w, h);

  const rows = presence.slice(0, MAX_PRESENCE_ROWS);
  for (let r = 0; r < Math.max(1, rows.length); r++) {
    ctx.fillStyle = r % 2 ? 'rgba(255,255,255,0.025)' : 'rgba(255,255,255,0.045)';
    ctx.fillRect(0, SCENE_CUT_H + r * SCENE_ROW_H, w, SCENE_ROW_H);
  }

  // Presence pills.
  rows.forEach((row, r) => {
    const y = SCENE_CUT_H + r * SCENE_ROW_H + 1.5;
    const ph = SCENE_ROW_H - 3;
    for (const [s, e] of visibleIntervals(row.intervals, Math.ceil(v.start - 1), Math.floor(v.end + 1))) {
      const x0 = xAtFrame(s - 0.5, w, v);
      const x1 = xAtFrame(e + 0.5, w, v);
      const px0 = Math.max(0, x0);
      const pw = Math.max(3, Math.min(w, x1) - px0);
      const selected = selectedPersonId === row.id
        && selectedInterval && selectedInterval[0] === s && selectedInterval[1] === e;
      ctx.globalAlpha = selectedPersonId && selectedPersonId !== row.id ? 0.5 : 0.92;
      ctx.fillStyle = row.color;
      roundRect(ctx, px0, y, pw, ph, 4);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (selected) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        roundRect(ctx, px0 + 0.75, y + 0.75, pw - 1.5, ph - 1.5, 4);
        ctx.stroke();
      }
    }
    // Sticky row label so a pill scrolled to the edge still says whose it is.
    const label = fitText(ctx, row.name, 70);
    if (label) {
      const lw = ctx.measureText(label).width + 8;
      ctx.fillStyle = 'rgba(0,0,0,0.62)';
      roundRect(ctx, 2, y, lw, ph, 3);
      ctx.fill();
      ctx.fillStyle = row.color;
      ctx.fillText(label, 6, y + ph - 2.5);
    }
  });
  if (!rows.length && emptyNote) {
    ctx.fillStyle = theme.muted;
    ctx.fillText(emptyNote, 8, SCENE_CUT_H + SCENE_ROW_H - 3);
  }

  // Scene cuts, over the pills. Dense clips (thousands of cuts) collapse to a density bar rather
  // than a wall of lines.
  const inView = [];
  for (const c of cuts) if (c >= v.start && c <= v.end) inView.push(c);
  ctx.fillStyle = theme.border;
  ctx.fillRect(0, SCENE_CUT_H - 1, w, 1);
  if (inView.length * 3 > w) {
    const bins = new Uint16Array(w);
    for (const c of inView) bins[clamp(Math.floor(((c - v.start) / viewSpan(v)) * w), 0, w - 1)]++;
    for (let x = 0; x < w; x++) {
      if (!bins[x]) continue;
      ctx.globalAlpha = clamp(0.25 + bins[x] * 0.2, 0.25, 1);
      ctx.fillStyle = CUT_COLOR;
      ctx.fillRect(x, 0, 1, h);
    }
    ctx.globalAlpha = 1;
  } else {
    ctx.fillStyle = CUT_COLOR;
    for (const c of inView) {
      // A cut is the first frame of the new shot: draw it on that frame's left edge.
      const x = Math.round(xAtFrame(c - (ppf >= 3 ? 0.5 : 0), w, v));
      ctx.fillRect(x, 0, 1.5, h);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x + 7, 0);
      ctx.lineTo(x, 7);
      ctx.closePath();
      ctx.fill();
    }
  }
}

// ── overrides ───────────────────────────────────────────────────────────────

export function drawOverrides(ctx, { w, h, theme }, { view, maxFrames, overrides, selectedId, inout, emptyNote }) {
  const v = normalizeView(view, maxFrames);
  ctx.font = FONT;
  ctx.fillStyle = theme.surface;
  ctx.fillRect(0, 0, w, h);

  // The In/Out work area, as a faint band: overrides outside it are not rendered by a ranged run.
  if (inout && (inout.start > 1 || inout.end < maxFrames)) {
    const x0 = clamp(xAtFrame(inout.start - 0.5, w, v), 0, w);
    const x1 = clamp(xAtFrame(inout.end + 0.5, w, v), 0, w);
    ctx.fillStyle = theme.accent;
    ctx.globalAlpha = 0.1;
    ctx.fillRect(x0, 0, Math.max(0, x1 - x0), h);
    ctx.globalAlpha = 1;
  }

  let any = false;
  for (const o of overrides) {
    if (o.end < v.start - 1 || o.start > v.end + 1) continue;
    any = true;
    const x0 = Math.max(0, xAtFrame(o.start - 0.5, w, v));
    const x1 = Math.min(w, xAtFrame(o.end + 0.5, w, v));
    const ow = Math.max(4, x1 - x0);
    const selected = o.id === selectedId;
    ctx.fillStyle = o.bypass ? '#7c3aed' : '#d97706';
    ctx.globalAlpha = selected ? 0.95 : 0.75;
    roundRect(ctx, x0, 3, ow, h - 6, 4);
    ctx.fill();
    ctx.globalAlpha = 1;
    if (o.bypass) {                       // hatching: this range is deliberately NOT swapped
      ctx.save();
      roundRect(ctx, x0, 3, ow, h - 6, 4);
      ctx.clip();
      ctx.strokeStyle = 'rgba(255,255,255,0.22)';
      ctx.lineWidth = 2;
      for (let hx = x0 - h; hx < x0 + ow; hx += 8) {
        ctx.beginPath(); ctx.moveTo(hx, h); ctx.lineTo(hx + h, 0); ctx.stroke();
      }
      ctx.restore();
    }
    if (selected) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      roundRect(ctx, x0 + 0.75, 3.75, ow - 1.5, h - 7.5, 4);
      ctx.stroke();
      ctx.fillStyle = '#ffffff';              // edge grips
      ctx.fillRect(x0 + 1.5, h / 2 - 5, 2, 10);
      ctx.fillRect(x0 + ow - 3.5, h / 2 - 5, 2, 10);
    }
    const text = fitText(ctx, `${o.label} · ${describeOverride(o)}`, ow - 12);
    if (text) {
      ctx.fillStyle = '#ffffff';
      ctx.fillText(text, x0 + 6, h / 2 + 3.5);
    }
  }
  if (!any && emptyNote) {
    ctx.fillStyle = theme.muted;
    ctx.fillText(emptyNote, 8, h / 2 + 3.5);
  }
}
