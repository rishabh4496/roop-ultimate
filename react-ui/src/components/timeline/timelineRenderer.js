// ── Virtualized Canvas Timeline 60 FPS Renderer ─────────────────────────────
//
// Renders the entire timeline hierarchy directly to an HTML5 <canvas>
// with zero DOM elements created per frame:
//   Track 1: Ruler & SMPTE Timecode Markings + Backend Scene Cuts
//   Track 2: Master Video Filmstrip with In/Out Clamps & Dimming
//   Track 3+: Identity Sub-Tracks (Clustered face presence blocks)
//   Track 4: Parameter Keyframe Curves (Bézier automation lanes)
//   Overlay: Playhead needle, SMPTE badge & Snapping Guides

import { formatSMPTE } from './timelineStore.js';

export const TRACK_CONFIG = {
  RULER_HEIGHT: 32,
  VIDEO_HEIGHT: 52,
  IDENTITY_HEIGHT: 28,
  CURVES_HEIGHT: 72,
  TRACK_GAP: 2,
};

const NICE_STEPS = [
  1, 2, 5, 10, 25, 50, 75, 100, 150, 250, 500, 750, 1000, 1500, 2500, 5000, 10000, 25000,
];

/** Selects an adaptive frame interval so tick labels stay ~70-140px apart */
export function getAdaptiveTickStep(zoom) {
  const minSpacingPx = 80;
  for (const step of NICE_STEPS) {
    if (step * zoom >= minSpacingPx) {
      return step;
    }
  }
  return NICE_STEPS[NICE_STEPS.length - 1];
}

export function frameToX(frame, zoom, scrollLeft) {
  return (frame - 1) * zoom - scrollLeft;
}

export function xToFrame(x, zoom, scrollLeft) {
  return 1 + (x + scrollLeft) / zoom;
}

export class TimelineRenderer {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.options = options;
    this.thumbnailCache = new Map(); // frame -> ImageBitmap
    this.pendingThumbnails = new Set();
    this.worker = null;
    this._initWorker();
  }

  _initWorker() {
    if (typeof Worker === 'undefined') return;
    try {
      this.worker = new Worker(new URL('./thumbnailWorker.js', import.meta.url), {
        type: 'module',
      });
      this.worker.onmessage = (e) => {
        const { type, frame, bitmap } = e.data || {};
        if (type === 'thumbnail_ready' && bitmap) {
          this.thumbnailCache.set(frame, bitmap);
          this.pendingThumbnails.delete(frame);
        }
      };
    } catch (err) {
      console.warn('[TimelineRenderer] Web Worker initialization skipped:', err);
    }
  }

  requestThumbnail(frame, width = 80, height = 48) {
    if (this.thumbnailCache.has(frame) || this.pendingThumbnails.has(frame)) return;
    this.pendingThumbnails.add(frame);
    if (this.worker) {
      this.worker.postMessage({ type: 'decode_thumbnail', frame, width, height });
    }
  }

  /**
   * Main 60 FPS Render Pass:
   * Virtualizes all tracks and draws strictly visible frame slices.
   */
  render(state) {
    const ctx = this.ctx;
    if (!ctx) return;

    const {
      frame,
      maxFrames,
      fps,
      inPoint,
      outPoint,
      zoom,
      scrollLeft,
      canvasWidth,
      canvasHeight,
      sceneCuts,
      identityTracks,
      curvesExpanded,
      curves,
      activeCurveId,
      activeSnapLine,
    } = state;

    const W = canvasWidth;
    const H = canvasHeight;

    // Background Fill
    ctx.fillStyle = '#090A0F';
    ctx.fillRect(0, 0, W, H);

    // Calculate Visible Frame Bounds
    const visibleStartFrame = Math.max(1, Math.floor(xToFrame(0, zoom, scrollLeft)));
    const visibleEndFrame = Math.min(maxFrames, Math.ceil(xToFrame(W, zoom, scrollLeft)));

    let currentY = 0;

    // ── Track 1: Ruler & SMPTE Timecode Markings + Scene Cuts ─────────────
    const rulerH = TRACK_CONFIG.RULER_HEIGHT;
    this._renderRuler(ctx, {
      y: currentY,
      height: rulerH,
      width: W,
      zoom,
      scrollLeft,
      fps,
      visibleStartFrame,
      visibleEndFrame,
      sceneCuts,
    });
    currentY += rulerH + TRACK_CONFIG.TRACK_GAP;

    // ── Track 2: Master Video Filmstrip & In/Out Clamps ───────────────────
    const videoH = TRACK_CONFIG.VIDEO_HEIGHT;
    this._renderMasterVideoTrack(ctx, {
      y: currentY,
      height: videoH,
      width: W,
      zoom,
      scrollLeft,
      inPoint,
      outPoint,
      maxFrames,
      visibleStartFrame,
      visibleEndFrame,
    });
    currentY += videoH + TRACK_CONFIG.TRACK_GAP;

    // ── Track 3+: Identity Sub-Tracks (Clustered Faces) ───────────────────
    const idH = TRACK_CONFIG.IDENTITY_HEIGHT;
    for (let i = 0; i < identityTracks.length; i++) {
      const track = identityTracks[i];
      this._renderIdentityTrack(ctx, {
        y: currentY,
        height: idH,
        width: W,
        zoom,
        scrollLeft,
        track,
        visibleStartFrame,
        visibleEndFrame,
        currentFrame: frame,
      });
      currentY += idH + TRACK_CONFIG.TRACK_GAP;
    }

    // ── Track 4: Parameter Keyframe Curves ────────────────────────────────
    if (curvesExpanded && curves && curves.length > 0) {
      const curveH = TRACK_CONFIG.CURVES_HEIGHT;
      const activeCurve = curves.find((c) => c.id === activeCurveId) || curves[0];
      this._renderKeyframeCurveTrack(ctx, {
        y: currentY,
        height: curveH,
        width: W,
        zoom,
        scrollLeft,
        curve: activeCurve,
        visibleStartFrame,
        visibleEndFrame,
      });
      currentY += curveH;
    }

    // ── Scene Cut Guide Lines (Extending down full canvas) ────────────────
    this._renderSceneCutLines(ctx, {
      sceneCuts,
      zoom,
      scrollLeft,
      canvasHeight: H,
      visibleStartFrame,
      visibleEndFrame,
    });

    // ── Active Magnetic Snap Line ─────────────────────────────────────────
    if (activeSnapLine) {
      const snapX = frameToX(activeSnapLine.frame, zoom, scrollLeft);
      if (snapX >= 0 && snapX <= W) {
        ctx.save();
        ctx.strokeStyle = '#10B981'; // Emerald Snap Guide
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(snapX, 0);
        ctx.lineTo(snapX, H);
        ctx.stroke();
        ctx.restore();
      }
    }

    // ── Playhead Needle & SMPTE Badge ─────────────────────────────────────
    this._renderPlayhead(ctx, {
      frame,
      zoom,
      scrollLeft,
      canvasHeight: H,
      fps,
    });
  }

  // ── Track 1: Ruler ──────────────────────────────────────────────────────
  _renderRuler(ctx, { y, height, width, zoom, scrollLeft, fps, visibleStartFrame, visibleEndFrame, sceneCuts }) {
    // Ruler background
    ctx.fillStyle = '#0F1117';
    ctx.fillRect(0, y, width, height);

    ctx.strokeStyle = '#1F2430';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y + height - 0.5);
    ctx.lineTo(width, y + height - 0.5);
    ctx.stroke();

    const step = getAdaptiveTickStep(zoom);
    const firstTick = Math.floor(visibleStartFrame / step) * step;

    ctx.fillStyle = '#94A3B8';
    ctx.font = '10px "Plus Jakarta Sans", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';

    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;

    for (let f = firstTick; f <= visibleEndFrame + step; f += step) {
      if (f < 1) continue;
      const x = Math.round(frameToX(f, zoom, scrollLeft)) + 0.5;
      if (x < -50 || x > width + 50) continue;

      // Major tick mark
      ctx.beginPath();
      ctx.moveTo(x, y + height - 10);
      ctx.lineTo(x, y + height);
      ctx.stroke();

      // SMPTE Timecode text
      const tc = formatSMPTE(f, fps);
      ctx.fillText(tc, x + 4, y + 4);

      // Minor tick marks
      const minorCount = 4;
      const minorStep = step / minorCount;
      if (minorStep * zoom >= 8) {
        ctx.strokeStyle = '#1E293B';
        for (let m = 1; m < minorCount; m++) {
          const mf = f + m * minorStep;
          const mx = Math.round(frameToX(mf, zoom, scrollLeft)) + 0.5;
          ctx.beginPath();
          ctx.moveTo(mx, y + height - 5);
          ctx.lineTo(mx, y + height);
          ctx.stroke();
        }
        ctx.strokeStyle = '#334155';
      }
    }

    // Scene cut flags on ruler
    for (const cutFrame of sceneCuts) {
      if (cutFrame < visibleStartFrame - 2 || cutFrame > visibleEndFrame + 2) continue;
      const cutX = Math.round(frameToX(cutFrame, zoom, scrollLeft));
      if (cutX < 0 || cutX > width) continue;

      // Red scene-cut indicator diamond
      ctx.fillStyle = '#EF4444';
      ctx.beginPath();
      ctx.moveTo(cutX, y + height - 12);
      ctx.lineTo(cutX + 4, y + height - 8);
      ctx.lineTo(cutX, y + height - 4);
      ctx.lineTo(cutX - 4, y + height - 8);
      ctx.closePath();
      ctx.fill();
    }
  }

  // ── Track 2: Master Video Track ─────────────────────────────────────────
  _renderMasterVideoTrack(ctx, {
    y,
    height,
    width,
    zoom,
    scrollLeft,
    inPoint,
    outPoint,
    maxFrames: _maxFrames,
    visibleStartFrame,
    visibleEndFrame,
  }) {
    // Video track background
    ctx.fillStyle = '#11131A';
    ctx.fillRect(0, y, width, height);

    // Filmstrip thumbnail cells
    const tileW = Math.max(64, Math.min(120, Math.round(80 * Math.max(0.5, zoom))));
    const framesPerTile = Math.max(1, Math.round(tileW / zoom));

    const firstTileFrame = Math.max(1, Math.floor((visibleStartFrame - 1) / framesPerTile) * framesPerTile + 1);

    for (let f = firstTileFrame; f <= visibleEndFrame; f += framesPerTile) {
      const tileX = frameToX(f, zoom, scrollLeft);
      const nextX = frameToX(f + framesPerTile, zoom, scrollLeft);
      const cellW = Math.max(tileW, nextX - tileX);

      const cached = this.thumbnailCache.get(f);
      if (cached) {
        ctx.drawImage(cached, tileX, y, cellW, height);
      } else {
        this.requestThumbnail(f, Math.round(cellW), height);

        // Filmstrip placeholder
        ctx.fillStyle = '#181A22';
        ctx.fillRect(tileX, y + 1, cellW - 1, height - 2);

        // Sprocket holes
        ctx.fillStyle = '#0B0C10';
        for (let sx = tileX + 4; sx < tileX + cellW - 6; sx += 14) {
          ctx.fillRect(sx, y + 2, 5, 3);
          ctx.fillRect(sx, y + height - 5, 5, 3);
        }

        // Frame number stamp
        ctx.fillStyle = '#475569';
        ctx.font = 'bold 9px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`#${f}`, tileX + cellW / 2, y + height / 2);
      }

      // Cell border
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.strokeRect(tileX, y, cellW, height);
    }

    // In / Out Point Trim Mask (Out-of-range is dimmed and hatched)
    const inX = frameToX(inPoint, zoom, scrollLeft);
    const outX = frameToX(outPoint, zoom, scrollLeft);

    // Dim before In-point
    if (inX > 0) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.72)';
      ctx.fillRect(0, y, inX, height);
      this._renderDiagonalHatch(ctx, 0, y, inX, height);
    }

    // Dim after Out-point
    if (outX < width) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.72)';
      ctx.fillRect(outX, y, width - outX, height);
      this._renderDiagonalHatch(ctx, outX, y, width - outX, height);
    }

    // Active Region Border Highlights
    ctx.strokeStyle = '#F59E0B'; // Amber Active Region
    ctx.lineWidth = 1.5;
    const activeLeft = Math.max(0, inX);
    const activeRight = Math.min(width, outX);
    if (activeRight > activeLeft) {
      ctx.strokeRect(activeLeft, y + 0.5, activeRight - activeLeft, height - 1);
    }

    // In-Point Bracket [
    if (inX >= -10 && inX <= width + 10) {
      ctx.fillStyle = '#10B981'; // Emerald In Bracket
      ctx.fillRect(inX, y, 3, height);
      ctx.fillRect(inX, y, 8, 3);
      ctx.fillRect(inX, y + height - 3, 8, 3);
    }

    // Out-Point Bracket ]
    if (outX >= -10 && outX <= width + 10) {
      ctx.fillStyle = '#EF4444'; // Red Out Bracket
      ctx.fillRect(outX - 3, y, 3, height);
      ctx.fillRect(outX - 8, y, 8, 3);
      ctx.fillRect(outX - 8, y + height - 3, 8, 3);
    }
  }

  _renderDiagonalHatch(ctx, x, y, w, h) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let d = x - h; d < x + w; d += 16) {
      ctx.beginPath();
      ctx.moveTo(d, y + h);
      ctx.lineTo(d + h, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  // ── Track 3+: Identity Sub-Tracks ───────────────────────────────────────
  _renderIdentityTrack(ctx, { y, height, width, zoom, scrollLeft, track, visibleStartFrame, visibleEndFrame, currentFrame }) {
    ctx.fillStyle = '#0D0E14';
    ctx.fillRect(0, y, width, height);

    // Track bottom separator
    ctx.strokeStyle = '#181A24';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, y, width, height);

    // Track label badge
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.fillRect(0, y, 70, height);
    ctx.fillStyle = track.color || '#38BDF8';
    ctx.fillRect(0, y + 4, 3, height - 8);

    ctx.fillStyle = '#E2E8F0';
    ctx.font = 'bold 10px "Plus Jakarta Sans", sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(track.name || 'Face', 8, y + height / 2);

    // Active Presence Blocks
    for (const [start, end] of track.intervals || []) {
      if (end < visibleStartFrame || start > visibleEndFrame) continue;

      const blockLeft = frameToX(start, zoom, scrollLeft);
      const blockRight = frameToX(end, zoom, scrollLeft);
      const blockWidth = Math.max(4, blockRight - blockLeft);

      const isActive = currentFrame >= start && currentFrame <= end;

      // Presence Block styling
      ctx.save();
      ctx.fillStyle = track.color || '#38BDF8';
      ctx.globalAlpha = isActive ? 0.45 : 0.25;

      const r = 4;
      ctx.beginPath();
      ctx.roundRect(blockLeft, y + 3, blockWidth, height - 6, r);
      ctx.fill();

      // Border
      ctx.globalAlpha = isActive ? 0.95 : 0.6;
      ctx.strokeStyle = track.color || '#38BDF8';
      ctx.lineWidth = isActive ? 1.5 : 1;
      ctx.stroke();

      // Text label inside block if wide enough
      if (blockWidth > 45) {
        ctx.fillStyle = '#FFFFFF';
        ctx.globalAlpha = 0.9;
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${start}–${end}`, blockLeft + blockWidth / 2, y + height / 2);
      }

      ctx.restore();
    }
  }

  // ── Track 4: Parameter Keyframe Curves (Bézier / Linear) ────────────────
  _renderKeyframeCurveTrack(ctx, { y, height, width, zoom, scrollLeft, curve, visibleStartFrame: _visibleStartFrame, visibleEndFrame: _visibleEndFrame }) {
    ctx.fillStyle = '#0A0B10';
    ctx.fillRect(0, y, width, height);

    ctx.strokeStyle = '#1E222D';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, y, width, height);

    // Track label badge
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.fillRect(0, y, 95, height);
    ctx.fillStyle = curve.color || '#F59E0B';
    ctx.fillRect(0, y + 4, 3, height - 8);

    ctx.fillStyle = '#E2E8F0';
    ctx.font = 'bold 10px "Plus Jakarta Sans", sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(curve.name || 'Parameter', 8, y + 6);

    ctx.fillStyle = '#94A3B8';
    ctx.font = '9px monospace';
    ctx.fillText(`${curve.min} — ${curve.max}`, 8, y + height - 14);

    const keyframes = curve.keyframes || [];
    if (keyframes.length === 0) return;

    // Helper: Value to Track Y
    const valToY = (v) => {
      const norm = (v - curve.min) / (curve.max - curve.min);
      const clamped = Math.max(0, Math.min(1, norm));
      const pad = 10;
      return y + height - pad - clamped * (height - 2 * pad);
    };

    // Draw Smooth Bézier Curve
    ctx.save();
    ctx.strokeStyle = curve.color || '#F59E0B';
    ctx.lineWidth = 2;

    const points = keyframes.map((k) => ({
      x: frameToX(k.frame, zoom, scrollLeft),
      y: valToY(k.value),
      frame: k.frame,
      value: k.value,
      type: k.type || 'bezier',
    }));

    if (points.length > 0) {
      // Area fill under curve
      ctx.beginPath();
      ctx.moveTo(points[0].x, y + height);
      ctx.lineTo(points[0].x, points[0].y);

      for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[i];
        const p1 = points[i + 1];

        if (p0.type === 'linear') {
          ctx.lineTo(p1.x, p1.y);
        } else {
          // Cubic Bézier control points
          const dx = (p1.x - p0.x) / 3;
          ctx.bezierCurveTo(p0.x + dx, p0.y, p1.x - dx, p1.y, p1.x, p1.y);
        }
      }

      ctx.lineTo(points[points.length - 1].x, y + height);
      ctx.closePath();

      const grad = ctx.createLinearGradient(0, y, 0, y + height);
      grad.addColorStop(0, `${curve.color || '#F59E0B'}33`);
      grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = grad;
      ctx.fill();

      // Stroke Line
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[i];
        const p1 = points[i + 1];
        if (p0.type === 'linear') {
          ctx.lineTo(p1.x, p1.y);
        } else {
          const dx = (p1.x - p0.x) / 3;
          ctx.bezierCurveTo(p0.x + dx, p0.y, p1.x - dx, p1.y, p1.x, p1.y);
        }
      }
      ctx.stroke();

      // Keyframe Diamond/Circle Nodes
      for (const pt of points) {
        if (pt.x < -10 || pt.x > width + 10) continue;

        ctx.fillStyle = '#0F172A';
        ctx.strokeStyle = curve.color || '#F59E0B';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  // ── Scene Cut Vertical Markers ──────────────────────────────────────────
  _renderSceneCutLines(ctx, { sceneCuts, zoom, scrollLeft, canvasHeight, visibleStartFrame, visibleEndFrame }) {
    ctx.save();
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.45)'; // Red dashed cut line
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);

    for (const cut of sceneCuts) {
      if (cut < visibleStartFrame - 2 || cut > visibleEndFrame + 2) continue;
      const x = Math.round(frameToX(cut, zoom, scrollLeft)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, TRACK_CONFIG.RULER_HEIGHT);
      ctx.lineTo(x, canvasHeight);
      ctx.stroke();
    }
    ctx.restore();
  }

  // ── Playhead Needle & SMPTE Badge ───────────────────────────────────────
  _renderPlayhead(ctx, { frame, zoom, scrollLeft, canvasHeight, fps }) {
    const x = frameToX(frame, zoom, scrollLeft);

    // Glowing Playhead Needle
    ctx.save();
    ctx.strokeStyle = '#F59E0B'; // Amber Needle
    ctx.lineWidth = 1.5;

    // Subtle glow line
    ctx.shadowColor = '#F59E0B';
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvasHeight);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Top Playhead Badge (Shield badge with SMPTE readout)
    const badgeW = 74;
    const badgeH = 18;
    const bx = Math.round(x - badgeW / 2);
    const by = 2;

    ctx.fillStyle = '#F59E0B';
    ctx.beginPath();
    ctx.moveTo(bx + 3, by);
    ctx.lineTo(bx + badgeW - 3, by);
    ctx.arcTo(bx + badgeW, by, bx + badgeW, by + 3, 3);
    ctx.lineTo(bx + badgeW, by + badgeH - 4);
    ctx.lineTo(x, by + badgeH + 4); // Triangle pointing to needle
    ctx.lineTo(bx, by + badgeH - 4);
    ctx.lineTo(bx, by + 3);
    ctx.arcTo(bx, by, bx + 3, by, 3);
    ctx.closePath();
    ctx.fill();

    // Timecode Text inside badge
    ctx.fillStyle = '#090A0F';
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(formatSMPTE(frame, fps), x, by + badgeH / 2);

    ctx.restore();
  }

  destroy() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    for (const bmp of this.thumbnailCache.values()) {
      bmp?.close?.();
    }
    this.thumbnailCache.clear();
    this.pendingThumbnails.clear();
  }
}
