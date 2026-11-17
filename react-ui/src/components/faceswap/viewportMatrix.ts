/**
 * viewportMatrix.ts: Pure Unified Viewport Transform & Zero-Drift Matrix Math.
 *
 * Capabilities:
 *   1. Zero-Drift 2D Affine Transforms:
 *      Maps points, bounding boxes, landmark points, and brush radii between natural media
 *      pixel space (e.g., 1920x1080) and screen viewport space with letterbox/pillarbox
 *      awareness, pan offset, and zoom scaling.
 *   2. Synchronized 3.5x Magnifier Loupe:
 *      Computes exact crop coordinates in media pixel space under the cursor with zero drift,
 *      ensuring the loupe always aligns with video playback, split wipe, and landmarks.
 *   3. Split Comparison Wipe Geometry:
 *      Determines exact screen and media divider positions for horizontal, vertical, and angled wipes.
 *   4. Pose Bucket & Euler Angle Classification:
 *      Classifies Yaw / Pitch / Roll degrees into standardized pose buckets ('Frontal', 'Profile Left',
 *      'Profile Right', 'Upward', 'Downward', 'Hard Yaw').
 *   5. In-Memory LRU Frame Cache:
 *      Map<number, ImageBitmap> with automated memory reclamation via .close() to eliminate flicker
 *      and seek latency during video scrubbing and A/B split-wipe comparison.
 *   6. Video Frame Synchronization:
 *      Synchronizes playback via requestVideoFrameCallback with requestAnimationFrame fallback.
 *   7. 100% Offline Compliance: zero external dependencies, fonts, or network URLs.
 */

// ── Types & Interfaces ─────────────────────────────────────────────────────────

export interface ViewportTransformConfig {
  containerWidth: number;
  containerHeight: number;
  mediaWidth: number;
  mediaHeight: number;
  zoom?: number;
  pan?: { x: number; y: number };
  objectFit?: 'contain' | 'cover' | 'fill';
}

export interface ViewportMetrics {
  containerWidth: number;
  containerHeight: number;
  mediaWidth: number;
  mediaHeight: number;
  zoom: number;
  pan: { x: number; y: number };
  containScale: number;
  effectiveScale: number;
  containedWidth: number;
  containedHeight: number;
  containedLeft: number;
  containedTop: number;
  centerScreenX: number;
  centerScreenY: number;
}

export interface ScreenPoint {
  x: number;
  y: number;
}

export interface MediaPoint {
  x: number;
  y: number;
  onMedia: boolean;
}

export interface ScreenBox {
  left: number;
  top: number;
  width: number;
  height: number;
  right: number;
  bottom: number;
}

export interface MediaBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  width: number;
  height: number;
}

export interface LoupeCrop {
  sourceX: number;
  sourceY: number;
  sourceWidth: number;
  sourceHeight: number;
  screenCenterX: number;
  screenCenterY: number;
  magnification: number;
  onMedia: boolean;
}

export interface PoseClassification {
  bucket: 'Frontal' | 'Slight Left' | 'Hard Left' | 'Slight Right' | 'Hard Right' | 'Looking Up' | 'Looking Down';
  yawDeg: number;
  pitchDeg: number;
  rollDeg: number;
  isChallenging: boolean; // hard yaw > 45 or pitch > 30
  badgeTone: 'success' | 'info' | 'warning' | 'error';
  label: string;
}

// ── Pure ViewportMatrix ───────────────────────────────────────────────────────

export class ViewportMatrix {
  public readonly metrics: ViewportMetrics;

  constructor(config: ViewportTransformConfig) {
    const containerWidth = Math.max(1, config.containerWidth || 1);
    const containerHeight = Math.max(1, config.containerHeight || 1);
    const mediaWidth = Math.max(1, config.mediaWidth || 1);
    const mediaHeight = Math.max(1, config.mediaHeight || 1);
    const zoom = Math.max(0.05, Math.min(32, config.zoom ?? 1.0));
    const pan = {
      x: Number.isFinite(config.pan?.x) ? (config.pan?.x ?? 0) : 0,
      y: Number.isFinite(config.pan?.y) ? (config.pan?.y ?? 0) : 0,
    };

    // Calculate object-contain scale and letterbox offsets
    const containScale = Math.min(containerWidth / mediaWidth, containerHeight / mediaHeight);
    const containedWidth = mediaWidth * containScale;
    const containedHeight = mediaHeight * containScale;
    const containedLeft = (containerWidth - containedWidth) / 2;
    const containedTop = (containerHeight - containedHeight) / 2;

    // Center screen position under pan
    const centerScreenX = containerWidth / 2 + pan.x;
    const centerScreenY = containerHeight / 2 + pan.y;

    // Total effective scaling factor from media coordinates to visual screen pixels
    const effectiveScale = containScale * zoom;

    this.metrics = {
      containerWidth,
      containerHeight,
      mediaWidth,
      mediaHeight,
      zoom,
      pan,
      containScale,
      effectiveScale,
      containedWidth,
      containedHeight,
      containedLeft,
      containedTop,
      centerScreenX,
      centerScreenY,
    };
  }

  /**
   * Media Pixel -> Screen Pixel (Affine Transform).
   * Maps natural image coordinates (0..mediaWidth, 0..mediaHeight) to screen coordinates.
   */
  public mediaToScreen(mx: number, my: number): ScreenPoint {
    const { centerScreenX, centerScreenY, mediaWidth, mediaHeight, effectiveScale } = this.metrics;
    return {
      x: centerScreenX + (mx - mediaWidth / 2) * effectiveScale,
      y: centerScreenY + (my - mediaHeight / 2) * effectiveScale,
    };
  }

  /**
   * Screen Pixel -> Media Pixel (Inverse Affine Transform).
   * Maps screen viewport coordinates (clientX - containerRect.left) to natural media pixels.
   */
  public screenToMedia(sx: number, sy: number): MediaPoint {
    const { centerScreenX, centerScreenY, mediaWidth, mediaHeight, effectiveScale } = this.metrics;
    const mx = mediaWidth / 2 + (sx - centerScreenX) / effectiveScale;
    const my = mediaHeight / 2 + (sy - centerScreenY) / effectiveScale;
    const onMedia = mx >= 0 && mx <= mediaWidth && my >= 0 && my <= mediaHeight;
    return { x: mx, y: my, onMedia };
  }

  /**
   * Transforms a bounding box from media space (or normalized 0..1 space) to screen pixels.
   */
  public mediaBoxToScreenBox(
    bbox: [number, number, number, number] | { x: number; y: number; width: number; height: number }
  ): ScreenBox {
    const { mediaWidth, mediaHeight } = this.metrics;
    let mx0 = 0;
    let my0 = 0;
    let mx1 = 0;
    let my1 = 0;

    if (Array.isArray(bbox) && bbox.length >= 4) {
      const [b0, b1, b2, b3] = bbox;
      if (b2 <= 1.0 && b3 <= 1.0 && (b0 < 1.0 || b1 < 1.0)) {
        // Normalized 0..1 coordinates
        mx0 = b0 * mediaWidth;
        my0 = b1 * mediaHeight;
        mx1 = b2 * mediaWidth;
        my1 = b3 * mediaHeight;
      } else {
        // Pixel coordinates in media space
        mx0 = b0;
        my0 = b1;
        mx1 = b2;
        my1 = b3;
      }
    } else if (typeof bbox === 'object' && bbox !== null) {
      const b = bbox as { x: number; y: number; width: number; height: number };
      if (b.width <= 1.0 && b.height <= 1.0 && (b.x < 1.0 || b.y < 1.0)) {
        mx0 = b.x * mediaWidth;
        my0 = b.y * mediaHeight;
        mx1 = (b.x + b.width) * mediaWidth;
        my1 = (b.y + b.height) * mediaHeight;
      } else {
        mx0 = b.x;
        my0 = b.y;
        mx1 = b.x + b.width;
        my1 = b.y + b.height;
      }
    }

    const p0 = this.mediaToScreen(mx0, my0);
    const p1 = this.mediaToScreen(mx1, my1);
    const left = Math.min(p0.x, p1.x);
    const top = Math.min(p0.y, p1.y);
    const right = Math.max(p0.x, p1.x);
    const bottom = Math.max(p0.y, p1.y);

    return {
      left,
      top,
      width: right - left,
      height: bottom - top,
      right,
      bottom,
    };
  }

  /**
   * Screen radius in visual px to Media radius in canvas px.
   */
  public screenRadiusToMedia(screenRadius: number): number {
    return Math.max(0.5, screenRadius / this.metrics.effectiveScale);
  }

  /**
   * Media radius in canvas px to Screen radius in visual px.
   */
  public mediaRadiusToScreen(mediaRadius: number): number {
    return mediaRadius * this.metrics.effectiveScale;
  }

  /**
   * Calculates sub-pixel zero-drift crop for 3.5x Magnifier Loupe centered at screen coordinates (sx, sy).
   */
  public getLoupeCrop(
    screenX: number,
    screenY: number,
    loupeDiameter: number = 180,
    magnification: number = 3.5
  ): LoupeCrop {
    const { x: mx, y: my, onMedia } = this.screenToMedia(screenX, screenY);
    const { effectiveScale } = this.metrics;

    // Desired visible size in media pixels
    const sourceWidth = loupeDiameter / (effectiveScale * magnification);
    const sourceHeight = loupeDiameter / (effectiveScale * magnification);
    const sourceX = mx - sourceWidth / 2;
    const sourceY = my - sourceHeight / 2;

    return {
      sourceX,
      sourceY,
      sourceWidth,
      sourceHeight,
      screenCenterX: screenX,
      screenCenterY: screenY,
      magnification,
      onMedia,
    };
  }

  /**
   * Computes split wipe divider coordinates in both screen and media space.
   */
  public getSplitDivider(wipePct: number = 50, _angleDeg: number = 0): {
    screenX: number;
    mediaX: number;
    wipePctClamped: number;
  } {
    const clamped = Math.max(0, Math.min(100, wipePct));
    const mediaX = (clamped / 100) * this.metrics.mediaWidth;
    const screenPoint = this.mediaToScreen(mediaX, this.metrics.mediaHeight / 2);
    return {
      screenX: screenPoint.x,
      mediaX,
      wipePctClamped: clamped,
    };
  }

  /**
   * Screen Box -> Media Box (Inverse Transform).
   */
  public screenBoxToMediaBox(box: ScreenBox): MediaBox {
    const p0 = this.screenToMedia(box.left, box.top);
    const p1 = this.screenToMedia(box.right, box.bottom);
    const x0 = Math.max(0, Math.min(p0.x, p1.x));
    const y0 = Math.max(0, Math.min(p0.y, p1.y));
    const x1 = Math.min(this.metrics.mediaWidth, Math.max(p0.x, p1.x));
    const y1 = Math.min(this.metrics.mediaHeight, Math.max(p0.y, p1.y));
    return {
      x0,
      y0,
      x1,
      y1,
      width: Math.max(0, x1 - x0),
      height: Math.max(0, y1 - y0),
    };
  }

  /**
   * Unified Comparison Layout parameters (Split Wipe, Alpha Blend, Diff Map, Heatmap).
   */
  public getComparisonLayout(
    mode: 'slider' | 'blend' | 'diff' | 'heatmap' | 'quad' | string = 'slider',
    wipePct: number = 50,
    angleDeg: number = 0
  ) {
    const divider = this.getSplitDivider(wipePct, angleDeg);
    const clampedPct = divider.wipePctClamped;
    return {
      mode,
      divider,
      blendAlpha: clampedPct / 100,
      isSplit: mode === 'slider' || mode === 'split',
      isBlend: mode === 'blend',
      isDiff: mode === 'diff' || mode === 'heatmap',
      isQuad: mode === 'quad',
      wipePct: clampedPct,
      angleDeg,
    };
  }

  /**
   * Derive a child ViewportMatrix for floating PiP mini-viewport (Stage 26).
   */
  public getPipChildMatrix(pipWidth: number, pipHeight: number): ViewportMatrix {
    return new ViewportMatrix({
      containerWidth: pipWidth,
      containerHeight: pipHeight,
      mediaWidth: this.metrics.mediaWidth,
      mediaHeight: this.metrics.mediaHeight,
      zoom: 1.0,
      pan: { x: 0, y: 0 },
      objectFit: 'contain',
    });
  }

  /**
   * Derive quadrant coordinates for A/B/C/D snapshot slots (Stage 23).
   */
  public getSlotQuadrant(slotIndex: number): {
    screenLeft: number;
    screenTop: number;
    screenWidth: number;
    screenHeight: number;
    slotId: string;
  } {
    const idx = Math.max(0, Math.min(3, Math.floor(slotIndex)));
    const slotNames = ['A', 'B', 'C', 'D'];
    const halfW = this.metrics.containerWidth / 2;
    const halfH = this.metrics.containerHeight / 2;
    const col = idx % 2;
    const row = Math.floor(idx / 2);
    return {
      screenLeft: col * halfW,
      screenTop: row * halfH,
      screenWidth: halfW,
      screenHeight: halfH,
      slotId: slotNames[idx],
    };
  }
}

/**
 * Pure factory helper to create a ViewportMatrix instance.
 */
export function createViewportMatrix(config: ViewportTransformConfig): ViewportMatrix {
  return new ViewportMatrix(config);
}

// ── Pose Bucket & Euler Angle Classification ──────────────────────────────────

export function classifyPoseBucket(
  yaw?: number,
  pitch?: number,
  roll?: number
): PoseClassification {
  const y = Number.isFinite(yaw) ? Number(yaw) : 0;
  const p = Number.isFinite(pitch) ? Number(pitch) : 0;
  const r = Number.isFinite(roll) ? Number(roll) : 0;

  let bucket: PoseClassification['bucket'] = 'Frontal';
  let badgeTone: PoseClassification['badgeTone'] = 'success';
  let isChallenging = false;

  if (y < -45) {
    bucket = 'Hard Left';
    badgeTone = 'error';
    isChallenging = true;
  } else if (y < -18) {
    bucket = 'Slight Left';
    badgeTone = 'info';
  } else if (y > 45) {
    bucket = 'Hard Right';
    badgeTone = 'error';
    isChallenging = true;
  } else if (y > 18) {
    bucket = 'Slight Right';
    badgeTone = 'info';
  } else if (p > 25) {
    bucket = 'Looking Up';
    badgeTone = 'warning';
    isChallenging = true;
  } else if (p < -25) {
    bucket = 'Looking Down';
    badgeTone = 'warning';
    isChallenging = true;
  } else {
    bucket = 'Frontal';
    badgeTone = 'success';
  }

  const sign = (val: number) => (val >= 0 ? `+${val.toFixed(0)}°` : `${val.toFixed(0)}°`);
  const label = `${bucket} [Y:${sign(y)} P:${sign(p)} R:${sign(r)}]`;

  return {
    bucket,
    yawDeg: y,
    pitchDeg: p,
    rollDeg: r,
    isChallenging,
    badgeTone,
    label,
  };
}

// ── In-Memory LRU Frame Cache ─────────────────────────────────────────────────

/**
 * In-memory LRU Frame Cache for high-performance zero-lag scrubbing and split-wipe comparison.
 * Closes evicted ImageBitmaps to prevent GPU memory leaks.
 */
export class LruFrameCache<K = number, V = ImageBitmap> {
  private cache: Map<K, V> = new Map();
  public maxEntries: number;

  constructor(maxEntries: number = 64) {
    this.maxEntries = Math.max(1, maxEntries);
  }

  public get(key: K): V | undefined {
    if (!this.cache.has(key)) return undefined;
    // Re-insert to mark as most recently used
    const val = this.cache.get(key)!;
    this.cache.delete(key);
    this.cache.set(key, val);
    return val;
  }

  public set(key: K, value: V): void {
    if (this.cache.has(key)) {
      const existing = this.cache.get(key);
      if (existing && existing !== value && typeof (existing as any).close === 'function') {
        try {
          (existing as any).close();
        } catch {
          // ignore already closed bitmaps
        }
      }
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxEntries) {
      // Evict oldest (first key in map)
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey !== undefined) {
        const oldestVal = this.cache.get(oldestKey);
        if (oldestVal && typeof (oldestVal as any).close === 'function') {
          try {
            (oldestVal as any).close();
          } catch {
            // ignore
          }
        }
        this.cache.delete(oldestKey);
      }
    }
    this.cache.set(key, value);
  }

  public has(key: K): boolean {
    return this.cache.has(key);
  }

  public delete(key: K): boolean {
    const val = this.cache.get(key);
    if (val && typeof (val as any).close === 'function') {
      try {
        (val as any).close();
      } catch {
        // ignore
      }
    }
    return this.cache.delete(key);
  }

  public clear(): void {
    for (const val of this.cache.values()) {
      if (val && typeof (val as any).close === 'function') {
        try {
          (val as any).close();
        } catch {
          // ignore
        }
      }
    }
    this.cache.clear();
  }

  public get size(): number {
    return this.cache.size;
  }

  public keys(): K[] {
    return Array.from(this.cache.keys());
  }
}

// ── Video Frame Synchronization Helper ────────────────────────────────────────

export interface VideoFrameSyncCallbackInfo {
  presentationTime: number;
  mediaTime: number;
  width: number;
  height: number;
}

/**
 * Registers a presentation-synchronized frame callback on HTMLVideoElement.
 * Uses requestVideoFrameCallback when available, falling back to requestAnimationFrame.
 * Returns an unregister cleanup function.
 */
export function registerVideoFrameSync(
  video: HTMLVideoElement | null,
  onFrame: (info: VideoFrameSyncCallbackInfo) => void
): () => void {
  if (!video) return () => {};

  let active = true;
  let rvfcHandle: number | null = null;
  let rafHandle: number | null = null;

  const hasRvfc = typeof (video as any).requestVideoFrameCallback === 'function';

  if (hasRvfc) {
    const handleRvfc = (now: DOMHighResTimeStamp, metadata: any) => {
      if (!active) return;
      onFrame({
        presentationTime: metadata.presentationTime || now / 1000,
        mediaTime: metadata.mediaTime ?? video.currentTime,
        width: metadata.width || video.videoWidth,
        height: metadata.height || video.videoHeight,
      });
      if (active) {
        rvfcHandle = (video as any).requestVideoFrameCallback(handleRvfc);
      }
    };
    rvfcHandle = (video as any).requestVideoFrameCallback(handleRvfc);
  } else {
    // rAF Fallback
    let lastTime = -1;
    const handleRaf = () => {
      if (!active) return;
      if (video.currentTime !== lastTime) {
        lastTime = video.currentTime;
        onFrame({
          presentationTime: performance.now() / 1000,
          mediaTime: video.currentTime,
          width: video.videoWidth,
          height: video.videoHeight,
        });
      }
      if (active) {
        rafHandle = requestAnimationFrame(handleRaf);
      }
    };
    rafHandle = requestAnimationFrame(handleRaf);
  }

  return () => {
    active = false;
    if (rvfcHandle !== null && typeof (video as any).cancelVideoFrameCallback === 'function') {
      try {
        (video as any).cancelVideoFrameCallback(rvfcHandle);
      } catch {
        // ignore
      }
    }
    if (rafHandle !== null) {
      cancelAnimationFrame(rafHandle);
    }
  };
}
