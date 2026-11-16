/**
 * VFR-safe PTS frame resolver and exact frame index math.
 * Eliminates 1-frame drift on 23.976, 29.97, and variable frame rate (VFR) clips.
 */

// ── Types ─────────────────────────────────────────────────────────────────────

export interface PtsTableEntry {
  frame_idx: number;
  pts: number;        // Presentation timestamp in seconds (float)
  num?: number;       // Exact numerator if available
  den?: number;       // Exact denominator if available
}

export type PtsTableRawTuple = [number, number, number] | [number, number]; // [frame_idx, num, den] or [frame_idx, pts]

export interface PtsTableRawObject {
  frame_idx?: number;
  frame?: number;
  pts?: number;
  time?: number;
  num?: number;
  den?: number;
}

export type PtsTableRawInput =
  | string                                      // base64/zlib encoded, or JSON string
  | PtsTableRawTuple[]
  | number[]                                    // array of timestamps [0.0, 0.0417, ...]
  | PtsTableRawObject[]
  | PtsTableEntry[]
  | PtsFrameMap
  | null
  | undefined;

export interface MaskBrushPoint {
  x: number;
  y: number;
  pressure?: number;
  time?: number;
}

export interface MaskBrushStroke {
  id: string;
  frame_idx: number;     // EXACT integer frame index
  tool: 'brush' | 'eraser' | string;
  color?: string;
  size: number;
  opacity?: number;
  points: MaskBrushPoint[] | Array<[number, number]>;
  created_at: number;
}

export interface KeyframePoint {
  id: string;
  frame_idx: number;     // EXACT integer frame index
  value?: number;
  weights?: Record<string, number> | number[];
  curve?: 'linear' | 'hold' | 'cubic_bezier' | 'ease' | 'ease-in' | 'ease-out' | 'step' | string;
  bezier?: [number, number, number, number];
  metadata?: Record<string, any>;
}

export interface RangeLockInterval {
  id: string;
  track_id: number;
  frame_start: number;   // EXACT integer frame index (trim in)
  frame_end: number;     // EXACT integer frame index (trim out, inclusive)
  label?: string;
  faceset_id?: string | null;
  source_index?: number | null;
  color?: string;
  locked?: boolean;
}

// ── Base64 & Decompression ────────────────────────────────────────────────────

/**
 * Cross-environment Base64 decoding into Uint8Array.
 */
export function base64ToUint8Array(b64: string): Uint8Array {
  const clean = b64.trim().replace(/\s+/g, '');
  if (typeof (globalThis as any).Buffer !== 'undefined') {
    return new Uint8Array((globalThis as any).Buffer.from(clean, 'base64'));
  }
  const binary = atob(clean);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

/**
 * Pure TypeScript RFC 1951 Deflate/RFC 1950 Zlib decompressor.
 * Works synchronously in any browser or Node.js environment with zero external dependencies.
 */
function inflateRaw(input: Uint8Array, offset = 0): Uint8Array {
  let bitBuf = 0;
  let bitLen = 0;
  let inIdx = offset;

  function readBits(n: number): number {
    while (bitLen < n) {
      if (inIdx >= input.length) return 0;
      bitBuf |= (input[inIdx++] << bitLen);
      bitLen += 8;
    }
    const val = bitBuf & ((1 << n) - 1);
    bitBuf >>>= n;
    bitLen -= n;
    return val;
  }

  function buildTree(lengths: Uint8Array | number[]): { maxLen: number; tree: Record<number, number> } {
    let maxLen = 0;
    for (let i = 0; i < lengths.length; i++) {
      if (lengths[i] > maxLen) maxLen = lengths[i];
    }
    const count = new Int32Array(maxLen + 1);
    for (let i = 0; i < lengths.length; i++) {
      if (lengths[i] > 0) count[lengths[i]]++;
    }
    const nextCode = new Int32Array(maxLen + 1);
    let code = 0;
    for (let i = 1; i <= maxLen; i++) {
      code = (code + count[i - 1]) << 1;
      nextCode[i] = code;
    }
    const tree: Record<number, number> = {};
    for (let i = 0; i < lengths.length; i++) {
      const len = lengths[i];
      if (len > 0) {
        const c = nextCode[len]++;
        let rev = 0;
        for (let j = 0; j < len; j++) {
          rev = (rev << 1) | ((c >> j) & 1);
        }
        tree[(len << 16) | rev] = i;
      }
    }
    return { maxLen, tree };
  }

  function decodeSymbol(treeObj: { maxLen: number; tree: Record<number, number> }): number {
    let code = 0;
    for (let len = 1; len <= treeObj.maxLen; len++) {
      code |= (readBits(1) << (len - 1));
      const key = (len << 16) | code;
      if (key in treeObj.tree) {
        return treeObj.tree[key];
      }
    }
    throw new Error('Invalid Huffman symbol in deflate stream');
  }

  // Pre-built fixed Huffman tables
  const fixedLitLens = new Uint8Array(288);
  for (let i = 0; i <= 143; i++) fixedLitLens[i] = 8;
  for (let i = 144; i <= 255; i++) fixedLitLens[i] = 9;
  for (let i = 256; i <= 279; i++) fixedLitLens[i] = 7;
  for (let i = 280; i <= 287; i++) fixedLitLens[i] = 8;
  const fixedLitTree = buildTree(fixedLitLens);

  const fixedDistLens = new Uint8Array(32);
  fixedDistLens.fill(5);
  const fixedDistTree = buildTree(fixedDistLens);

  const LEXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
  const LBASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
  const DEXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
  const DBASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
  const CLEN_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

  const out: number[] = [];
  let isFinal = 0;

  while (!isFinal) {
    isFinal = readBits(1);
    const type = readBits(2);

    if (type === 0) {
      // Uncompressed block
      bitBuf = 0;
      bitLen = 0;
      if (inIdx + 4 > input.length) break;
      const len = input[inIdx] | (input[inIdx + 1] << 8);
      inIdx += 4;
      for (let i = 0; i < len && inIdx < input.length; i++) {
        out.push(input[inIdx++]);
      }
    } else if (type === 1 || type === 2) {
      let litTree: { maxLen: number; tree: Record<number, number> };
      let distTree: { maxLen: number; tree: Record<number, number> };

      if (type === 1) {
        litTree = fixedLitTree;
        distTree = fixedDistTree;
      } else {
        const hlit = readBits(5) + 257;
        const hdist = readBits(5) + 1;
        const hclen = readBits(4) + 4;
        const clen = new Uint8Array(19);
        for (let i = 0; i < hclen; i++) {
          clen[CLEN_ORDER[i]] = readBits(3);
        }
        const clenTree = buildTree(clen);
        const codeLens = new Uint8Array(hlit + hdist);
        let idx = 0;
        while (idx < hlit + hdist) {
          const sym = decodeSymbol(clenTree);
          if (sym < 16) {
            codeLens[idx++] = sym;
          } else if (sym === 16) {
            let rep = readBits(2) + 3;
            const prev = codeLens[idx - 1];
            while (rep-- > 0 && idx < hlit + hdist) {
              codeLens[idx++] = prev;
            }
          } else if (sym === 17) {
            let rep = readBits(3) + 3;
            while (rep-- > 0 && idx < hlit + hdist) {
              codeLens[idx++] = 0;
            }
          } else if (sym === 18) {
            let rep = readBits(7) + 11;
            while (rep-- > 0 && idx < hlit + hdist) {
              codeLens[idx++] = 0;
            }
          }
        }
        litTree = buildTree(codeLens.subarray(0, hlit));
        distTree = buildTree(codeLens.subarray(hlit));
      }

      while (true) {
        const sym = decodeSymbol(litTree);
        if (sym < 256) {
          out.push(sym);
        } else if (sym === 256) {
          break; // End of block
        } else {
          const lSym = sym - 257;
          const matchLen = LBASE[lSym] + readBits(LEXTRA[lSym]);
          const dSym = decodeSymbol(distTree);
          const matchDist = DBASE[dSym] + readBits(DEXTRA[dSym]);
          let src = out.length - matchDist;
          for (let i = 0; i < matchLen; i++) {
            out.push(out[src++]);
          }
        }
      }
    } else {
      throw new Error(`Unsupported deflate block type ${type}`);
    }
  }

  return new Uint8Array(out);
}

/**
 * Decompress zlib formatted bytes synchronously.
 */
export function decompressZlibSync(data: Uint8Array): Uint8Array {
  if (!data || data.length === 0) return new Uint8Array(0);
  let offset = 0;
  // RFC 1950 zlib header check: CMF (0x78) and FLG
  if (data[0] === 0x78) {
    const cmf = data[0];
    const flg = data[1];
    if (((cmf << 8) | flg) % 31 === 0) {
      offset = 2;
      if (flg & 0x20) offset += 4; // FDICT
    }
  }
  return inflateRaw(data, offset);
}

/**
 * Decompress zlib formatted bytes asynchronously using DecompressionStream if available.
 */
export async function decompressZlibAsync(data: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream !== 'undefined') {
    try {
      const ds = new DecompressionStream('deflate');
      const writer = ds.writable.getWriter();
      writer.write(data as any);
      writer.close();
      const res = await new Response(ds.readable).arrayBuffer();
      return new Uint8Array(res);
    } catch {
      // Fallback to sync inflate
    }
  }
  return decompressZlibSync(data);
}

// ── Decoder ───────────────────────────────────────────────────────────────────

/**
 * Parse an array of raw items into structured PtsTableEntry objects.
 */
function parseRawArray(arr: any[]): PtsTableEntry[] {
  if (!Array.isArray(arr) || arr.length === 0) return [];
  const entries: PtsTableEntry[] = [];

  for (let i = 0; i < arr.length; i++) {
    const item = arr[i];
    if (item == null) continue;

    if (Array.isArray(item)) {
      if (item.length >= 3) {
        // [frame_idx, num, den]
        const frame_idx = Math.round(Number(item[0]));
        const num = Number(item[1]);
        const den = Number(item[2]);
        const pts = den !== 0 ? num / den : 0;
        entries.push({ frame_idx, pts, num, den });
      } else if (item.length === 2) {
        // [frame_idx, pts]
        const frame_idx = Math.round(Number(item[0]));
        const pts = Number(item[1]);
        entries.push({ frame_idx, pts });
      } else if (item.length === 1) {
        entries.push({ frame_idx: i, pts: Number(item[0]) });
      }
    } else if (typeof item === 'number') {
      // Direct timestamp number: index is frame_idx
      entries.push({ frame_idx: i, pts: item });
    } else if (typeof item === 'object') {
      const frame_idx = Math.round(
        Number(item.frame_idx !== undefined ? item.frame_idx : item.frame !== undefined ? item.frame : i)
      );
      let pts = 0;
      let num: number | undefined;
      let den: number | undefined;

      if (item.num !== undefined && item.den !== undefined && item.den !== 0) {
        num = Number(item.num);
        den = Number(item.den);
        pts = num / den;
      } else if (item.pts !== undefined) {
        pts = Number(item.pts);
      } else if (item.time !== undefined) {
        pts = Number(item.time);
      }
      entries.push({ frame_idx, pts, num, den });
    }
  }

  // Sort ascending by frame_idx
  return entries.sort((a, b) => a.frame_idx - b.frame_idx);
}

/**
 * Decode and consume `pts_table` returned by `1 Pre-Scan` synchronously.
 * Supports:
 * - base64/zlib encoded string
 * - JSON string array
 * - raw `[[frame_idx, num, den], ...]`
 * - array of timestamps `[0.0, 0.0417, ...]`
 * - array of objects `{ frame_idx, pts }`
 * - existing `PtsTableEntry[]` or `PtsFrameMap`
 */
export function decodePtsTable(input: PtsTableRawInput): PtsTableEntry[] {
  if (!input) return [];

  if (input instanceof PtsFrameMap) {
    return [...input.entries];
  }

  if (Array.isArray(input)) {
    return parseRawArray(input);
  }

  if (typeof input === 'string') {
    const trimmed = input.trim();
    if (!trimmed) return [];

    // Check if direct JSON string
    if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
      try {
        const parsed = JSON.parse(trimmed);
        return parseRawArray(Array.isArray(parsed) ? parsed : (parsed.pts_table || parsed.entries || []));
      } catch {
        return [];
      }
    }

    // Try Base64 / Zlib decode
    try {
      const bytes = base64ToUint8Array(trimmed);
      // Check if uncompressed ASCII JSON
      if (bytes.length > 0 && (bytes[0] === 0x5b || bytes[0] === 0x7b)) {
        const text = new TextDecoder().decode(bytes);
        const parsed = JSON.parse(text);
        return parseRawArray(Array.isArray(parsed) ? parsed : (parsed.pts_table || parsed.entries || []));
      }
      // Decompress zlib
      const decompressed = decompressZlibSync(bytes);
      const text = new TextDecoder().decode(decompressed);
      const parsed = JSON.parse(text);
      return parseRawArray(Array.isArray(parsed) ? parsed : (parsed.pts_table || parsed.entries || []));
    } catch {
      return [];
    }
  }

  return [];
}

/**
 * Asynchronously decode pts_table (useful for large tables using native DecompressionStream).
 */
export async function decodePtsTableAsync(input: PtsTableRawInput): Promise<PtsTableEntry[]> {
  if (!input) return [];

  if (input instanceof PtsFrameMap) {
    return [...input.entries];
  }

  if (Array.isArray(input)) {
    return parseRawArray(input);
  }

  if (typeof input === 'string') {
    const trimmed = input.trim();
    if (!trimmed) return [];

    if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
      try {
        const parsed = JSON.parse(trimmed);
        return parseRawArray(Array.isArray(parsed) ? parsed : (parsed.pts_table || parsed.entries || []));
      } catch {
        return [];
      }
    }

    try {
      const bytes = base64ToUint8Array(trimmed);
      if (bytes.length > 0 && (bytes[0] === 0x5b || bytes[0] === 0x7b)) {
        const text = new TextDecoder().decode(bytes);
        const parsed = JSON.parse(text);
        return parseRawArray(Array.isArray(parsed) ? parsed : (parsed.pts_table || parsed.entries || []));
      }
      const decompressed = await decompressZlibAsync(bytes);
      const text = new TextDecoder().decode(decompressed);
      const parsed = JSON.parse(text);
      return parseRawArray(Array.isArray(parsed) ? parsed : (parsed.pts_table || parsed.entries || []));
    } catch {
      return [];
    }
  }

  return [];
}

// ── Core Frame / Time Resolution ──────────────────────────────────────────────

/**
 * Binary search (O(log N)) to find the exact integer frame_idx for a given time in seconds.
 * Eliminates 1-frame drift on 23.976, 29.97, and variable frame rate (VFR) clips.
 *
 * @param currentTimeSeconds Current playback / seek time in seconds
 * @param ptsTable Decoded entries, raw table input, or PtsFrameMap instance
 * @param fallbackFps FPS to use if table is absent or empty (defaults to 30)
 * @param mode 'floor' (default, video frame active interval [pts[i], pts[i+1])) or 'nearest'
 */
export function resolveFrameFromTime(
  currentTimeSeconds: number,
  ptsTable?: PtsTableRawInput | PtsFrameMap | PtsTableEntry[],
  fallbackFps = 30,
  mode: 'floor' | 'nearest' = 'floor'
): number {
  const time = Number.isFinite(currentTimeSeconds) ? currentTimeSeconds : 0;
  const fps = Number.isFinite(fallbackFps) && fallbackFps > 0 ? fallbackFps : 30;

  let entries: PtsTableEntry[];
  if (ptsTable instanceof PtsFrameMap) {
    entries = ptsTable.entries;
  } else if (Array.isArray(ptsTable) && ptsTable.length > 0 && typeof (ptsTable[0] as any)?.pts === 'number') {
    entries = ptsTable as PtsTableEntry[];
  } else {
    entries = decodePtsTable(ptsTable as PtsTableRawInput);
  }

  if (!entries || entries.length === 0) {
    return Math.max(0, mode === 'nearest' ? Math.round(time * fps) : Math.floor(time * fps));
  }

  const N = entries.length;
  // Boundary clamps
  if (time <= entries[0].pts) {
    return entries[0].frame_idx;
  }
  if (time >= entries[N - 1].pts) {
    return entries[N - 1].frame_idx;
  }

  // 1 microsecond epsilon to avoid float rounding edge flicker
  const EPSILON = 1e-6;

  let low = 0;
  let high = N - 1;
  let bestIdx = 0;

  while (low <= high) {
    const mid = (low + high) >>> 1;
    if (entries[mid].pts <= time + EPSILON) {
      bestIdx = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  if (mode === 'nearest' && bestIdx < N - 1) {
    const diffCurr = Math.abs(time - entries[bestIdx].pts);
    const diffNext = Math.abs(entries[bestIdx + 1].pts - time);
    if (diffNext < diffCurr) {
      return entries[bestIdx + 1].frame_idx;
    }
  }

  return entries[bestIdx].frame_idx;
}

/**
 * Return the exact presentation timestamp in seconds for a given integer frame index.
 *
 * @param frameIdx Exact integer frame index
 * @param ptsTable Decoded entries, raw table input, or PtsFrameMap instance
 * @param fallbackFps FPS to use if table is absent or index out of range (defaults to 30)
 */
export function resolveTimeFromFrame(
  frameIdx: number,
  ptsTable?: PtsTableRawInput | PtsFrameMap | PtsTableEntry[],
  fallbackFps = 30
): number {
  const f = Math.round(Number.isFinite(frameIdx) ? frameIdx : 0);
  const fps = Number.isFinite(fallbackFps) && fallbackFps > 0 ? fallbackFps : 30;

  let entries: PtsTableEntry[];
  if (ptsTable instanceof PtsFrameMap) {
    entries = ptsTable.entries;
  } else if (Array.isArray(ptsTable) && ptsTable.length > 0 && typeof (ptsTable[0] as any)?.pts === 'number') {
    entries = ptsTable as PtsTableEntry[];
  } else {
    entries = decodePtsTable(ptsTable as PtsTableRawInput);
  }

  if (!entries || entries.length === 0) {
    return Math.max(0, f) / fps;
  }

  // Fast direct array lookup if 0-based sequential
  if (f >= 0 && f < entries.length && entries[f].frame_idx === f) {
    const entry = entries[f];
    return entry.num !== undefined && entry.den !== undefined && entry.den !== 0
      ? entry.num / entry.den
      : entry.pts;
  }

  // Binary search by frame_idx in case table has sparse indices or frame_start offset
  let low = 0;
  let high = entries.length - 1;

  while (low <= high) {
    const mid = (low + high) >>> 1;
    const entry = entries[mid];
    if (entry.frame_idx === f) {
      return entry.num !== undefined && entry.den !== undefined && entry.den !== 0
        ? entry.num / entry.den
        : entry.pts;
    }
    if (entry.frame_idx < f) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  // Extrapolate if outside table range
  if (f < entries[0].frame_idx) {
    if (f <= 0) return 0;
    const dt = entries.length > 1
      ? (entries[1].pts - entries[0].pts) / Math.max(1, entries[1].frame_idx - entries[0].frame_idx)
      : 1 / fps;
    return Math.max(0, entries[0].pts - (entries[0].frame_idx - f) * dt);
  }

  const last = entries[entries.length - 1];
  const dt = entries.length > 1
    ? (last.pts - entries[entries.length - 2].pts) / Math.max(1, last.frame_idx - entries[entries.length - 2].frame_idx)
    : 1 / fps;
  return last.pts + (f - last.frame_idx) * dt;
}

// ── Helper Math & Clamp Functions ─────────────────────────────────────────────

/**
 * Coerce any value to a strictly valid, safe integer frame index clamped within [minFrame, maxFrames - 1].
 */
export function clampFrameIdx(frameIdx: unknown, minFrame = 0, maxFrames = Infinity): number {
  const n = typeof frameIdx === 'number' ? frameIdx : Number(frameIdx);
  if (!Number.isFinite(n)) return Math.max(0, Math.floor(minFrame));
  const rounded = Math.round(n);
  const min = Math.max(0, Math.floor(minFrame));
  const max = Number.isFinite(maxFrames) && maxFrames > 0 ? Math.floor(maxFrames) - 1 : Infinity;
  return Math.max(min, Math.min(max, rounded));
}

/**
 * Format a frame index into SMPTE timecode string HH:MM:SS:FF.
 */
export function frameToTimecode(frameIdx: number, fps = 30): string {
  const f = Math.max(0, Math.round(frameIdx));
  const roundedFps = Math.max(1, Math.round(fps));
  const frames = f % roundedFps;
  const totalSeconds = Math.floor(f / roundedFps);
  const s = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const m = totalMinutes % 60;
  const h = Math.floor(totalMinutes / 60);

  const pad = (v: number, len = 2) => String(v).padStart(len, '0');
  return `${pad(h)}:${pad(m)}:${pad(s)}:${pad(frames)}`;
}

/**
 * Parse an SMPTE timecode HH:MM:SS:FF into an integer frame index.
 */
export function timecodeToFrame(timecode: string, fps = 30): number {
  if (!timecode) return 0;
  const parts = timecode.trim().split(':').map((p) => Number(p));
  if (parts.some((p) => !Number.isFinite(p))) return 0;

  const roundedFps = Math.max(1, Math.round(fps));
  if (parts.length === 4) {
    const [h, m, s, f] = parts;
    return Math.round((h * 3600 + m * 60 + s) * roundedFps + f);
  }
  if (parts.length === 3) {
    const [h, m, s] = parts;
    return Math.round((h * 3600 + m * 60 + s) * roundedFps);
  }
  if (parts.length === 2) {
    const [m, s] = parts;
    return Math.round((m * 60 + s) * roundedFps);
  }
  return 0;
}

/**
 * Format seconds into human-readable MM:SS.mmm or SS.mmm.
 */
export function formatSeconds(seconds: number): string {
  const sec = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(3).padStart(6, '0')}`;
}

// ── Mask Brush Stroke Helpers ─────────────────────────────────────────────────

export function createMaskBrushStroke(
  stroke: Partial<MaskBrushStroke> & { frame_idx: number; size: number; points: any[] },
  maxFrames = Infinity
): MaskBrushStroke {
  return {
    id: stroke.id || `stroke_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    frame_idx: clampFrameIdx(stroke.frame_idx, 0, maxFrames),
    tool: stroke.tool || 'brush',
    color: stroke.color || '#ff0055',
    size: Math.max(1, Math.round(stroke.size || 20)),
    opacity: typeof stroke.opacity === 'number' ? Math.max(0, Math.min(1, stroke.opacity)) : 1.0,
    points: Array.isArray(stroke.points) ? stroke.points : [],
    created_at: typeof stroke.created_at === 'number' ? stroke.created_at : Date.now(),
  };
}

export function filterStrokesByFrame(strokes: MaskBrushStroke[], frameIdx: number): MaskBrushStroke[] {
  const target = clampFrameIdx(frameIdx);
  return (strokes || []).filter((s) => s && s.frame_idx === target);
}

// ── Keyframe Helpers ──────────────────────────────────────────────────────────

export function createKeyframePoint(
  kf: Partial<KeyframePoint> & { frame_idx: number },
  maxFrames = Infinity
): KeyframePoint {
  return {
    id: kf.id || `kf_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    frame_idx: clampFrameIdx(kf.frame_idx, 0, maxFrames),
    value: typeof kf.value === 'number' ? kf.value : 1.0,
    weights: kf.weights,
    curve: kf.curve || 'linear',
    bezier: kf.bezier,
    metadata: kf.metadata,
  };
}

export function clampKeyframe(kf: KeyframePoint, minFrame = 0, maxFrames = Infinity): KeyframePoint {
  return {
    ...kf,
    frame_idx: clampFrameIdx(kf.frame_idx, minFrame, maxFrames),
  };
}

export function normalizeKeyframePoints(list: KeyframePoint[], maxFrames = Infinity): KeyframePoint[] {
  const byFrame = new Map<number, KeyframePoint>();
  for (const item of list || []) {
    if (!item) continue;
    const clamped = clampKeyframe(item, 0, maxFrames);
    byFrame.set(clamped.frame_idx, clamped);
  }
  return [...byFrame.values()].sort((a, b) => a.frame_idx - b.frame_idx);
}

// ── Range Lock Trim Handles Helpers ───────────────────────────────────────────

export function createRangeLockInterval(
  interval: Partial<RangeLockInterval> & { frame_start: number; frame_end: number; track_id?: number },
  maxFrames = Infinity
): RangeLockInterval {
  const inFrame = clampFrameIdx(interval.frame_start, 0, maxFrames);
  const outFrame = clampFrameIdx(interval.frame_end, inFrame, maxFrames);
  return {
    id: interval.id || `range_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    track_id: typeof interval.track_id === 'number' ? Math.round(interval.track_id) : 0,
    frame_start: inFrame,
    frame_end: Math.max(inFrame, outFrame),
    label: interval.label || `Track ${interval.track_id ?? 0}`,
    faceset_id: interval.faceset_id ?? null,
    source_index: interval.source_index ?? null,
    color: interval.color,
    locked: Boolean(interval.locked),
  };
}

export function clampRangeLockInterval(
  interval: RangeLockInterval,
  minFrame = 0,
  maxFrames = Infinity
): RangeLockInterval {
  const start = clampFrameIdx(interval.frame_start, minFrame, maxFrames);
  const end = clampFrameIdx(interval.frame_end, start, maxFrames);
  return {
    ...interval,
    frame_start: start,
    frame_end: Math.max(start, end),
  };
}

// ── PtsFrameMap Container Class ───────────────────────────────────────────────

export class PtsFrameMap {
  readonly entries: PtsTableEntry[];
  readonly isVfr: boolean;
  readonly fps: number;
  readonly duration: number;

  constructor(rawInput?: PtsTableRawInput, fallbackFps = 30, isVfrHint?: boolean) {
    this.entries = decodePtsTable(rawInput);
    this.fps = Number.isFinite(fallbackFps) && fallbackFps > 0 ? fallbackFps : 30;

    if (this.entries.length > 1) {
      this.duration = this.entries[this.entries.length - 1].pts;
      // Detect VFR if delta between frames has variance > 5%
      if (typeof isVfrHint === 'boolean') {
        this.isVfr = isVfrHint;
      } else {
        const deltas: number[] = [];
        for (let i = 1; i < Math.min(100, this.entries.length); i++) {
          deltas.push(this.entries[i].pts - this.entries[i - 1].pts);
        }
        const avg = deltas.reduce((a, b) => a + b, 0) / deltas.length;
        const maxDev = Math.max(...deltas.map((d) => Math.abs(d - avg)));
        this.isVfr = avg > 0 && maxDev / avg > 0.05;
      }
    } else {
      this.duration = this.entries.length === 1 ? this.entries[0].pts : 0;
      this.isVfr = Boolean(isVfrHint);
    }
  }

  get length(): number {
    return this.entries.length;
  }

  getEntry(frameIdx: number): PtsTableEntry | undefined {
    const f = clampFrameIdx(frameIdx);
    if (f >= 0 && f < this.entries.length && this.entries[f].frame_idx === f) {
      return this.entries[f];
    }
    return this.entries.find((e) => e.frame_idx === f);
  }

  resolveFrameFromTime(currentTimeSeconds: number, mode: 'floor' | 'nearest' = 'floor'): number {
    return resolveFrameFromTime(currentTimeSeconds, this.entries, this.fps, mode);
  }

  resolveTimeFromFrame(frameIdx: number): number {
    return resolveTimeFromFrame(frameIdx, this.entries, this.fps);
  }

  hasFrame(frameIdx: number): boolean {
    const f = clampFrameIdx(frameIdx);
    if (this.entries.length === 0) return false;
    return f >= this.entries[0].frame_idx && f <= this.entries[this.entries.length - 1].frame_idx;
  }

  clampFrame(frameIdx: number): number {
    if (this.entries.length === 0) return clampFrameIdx(frameIdx);
    return clampFrameIdx(
      frameIdx,
      this.entries[0].frame_idx,
      this.entries[this.entries.length - 1].frame_idx + 1
    );
  }

  toTimes(): number[] {
    return this.entries.map((e) => e.pts);
  }
}

/**
 * Factory function to create a PtsFrameMap instance.
 */
export function createPtsFrameMap(
  input?: PtsTableRawInput,
  fallbackFps = 30,
  isVfrHint?: boolean
): PtsFrameMap {
  return new PtsFrameMap(input, fallbackFps, isVfrHint);
}

export default PtsFrameMap;
