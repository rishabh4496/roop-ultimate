import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  Paintbrush,
  Eraser,
  Undo2,
  Trash2,
  Send,
  Download,
  UploadCloud,
  FlipHorizontal,
  Check,
  AlertCircle,
  Loader2,
  Eye,
  Sliders,
  Sparkles,
} from 'lucide-react';

export type DrawTool = 'draw' | 'erase';
export type TransmitStatus = 'idle' | 'uploading' | 'success' | 'error';

export interface OcclusionPainterRef {
  /** Exports the drawn mask as an 8-bit single-channel grayscale PNG Blob */
  exportMaskBlob: () => Promise<Blob>;
  /** Clears the current mask overlay (pushing previous state to undo stack) */
  clear: () => void;
  /** Undoes the last paint, clear, or invert action */
  undo: () => void;
  /** Inverts mask alpha values */
  invert: () => void;
  /** Retrieves the interactive overlay canvas DOM node */
  getOverlayCanvas: () => HTMLCanvasElement | null;
  /** Retrieves the reference video frame canvas DOM node */
  getReferenceCanvas: () => HTMLCanvasElement | null;
  /** Programmatically sets brush diameter (2 to 100) */
  setBrushSize: (size: number) => void;
  /** Programmatically sets brush hardness percentage (0 to 100) */
  setHardness: (val: number) => void;
  /** Programmatically sets active draw tool */
  setTool: (tool: DrawTool) => void;
}

export interface OcclusionPainterProps {
  /**
   * Reference video frame or face image source.
   * Can be an image/video URL, HTMLImageElement, HTMLVideoElement, or ImageBitmap.
   */
  referenceSource?: string | HTMLImageElement | HTMLVideoElement | ImageBitmap | null;
  /** Canvas native pixel width if source is omitted (default: 1024) */
  width?: number;
  /** Canvas native pixel height if source is omitted (default: 1024) */
  height?: number;
  /** Initial brush diameter in pixels (default: 28, clamped 2..100) */
  initialBrushSize?: number;
  /** Initial brush hardness/feathering percentage (default: 75, clamped 0..100) */
  initialHardness?: number;
  /** Initial visual overlay opacity for Layer 2 (default: 0.65, clamped 0.1..1.0) */
  initialOpacity?: number;
  /** Visual mask tint hex in the editor (default: '#ef4444' Ruby Red) */
  overlayColor?: string;
  /** Backend API endpoint to transmit the mask blob to (default: '/api/mask/custom') */
  apiEndpoint?: string;
  /** Additional metadata fields to append to multipart upload */
  metadata?: Record<string, string | number | boolean>;
  /** Callback fired after exportMaskBlob() successfully resolves */
  onExport?: (blob: Blob) => void;
  /** Callback fired when backend transmission succeeds */
  onUploadSuccess?: (response: unknown) => void;
  /** Callback fired when backend transmission fails */
  onUploadError?: (error: Error) => void;
  /** Callback fired when mask pixels are modified */
  onMaskChange?: () => void;
  /** Optional custom CSS classes */
  className?: string;
}

// ── Grayscale PNG 8-Bit Encoder (Conforming to RFC 2083 Color Type 0) ──────
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[i] = c >>> 0;
  }
  return table;
})();

function computeCrc32(data: Uint8Array, offset = 0, length = data.length - offset): number {
  let crc = 0xffffffff;
  for (let i = offset; i < offset + length; i++) {
    crc = (crc >>> 8) ^ CRC_TABLE[(crc ^ data[i]) & 0xff];
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function buildPngChunk(type: string, data: Uint8Array): Uint8Array {
  const typeBytes = new Uint8Array([
    type.charCodeAt(0),
    type.charCodeAt(1),
    type.charCodeAt(2),
    type.charCodeAt(3),
  ]);
  const chunkLength = data.length;
  const chunk = new Uint8Array(4 + 4 + chunkLength + 4);
  const view = new DataView(chunk.buffer);

  view.setUint32(0, chunkLength, false);
  chunk.set(typeBytes, 4);
  chunk.set(data, 8);

  const crcVal = computeCrc32(chunk, 4, 4 + chunkLength);
  view.setUint32(8 + chunkLength, crcVal, false);

  return chunk;
}

async function encodeGrayscalePng(
  width: number,
  height: number,
  grayscaleData: Uint8Array
): Promise<Blob> {
  const scanlineLength = 1 + width;
  const rawScanlines = new Uint8Array(height * scanlineLength);

  for (let y = 0; y < height; y++) {
    const rowOffset = y * scanlineLength;
    rawScanlines[rowOffset] = 0; // Filter: None
    const srcOffset = y * width;
    rawScanlines.set(grayscaleData.subarray(srcOffset, srcOffset + width), rowOffset + 1);
  }

  let compressedIdat: Uint8Array;

  if (typeof CompressionStream !== 'undefined') {
    try {
      const cs = new CompressionStream('deflate');
      const writer = cs.writable.getWriter();
      await writer.write(rawScanlines);
      await writer.close();

      const reader = cs.readable.getReader();
      const chunks: Uint8Array[] = [];
      let totalLength = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          chunks.push(value);
          totalLength += value.byteLength;
        }
      }

      compressedIdat = new Uint8Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {
        compressedIdat.set(chunk, offset);
        offset += chunk.byteLength;
      }
    } catch {
      return encodeCanvasGrayscaleFallback(width, height, grayscaleData);
    }
  } else {
    return encodeCanvasGrayscaleFallback(width, height, grayscaleData);
  }

  const signature = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

  // IHDR: 13 bytes (Width, Height, BitDepth 8, ColorType 0 = Grayscale)
  const ihdrData = new Uint8Array(13);
  const ihdrView = new DataView(ihdrData.buffer);
  ihdrView.setUint32(0, width, false);
  ihdrView.setUint32(4, height, false);
  ihdrView.setUint8(8, 8); // 8-bit depth
  ihdrView.setUint8(9, 0); // Color Type 0 = Grayscale
  ihdrView.setUint8(10, 0);
  ihdrView.setUint8(11, 0);
  ihdrView.setUint8(12, 0);

  const ihdrChunk = buildPngChunk('IHDR', ihdrData);
  const idatChunk = buildPngChunk('IDAT', compressedIdat);
  const iendChunk = buildPngChunk('IEND', new Uint8Array(0));

  const totalLength = signature.length + ihdrChunk.length + idatChunk.length + iendChunk.length;
  const pngBuffer = new Uint8Array(totalLength);

  let cursor = 0;
  pngBuffer.set(signature, cursor);
  cursor += signature.length;
  pngBuffer.set(ihdrChunk, cursor);
  cursor += ihdrChunk.length;
  pngBuffer.set(idatChunk, cursor);
  cursor += idatChunk.length;
  pngBuffer.set(iendChunk, cursor);

  return new Blob([pngBuffer], { type: 'image/png' });
}

function encodeCanvasGrayscaleFallback(
  width: number,
  height: number,
  grayscaleData: Uint8Array
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      reject(new Error('Canvas 2D context creation failed'));
      return;
    }
    const imgData = ctx.createImageData(width, height);
    const dst = imgData.data;
    for (let i = 0, j = 0; i < dst.length; i += 4, j++) {
      const v = grayscaleData[j];
      dst[i] = v;
      dst[i + 1] = v;
      dst[i + 2] = v;
      dst[i + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error('Canvas export toBlob returned null'));
      }
    }, 'image/png');
  });
}

function parseHexColor(hex: string): { r: number; g: number; b: number } {
  let clean = hex.replace('#', '');
  if (clean.length === 3) {
    clean = clean.split('').map((c) => c + c).join('');
  }
  const num = parseInt(clean, 16);
  if (Number.isNaN(num) || clean.length !== 6) {
    return { r: 239, g: 68, b: 68 }; // Default ruby red
  }
  return {
    r: (num >> 16) & 255,
    g: (num >> 8) & 255,
    b: num & 255,
  };
}

const TINT_PRESETS = [
  { name: 'Ruby Red', hex: '#ef4444' },
  { name: 'Neon Cyan', hex: '#06b6d4' },
  { name: 'Emerald', hex: '#10b981' },
  { name: 'Pure White', hex: '#ffffff' },
];

export const OcclusionPainter = forwardRef<OcclusionPainterRef, OcclusionPainterProps>(
  (
    {
      referenceSource,
      width = 1024,
      height = 1024,
      initialBrushSize = 28,
      initialHardness = 75,
      initialOpacity = 0.65,
      overlayColor = '#ef4444',
      apiEndpoint = '/api/mask/custom',
      metadata,
      onExport,
      onUploadSuccess,
      onUploadError,
      onMaskChange,
      className = '',
    },
    ref
  ) => {
    // ── Native Canvas Resolutions ──────────────────────────────────────────
    const [resolution, setResolution] = useState<{ width: number; height: number }>({
      width,
      height,
    });

    // ── Tooling States ─────────────────────────────────────────────────────
    const [tool, setTool] = useState<DrawTool>('draw');
    const [brushSize, setBrushSize] = useState<number>(
      Math.max(2, Math.min(100, initialBrushSize))
    );
    const [hardness, setHardness] = useState<number>(
      Math.max(0, Math.min(100, initialHardness))
    );
    const [overlayOpacity, setOverlayOpacity] = useState<number>(
      Math.max(0.1, Math.min(1.0, initialOpacity))
    );
    const [tintColor, setTintColor] = useState<string>(overlayColor);
    const [undoCount, setUndoCount] = useState<number>(0);

    // ── Cursor / Viewport Tracking ─────────────────────────────────────────
    const [cursor, setCursor] = useState<{
      x: number;
      y: number;
      visible: boolean;
      cssScale: number;
    }>({
      x: 0,
      y: 0,
      visible: false,
      cssScale: 1,
    });

    // ── Transmission State ─────────────────────────────────────────────────
    const [transmitStatus, setTransmitStatus] = useState<TransmitStatus>('idle');
    const [transmitMessage, setTransmitMessage] = useState<string>('');

    // ── Canvas DOM References ──────────────────────────────────────────────
    const refCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);

    // ── Drawing Execution State ────────────────────────────────────────────
    const isDrawingRef = useRef<boolean>(false);
    const lastPointRef = useRef<{ x: number; y: number } | null>(null);
    const undoStackRef = useRef<ImageData[]>([]);

    // Memoized RGB color for stamping
    const parsedRgb = useMemo(() => parseHexColor(tintColor), [tintColor]);

    // ── Snapshot Management (Max 10 Undo History) ──────────────────────────
    const pushUndoSnapshot = useCallback(() => {
      const canvas = overlayCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;

      try {
        const snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const stack = undoStackRef.current;
        if (stack.length >= 10) {
          stack.shift();
        }
        stack.push(snapshot);
        setUndoCount(stack.length);
      } catch {
        // Fallback gracefully if canvas context read fails
      }
    }, []);

    const handleUndo = useCallback(() => {
      const canvas = overlayCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;

      const stack = undoStackRef.current;
      if (stack.length === 0) return;

      const previous = stack.pop();
      if (previous) {
        ctx.putImageData(previous, 0, 0);
        setUndoCount(stack.length);
        onMaskChange?.();
      }
    }, [onMaskChange]);

    const handleClear = useCallback(() => {
      const canvas = overlayCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;

      pushUndoSnapshot();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      onMaskChange?.();
    }, [onMaskChange, pushUndoSnapshot]);

    const handleInvert = useCallback(() => {
      const canvas = overlayCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;

      pushUndoSnapshot();
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imgData.data;
      const { r, g, b } = parsedRgb;

      for (let i = 0; i < data.length; i += 4) {
        // Invert alpha transition: 0 becomes 255, 255 becomes 0, feathered edges invert smoothly
        const invertedAlpha = 255 - data[i + 3];
        data[i] = r;
        data[i + 1] = g;
        data[i + 2] = b;
        data[i + 3] = invertedAlpha;
      }

      ctx.putImageData(imgData, 0, 0);
      onMaskChange?.();
    }, [onMaskChange, parsedRgb, pushUndoSnapshot]);

    // ── Radial Gradient Stamp Drawing ──────────────────────────────────────
    const drawStamp = useCallback(
      (
        ctx: CanvasRenderingContext2D,
        x: number,
        y: number,
        radius: number,
        hFraction: number,
        isErase: boolean
      ) => {
        ctx.save();
        if (isErase) {
          ctx.globalCompositeOperation = 'destination-out';
          const grad = ctx.createRadialGradient(x, y, 0, x, y, radius);
          grad.addColorStop(0, 'rgba(0, 0, 0, 1.0)');
          const clampedH = Math.max(0, Math.min(0.999, hFraction));
          grad.addColorStop(clampedH, 'rgba(0, 0, 0, 1.0)');
          grad.addColorStop(1, 'rgba(0, 0, 0, 0.0)');
          ctx.fillStyle = grad;
        } else {
          ctx.globalCompositeOperation = 'source-over';
          const grad = ctx.createRadialGradient(x, y, 0, x, y, radius);
          const { r, g, b } = parsedRgb;
          grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, 1.0)`);
          const clampedH = Math.max(0, Math.min(0.999, hFraction));
          grad.addColorStop(clampedH, `rgba(${r}, ${g}, ${b}, 1.0)`);
          grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0.0)`);
          ctx.fillStyle = grad;
        }

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      },
      [parsedRgb]
    );

    // ── Continuous Path Interpolation ──────────────────────────────────────
    const drawSegment = useCallback(
      (
        ctx: CanvasRenderingContext2D,
        p0: { x: number; y: number },
        p1: { x: number; y: number },
        radius: number,
        hFraction: number,
        isErase: boolean
      ) => {
        const dx = p1.x - p0.x;
        const dy = p1.y - p0.y;
        const dist = Math.hypot(dx, dy);

        // Step distance at 15% of radius prevents dotted gaps during swift mouse movement
        const stepDist = Math.max(1, radius * 0.15);
        const steps = Math.max(1, Math.ceil(dist / stepDist));

        for (let i = 1; i <= steps; i++) {
          const t = i / steps;
          const currX = p0.x + dx * t;
          const currY = p0.y + dy * t;
          drawStamp(ctx, currX, currY, radius, hFraction, isErase);
        }
      },
      [drawStamp]
    );

    // ── Viewport to Native Canvas Coordinate Normalization ─────────────────
    const getNormalizedCoordinates = useCallback(
      (e: React.PointerEvent<HTMLCanvasElement>): { x: number; y: number; cssScale: number } | null => {
        const canvas = overlayCanvasRef.current;
        if (!canvas) return null;
        const rect = canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return null;

        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        const rawX = (e.clientX - rect.left) * scaleX;
        const rawY = (e.clientY - rect.top) * scaleY;

        return {
          x: Math.max(0, Math.min(canvas.width, rawX)),
          y: Math.max(0, Math.min(canvas.height, rawY)),
          cssScale: rect.width / canvas.width,
        };
      },
      []
    );

    // ── Pointer Event Handlers ─────────────────────────────────────────────
    const handlePointerDown = useCallback(
      (e: React.PointerEvent<HTMLCanvasElement>) => {
        if (e.button !== 0) return; // Primary button only

        const canvas = overlayCanvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) return;

        const coords = getNormalizedCoordinates(e);
        if (!coords) return;

        try {
          e.currentTarget.setPointerCapture(e.pointerId);
        } catch {
          // Pointer capture unsupported or unavailable
        }

        pushUndoSnapshot();

        const radius = brushSize / 2;
        const hFraction = hardness / 100;
        const isErase = tool === 'erase';

        // Stamp once at the initial point
        drawStamp(ctx, coords.x, coords.y, radius, hFraction, isErase);

        isDrawingRef.current = true;
        lastPointRef.current = { x: coords.x, y: coords.y };
        onMaskChange?.();
      },
      [brushSize, drawStamp, getNormalizedCoordinates, hardness, onMaskChange, pushUndoSnapshot, tool]
    );

    const handlePointerMove = useCallback(
      (e: React.PointerEvent<HTMLCanvasElement>) => {
        const canvas = overlayCanvasRef.current;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const clientX = e.clientX - rect.left;
        const clientY = e.clientY - rect.top;
        const cssScale = rect.width / canvas.width;

        setCursor({
          x: clientX,
          y: clientY,
          visible: true,
          cssScale,
        });

        if (!isDrawingRef.current || !lastPointRef.current) return;

        const coords = getNormalizedCoordinates(e);
        if (!coords) return;

        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) return;

        const radius = brushSize / 2;
        const hFraction = hardness / 100;
        const isErase = tool === 'erase';

        drawSegment(ctx, lastPointRef.current, { x: coords.x, y: coords.y }, radius, hFraction, isErase);

        lastPointRef.current = { x: coords.x, y: coords.y };
        onMaskChange?.();
      },
      [brushSize, drawSegment, getNormalizedCoordinates, hardness, onMaskChange, tool]
    );

    const handlePointerUp = useCallback(
      (e: React.PointerEvent<HTMLCanvasElement>) => {
        if (!isDrawingRef.current) return;
        isDrawingRef.current = false;
        lastPointRef.current = null;

        try {
          if (e.currentTarget.hasPointerCapture(e.pointerId)) {
            e.currentTarget.releasePointerCapture(e.pointerId);
          }
        } catch {
          // Release pointer capture safe fallback
        }
      },
      []
    );

    const handlePointerLeave = useCallback(() => {
      setCursor((prev) => ({ ...prev, visible: false }));
      if (!isDrawingRef.current) {
        lastPointRef.current = null;
      }
    }, []);

    // ── Reference Frame Drawing (Layer 1) ──────────────────────────────────
    useEffect(() => {
      const refCanvas = refCanvasRef.current;
      const overlayCanvas = overlayCanvasRef.current;
      if (!refCanvas || !overlayCanvas) return;
      const ctx = refCanvas.getContext('2d');
      if (!ctx) return;

      if (!referenceSource) {
        // Render neutral checkerboard pattern when no frame is loaded
        const w = resolution.width;
        const h = resolution.height;
        refCanvas.width = w;
        refCanvas.height = h;
        overlayCanvas.width = w;
        overlayCanvas.height = h;

        ctx.fillStyle = '#0f1115';
        ctx.fillRect(0, 0, w, h);

        const grid = 32;
        ctx.fillStyle = '#171a21';
        for (let y = 0; y < h; y += grid) {
          for (let x = 0; x < w; x += grid) {
            if ((Math.floor(x / grid) + Math.floor(y / grid)) % 2 === 0) {
              ctx.fillRect(x, y, grid, grid);
            }
          }
        }
        return;
      }

      if (typeof referenceSource === 'string') {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          const natW = img.naturalWidth || width;
          const natH = img.naturalHeight || height;
          setResolution({ width: natW, height: natH });
          refCanvas.width = natW;
          refCanvas.height = natH;
          overlayCanvas.width = natW;
          overlayCanvas.height = natH;
          ctx.drawImage(img, 0, 0);
        };
        img.src = referenceSource;
      } else if (referenceSource instanceof HTMLImageElement) {
        const natW = referenceSource.naturalWidth || referenceSource.width || width;
        const natH = referenceSource.naturalHeight || referenceSource.height || height;
        setResolution({ width: natW, height: natH });
        refCanvas.width = natW;
        refCanvas.height = natH;
        overlayCanvas.width = natW;
        overlayCanvas.height = natH;
        ctx.drawImage(referenceSource, 0, 0);
      } else if (referenceSource instanceof HTMLVideoElement) {
        const natW = referenceSource.videoWidth || referenceSource.width || width;
        const natH = referenceSource.videoHeight || referenceSource.height || height;
        setResolution({ width: natW, height: natH });
        refCanvas.width = natW;
        refCanvas.height = natH;
        overlayCanvas.width = natW;
        overlayCanvas.height = natH;
        ctx.drawImage(referenceSource, 0, 0);
      }
    }, [referenceSource, resolution.width, resolution.height, width, height]);

    // ── Mask Generation & Off-Screen 8-Bit Grayscale Export ─────────────────
    const exportMaskBlob = useCallback(async (): Promise<Blob> => {
      const canvas = overlayCanvasRef.current;
      if (!canvas) throw new Error('Overlay canvas is not initialized');

      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) throw new Error('Canvas 2D context unavailable for mask extraction');

      const w = canvas.width;
      const h = canvas.height;
      const imgData = ctx.getImageData(0, 0, w, h);
      const src = imgData.data;

      // Extract 8-bit alpha transitions: Masked = 255 (White), Unmasked = 0 (Black)
      const grayscale = new Uint8Array(w * h);
      for (let i = 0, j = 0; i < src.length; i += 4, j++) {
        grayscale[j] = src[i + 3];
      }

      // Encode into RFC 2083 single-channel 8-bit grayscale PNG Blob
      const blob = await encodeGrayscalePng(w, h, grayscale);
      onExport?.(blob);
      return blob;
    }, [onExport]);

    // ── Transmit Mask to Backend API (/api/mask/custom) ─────────────────────
    const handleTransmit = useCallback(async () => {
      setTransmitStatus('uploading');
      setTransmitMessage('Encoding 8-bit mask and sending to backend...');

      try {
        const blob = await exportMaskBlob();
        const formData = new FormData();
        formData.append('file', blob, 'occlusion_mask.png');
        formData.append('mask', blob, 'occlusion_mask.png');
        formData.append('width', String(resolution.width));
        formData.append('height', String(resolution.height));
        formData.append('timestamp', String(Date.now()));

        if (metadata) {
          for (const [key, val] of Object.entries(metadata)) {
            formData.append(key, String(val));
          }
        }

        const response = await fetch(apiEndpoint, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errText = await response.text().catch(() => 'Backend transmission rejected');
          throw new Error(`HTTP ${response.status}: ${errText}`);
        }

        const responseData = await response.json().catch(() => ({ status: 'ok' }));
        setTransmitStatus('success');
        setTransmitMessage(`Successfully uploaded mask to ${apiEndpoint}`);
        onUploadSuccess?.(responseData);

        setTimeout(() => {
          setTransmitStatus('idle');
          setTransmitMessage('');
        }, 4000);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Transmission failed';
        setTransmitStatus('error');
        setTransmitMessage(msg);
        onUploadError?.(err instanceof Error ? err : new Error(msg));
      }
    }, [apiEndpoint, exportMaskBlob, metadata, onUploadError, onUploadSuccess, resolution.height, resolution.width]);

    // ── Local File Download (Testing & Inspection) ─────────────────────────
    const handleDownload = useCallback(async () => {
      try {
        const blob = await exportMaskBlob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `occlusion_mask_${resolution.width}x${resolution.height}_${Date.now()}.png`;
        link.click();
        URL.revokeObjectURL(url);
      } catch (err) {
        console.error('Failed to download mask:', err);
      }
    }, [exportMaskBlob, resolution.height, resolution.width]);

    // ── User Local Frame Upload ────────────────────────────────────────────
    const handleLocalImageUpload = useCallback(
      (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (ev) => {
          const dataUrl = ev.target?.result;
          if (typeof dataUrl === 'string') {
            const img = new Image();
            img.onload = () => {
              const refCanvas = refCanvasRef.current;
              const overlayCanvas = overlayCanvasRef.current;
              if (!refCanvas || !overlayCanvas) return;
              const ctx = refCanvas.getContext('2d');
              if (!ctx) return;

              setResolution({ width: img.naturalWidth, height: img.naturalHeight });
              refCanvas.width = img.naturalWidth;
              refCanvas.height = img.naturalHeight;
              overlayCanvas.width = img.naturalWidth;
              overlayCanvas.height = img.naturalHeight;
              ctx.drawImage(img, 0, 0);

              // Clear existing mask and reset undo stack on new image load
              const overlayCtx = overlayCanvas.getContext('2d');
              overlayCtx?.clearRect(0, 0, img.naturalWidth, img.naturalHeight);
              undoStackRef.current = [];
              setUndoCount(0);
            };
            img.src = dataUrl;
          }
        };
        reader.readAsDataURL(file);
      },
      []
    );

    // ── Keyboard Shortcuts ─────────────────────────────────────────────────
    useEffect(() => {
      const handleKeyDown = (e: KeyboardEvent) => {
        // Avoid intercepting keystrokes if focused inside an input or textarea
        if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement)?.tagName)) {
          return;
        }

        if (e.key === '[' || e.key === '{') {
          setBrushSize((s) => Math.max(2, s - 5));
        } else if (e.key === ']' || e.key === '}') {
          setBrushSize((s) => Math.min(100, s + 5));
        } else if (e.key.toLowerCase() === 'b') {
          setTool('draw');
        } else if (e.key.toLowerCase() === 'e') {
          setTool('erase');
        } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
          e.preventDefault();
          handleUndo();
        } else if (e.key.toLowerCase() === 'i') {
          handleInvert();
        }
      };

      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }, [handleInvert, handleUndo]);

    // ── Forwarded Imperative Handle ────────────────────────────────────────
    useImperativeHandle(
      ref,
      () => ({
        exportMaskBlob,
        clear: handleClear,
        undo: handleUndo,
        invert: handleInvert,
        getOverlayCanvas: () => overlayCanvasRef.current,
        getReferenceCanvas: () => refCanvasRef.current,
        setBrushSize: (size) => setBrushSize(Math.max(2, Math.min(100, size))),
        setHardness: (val) => setHardness(Math.max(0, Math.min(100, val))),
        setTool: (newTool) => setTool(newTool),
      }),
      [exportMaskBlob, handleClear, handleInvert, handleUndo]
    );

    // Calculate CSS cursor dimensions
    const cursorDiameterCss = (brushSize * cursor.cssScale);
    const cursorInnerDiameterCss = cursorDiameterCss * (hardness / 100);

    return (
      <div
        className={`flex flex-col rounded-2xl border border-white/10 bg-neutral-950/80 text-white shadow-2xl backdrop-blur-xl ${className}`}
      >
        {/* ── Top Header & Telemetry Bar ──────────────────────────────────── */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-5 py-3.5">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-rose-500/20 to-amber-500/20 text-rose-400 border border-rose-500/30">
              <Sparkles className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold tracking-wide text-white">Occlusion Mask Painter</h2>
              <p className="text-xs text-white/50">
                Precision 8-bit alpha mask for difficult face occlusions
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="rounded-md border border-white/10 bg-white/5 px-2.5 py-1 font-mono text-xs text-white/70">
              {resolution.width} × {resolution.height} px
            </span>

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleLocalImageUpload}
              accept="image/*"
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-white/80 transition-all hover:bg-white/10 hover:text-white"
              title="Load custom reference image"
            >
              <UploadCloud className="h-3.5 w-3.5" />
              <span>Load Frame</span>
            </button>
          </div>
        </div>

        {/* ── Main Toolbar Controls ────────────────────────────────────────── */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 bg-neutral-900/40 px-5 py-3">
          {/* Tool Segment & Action Buttons */}
          <div className="flex flex-wrap items-center gap-2">
            {/* Draw / Erase Toggle */}
            <div className="flex rounded-xl border border-white/10 bg-neutral-950/80 p-0.5 shadow-inner">
              <button
                type="button"
                onClick={() => setTool('draw')}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                  tool === 'draw'
                    ? 'bg-rose-500 text-white shadow-md'
                    : 'text-white/60 hover:text-white'
                }`}
                title="Brush mode (Paint mask) [Hotkey: B]"
              >
                <Paintbrush className="h-3.5 w-3.5" />
                <span>Draw</span>
              </button>
              <button
                type="button"
                onClick={() => setTool('erase')}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                  tool === 'erase'
                    ? 'bg-rose-500 text-white shadow-md'
                    : 'text-white/60 hover:text-white'
                }`}
                title="Eraser mode (Remove mask) [Hotkey: E]"
              >
                <Eraser className="h-3.5 w-3.5" />
                <span>Erase</span>
              </button>
            </div>

            {/* Invert Mask Button */}
            <button
              type="button"
              onClick={handleInvert}
              className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-white/80 transition-all hover:bg-white/10 hover:text-white"
              title="Invert mask coverage [Hotkey: I]"
            >
              <FlipHorizontal className="h-3.5 w-3.5" />
              <span>Invert</span>
            </button>

            {/* Undo Button */}
            <button
              type="button"
              onClick={handleUndo}
              disabled={undoCount === 0}
              className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-white/80 transition-all hover:bg-white/10 hover:text-white disabled:pointer-events-none disabled:opacity-30"
              title="Undo previous stroke (up to 10 steps) [Hotkey: Ctrl+Z]"
            >
              <Undo2 className="h-3.5 w-3.5" />
              <span>Undo</span>
              {undoCount > 0 && (
                <span className="ml-0.5 rounded bg-white/15 px-1 py-0.2 font-mono text-[10px] text-white/90">
                  {undoCount}
                </span>
              )}
            </button>

            {/* Clear Button */}
            <button
              type="button"
              onClick={handleClear}
              className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-white/80 transition-all hover:border-red-500/40 hover:bg-red-500/15 hover:text-red-300"
              title="Clear entire mask canvas"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span>Clear</span>
            </button>
          </div>

          {/* Sliders: Size, Hardness, Opacity */}
          <div className="flex flex-wrap items-center gap-5">
            {/* Brush Size Slider (2px - 100px) */}
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-white/60">Size</span>
              <input
                type="range"
                min="2"
                max="100"
                step="1"
                value={brushSize}
                onChange={(e) => setBrushSize(Number(e.target.value))}
                className="h-1.5 w-24 cursor-pointer appearance-none rounded-lg bg-white/20 accent-rose-500 focus:outline-none"
              />
              <span className="w-8 font-mono text-xs text-white/80">{brushSize}px</span>
            </div>

            {/* Brush Hardness / Feathering Slider (0% - 100%) */}
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-white/60">Hardness</span>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={hardness}
                onChange={(e) => setHardness(Number(e.target.value))}
                className="h-1.5 w-24 cursor-pointer appearance-none rounded-lg bg-white/20 accent-rose-500 focus:outline-none"
              />
              <span className="w-8 font-mono text-xs text-white/80">{hardness}%</span>
            </div>

            {/* Overlay Opacity Slider */}
            <div className="flex items-center gap-2">
              <Eye className="h-3.5 w-3.5 text-white/60" />
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                className="h-1.5 w-20 cursor-pointer appearance-none rounded-lg bg-white/20 accent-rose-500 focus:outline-none"
              />
              <span className="w-7 font-mono text-xs text-white/80">
                {Math.round(overlayOpacity * 100)}%
              </span>
            </div>

            {/* Tint Color Selector */}
            <div className="flex items-center gap-1.5">
              {TINT_PRESETS.map((tint) => (
                <button
                  key={tint.hex}
                  type="button"
                  onClick={() => setTintColor(tint.hex)}
                  className={`h-4 w-4 rounded-full border transition-transform ${
                    tintColor === tint.hex
                      ? 'scale-125 border-white ring-2 ring-white/30'
                      : 'border-white/20 hover:scale-110'
                  }`}
                  style={{ backgroundColor: tint.hex }}
                  title={`Mask tint: ${tint.name}`}
                />
              ))}
            </div>
          </div>
        </div>

        {/* ── Dual-Canvas Canvas Stack Viewport ────────────────────────────── */}
        <div
          ref={containerRef}
          className="relative flex min-h-[460px] max-h-[72vh] w-full flex-1 items-center justify-center overflow-hidden bg-neutral-950 p-4 select-none"
        >
          <div
            className="relative flex items-center justify-center shadow-2xl rounded-xl overflow-hidden border border-white/10"
            style={{
              aspectRatio: `${resolution.width} / ${resolution.height}`,
              maxWidth: '100%',
              maxHeight: '100%',
            }}
          >
            {/* Layer 1: Reference Video Frame Canvas */}
            <canvas
              ref={refCanvasRef}
              className="block w-full h-full object-contain pointer-events-none"
            />

            {/* Layer 2: Interactive Overlay Mask Canvas */}
            <canvas
              ref={overlayCanvasRef}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerLeave}
              style={{ opacity: overlayOpacity }}
              className="absolute inset-0 block w-full h-full object-contain cursor-none touch-none"
            />

            {/* Custom Interactive Brush Cursor Preview */}
            {cursor.visible && (
              <div
                className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/80 shadow-sm"
                style={{
                  left: `${cursor.x}px`,
                  top: `${cursor.y}px`,
                  width: `${cursorDiameterCss}px`,
                  height: `${cursorDiameterCss}px`,
                  backgroundColor: tool === 'draw' ? `${tintColor}22` : 'rgba(0, 0, 0, 0.25)',
                }}
              >
                {/* Inner Hardness Boundary Circle */}
                {cursorInnerDiameterCss > 2 && (
                  <div
                    className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/40 border-dashed"
                    style={{
                      width: `${cursorInnerDiameterCss}px`,
                      height: `${cursorInnerDiameterCss}px`,
                    }}
                  />
                )}
                {/* Crosshair Center Point */}
                <div className="absolute left-1/2 top-1/2 h-1 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/90" />
              </div>
            )}
          </div>
        </div>

        {/* ── Status Banner (Transmission / Diagnostics) ───────────────────── */}
        {transmitStatus !== 'idle' && (
          <div
            className={`flex items-center gap-2 border-t px-5 py-2.5 text-xs font-medium transition-all ${
              transmitStatus === 'uploading'
                ? 'border-amber-500/20 bg-amber-500/10 text-amber-300'
                : transmitStatus === 'success'
                ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300'
                : 'border-rose-500/20 bg-rose-500/10 text-rose-300'
            }`}
          >
            {transmitStatus === 'uploading' && <Loader2 className="h-4 w-4 animate-spin" />}
            {transmitStatus === 'success' && <Check className="h-4 w-4" />}
            {transmitStatus === 'error' && <AlertCircle className="h-4 w-4" />}
            <span>{transmitMessage}</span>
          </div>
        )}

        {/* ── Bottom Action Footer ─────────────────────────────────────────── */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 bg-neutral-900/60 px-5 py-3.5">
          <div className="flex items-center gap-4 text-xs text-white/50">
            <span className="flex items-center gap-1.5">
              <Sliders className="h-3 w-3" />
              <span>
                Hotkeys: <strong className="text-white/70">[ / ]</strong> Size &bull;{' '}
                <strong className="text-white/70">B</strong> Draw &bull;{' '}
                <strong className="text-white/70">E</strong> Erase &bull;{' '}
                <strong className="text-white/70">Ctrl+Z</strong> Undo
              </span>
            </span>
          </div>

          <div className="flex items-center gap-2.5">
            {/* Download Grayscale PNG locally */}
            <button
              type="button"
              onClick={handleDownload}
              className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3.5 py-2 text-xs font-medium text-white/90 transition-all hover:bg-white/10 hover:text-white"
              title="Download 8-bit single channel grayscale PNG mask"
            >
              <Download className="h-3.5 w-3.5" />
              <span>Download PNG</span>
            </button>

            {/* Transmit directly to backend API (/api/mask/custom) */}
            <button
              type="button"
              onClick={handleTransmit}
              disabled={transmitStatus === 'uploading'}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-rose-500 to-red-600 px-4 py-2 text-xs font-semibold text-white shadow-lg shadow-rose-950/40 transition-all hover:from-rose-400 hover:to-red-500 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50"
            >
              {transmitStatus === 'uploading' ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span>Transmitting...</span>
                </>
              ) : (
                <>
                  <Send className="h-3.5 w-3.5" />
                  <span>Transmit to Backend</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    );
  }
);

OcclusionPainter.displayName = 'OcclusionPainter';
export default OcclusionPainter;
