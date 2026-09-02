import React, { forwardRef, useImperativeHandle, useRef } from 'react';

/**
 * Interactive Dual-Canvas Face Mask Drawing and Erasing Subsystem.
 *  - Canvas 1 (Screen): Translucent pink visual feedback layer.
 *  - Canvas 2 (Export): Off-screen solid white on transparent (0 / 255) for backend.
 */
export const MaskBrushOverlay = forwardRef(function MaskBrushOverlay(
  {
    active = false,
    brushMode = 'paint', // 'paint' | 'erase'
    brushSize = 30,
    imgDim = null,
    onMaskChange,
  },
  ref,
) {
  const maskCanvasRef = useRef(null);
  const maskExportRef = useRef(null);
  const lastPointRef = useRef(null);

  useImperativeHandle(ref, () => ({
    clear: () => {
      for (const c of [maskCanvasRef.current, maskExportRef.current]) {
        const ctx = c?.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, c.width, c.height);
      }
      lastPointRef.current = null;
      onMaskChange?.(null);
    },
    commit: () => {
      const c = maskExportRef.current;
      if (!c || !onMaskChange) return;
      let hasPixels = false;
      try {
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        for (let i = 3; i < d.length; i += 16) {
          if (d[i] > 8) {
            hasPixels = true;
            break;
          }
        }
      } catch {
        hasPixels = true;
      }
      onMaskChange(hasPixels ? c.toDataURL('image/png') : null);
    },
    drawStroke: (e, isStart = false) => {
      const canvas = maskCanvasRef.current;
      const exportCanvas = maskExportRef.current;
      if (!canvas || !exportCanvas || !imgDim) return;

      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX ?? e.touches?.[0]?.clientX;
      const cy = e.clientY ?? e.touches?.[0]?.clientY;
      if (cx === undefined || !rect.width) return;

      // Scale screen pointer coordinates to native media pixel resolution
      const scale = canvas.width / rect.width;
      const x = (cx - rect.left) * scale;
      const y = (cy - rect.top) * scale;
      const r = Math.max(0.5, (brushSize / 2) * scale);

      const prev = isStart ? null : lastPointRef.current;
      const isErase = brushMode === 'erase';

      for (const [c, color] of [
        [canvas, 'rgba(233, 69, 96, 0.5)'],
        [exportCanvas, '#ffffff'],
      ]) {
        const ctx = c.getContext('2d');
        if (!ctx) continue;
        ctx.globalCompositeOperation = isErase ? 'destination-out' : 'source-over';
        ctx.fillStyle = color;
        ctx.strokeStyle = color;
        ctx.lineWidth = r * 2;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        if (prev) {
          ctx.beginPath();
          ctx.moveTo(prev.x, prev.y);
          ctx.lineTo(x, y);
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
      }

      lastPointRef.current = { x, y };
    },
    endStroke: () => {
      lastPointRef.current = null;
    },
  }));

  if (!active || !imgDim || !imgDim.w || !imgDim.h) return null;

  return (
    <>
      {/* Visual On-Screen Canvas */}
      <canvas
        ref={maskCanvasRef}
        width={imgDim.w}
        height={imgDim.h}
        className="absolute inset-0 w-full h-full pointer-events-none z-30"
      />
      {/* Hidden Off-Screen Export Canvas */}
      <canvas
        ref={maskExportRef}
        width={imgDim.w}
        height={imgDim.h}
        className="hidden"
      />
    </>
  );
});
