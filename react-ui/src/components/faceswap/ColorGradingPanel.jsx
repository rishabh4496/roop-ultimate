import React, { useEffect, useId, useRef, useState } from 'react';
import { parseCubeLut } from './cubeLut.js';
import { GRADING_DEFAULTS } from './webgl2GradingPipeline.js';
import { useApi } from '../../useApi';

const clamp01 = (v) => Math.max(0, Math.min(1, v));

function lutColor(lut, rgb) {
  const { size, data, domainMin, domainMax } = lut;
  const p = rgb.map((v, c) => clamp01((v - domainMin[c]) / (domainMax[c] - domainMin[c])) * (size - 1));
  const lo = p.map(Math.floor);
  const t = p.map((v, c) => v - lo[c]);
  const [first, second, third] = [0, 1, 2].sort((a, b) => t[b] - t[a]);
  const strides = [3, size * 3, size * size * 3];
  const base = lo[0] * strides[0] + lo[1] * strides[1] + lo[2] * strides[2];
  const step = (axis) => lo[axis] < size - 1 ? strides[axis] : 0;
  const a = base + step(first);
  const b = a + step(second);
  const c = b + step(third);
  return [0, 1, 2].map((i) => data[base + i] * (1 - t[first])
    + data[a + i] * (t[first] - t[second]) + data[b + i] * (t[second] - t[third]) + data[c + i] * t[third]);
}

/**
 * Stage 18: WebGL2 3D .cube LUT Grading Panel.
 *
 * Controls:
 *   - .cube LUT file loading with strict DOMAIN_MIN / DOMAIN_MAX validation
 *   - 3D LUT Blend Opacity slider (0.0 .. 1.0)
 *   - Display-Only Preview vs Bake Tetrahedral 3D LUT into 10-bit FFmpeg Encode toggle
 *   - 3-Way Lift/Gamma/Gain controls (Shadows / Midtones / Highlights)
 *   - Kelvin Temperature & Tint sliders
 *   - Vibrance with Skin-Tone Hue Protection (20°-50° melanin corridor)
 */
export default function ColorGradingPanel({
  params = GRADING_DEFAULTS,
  onChange,
  notify,
  previewSource = null,
  className = '',
}) {
  const fileInputRef = useRef(null);
  const fileInputId = useId();
  const originalCanvasRef = useRef(null);
  const gradedCanvasRef = useRef(null);
  const [activeLut, setActiveLut] = useState(null);
  const [shelf, setShelf] = useState([]);
  const { getJSON, postFile, postJSON } = useApi();

  useEffect(() => {
    let live = true;
    getJSON('/api/lut/shelf').then((result) => {
      if (live) setShelf(result.files || []);
    }).catch(() => {});
    return () => { live = false; };
  }, [getJSON]);

  const update = (key, value) => {
    if (onChange) {
      onChange({ [key]: value });
    }
  };

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const parsed = parseCubeLut(await file.text());
      const saved = await postFile('/api/lut/upload', file);
      setActiveLut(parsed);
      setShelf((previous) => previous.some((entry) => entry.path === saved.path) ? previous
        : [...previous, { path: saved.path, name: file.name }]);
      update('lut_path', saved.path);
      notify?.(`Loaded ${parsed.title} · ${saved.size}³`);
    } catch (err) {
      notify?.(`LUT Error: ${err.message}`, 'error');
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const clearLut = () => {
    onChange?.({ lut_path: '', lut_bake: false });
    setActiveLut(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    notify?.('Cleared 3D LUT');
  };

  const resetGrading = () => {
    if (onChange) onChange({ ...GRADING_DEFAULTS, lut_path: params.lut_path });
    notify?.('Reset Color Grading parameters');
  };

  const lutName = params.lut_path ? params.lut_path.split(/[/\\]/).pop() : '';

  useEffect(() => {
    if (!params.lut_path) { setActiveLut(null); return undefined; }
    let live = true;
    postJSON('/api/lut/read', { path: params.lut_path }, { abortOnUnmount: true })
      .then((result) => {
        if (!live) return;
        const parsed = parseCubeLut(result.cube_text);
        setActiveLut(parsed);
      })
      .catch(() => { if (live) setActiveLut(null); });
    return () => { live = false; };
  }, [params.lut_path, postJSON]);

  useEffect(() => {
    if (!activeLut) return undefined;
    const previewImage = typeof previewSource === 'string' && previewSource ? new Image() : null;
    if (previewImage) previewImage.src = previewSource;
    let paintedSource;
    let paintedTime;
    const paint = () => {
      const originalCanvas = originalCanvasRef.current;
      const gradedCanvas = gradedCanvasRef.current;
      const source = previewImage || (typeof previewSource === 'function' ? previewSource() : previewSource);
      const sourceReady = source && (source.tagName !== 'VIDEO' || source.readyState >= 2)
        && (source.tagName !== 'IMG' || (source.complete && source.naturalWidth > 0));
      if (sourceReady && originalCanvas && gradedCanvas) {
        const mediaTime = source.tagName === 'VIDEO' ? source.currentTime : source.currentSrc;
        if (paintedSource === source && paintedTime === mediaTime) return;
        const width = source.videoWidth || source.naturalWidth || source.width;
        const height = source.videoHeight || source.naturalHeight || source.height;
        if (!width || !height) return;
        for (const canvas of [originalCanvas, gradedCanvas]) {
          const ctx = canvas.getContext('2d', { willReadFrequently: canvas === gradedCanvas });
          const scale = Math.min(canvas.width / width, canvas.height / height);
          try {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(source, (canvas.width - width * scale) / 2, (canvas.height - height * scale) / 2,
              width * scale, height * scale);
          } catch { return; }
        }
        paintedSource = source;
        paintedTime = mediaTime;
        try {
          const ctx = gradedCanvas.getContext('2d', { willReadFrequently: true });
          const pixels = ctx.getImageData(0, 0, gradedCanvas.width, gradedCanvas.height);
          const data = pixels.data;
          const strength = clamp01(Number(params.lut_opacity ?? 1));
          for (let i = 0; i < data.length; i += 4) {
            const input = [data[i] / 255, data[i + 1] / 255, data[i + 2] / 255];
            const graded = lutColor(activeLut, input);
            data[i] = Math.round(clamp01(input[0] * (1 - strength) + graded[0] * strength) * 255);
            data[i + 1] = Math.round(clamp01(input[1] * (1 - strength) + graded[1] * strength) * 255);
            data[i + 2] = Math.round(clamp01(input[2] * (1 - strength) + graded[2] * strength) * 255);
          }
          ctx.putImageData(pixels, 0, 0);
        } catch { /* keep the ungraded thumbnail visible if readback is unavailable */ }
      }
    };
    if (previewImage) {
      previewImage.onload = paint;
      paint();
      return () => { previewImage.onload = null; };
    }
    paint();
    const timer = window.setInterval(paint, 250);
    return () => window.clearInterval(timer);
  }, [activeLut, previewSource, params.lut_opacity]);

  return (
    <div
      className={`glass-panel border border-white/15 rounded-2xl p-3 flex flex-col gap-3 select-none text-xs text-white/80 ${className}`}
      data-testid="color-grading-panel"
    >
      {/* Header & Reset */}
      <div className="flex items-center justify-between border-b border-white/10 pb-2">
        <div className="flex items-center gap-1.5 font-bold text-mini tracking-wide text-white">
          <span className="h-2 w-2 rounded-full bg-[var(--accent)]" />
          <span>Color & 3D LUT grading</span>
        </div>

        {shelf.length > 0 && <select aria-label="Saved LUTs" value={shelf.some((entry) => entry.path === params.lut_path) ? params.lut_path : ''}
          onChange={(event) => update('lut_path', event.target.value)} className="w-full rounded-lg glass-input px-2 py-1.5 text-mini">
          <option value="" disabled>Choose a saved LUT</option>
          {shelf.map((entry) => <option key={entry.path} value={entry.path}>{entry.name}</option>)}
        </select>}
        <button
          type="button"
          onClick={resetGrading}
          className="px-2 py-0.5 rounded-md hover:bg-white/10 text-nano text-white/50 hover:text-white transition-colors"
          title="Reset Lift/Gamma/Gain and Color sliders"
        >
          Reset
        </button>
      </div>

      {/* .cube LUT File Loader & Encode Baking Toggle */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-mini">
          <span className="font-semibold text-white/70">3D .cube LUT</span>
          {lutName ? (
            <div className="flex items-center gap-1.5">
              <span className="text-emerald-400 font-mono text-nano truncate max-w-[130px]">{lutName}</span>
              <button
                type="button"
                onClick={clearLut}
                className="text-white/40 hover:text-rose-400 text-nano px-1"
                title="Remove LUT"
                aria-label="Remove LUT"
              >
                ✕
              </button>
            </div>
          ) : (
            <span className="text-white/40 text-nano">None loaded</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".cube,.CUBE"
            onChange={handleFileChange}
            className="sr-only"
            id={fileInputId}
          />
          <label
            htmlFor={fileInputId}
            className="flex-1 py-1.5 px-3 rounded-lg bg-white/10 hover:bg-white/15 text-center text-nano font-bold text-white/90 cursor-pointer border border-white/10 transition-colors"
          >
            {lutName ? 'Replace .cube LUT' : 'Load .cube LUT file'}
          </label>
        </div>

        {activeLut && (
          <div className="grid grid-cols-2 gap-2" data-testid="lut-live-thumbnails">
            <div className="min-w-0 rounded-lg border border-white/10 bg-black/30 p-1.5">
              <div className="mb-1 truncate text-micro uppercase tracking-wider text-white/45">Current frame</div>
              <canvas ref={originalCanvasRef} width="144" height="82" className="aspect-video w-full rounded object-cover" />
            </div>
            <div className="min-w-0 rounded-lg border border-cyan-300/20 bg-cyan-950/20 p-1.5">
              <div className="mb-1 truncate text-micro uppercase tracking-wider text-cyan-100/70">LUT preview</div>
              <canvas ref={gradedCanvasRef} width="144" height="82" className="aspect-video w-full rounded object-cover" />
            </div>
          </div>
        )}

        {/* LUT Blend Opacity */}
        {params.lut_path && (
          <div className="space-y-1 pt-1">
            <div className="flex justify-between text-nano font-mono text-white/60">
              <span>LUT Opacity</span>
              <span>{Math.round((params.lut_opacity ?? 1.0) * 100)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              aria-label="LUT intensity"
              value={params.lut_opacity ?? 1.0}
              onChange={(e) => update('lut_opacity', parseFloat(e.target.value))}
              className="w-full h-1.5 rounded-lg appearance-none bg-white/10 accent-[var(--accent)] cursor-pointer"
            />
          </div>
        )}

        <div className="space-y-1 rounded-xl border border-white/10 bg-black/30 p-2">
          <div className="text-nano font-semibold uppercase tracking-wider text-white/50">LUT output mode</div>
          <div className="grid grid-cols-2 gap-1" role="group" aria-label="LUT output mode">
            {[[false, 'Display-Only Preview LUT'], [true, 'Bake LUT into Hardware Encode']].map(([bake, label]) => (
              <button key={label} type="button" aria-pressed={!!params.lut_bake === bake} disabled={bake && !params.lut_path}
                onClick={() => update('lut_bake', bake)}
                className={`rounded-lg border px-2 py-1.5 text-micro font-bold transition-colors ${!!params.lut_bake === bake ? 'border-cyan-300/50 bg-cyan-400/15 text-cyan-100' : 'border-white/10 text-white/55 hover:border-white/25'}`}>
                {label}
              </button>
            ))}
          </div>
          <p className="m-0 text-micro text-white/40">Baking uses the same intensity value above in NVHardwareVideoWriter or its software fallback.</p>
        </div>
      </div>

      {/* 3-Way Lift / Gamma / Gain */}
      <div className="space-y-2 pt-1 border-t border-white/10">
        <span className="font-semibold text-white/70 text-mini">Lift / Gamma / Gain</span>
        <div className="grid grid-cols-3 gap-2">
          {/* Lift (Shadows) */}
          <div className="p-2 rounded-xl bg-black/30 border border-white/10 flex flex-col items-center gap-1">
            <span className="text-nano font-bold text-white/70">Lift</span>
            <span className="text-micro font-mono text-white/50">{(params.lift_m ?? 0.0).toFixed(2)}</span>
            <input
              type="range"
              min="-0.5"
              max="0.5"
              step="0.01"
              value={params.lift_m ?? 0.0}
              onChange={(e) => update('lift_m', parseFloat(e.target.value))}
              className="w-full h-1.5 rounded-lg appearance-none bg-white/10 accent-indigo-400 cursor-pointer"
              title="Lift: shadow offset"
            />
          </div>

          {/* Gamma (Midtones) */}
          <div className="p-2 rounded-xl bg-black/30 border border-white/10 flex flex-col items-center gap-1">
            <span className="text-nano font-bold text-white/70">Gamma</span>
            <span className="text-micro font-mono text-white/50">{(params.gamma_m ?? 1.0).toFixed(2)}</span>
            <input
              type="range"
              min="0.2"
              max="2.5"
              step="0.01"
              value={params.gamma_m ?? 1.0}
              onChange={(e) => update('gamma_m', parseFloat(e.target.value))}
              className="w-full h-1.5 rounded-lg appearance-none bg-white/10 accent-emerald-400 cursor-pointer"
              title="Gamma: midtone power"
            />
          </div>

          {/* Gain (Highlights) */}
          <div className="p-2 rounded-xl bg-black/30 border border-white/10 flex flex-col items-center gap-1">
            <span className="text-nano font-bold text-white/70">Gain</span>
            <span className="text-micro font-mono text-white/50">{(params.gain_m ?? 1.0).toFixed(2)}</span>
            <input
              type="range"
              min="0.0"
              max="2.5"
              step="0.01"
              value={params.gain_m ?? 1.0}
              onChange={(e) => update('gain_m', parseFloat(e.target.value))}
              className="w-full h-1.5 rounded-lg appearance-none bg-white/10 accent-amber-400 cursor-pointer"
              title="Gain: highlight slope"
            />
          </div>
        </div>
      </div>

      {/* Temperature & Tint */}
      <div className="space-y-2 pt-1 border-t border-white/10">
        <div className="flex justify-between text-mini font-semibold text-white/70">
          <span>Kelvin Temp / Tint</span>
        </div>

        <div className="space-y-1.5">
          <div className="flex justify-between text-nano font-mono text-white/50">
            <span className="text-sky-300">Cool</span>
            <span>Temperature: {(params.color_temperature ?? 0.0) > 0 ? '+' : ''}{(params.color_temperature ?? 0.0).toFixed(2)}</span>
            <span className="text-amber-300">Warm</span>
          </div>
          <input
            type="range"
            min="-1"
            max="1"
            step="0.02"
            value={params.color_temperature ?? 0.0}
            onChange={(e) => update('color_temperature', parseFloat(e.target.value))}
            className="w-full h-1.5 rounded-lg appearance-none bg-gradient-to-r from-sky-400 via-white/20 to-amber-400 accent-amber-400 cursor-pointer"
          />

          <div className="flex justify-between text-nano font-mono text-white/50 pt-1">
            <span className="text-emerald-300">Green</span>
            <span>Tint: {(params.color_tint ?? 0.0) > 0 ? '+' : ''}{(params.color_tint ?? 0.0).toFixed(2)}</span>
            <span className="text-fuchsia-300">Magenta</span>
          </div>
          <input
            type="range"
            min="-1"
            max="1"
            step="0.02"
            value={params.color_tint ?? 0.0}
            onChange={(e) => update('color_tint', parseFloat(e.target.value))}
            className="w-full h-1.5 rounded-lg appearance-none bg-gradient-to-r from-emerald-400 via-white/20 to-fuchsia-400 accent-fuchsia-400 cursor-pointer"
          />
        </div>
      </div>

      {/* Vibrance with Skin-Tone Corridor Protection */}
      <div className="space-y-2 pt-1 border-t border-white/10">
        <div className="flex justify-between text-nano font-mono text-white/60">
          <span className="font-semibold text-white/70 text-mini">Vibrance (20°-50° Skin Protected)</span>
          <span>{(params.color_vibrance ?? 0.0) > 0 ? '+' : ''}{(params.color_vibrance ?? 0.0).toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="-1"
          max="1"
          step="0.02"
          value={params.color_vibrance ?? 0.0}
          onChange={(e) => update('color_vibrance', parseFloat(e.target.value))}
          className="w-full h-1.5 rounded-lg appearance-none bg-white/10 accent-[var(--accent)] cursor-pointer"
        />

        <div className="flex items-center justify-between text-nano text-white/60 pt-0.5">
          <span>Skin Corridor Protection</span>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <span className="text-amber-400 font-mono text-micro font-bold">
              {Math.round((params.color_skin_protect ?? 1.0) * 100)}%
            </span>
            <input
              type="checkbox"
              checked={(params.color_skin_protect ?? 1.0) > 0.05}
              onChange={(e) => update('color_skin_protect', e.target.checked ? 1.0 : 0.0)}
              className="h-3.5 w-3.5 rounded accent-[var(--accent)] cursor-pointer"
            />
          </label>
        </div>
      </div>
    </div>
  );
}
