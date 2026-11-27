import React, { useRef } from 'react';
import { parseCubeLut } from './cubeLut.js';
import { GRADING_DEFAULTS } from './webgl2GradingPipeline.js';

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
  onLutLoaded,
  notify,
  className = '',
}) {
  const fileInputRef = useRef(null);

  const update = (key, value) => {
    if (onChange) {
      onChange({ [key]: value });
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result;
      if (!text || typeof text !== 'string') return;

      try {
        const parsed = parseCubeLut(text);
        update('lut_path', file.path || file.name);
        if (onLutLoaded) {
          onLutLoaded(parsed);
        }
        notify?.(`Loaded 3D LUT: ${parsed.title} (${parsed.size}×${parsed.size}×${parsed.size})`);
      } catch (err) {
        // Refuse invalid domain or malformed LUT with a clear UI notification
        notify?.(`LUT Error: ${err.message}`, 'error');
        if (fileInputRef.current) fileInputRef.current.value = '';
      }
    };
    reader.readAsText(file);
  };

  const clearLut = () => {
    update('lut_path', '');
    if (onLutLoaded) onLutLoaded(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    notify?.('Cleared 3D LUT');
  };

  const resetGrading = () => {
    if (onChange) onChange({ ...GRADING_DEFAULTS, lut_path: params.lut_path });
    notify?.('Reset Color Grading parameters');
  };

  const lutName = params.lut_path ? params.lut_path.split(/[/\\]/).pop() : '';

  return (
    <div
      className={`glass-panel border border-white/15 rounded-2xl p-3 flex flex-col gap-3 select-none text-xs text-white/80 ${className}`}
      data-testid="color-grading-panel"
    >
      {/* Header & Reset */}
      <div className="flex items-center justify-between border-b border-white/10 pb-2">
        <div className="flex items-center gap-1.5 font-bold text-mini tracking-wide text-white">
          <span className="h-2 w-2 rounded-full bg-[var(--accent)]" />
          <span>3D .cube LUT & Color Grade</span>
        </div>
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
            id="cube-lut-file-input"
          />
          <label
            htmlFor="cube-lut-file-input"
            className="flex-1 py-1.5 px-3 rounded-lg bg-white/10 hover:bg-white/15 text-center text-nano font-bold text-white/90 cursor-pointer border border-white/10 transition-colors"
          >
            {lutName ? 'Replace .cube LUT' : 'Load .cube LUT file'}
          </label>
        </div>

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
              value={params.lut_opacity ?? 1.0}
              onChange={(e) => update('lut_opacity', parseFloat(e.target.value))}
              className="w-full h-1.5 rounded-lg appearance-none bg-white/10 accent-[var(--accent)] cursor-pointer"
            />
          </div>
        )}

        {/* Bake LUT into 10-bit FFmpeg Encode toggle */}
        <label
          htmlFor="lut-bake-checkbox"
          aria-label="Bake into 10-bit FFmpeg Encode"
          className="flex items-center justify-between p-2 rounded-xl bg-black/30 border border-white/10 cursor-pointer group"
        >
          <div className="space-y-0.5 pr-2">
            <span className="block text-nano font-bold text-white/90 group-hover:text-white">Bake into 10-bit FFmpeg Encode</span>
            <span className="block text-micro text-white/50">Tetrahedral 3D interpolation baked into output vs display preview</span>
          </div>
          <input
            id="lut-bake-checkbox"
            type="checkbox"
            checked={!!params.lut_bake}
            onChange={(e) => update('lut_bake', e.target.checked)}
            className="h-4 w-4 rounded accent-[var(--accent)] bg-black/40 border-white/20 cursor-pointer"
          />
        </label>
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
