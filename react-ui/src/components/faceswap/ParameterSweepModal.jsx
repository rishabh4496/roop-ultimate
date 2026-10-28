import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { useApi } from '../../useApi';
import { SWEEP_PARAMETERS } from './sweepParameters';

export default function ParameterSweepModal({
  isOpen,
  onClose,
  targetIndex = 0,
  frame = 1,
  currentSettings = {},
  onApplyRecipe,
  notify,
}) {
  const { postJSON } = useApi();

  const [xParamId, setXParamId] = useState('restore_ultra_detail_weight');
  const [xValues, setXValues] = useState([0.25, 0.65, 1.0]);

  const [yParamId, setYParamId] = useState('mask_erode_dilate_radius');
  const [yValues, setYValues] = useState([-8, 0, 8]);

  const [loading, setLoading] = useState(false);
  const [sweepResult, setSweepResult] = useState(null);
  const [error, setError] = useState(null);

  // Synchronized Zoom & Pan state across all 9 tiles
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const panStartRef = useRef({ x: 0, y: 0 });

  const xDef = SWEEP_PARAMETERS.find((p) => p.id === xParamId) || SWEEP_PARAMETERS[0];
  const yDef = SWEEP_PARAMETERS.find((p) => p.id === yParamId) || SWEEP_PARAMETERS[6];

  // Update values when parameter selection changes
  const handleXParamChange = (newId) => {
    setXParamId(newId);
    const def = SWEEP_PARAMETERS.find((p) => p.id === newId);
    if (def) setXValues([...def.defaultRange]);
  };

  const handleYParamChange = (newId) => {
    setYParamId(newId);
    const def = SWEEP_PARAMETERS.find((p) => p.id === newId);
    if (def) setYValues([...def.defaultRange]);
  };

  // Run Sweep
  const runSweep = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = {
        ...currentSettings,
        index: targetIndex,
        frame,
        x_param: xParamId,
        x_values: xValues,
        y_param: yParamId,
        y_values: yValues,
      };
      const res = await postJSON('/api/preview/sweep', payload);
      setSweepResult(res);
      // Reset zoom & pan on new sweep
      setZoom(1.0);
      setPan({ x: 0, y: 0 });
    } catch (err) {
      setError(err?.message || 'Failed to generate 3×3 parameter sweep');
    } finally {
      setLoading(false);
    }
  }, [postJSON, currentSettings, targetIndex, frame, xParamId, xValues, yParamId, yValues]);

  // Initial trigger when modal opens
  useEffect(() => {
    if (isOpen && !sweepResult && !loading) {
      runSweep();
    }
  }, [isOpen, sweepResult, loading, runSweep]);

  // Keyboard navigation: Esc to close, 1-9 to select tile
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
        return;
      }
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
      const num = parseInt(e.key, 10);
      if (num >= 1 && num <= 9 && sweepResult?.tiles?.length === 9) {
        e.preventDefault();
        const tile = sweepResult.tiles[num - 1];
        if (tile) {
          onApplyRecipe?.(tile.recipe);
          notify?.(`Applied Tile ${num} recipe: ${xDef.label}=${tile.x_value}, ${yDef.label}=${tile.y_value}`);
          onClose();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, sweepResult, onApplyRecipe, notify, xDef, yDef]);

  // Synchronized Pan & Zoom Handlers
  const handlePointerDown = (e) => {
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX, y: e.clientY };
    panStartRef.current = { ...pan };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };

  const handlePointerMove = (e) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;
    setPan({
      x: panStartRef.current.x + dx,
      y: panStartRef.current.y + dy,
    });
  };

  const handlePointerUp = (e) => {
    setIsDragging(false);
    e.currentTarget.releasePointerCapture?.(e.pointerId);
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    setZoom((z) => Math.min(8.0, Math.max(1.0, z * factor)));
  };

  // Facial Landmark Quick-Focus Handlers
  const focusEyes = () => {
    setZoom(2.8);
    if (sweepResult?.face_coordinates?.eyes) {
      const { left, right } = sweepResult.face_coordinates.eyes;
      const eyeMidX = (left[0] + right[0]) / 2;
      const eyeMidY = (left[1] + right[1]) / 2;
      setPan({ x: -(eyeMidX - 120) * 1.5, y: -(eyeMidY - 120) * 1.5 });
    } else {
      setPan({ x: 0, y: 40 });
    }
  };

  const focusMouth = () => {
    setZoom(3.0);
    if (sweepResult?.face_coordinates?.mouth) {
      const { center } = sweepResult.face_coordinates.mouth;
      setPan({ x: -(center[0] - 120) * 1.8, y: -(center[1] - 120) * 1.8 });
    } else {
      setPan({ x: 0, y: -60 });
    }
  };

  const focusJawline = () => {
    setZoom(2.6);
    if (sweepResult?.face_coordinates?.jawline) {
      const { center } = sweepResult.face_coordinates.jawline;
      setPan({ x: -(center[0] - 120) * 1.5, y: -(center[1] - 120) * 1.5 });
    } else {
      setPan({ x: 0, y: -100 });
    }
  };

  const fitReset = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };

  const handleApplyTile = (tile) => {
    onApplyRecipe?.(tile.recipe);
    notify?.(`Applied Tile recipe: ${xDef.label}=${tile.x_value}, ${yDef.label}=${tile.y_value}`);
    onClose();
  };

  if (!isOpen) return null;

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="parameter-sweep-title"
      data-testid="parameter-sweep-modal"
      className="fixed inset-0 z-[120] flex items-center justify-center bg-black/85 backdrop-blur-md p-4 sm:p-6 select-none"
    >
      <div
        className="relative flex flex-col w-full max-w-[1300px] max-h-[95vh] rounded-2xl border border-white/15 bg-[#10121d] shadow-2xl overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-white/10 bg-white/[0.03]">
          <div className="flex items-center gap-3">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-[var(--accent)]/20 border border-[var(--accent)]/40 text-[var(--accent)] text-sm font-bold">
              3×3
            </span>
            <div>
              <h2 id="parameter-sweep-title" className="text-sm font-bold text-white tracking-wide">
                Parameter Sweep Studio (3×3 Contact Sheet Grid)
              </h2>
              <p className="text-micro text-white/50">
                Synchronized lockstep zoom/pan evaluation · Reusing Stage 3 detection crop · Press 1–9 to commit recipe
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-lg text-white/60 hover:text-white hover:bg-white/10 transition-colors"
            title="Close (Esc)"
          >
            ✕
          </button>
        </div>

        {/* Toolbar & Axis Configuration */}
        <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3 border-b border-white/10 bg-white/[0.015]">
          <div className="flex flex-wrap items-center gap-4">
            {/* X-Axis Parameter */}
            <div className="flex items-center gap-2">
              <span className="text-micro font-bold text-[var(--accent)] uppercase tracking-wider">X-Axis:</span>
              <select
                value={xParamId}
                onChange={(e) => handleXParamChange(e.target.value)}
                className="px-2.5 py-1 rounded-lg bg-black/60 border border-white/15 text-xs text-white focus:outline-none cursor-pointer"
              >
                {SWEEP_PARAMETERS.map((p) => (
                  <option key={p.id} value={p.id} className="bg-[#121420]">
                    {p.label}
                  </option>
                ))}
              </select>

              {/* X Values [Min, Mid, Max] */}
              <div className="flex items-center gap-1">
                {xDef.type === 'categorical' ? (
                  xValues.map((val, idx) => (
                    <select
                      key={idx}
                      value={val}
                      onChange={(e) => {
                        const next = [...xValues];
                        next[idx] = e.target.value;
                        setXValues(next);
                      }}
                      className="px-2 py-0.5 rounded bg-black/60 border border-white/15 text-micro text-white cursor-pointer"
                    >
                      {xDef.options.map((opt) => (
                        <option key={opt} value={opt} className="bg-[#121420]">
                          {opt}
                        </option>
                      ))}
                    </select>
                  ))
                ) : (
                  xValues.map((val, idx) => (
                    <input
                      key={idx}
                      type="number"
                      step={xDef.step}
                      min={xDef.min}
                      max={xDef.max}
                      value={val}
                      onChange={(e) => {
                        const next = [...xValues];
                        next[idx] = parseFloat(e.target.value) || 0;
                        setXValues(next);
                      }}
                      className="w-14 px-1.5 py-0.5 rounded bg-black/60 border border-white/15 text-micro font-mono text-center text-white focus:outline-none"
                      title={`${['Min', 'Mid', 'Max'][idx]} ${xDef.label}`}
                    />
                  ))
                )}
              </div>
            </div>

            {/* Y-Axis Parameter */}
            <div className="flex items-center gap-2">
              <span className="text-micro font-bold text-amber-300 uppercase tracking-wider">Y-Axis:</span>
              <select
                value={yParamId}
                onChange={(e) => handleYParamChange(e.target.value)}
                className="px-2.5 py-1 rounded-lg bg-black/60 border border-white/15 text-xs text-white focus:outline-none cursor-pointer"
              >
                {SWEEP_PARAMETERS.map((p) => (
                  <option key={p.id} value={p.id} className="bg-[#121420]">
                    {p.label}
                  </option>
                ))}
              </select>

              {/* Y Values [Min, Mid, Max] */}
              <div className="flex items-center gap-1">
                {yDef.type === 'categorical' ? (
                  yValues.map((val, idx) => (
                    <select
                      key={idx}
                      value={val}
                      onChange={(e) => {
                        const next = [...yValues];
                        next[idx] = e.target.value;
                        setYValues(next);
                      }}
                      className="px-2 py-0.5 rounded bg-black/60 border border-white/15 text-micro text-white cursor-pointer"
                    >
                      {yDef.options.map((opt) => (
                        <option key={opt} value={opt} className="bg-[#121420]">
                          {opt}
                        </option>
                      ))}
                    </select>
                  ))
                ) : (
                  yValues.map((val, idx) => (
                    <input
                      key={idx}
                      type="number"
                      step={yDef.step}
                      min={yDef.min}
                      max={yDef.max}
                      value={val}
                      onChange={(e) => {
                        const next = [...yValues];
                        next[idx] = parseFloat(e.target.value) || 0;
                        setYValues(next);
                      }}
                      className="w-14 px-1.5 py-0.5 rounded bg-black/60 border border-white/15 text-micro font-mono text-center text-white focus:outline-none"
                      title={`${['Min', 'Mid', 'Max'][idx]} ${yDef.label}`}
                    />
                  ))
                )}
              </div>
            </div>

            <button
              type="button"
              onClick={runSweep}
              disabled={loading}
              className="px-3 py-1 rounded-lg bg-[var(--accent)] text-black text-xs font-bold hover:brightness-110 active:scale-95 transition-all shadow-md disabled:opacity-50"
            >
              {loading ? 'Evaluating 9 Variations…' : 'Generate 3×3 Grid'}
            </button>
          </div>

          {/* Synchronized Zoom & Facial Feature Focus Controls */}
          <div className="flex items-center gap-2">
            <span className="text-micro text-white/50 font-mono tabular-nums">
              {Math.round(zoom * 100)}%
            </span>
            <div className="flex items-center rounded-lg bg-black/60 border border-white/15 p-0.5">
              <button
                type="button"
                onClick={fitReset}
                className="px-2 py-0.5 rounded text-micro text-white/80 hover:text-white hover:bg-white/10 transition-colors"
                title="Fit full face crop (Reset zoom/pan)"
              >
                Fit
              </button>
              <button
                type="button"
                onClick={focusEyes}
                className="px-2 py-0.5 rounded text-micro text-cyan-300 hover:text-cyan-100 hover:bg-white/10 transition-colors font-semibold"
                title="Synchronized zoom onto Eyes (2.8x)"
              >
                Eyes
              </button>
              <button
                type="button"
                onClick={focusMouth}
                className="px-2 py-0.5 rounded text-micro text-emerald-300 hover:text-emerald-100 hover:bg-white/10 transition-colors font-semibold"
                title="Synchronized zoom onto Mouth (3.0x)"
              >
                Mouth
              </button>
              <button
                type="button"
                onClick={focusJawline}
                className="px-2 py-0.5 rounded text-micro text-amber-300 hover:text-amber-100 hover:bg-white/10 transition-colors font-semibold"
                title="Synchronized zoom onto Jawline (2.6x)"
              >
                Jawline
              </button>
            </div>
          </div>
        </div>

        {/* 3x3 Contact Sheet Canvas Grid */}
        <div className="relative flex-1 p-4 overflow-auto min-h-[460px] max-h-[72vh] bg-black/50">
          {error && (
            <div className="mb-3 p-3 rounded-xl bg-rose-500/20 border border-rose-500/30 text-rose-200 text-xs">
              {error}
            </div>
          )}

          {loading && !sweepResult && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/60 backdrop-blur-sm z-20">
              <span className="h-8 w-8 rounded-full border-2 border-[var(--accent)] border-t-transparent animate-spin" />
              <p className="text-sm font-semibold text-white/90">
                Coalescing 9 parameter variations on current frame…
              </p>
              <p className="text-micro text-white/50">Reusing Stage 3 face detection & alignment crop once</p>
            </div>
          )}

          {/* Matrix Container */}
          <div
            className="flex flex-col gap-2 max-w-[1200px] mx-auto cursor-grab active:cursor-grabbing"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onWheel={handleWheel}
          >
            {/* Top X-Axis Column Headers */}
            <div className="grid grid-cols-[80px_repeat(3,1fr)] gap-2">
              <div />
              {xValues.map((xv, colIdx) => (
                <div
                  key={colIdx}
                  className="py-1 px-2 rounded-lg bg-[var(--accent)]/10 border border-[var(--accent)]/20 text-center"
                >
                  <span className="text-micro font-bold text-[var(--accent)]">
                    X{colIdx + 1}: {xDef.label} = <span className="font-mono">{xv}</span>
                  </span>
                </div>
              ))}
            </div>

            {/* 3 Rows with Left Y-Axis Header */}
            {[0, 1, 2].map((rowIdx) => {
              const yVal = yValues[rowIdx];
              return (
                <div key={rowIdx} className="grid grid-cols-[80px_repeat(3,1fr)] gap-2 items-center">
                  {/* Left Y Header */}
                  <div className="py-2 px-1 rounded-lg bg-amber-400/10 border border-amber-400/20 text-center flex flex-col justify-center">
                    <span className="text-micro font-bold text-amber-300">Y{rowIdx + 1}</span>
                    <span className="text-[10px] font-mono text-white/70">{yVal}</span>
                  </div>

                  {/* 3 Tiles in this row */}
                  {[0, 1, 2].map((colIdx) => {
                    const tileIndex = rowIdx * 3 + colIdx;
                    const tile = sweepResult?.tiles?.[tileIndex] || null;
                    const xVal = xValues[colIdx];

                    return (
                      <div
                        key={colIdx}
                        data-testid={`sweep-tile-${colIdx}-${rowIdx}`}
                        className="group relative aspect-square rounded-xl overflow-hidden bg-black/80 border border-white/10 hover:border-[var(--accent)] transition-all shadow-md flex flex-col"
                      >
                        {/* Synchronized viewport frame */}
                        <div className="relative flex-1 overflow-hidden w-full h-full flex items-center justify-center">
                          {tile?.image ? (
                            <img
                              src={tile.image}
                              alt={`Tile X=${xVal} Y=${yVal}`}
                              className="w-full h-full object-contain pointer-events-none"
                              style={{
                                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                                transformOrigin: 'center center',
                                transition: isDragging ? 'none' : 'transform 0.08s ease-out',
                              }}
                            />
                          ) : (
                            <div className="flex flex-col items-center justify-center text-white/30 text-micro">
                              <span>Tile ({colIdx + 1}, {rowIdx + 1})</span>
                            </div>
                          )}
                        </div>

                        {/* Top-Left Slot Tag */}
                        <div className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded bg-black/75 border border-white/15 text-[10px] font-mono font-bold text-white/80 pointer-events-none">
                          #{tileIndex + 1}
                        </div>

                        {/* Bottom Recipe Bar & Apply Button */}
                        <div className="absolute bottom-0 inset-x-0 p-2 bg-gradient-to-t from-black/90 via-black/75 to-transparent flex items-center justify-between gap-1 opacity-90 group-hover:opacity-100 transition-opacity">
                          <div className="flex flex-col text-[10px] text-white/70 font-mono leading-tight">
                            <span>{xDef.label.slice(0, 12)}: {xVal}</span>
                            <span>{yDef.label.slice(0, 12)}: {yVal}</span>
                          </div>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              if (tile) handleApplyTile(tile);
                            }}
                            className="px-2.5 py-1 rounded bg-[var(--accent)] text-black text-micro font-bold hover:brightness-110 active:scale-95 transition-all shadow-md"
                            title={`Apply recipe to FaceSwap (#${tileIndex + 1})`}
                          >
                            Apply Recipe
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>

        {/* Footer info bar */}
        <div className="flex items-center justify-between px-5 py-2.5 border-t border-white/10 bg-white/[0.02] text-micro text-white/50">
          <span>Click & drag inside any tile to pan all 9 tiles · Mouse wheel to zoom in/out</span>
          <span>Click &quot;Apply Recipe&quot; or press 1–9 to commit selected variation</span>
        </div>
      </div>
    </div>,
    document.body
  );
}
