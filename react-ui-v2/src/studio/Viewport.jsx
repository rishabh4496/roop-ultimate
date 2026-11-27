import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Film, ImagePlus, ScanFace, SplitSquareHorizontal, Sparkles, Image as ImageIcon, Maximize2, ZoomIn, Users } from 'lucide-react';
import { useStudio, currentTarget } from '../engine/studio';
import { sourceNameAt } from '../engine/payload';
import { useRun } from '../engine/run';
import { peopleOf, hueOf, mappedSource, timecode, frameUrl, isSelectedPerson } from './people';
import { Meteors } from '../fx/effects';

// ── The viewport ────────────────────────────────────────────────────────────
// The picture the engine will render, for the frame under the playhead:
//   * Swap      - the swapped frame from /api/preview, with every detected face reticled;
//   * Original  - the untouched frame and its faces (what you pick people on);
//   * Compare   - original left, swap right, a wipe you drag.
// While the playhead is being dragged it shows raw frames (a cheap JPEG GET) and asks for the swap
// once you let go, so scrubbing never queues swap inference. Clicking a face makes it a target
// person (or selects the person it already is). Zoom: wheel / double-click; pan: drag when zoomed.

const MODES = [
  { id: 'swap', label: 'Swap', icon: Sparkles },
  { id: 'original', label: 'Original', icon: ImageIcon },
  { id: 'compare', label: 'Compare', icon: SplitSquareHorizontal },
];

// The engine's boxes are [x1, y1, x2, y2] (insightface). A box whose far corner is not past its
// near one cannot be that, so it is read as [x, y, w, h] - only the mock backend sends that form.
const asCorners = (b) => {
  const [a, c, d, e] = (b || []).map(Number);
  return d > a && e > c ? [a, c, d, e] : [a, c, a + Math.abs(d), c + Math.abs(e)];
};

function fitRect(boxW, boxH, w, h) {
  if (!w || !h || !boxW || !boxH) return null;
  const s = Math.min(boxW / w, boxH / h);
  return { w: w * s, h: h * s, x: (boxW - w * s) / 2, y: (boxH - h * s) / 2 };
}

export default function Viewport({ scrubbing }) {
  const s = useStudio();
  const target = currentTarget(s);
  const processing = useRun((r) => r.progress.processing);
  const { preview, previewMode, frame, selectedTarget } = s;
  const box = useRef(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [nat, setNat] = useState({ w: 0, h: 0 });
  const [wipe, setWipe] = useState(50);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [over, setOver] = useState(false);
  const [hoverFace, setHoverFace] = useState(null);

  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Ask for the swap preview when what it depends on settles (debounced; scrubbing shows raw frames).
  const deps = JSON.stringify([frame, selectedTarget, previewMode, s.targetMediaId, s.faceMapping, s.selectedTargetPersonId, s.selTargetFace,
    s.selectedSource, s.sources.length, s.targetPersonIds, s.settings]);
  useEffect(() => {
    if (scrubbing || !target) return undefined;
    const h = setTimeout(() => useStudio.getState().requestPreview(), 220);
    return () => clearTimeout(h);
  }, [deps, scrubbing, target]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { setZoom(1); setPan({ x: 0, y: 0 }); }, [selectedTarget]);

  const rawUrl = target ? frameUrl(selectedTarget, frame, 1280) : null;
  const showRaw = scrubbing || (!preview.image && !!target);
  const mainSrc = showRaw ? rawUrl : preview.image;
  const rect = fitRect(size.w, size.h, nat.w, nat.h);
  const people = useMemo(() => peopleOf(s), [s.targetPersonIds, s.targetGroups, s.targetNames, s.targetFaces]); // eslint-disable-line react-hooks/exhaustive-deps
  const rankOf = useMemo(() => new Map(people.map((p) => [p.id, p])), [people]);
  const faces = !showRaw && preview.frame === frame ? (preview.faces || []).map(asCorners) : [];

  const onPickFace = useCallback((i) => {
    const pid = preview.personIds?.[i];
    if (pid && rankOf.has(pid)) {
      const p = rankOf.get(pid);
      useStudio.getState().selectPerson(pid, p.faceIndex);
    } else {
      useStudio.getState().captureFace({ faceIndex: i });
    }
  }, [preview.personIds, rankOf]);

  // Zoom / pan
  const onWheel = (e) => {
    if (!rect) return;
    e.preventDefault();
    setZoom((z) => Math.min(6, Math.max(1, z * (e.deltaY < 0 ? 1.12 : 1 / 1.12))));
  };
  useEffect(() => {
    const el = box.current;
    if (!el) return undefined;
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  });
  useEffect(() => { if (zoom === 1) setPan({ x: 0, y: 0 }); }, [zoom]);
  const drag = useRef(null);
  const onPointerDown = (e) => {
    if (zoom <= 1 || e.button !== 0 || e.target.closest('[data-face],[data-wipe],button')) return;
    drag.current = { x: e.clientX, y: e.clientY, pan };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e) => {
    if (!drag.current) return;
    setPan({ x: drag.current.pan.x + (e.clientX - drag.current.x), y: drag.current.pan.y + (e.clientY - drag.current.y) });
  };
  const onPointerUp = () => { drag.current = null; };

  // Wipe
  const wipeDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const stage = e.currentTarget.parentElement.getBoundingClientRect();
    const move = (ev) => setWipe(Math.max(0, Math.min(100, ((ev.clientX - stage.left) / stage.width) * 100)));
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  // Drop media
  const onDrop = (e) => {
    e.preventDefault();
    setOver(false);
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length) useStudio.getState().addTargets(files);
  };

  const fileInput = useRef(null);

  if (!target) {
    return (
      <div className={`viewport dropzone h-full ${over ? 'over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={onDrop}
        data-testid="viewport-empty">
        <Meteors count={14} />
        <div className="relative grid h-16 w-16 place-items-center rounded-2xl border border-[var(--rule-strong)] bg-black/60">
          <Film size={28} strokeWidth={1.6} className="text-[var(--accent-2)]" />
        </div>
        <div className="relative text-lg font-semibold text-white">Drop a video or image to begin</div>
        <div className="relative max-w-md text-[13px] text-[var(--fg-dim)]">It becomes the target. Then add a source face to the cast and click the person to swap.</div>
        <button type="button" className="btn btn-primary relative mt-2" onClick={() => fileInput.current?.click()}>
          <ImagePlus size={16} /> Choose media
        </button>
        <input ref={fileInput} type="file" multiple accept="video/*,image/*" className="hidden"
          onChange={(e) => { useStudio.getState().addTargets(Array.from(e.target.files || [])); e.target.value = ''; }} />
      </div>
    );
  }

  const stageStyle = rect ? {
    left: rect.x, top: rect.y, width: rect.w, height: rect.h,
    transform: zoom !== 1 || pan.x || pan.y ? `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` : undefined,
    transformOrigin: 'center center',
  } : { inset: 0, opacity: 0 };
  const audit = preview.audit || {};
  const swapped = Object.entries(audit).filter(([k]) => /swapped/i.test(k)).reduce((a, [, v]) => a + v, 0);
  const refused = Object.entries(audit).filter(([k]) => /refused|skipped/i.test(k)).reduce((a, [, v]) => a + v, 0);

  return (
    <div ref={box} className={`viewport h-full ${preview.loading ? 'beam' : ''} ${over ? 'ring-2 ring-[var(--accent-2)]' : ''}`}
      onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
      onDoubleClick={(e) => { if (!e.target.closest('[data-face],[data-wipe],button')) setZoom((z) => (z > 1 ? 1 : 2)); }}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={onDrop}
      style={{ cursor: zoom > 1 ? (drag.current ? 'grabbing' : 'grab') : 'default' }}
      data-testid="viewport">
      <span className="viewport-corner tl" /><span className="viewport-corner tr" />
      <span className="viewport-corner bl" /><span className="viewport-corner br" />

      <div className="absolute" style={stageStyle}>
        {previewMode === 'compare' && !showRaw ? (
          <>
            <img src={rawUrl} alt="Original frame" className="absolute inset-0 h-full w-full select-none" draggable={false} />
            <img src={preview.image} alt="Swapped frame" className="absolute inset-0 h-full w-full select-none" draggable={false}
              style={{ clipPath: `inset(0 0 0 ${wipe}%)` }}
              onLoad={(e) => setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
            <div data-wipe className="wipe-handle" style={{ left: `calc(${wipe}% - 1px)` }} onPointerDown={wipeDrag}
              role="slider" aria-label="Compare wipe" aria-valuenow={Math.round(wipe)} aria-valuemin={0} aria-valuemax={100} tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'ArrowLeft') setWipe((w) => Math.max(0, w - 5)); if (e.key === 'ArrowRight') setWipe((w) => Math.min(100, w + 5)); }} />
            <span className="hud left-3 top-3" style={{ top: 'auto', bottom: 12 }}>original</span>
            <span className="hud right-3" style={{ bottom: 12 }}>swap</span>
          </>
        ) : mainSrc ? (
          <img src={mainSrc} alt={showRaw ? 'Target frame' : 'Preview'} className="absolute inset-0 h-full w-full select-none" draggable={false}
            onLoad={(e) => setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
        ) : null}

        {/* Face reticles (frame-pixel coordinates, scaled with the picture). */}
        {nat.w > 0 && faces.length > 0 && (
          <svg className="absolute inset-0 h-full w-full" viewBox={`0 0 ${nat.w} ${nat.h}`} preserveAspectRatio="none">
            {faces.map((b, i) => {
              const [x1, y1, x2, y2] = b;
              const pid = preview.personIds?.[i];
              const p = pid ? rankOf.get(pid) : null;
              const color = p ? hueOf(p.rank) : '#ffffff';
              const sel = p && isSelectedPerson(s, p);
              const w = x2 - x1; const h = y2 - y1; const c = Math.min(w, h) * 0.22;
              const corners = `M${x1},${y1 + c} V${y1} H${x1 + c} M${x2 - c},${y1} H${x2} V${y1 + c} M${x2},${y2 - c} V${y2} H${x2 - c} M${x1 + c},${y2} H${x1} V${y2 - c}`;
              return (
                <g key={i} data-face className={`face-box ${sel ? 'selected' : ''}`} style={{ color, cursor: 'pointer' }}
                  onClick={(e) => { e.stopPropagation(); onPickFace(i); }}
                  onPointerEnter={() => setHoverFace(i)} onPointerLeave={() => setHoverFace(null)}>
                  <rect className="frame" x={x1} y={y1} width={w} height={h} stroke={color} fill={hoverFace === i ? `${color}14` : 'transparent'} style={{ pointerEvents: 'all' }} />
                  <path className="corner" d={corners} stroke={color} />
                </g>
              );
            })}
          </svg>
        )}
        {nat.w > 0 && faces.map((b, i) => {
          const pid = preview.personIds?.[i];
          const p = pid ? rankOf.get(pid) : null;
          const color = p ? hueOf(p.rank) : '#ffffff';
          const src = p ? mappedSource(s, p.id) : null;
          const becomes = p ? (src == null ? (s.settings?.face_detection_mode === 'Selected face' && isSelectedPerson(s, p) ? sourceNameAt(s, s.selectedSource) : null) : src < 0 ? 'untouched' : sourceNameAt(s, src)) : null;
          return (
            <button key={`t${i}`} type="button" data-face className="face-tag" style={{ color, left: `${(b[0] / nat.w) * 100}%`, top: `${(b[1] / nat.h) * 100}%` }}
              onClick={(e) => { e.stopPropagation(); onPickFace(i); }} title={p ? `${p.name}: click to select` : 'Click to make this face a target person'}>
              {p?.thumb ? <img src={p.thumb} alt="" className="h-4 w-4 rounded-full object-cover" /> : <ScanFace size={13} />}
              <span className="text-white">{p ? p.name : `Face ${i + 1}`}</span>
              {p && <span className="font-mono text-[10.5px] text-[var(--fg-dim)]">→ {becomes || 'map…'}</span>}
              {!p && <span className="font-mono text-[10.5px] text-[var(--accent-2)]">+ pick</span>}
            </button>
          );
        })}
      </div>

      {preview.loading && <div className="scanline" aria-hidden="true" />}

      {/* HUD */}
      <div className="hud left-3 top-3" data-testid="viewport-hud">
        <span className="text-[var(--accent-2)]">F</span><span className="tnum">{frame}</span>
        <span className="text-[var(--fg-faint)]">/ {target.frames || 1}</span>
        <span className="text-[var(--fg-faint)]">·</span>
        <span className="tnum">{timecode(frame, target.fps)}</span>
        {nat.w > 0 && <><span className="text-[var(--fg-faint)]">·</span><span>{nat.w}×{nat.h}</span></>}
      </div>
      <div className="hud right-3 top-3 !p-1" role="radiogroup" aria-label="Viewport mode">
        {MODES.map((m) => (
          <button key={m.id} type="button" role="radio" aria-checked={previewMode === m.id}
            className={`chip !h-7 ${previewMode === m.id ? 'on' : ''}`} onClick={() => s.setPreviewMode(m.id)} data-testid={`mode-${m.id}`}>
            <m.icon size={13} /> {m.label}
          </button>
        ))}
        <span className="mx-1 h-4 w-px bg-white/15" aria-hidden="true" />
        <button type="button" className="chip !h-7" onClick={() => setZoom((z) => (z > 1 ? 1 : 2))} aria-label={zoom > 1 ? 'Fit to view' : 'Zoom 2x'} title="Zoom (wheel / double-click)">
          {zoom > 1 ? <Maximize2 size={13} /> : <ZoomIn size={13} />} {Math.round(zoom * 100)}%
        </button>
      </div>
      <div className="hud bottom-3 left-3 max-w-[70%]" data-testid="viewport-status">
        {processing ? <span className="text-[var(--warn)]">render running — preview paused</span>
          : preview.error ? <span className="text-[var(--danger)]">{preview.error}</span>
          : preview.message && preview.diagnostic ? <span className="text-[var(--warn)]">{preview.message}</span>
          : scrubbing ? <span className="text-[var(--fg-dim)]">scrubbing · raw frames</span>
          : preview.loading ? <span className="text-[var(--accent-2)]">rendering preview…</span>
          : !s.sources.length ? <span className="text-[var(--warn)]">add a source face to see the swap</span>
          : faces.length ? <span><Users size={12} className="mr-1 inline" />{faces.length} face{faces.length > 1 ? 's' : ''}{swapped ? <> · <span className="text-[var(--ok)]">{swapped} swapped</span></> : ''}{refused ? <> · <span className="text-[var(--warn)]">{refused} refused</span></> : ''}</span>
          : preview.image ? <span className="text-[var(--fg-dim)]">no face on this frame</span> : null}
      </div>
    </div>
  );
}
