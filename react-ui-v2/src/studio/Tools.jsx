import React, { useEffect, useMemo, useState } from 'react';
import { Brush, Eraser, Check, PenTool, Pin, Blend, Clapperboard, Spline, SplitSquareHorizontal, Trash2, Plus, X, Layers } from 'lucide-react';
import { getJSON } from '@engine/api';
import { setMode, setFeather, setRange, describe as describeZone, MODE_COLOR } from '@engine/components/faceswap/rotoModel.js';
import { makePin, addPin, removePin } from '@engine/components/faceswap/identityLock.js';
import { BLEND_MODES, BLEND_MODE_LABELS, MAX_BLEND_SOURCES } from '@engine/components/faceswap/identityBlend.js';
import { addKeyframe, removeKeyframe, updateKeyframe, seedWeightsAt, CURVES, CURVE_LABELS, uiFrameOf } from '@engine/components/faceswap/identityKeyframes.js';
import { OVERRIDE_FIELDS, addOverride, updateOverride, removeOverride, describeOverride, overrideAt } from '@engine/components/faceswap/timeline/overrides.js';
import { DIRECT_PARAMS, INTERPS, INTERP_LABELS, addKey, removeKey, setInterp, setValue, evaluate, clampValue, curveIneffective } from '@engine/components/faceswap/curves/curveModel.js';
import { swapperOptions } from '@engine/components/faceswap/pipelineRules.js';
import { useStudio, currentTarget } from '../engine/studio';
import { clipFxOf, renderPlan, sourceIdAt, sourceNameAt, inOutOf } from '../engine/payload';
import { Panel } from '../fx/effects';
import { Slider, Switch, Select, notify } from '../ui/controls';
import { exportMask } from './Overlays';
import { peopleOf, mappedSource, asCorners } from './people';

// ── The studio's precision tools ─────────────────────────────────────────────
// Every tab edits state that rides in the render (payload.js): the brush mask (imagemask), roto
// zones (roto_polygons), identity pins (identity_pins), the identity blend and its keyframes
// (identity_blend), scene overrides and parameter curves (which cut the render into queued pieces).
// The maths is React UI 1.0's (rotoModel, identityLock, identityKeyframes, overrides, curveModel) -
// what is drawn here is what renders.

const TABS = [
  { id: 'mask', label: 'Mask', icon: Brush },
  { id: 'roto', label: 'Roto', icon: PenTool },
  { id: 'pins', label: 'Pins', icon: Pin },
  { id: 'blend', label: 'Blend', icon: Blend },
  { id: 'scenes', label: 'Scenes', icon: Clapperboard },
  { id: 'curves', label: 'Curves', icon: Spline },
  { id: 'reference', label: 'Ref', icon: SplitSquareHorizontal },
];
const TOOL_OF_TAB = { mask: 'mask', roto: 'roto', pins: 'pin' };

function MaskTab() {
  const brush = useStudio((s) => s.brush);
  const mask = useStudio((s) => s.mask);
  const { setBrush, applyMask } = useStudio.getState();
  return (
    <div className="space-y-2.5" data-testid="tool-mask">
      <p className="text-[12px] text-[var(--fg-dim)]">Paint on the viewport where the <b>original</b> face must stay (a hand, a mic, hair). It is bound to the selected person's landmarks, so it follows them.</p>
      <div className="flex gap-1.5">
        <button type="button" className={`chip !h-7 ${!brush.erase ? 'on' : ''}`} onClick={() => setBrush({ erase: false })}><Brush size={12} /> Paint</button>
        <button type="button" className={`chip !h-7 ${brush.erase ? 'on' : ''}`} onClick={() => setBrush({ erase: true })}><Eraser size={12} /> Erase</button>
      </div>
      <Slider label="Brush size" value={brush.size} min={4} max={160} step={1} format={(v) => `${v}px`} onChange={(v) => setBrush({ size: v })} />
      <div className="flex gap-2">
        <button type="button" className="btn btn-primary btn-sm flex-1" data-testid="mask-apply"
          onClick={() => { const d = exportMask(); applyMask(d); notify(d ? 'Mask applied - painted areas keep the original' : 'Nothing painted - mask cleared', d ? 'success' : 'info'); }}>
          <Check size={13} /> Apply mask
        </button>
        <button type="button" className="btn btn-sm" onClick={() => { applyMask(null); notify('Mask cleared', 'info'); }}><Trash2 size={13} /> Clear</button>
      </div>
      <div className="font-mono text-[11px] text-[var(--fg-faint)]">{mask.dataUrl ? `mask v${mask.version} in the render${mask.refKps ? ' · bound to landmarks' : ''}` : 'no mask in the render'}</div>
    </div>
  );
}

function RotoTab() {
  const s = useStudio();
  const zones = clipFxOf(s).roto;
  const t = currentTarget(s);
  const up = (id, fn) => s.updateClipFx((fx) => ({ roto: fx.roto.map((z) => (z.id === id ? fn(z) : z)) }));
  return (
    <div className="space-y-2" data-testid="tool-roto">
      <p className="text-[12px] text-[var(--fg-dim)]">Click on the viewport to draw a zone; click its first point to close it. <b style={{ color: MODE_COLOR.exclude }}>Exclude</b> restores the original inside; <b style={{ color: MODE_COLOR.include }}>include</b> forces the swap.</p>
      {zones.map((z, i) => (
        <div key={z.id} className="tile p-2">
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: MODE_COLOR[z.mode] }} />
            <span className="min-w-0 flex-1 truncate font-mono text-[11px]">{describeZone(z, i)}</span>
            <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Delete zone" onClick={() => s.updateClipFx((fx) => ({ roto: fx.roto.filter((x) => x.id !== z.id) }))}><Trash2 size={13} /></button>
          </div>
          <div className="mt-1.5 flex flex-wrap gap-1">
            {['exclude', 'include'].map((m) => <button key={m} type="button" className={`chip ${z.mode === m ? 'on' : ''}`} onClick={() => up(z.id, (x) => setMode(x, m))}>{m}</button>)}
            <span className="mx-1 w-px bg-white/10" />
            <button type="button" className={`chip ${!z.frames ? 'on' : ''}`} onClick={() => up(z.id, (x) => setRange(x, null))}>whole clip</button>
            <button type="button" className="chip" onClick={() => up(z.id, (x) => setRange(x, [inOutOf(t).a, inOutOf(t).b]))}>in–out</button>
            <button type="button" className="chip" onClick={() => up(z.id, (x) => setRange(x, [s.frame, s.frame]))}>this frame</button>
          </div>
          <div className="mt-1.5"><Slider label="Feather" value={z.feather} min={0} max={64} step={1} format={(v) => `${v}px`} onChange={(v) => up(z.id, (x) => setFeather(x, v))} /></div>
        </div>
      ))}
      {!zones.length && <div className="text-[12px] text-[var(--fg-faint)]">No zones on this clip.</div>}
    </div>
  );
}

function PinsTab() {
  const s = useStudio();
  const pins = clipFxOf(s).pins;
  return (
    <div className="space-y-2" data-testid="tool-pins">
      <p className="text-[12px] text-[var(--fg-dim)]">With this tab open, click a face on the viewport to lock it to the source it is mapped to from this frame on - so a crossing or a turn cannot hand it to someone else.</p>
      {pins.map((p) => (
        <div key={p.id} className="tile flex items-center gap-2 p-2 font-mono text-[11.5px]">
          <Pin size={13} className="text-[var(--accent-2)]" />
          <span className="flex-1">frame {p.frame} → {sourceNameAt(s, p.source_index) || `source ${p.source_index + 1}`}</span>
          <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Remove pin" onClick={() => s.updateClipFx((fx) => ({ pins: removePin(fx.pins, p.id) }))}><X size={13} /></button>
        </div>
      ))}
      {!pins.length && <div className="text-[12px] text-[var(--fg-faint)]">No pins on this clip.</div>}
    </div>
  );
}

/** Pin face `i` of the current preview (called from the viewport in the Pins tool). */
export function pinFace(i) {
  const s = useStudio.getState();
  const raw = s.preview.faces?.[i];
  const bbox = raw ? asCorners(raw) : null;
  const personId = s.preview.personIds?.[i] ?? null;
  const people = peopleOf(s);
  const person = people.find((p) => p.id === personId);
  const mapped = person ? mappedSource(s, person.id) : null;
  const sourceIndex = mapped != null && mapped >= 0 ? mapped : s.selectedSource;
  const pin = makePin({ frame: s.frame, bbox, personId, sourceIndex, faceIndex: i });
  if (!pin) { notify('That face cannot be pinned (no box or no source)', 'error'); return; }
  s.updateClipFx((fx) => ({ pins: addPin(fx.pins, pin) }));
  notify(`Pinned to ${sourceNameAt(s, sourceIndex)} from frame ${s.frame}`, 'info');
}

function BlendTab() {
  const s = useStudio();
  const b = s.identityBlend;
  const total = Math.max(1, currentTarget(s)?.frames || 1);
  const ids = s.sources.map((_, i) => sourceIdAt(s, i));
  const comp = (id) => b.components.find((c) => c.source_id === id);
  const set = (patch) => s.setIdentityBlend({ ...b, ...patch });
  const toggle = (id) => {
    const has = comp(id);
    if (!has && b.components.length >= MAX_BLEND_SOURCES) { notify(`At most ${MAX_BLEND_SOURCES} identities`, 'warn'); return; }
    set({ components: has ? b.components.filter((c) => c.source_id !== id) : [...b.components, { source_id: id, weight: 1 }] });
  };
  return (
    <div className="space-y-2.5" data-testid="tool-blend">
      <Switch label="Blend identities" hint="The swap becomes a weighted mix of these cast faces" checked={b.enabled} onChange={(v) => set({ enabled: v })} />
      <div className="flex flex-wrap gap-1.5">
        {s.sources.map((src, i) => (
          <button key={ids[i]} type="button" aria-pressed={!!comp(ids[i])} onClick={() => toggle(ids[i])} className={`tile flex items-center gap-1.5 py-1 pl-1 pr-2 ${comp(ids[i]) ? 'selected' : ''}`}>
            <img src={src} alt="" className="h-6 w-6 rounded-full object-cover" /><span className="text-[11.5px]">{sourceNameAt(s, i)}</span>
          </button>
        ))}
      </div>
      {b.components.map((c) => (
        <Slider key={c.source_id} label={sourceNameAt(s, ids.indexOf(c.source_id)) || c.source_id} value={c.weight} min={0} max={1} step={0.01} format={(v) => v.toFixed(2)}
          onChange={(v) => set({ components: b.components.map((x) => (x.source_id === c.source_id ? { ...x, weight: v } : x)) })} />
      ))}
      <div className="flex gap-1.5">{BLEND_MODES.map((m) => <button key={m} type="button" className={`chip !h-7 ${b.mode === m ? 'on' : ''}`} onClick={() => set({ mode: m })}>{BLEND_MODE_LABELS[m]}</button>)}</div>
      <Slider label="Jawline" value={b.dials?.jawline ?? 0} min={-1} max={1} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => set({ dials: { ...b.dials, jawline: v } })} />
      <Slider label="Minimum similarity to the primary" value={b.min_cosine} min={0} max={1} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => set({ min_cosine: v })} />
      <div className="rounded-xl border border-[var(--rule)] bg-black/30 p-2">
        <div className="flex items-center justify-between">
          <span className="text-[12px] font-semibold">Keyframes · {b.keyframes.length}</span>
          <button type="button" className="btn btn-sm" disabled={b.components.length < 2} title="Key the current weights at the playhead"
            onClick={() => {
              const r = addKeyframe(b.keyframes, s.frame, seedWeightsAt(b.keyframes, b.components, s.frame), total);
              if (r.full) notify('Keyframe limit reached', 'warn'); else set({ keyframes: r.list });
            }}><Plus size={12} /> Key at f{s.frame}</button>
        </div>
        {b.keyframes.map((k) => (
          <div key={k.frame_index} className="mt-1.5 flex items-center gap-1.5 font-mono text-[11px]">
            <button type="button" className="chip" onClick={() => s.setFrame(uiFrameOf(k))}>f{uiFrameOf(k)}</button>
            <span className="min-w-0 flex-1 truncate text-[var(--fg-dim)]">{Object.entries(k.weights || {}).map(([id, w]) => `${sourceNameAt(s, ids.indexOf(id)) || id.slice(0, 6)} ${Math.round(w)}%`).join(' · ')}</span>
            <select className="select !h-6 !py-0 !pl-1.5 !pr-6 text-[11px]" style={{ width: 'auto' }} value={k.curve} aria-label="Curve" onChange={(e) => set({ keyframes: updateKeyframe(b.keyframes, k.frame_index, { curve: e.target.value }) })}>
              {CURVES.map((c) => <option key={c} value={c}>{CURVE_LABELS[c]}</option>)}
            </select>
            <button type="button" className="btn btn-ghost btn-sm btn-icon !h-6 !w-6" aria-label="Remove keyframe" onClick={() => set({ keyframes: removeKeyframe(b.keyframes, k.frame_index) })}><X size={12} /></button>
          </div>
        ))}
      </div>
      {s.blendDiag && <div className={`font-mono text-[11px] ${s.blendDiag.error || s.blendDiag.warning ? 'text-[var(--warn)]' : 'text-[var(--fg-faint)]'}`}>{s.blendDiag.error || s.blendDiag.warning || s.blendDiag.message || (s.blendDiag.cosine_to_each ? `similarity ${s.blendDiag.cosine_to_each.map((c) => c.toFixed(2)).join(' / ')}` : 'blend accepted')}</div>}
    </div>
  );
}

function ScenesTab() {
  const s = useStudio();
  const t = currentTarget(s);
  const total = Math.max(1, t?.frames || 1);
  const list = clipFxOf(s).overrides;
  const here = overrideAt(list, s.frame);
  const up = (fn) => s.updateClipFx((fx) => ({ overrides: fn(fx.overrides) }));
  return (
    <div className="space-y-2" data-testid="tool-scenes">
      <p className="text-[12px] text-[var(--fg-dim)]">Different settings for a stretch of the clip. Set In/Out on the timeline, then add. The render is cut into one queued piece per change.</p>
      <button type="button" className="btn btn-sm w-full" onClick={() => up((l) => addOverride(l, { start: inOutOf(t).a, end: inOutOf(t).b }, total).list)}><Plus size={13} /> Override In–Out ({inOutOf(t).a}–{inOutOf(t).b})</button>
      {list.map((o) => (
        <div key={o.id} className={`tile p-2 ${here?.id === o.id ? 'selected' : ''}`}>
          <div className="flex items-center gap-2">
            <input className="input !h-7 flex-1 text-[12px]" value={o.label} aria-label="Override name" onChange={(e) => up((l) => updateOverride(l, o.id, { label: e.target.value }, total))} />
            <button type="button" className="chip" onClick={() => s.setFrame(o.start)}>{o.start}–{o.end}</button>
            <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Delete override" onClick={() => up((l) => removeOverride(l, o.id))}><Trash2 size={13} /></button>
          </div>
          <Switch label="Bypass swap" hint="Leave this stretch untouched" checked={o.bypass} onChange={(v) => up((l) => updateOverride(l, o.id, { bypass: v }, total))} />
          {!o.bypass && OVERRIDE_FIELDS.map((f) => {
            const on = o.params?.[f.id] !== undefined;
            const setParam = (v) => up((l) => updateOverride(l, o.id, { params: { ...o.params, [f.id]: v } }, total));
            const clear = () => { const p = { ...o.params }; delete p[f.id]; up((l) => updateOverride(l, o.id, { params: p }, total)); };
            return (
              <div key={f.id} className="mt-1">
                {!on ? <button type="button" className="chip" onClick={() => setParam(f.kind === 'model' ? (s.settings?.swap_model || 'realswap') : f.base)}><Plus size={11} /> {f.label}</button> : (
                  <div className="flex items-end gap-1.5">
                    <div className="flex-1">
                      {f.kind === 'model'
                        ? <Select label={f.label} value={o.params[f.id]} options={swapperOptions(s.meta?.swap_models, o.params[f.id])} onChange={setParam} />
                        : <Slider label={f.label} value={o.params[f.id]} min={f.min} max={f.max} step={f.step} onChange={setParam} />}
                    </div>
                    <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label={`Remove ${f.label}`} onClick={clear}><X size={12} /></button>
                  </div>
                )}
              </div>
            );
          })}
          <div className="mt-1 font-mono text-[10.5px] text-[var(--fg-faint)]">{describeOverride(o)}</div>
        </div>
      ))}
    </div>
  );
}

function CurvePlot({ list, param, total, frame }) {
  const W = 300; const H = 70;
  const pts = useMemo(() => {
    if (!list.length) return '';
    const n = 80;
    return Array.from({ length: n + 1 }, (_, i) => {
      const f = 1 + (i / n) * (total - 1);
      const v = clampValue(param, evaluate(list, f));
      return `${(i / n) * W},${H - ((v - param.min) / (param.max - param.min || 1)) * H}`;
    }).join(' ');
  }, [list, param, total]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-[70px] w-full rounded-lg border border-[var(--rule)] bg-black/40" preserveAspectRatio="none" aria-hidden="true">
      <polyline points={pts} fill="none" stroke={param.color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
      {list.map((k) => <circle key={k.frame} cx={((k.frame - 1) / Math.max(1, total - 1)) * W} cy={H - ((k.value - param.min) / (param.max - param.min || 1)) * H} r="3.5" fill={param.color} />)}
      <line x1={((frame - 1) / Math.max(1, total - 1)) * W} x2={((frame - 1) / Math.max(1, total - 1)) * W} y1="0" y2={H} stroke="#fff" strokeOpacity="0.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function CurvesTab() {
  const s = useStudio();
  const total = Math.max(1, currentTarget(s)?.frames || 1);
  const [pid, setPid] = useState(DIRECT_PARAMS[0].id);
  const param = DIRECT_PARAMS.find((p) => p.id === pid);
  const tracks = clipFxOf(s).curves;
  const list = tracks[pid] || [];
  const setList = (next) => s.updateClipFx((fx) => ({ curves: { ...fx.curves, [pid]: next } }));
  const warn = curveIneffective(param, s.settings || {});
  return (
    <div className="space-y-2" data-testid="tool-curves">
      <div className="flex flex-wrap gap-1">
        {DIRECT_PARAMS.map((p) => <button key={p.id} type="button" className={`chip ${pid === p.id ? 'on' : ''}`} onClick={() => setPid(p.id)}><span className="h-2 w-2 rounded-full" style={{ background: p.color }} />{p.label}{(tracks[p.id] || []).length ? <b>{tracks[p.id].length}</b> : null}</button>)}
      </div>
      {warn && <div className="text-[11.5px] text-[var(--warn)]">{typeof warn === 'string' ? warn : `Needs ${param.needs?.label} on to do anything.`}</div>}
      <CurvePlot list={list} param={param} total={total} frame={s.frame} />
      <button type="button" className="btn btn-sm w-full" onClick={() => { const r = addKey(list, param, s.frame, s.settings?.[param.key] ?? param.base, total); if (r.created) setList(r.list); }}><Plus size={12} /> Key {param.label} at f{s.frame}</button>
      {list.map((k) => (
        <div key={k.frame} className="flex items-center gap-1.5">
          <button type="button" className="chip" onClick={() => s.setFrame(k.frame)}>f{k.frame}</button>
          <div className="flex-1"><Slider label={param.label} value={k.value} min={param.min} max={param.max} step={param.quantum / 5} format={(v) => `${v.toFixed(param.digits)}${param.unit}`} onChange={(v) => setList(setValue(list, param, k.frame, v))} /></div>
          <select className="select !h-6 !py-0 !pl-1.5 !pr-6 text-[11px]" style={{ width: 'auto' }} value={k.interp} aria-label="Interpolation" onChange={(e) => setList(setInterp(list, k.frame, e.target.value))}>
            {INTERPS.map((i) => <option key={i} value={i}>{INTERP_LABELS[i]}</option>)}
          </select>
          <button type="button" className="btn btn-ghost btn-sm btn-icon !h-6 !w-6" aria-label="Remove key" onClick={() => setList(removeKey(list, k.frame))}><X size={12} /></button>
        </div>
      ))}
    </div>
  );
}

function ReferenceTab() {
  const ref = useStudio((s) => s.reference);
  const [files, setFiles] = useState(null);
  useEffect(() => { getJSON('/api/output').then((r) => setFiles(r.files || [])).catch(() => setFiles([])); }, []);
  return (
    <div className="space-y-2" data-testid="tool-reference">
      <p className="text-[12px] text-[var(--fg-dim)]">Pick a finished render to compare against: the viewport's <b>Ref</b> mode wipes between it and the live preview at the same moment.</p>
      {ref && <div className="flex items-center gap-2 font-mono text-[11.5px]"><span className="min-w-0 flex-1 truncate text-[var(--accent-2)]">{ref.name}</span><button type="button" className="btn btn-ghost btn-sm" onClick={() => useStudio.getState().setReference(null)}>Clear</button></div>}
      <div className="max-h-[200px] space-y-1 overflow-y-auto">
        {(files || []).map((f) => (
          <button key={f.name} type="button" className={`tile block w-full truncate px-2 py-1.5 text-left font-mono text-[11.5px] ${ref?.name === f.name ? 'selected' : ''}`}
            onClick={() => { useStudio.getState().setReference({ url: f.url, name: f.name, kind: f.kind }); useStudio.getState().setPreviewMode('reference'); }}>{f.name}</button>
        ))}
        {files && !files.length && <div className="text-[12px] text-[var(--fg-faint)]">No outputs yet.</div>}
      </div>
    </div>
  );
}

export default function ToolsPanel() {
  const [tab, setTab] = useState('mask');
  // Not a store selector: renderPlan returns a fresh object, which a selector would re-read forever.
  const s = useStudio();
  const plan = useMemo(() => (s.targets.length ? renderPlan(s) : null),
    [s.targets, s.selectedTarget, s.clipFx, s.settings, s.targetMediaId]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { useStudio.getState().setTool(TOOL_OF_TAB[tab] || 'pick'); }, [tab]);
  useEffect(() => () => useStudio.getState().setTool('pick'), []);
  const Body = { mask: MaskTab, roto: RotoTab, pins: PinsTab, blend: BlendTab, scenes: ScenesTab, curves: CurvesTab, reference: ReferenceTab }[tab];
  return (
    <Panel title="Tools" accent={plan?.has ? `renders in ${plan.pieces.length} pieces` : undefined} data-testid="tools">
      <div className="mb-3 flex flex-wrap gap-1" role="tablist" aria-label="Tools">
        {TABS.map((t) => (
          <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} className={`chip !h-7 ${tab === t.id ? 'on' : ''}`} onClick={() => setTab(t.id)} data-testid={`tool-tab-${t.id}`}>
            <t.icon size={12} /> {t.label}
          </button>
        ))}
      </div>
      <Body />
      {plan?.has && <div className="mt-3 flex items-center gap-1.5 text-[11.5px] text-[var(--accent-2)]"><Layers size={12} /> Start renders {plan.pieces.length} queued piece{plan.pieces.length === 1 ? '' : 's'} ({plan.lo}–{plan.hi}).</div>}
    </Panel>
  );
}
