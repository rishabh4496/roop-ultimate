import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Section, Slider, Toggle, Button } from '../ui';
import { postJSON } from '../../api';
import { EMPTY_BLEND } from './identityBlend';

// Identity Blender: latent blend of up to four source identities on the ArcFace
// hypersphere, plus attribute offsets along fitted directions. The maths and its
// caveats live in app/roop/identity_algebra.py; this panel only edits the
// recipe. The recipe travels inside every preview/swap payload (so the preview
// cache keys on it and a queued job freezes it) and is also POSTed here for the
// diagnostics readout (share of each source, identity-guard clamping).

const MAX_SLOTS = 4;

// `jawline` is the "Feature Weight" direction (68-landmark jaw squareness);
// `gender` is the Femininity <-> Masculinity axis the brief asks for.
const DIALS = [
  { key: 'age', label: 'Age Shift', min: -30, max: 30, step: 1, unit: 'yr',
    info: 'Moves the identity along the direction genderage reads as older (+) or younger (-), calibrated in years on the fitting corpus. Bounded to ±30 years and by the identity guard.' },
  { key: 'gender', label: 'Femininity ↔ Masculinity', min: -1, max: 1, step: 0.05,
    info: '±1 = two standard deviations of how real faces vary along the fitted sex direction.' },
  { key: 'jawline', label: 'Feature Dominance', min: -1, max: 1, step: 0.05,
    info: 'Facial-structure weight: the direction that tracks jaw squareness in the fitting corpus. ±1 = two standard deviations.' },
  { key: 'expression', label: 'Expression Intensity', min: -1, max: 1, step: 0.05,
    info: 'ArcFace is trained to IGNORE expression, and the swapper takes expression from the target, so this direction is weak by construction. Check the fit score beside it.' },
];

const pct = (v) => `${Math.round((v || 0) * 100)}%`;

function scoreBadge(entry) {
  if (!entry || entry.score == null || Number.isNaN(entry.score)) return null;
  const s = Number(entry.score);
  // Scale-free held-out scores: Pearson r for continuous labels, AUC for sex.
  const weak = entry.metric === 'auc' ? s < 0.65 : s < 0.2;
  return { text: `${entry.metric === 'auc' ? 'AUC' : 'r'} ${s.toFixed(2)}`, weak };
}

export default function IdentityBlender({ sourceFaces = [], sourceFacesInfo = [], value, onChange, onUploadSource, notify }) {
  const recipe = value || EMPTY_BLEND;
  const [diag, setDiag] = useState(null);
  const [busySlot, setBusySlot] = useState(-1);
  const [pickSlot, setPickSlot] = useState(-1);
  const fileRef = useRef(null);
  const pendingSlot = useRef(-1);

  const infoById = useMemo(() => {
    const m = new Map();
    sourceFacesInfo.forEach((info, index) => { if (info?.id) m.set(info.id, { ...info, index }); });
    return m;
  }, [sourceFacesInfo]);

  const update = (patch) => onChange({ ...recipe, ...patch });
  const setComponents = (components) => update({ components: components.slice(0, MAX_SLOTS) });
  const setDial = (key, v) => update({ dials: { ...recipe.dials, [key]: v } });

  // Diagnostics: debounced, and re-read whenever the recipe or gallery changes.
  const sig = JSON.stringify(recipe) + '|' + sourceFacesInfo.map((i) => i?.id).join(',');
  useEffect(() => {
    const t = setTimeout(() => {
      postJSON('/api/identity/blend', recipe).then(setDiag).catch(() => {});
    }, 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

  const components = recipe.components || [];
  const total = components.reduce((a, c) => a + (Number(c.weight) || 0), 0);
  const shareOf = (c) => (total > 0 ? (Number(c.weight) || 0) / total : 1 / Math.max(1, components.length));
  const diagById = new Map((diag?.diagnostics?.components || []).map((c) => [c.source_id, c]));
  const dirInfo = diag?.directions || {};
  const available = new Set(dirInfo.available || []);

  const bindSlot = (slot, sourceId) => {
    if (!sourceId) return;
    const next = components.filter((c) => c.source_id !== sourceId);
    const entry = { source_id: sourceId, weight: slot < components.length ? components[slot].weight : (next.length ? 50 : 100) };
    if (slot < next.length) next[slot] = entry; else next.push(entry);
    setComponents(next);
    setPickSlot(-1);
  };

  const uploadInto = async (slot, files) => {
    if (!files?.length || !onUploadSource) return;
    const before = new Set(sourceFacesInfo.map((i) => i?.id));
    setBusySlot(slot);
    try {
      const res = await onUploadSource([files[0]]);
      const added = (res?.source_faces_info || []).filter((i) => i?.id && !before.has(i.id));
      if (added.length) bindSlot(slot, added[0].id);
      else notify?.('No new face was added from that photo', 'error');
    } finally { setBusySlot(-1); }
  };

  const onDrop = (slot) => (e) => {
    e.preventDefault();
    e.nativeEvent.roopConsumed = true;
    uploadInto(slot, Array.from(e.dataTransfer.files || []));
  };

  const slots = Array.from({ length: MAX_SLOTS }, (_, i) => components[i] || null);
  const clamped = diag?.diagnostics?.clamped;
  const applied = diag?.diagnostics?.applied || {};

  return (
    <Section title="Identity Blender" icon={null} collapsible defaultOpen={recipe.enabled}>
      <div className="space-y-3">
        <Toggle label="Blend identities & shift attributes" checked={!!recipe.enabled}
          info="Replaces the swap identity with normalize(Σ wᵢ·zᵢ) over the sources below, then offsets it along fitted attribute directions. Applies to faces assigned to ANY blended source; with one source, the dials edit whichever source a face was assigned."
          onChange={(v) => update({ enabled: v })} />

        <div className="grid grid-cols-2 gap-2">
          {slots.map((comp, slot) => {
            const info = comp ? infoById.get(comp.source_id) : null;
            const thumb = info ? sourceFaces[info.index] : null;
            const d = comp ? diagById.get(comp.source_id) : null;
            if (!comp) {
              const disabled = slot > components.length;
              return (
                <div key={slot}
                  onDragOver={(e) => { e.preventDefault(); }}
                  onDrop={disabled ? undefined : onDrop(slot)}
                  className={`relative flex min-h-24 flex-col items-center justify-center gap-1.5 rounded-xl border border-dashed p-2 text-center text-mini ${disabled ? 'border-white/5 text-white/20' : 'border-white/15 text-white/50 hover:border-[var(--accent)]'}`}>
                  {busySlot === slot ? <span>Detecting face…</span> : (
                    <>
                      <span>Drop a photo</span>
                      {!disabled && (
                        <div className="flex gap-1.5">
                          <button type="button" className="rounded-md bg-white/10 px-2 py-0.5 hover:bg-white/20"
                            onClick={() => { pendingSlot.current = slot; fileRef.current?.click(); }}>Browse</button>
                          {sourceFaces.length > 0 && (
                            <button type="button" className="rounded-md bg-white/10 px-2 py-0.5 hover:bg-white/20"
                              onClick={() => setPickSlot(pickSlot === slot ? -1 : slot)}>From gallery</button>
                          )}
                        </div>
                      )}
                    </>
                  )}
                  {pickSlot === slot && (
                    <div className="absolute inset-x-1 top-full z-20 mt-1 flex flex-wrap gap-1.5 rounded-lg border border-white/15 bg-black/90 p-1.5">
                      {sourceFacesInfo.map((i, idx) => (
                        <button key={i?.id || idx} type="button" title={i?.name}
                          disabled={components.some((c) => c.source_id === i?.id)}
                          onClick={() => bindSlot(slot, i?.id)}
                          className="h-10 w-10 overflow-hidden rounded-md border border-white/20 hover:border-[var(--accent)] disabled:opacity-30">
                          {sourceFaces[idx] && <img src={sourceFaces[idx]} alt={i?.name || ''} className="h-full w-full object-cover" />}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            }
            return (
              <div key={comp.source_id} className="space-y-1.5 rounded-xl border border-white/10 bg-black/35 p-2">
                <div className="flex items-center gap-2">
                  <div className="h-10 w-10 shrink-0 overflow-hidden rounded-lg border border-white/15 bg-white/5">
                    {thumb ? <img src={thumb} alt="" className="h-full w-full object-cover" /> : <span className="text-nano text-rose-300">missing</span>}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-mini text-white/75" title={info?.name || comp.source_id}>{info?.name || 'Not in gallery'}</div>
                    <div className="text-nano tabular-nums text-white/45">
                      share {pct(shareOf(comp))}{d?.cosine_to_blend != null ? ` · cos ${d.cosine_to_blend.toFixed(2)}` : ''}
                    </div>
                  </div>
                  <button type="button" aria-label="Remove from blend" className="text-white/35 hover:text-white"
                    onClick={() => setComponents(components.filter((c) => c.source_id !== comp.source_id))}>✕</button>
                </div>
                <input type="range" min={0} max={100} step={1} value={Number(comp.weight) || 0}
                  aria-label={`Weight of ${info?.name || 'source'}`}
                  onChange={(e) => setComponents(components.map((c) => (c.source_id === comp.source_id ? { ...c, weight: Number(e.target.value) } : c)))}
                  className="w-full accent-[var(--accent)] h-1.5" />
              </div>
            );
          })}
        </div>
        <input ref={fileRef} type="file" accept="image/*" className="sr-only"
          onChange={(e) => { uploadInto(pendingSlot.current, Array.from(e.target.files || [])); e.target.value = ''; }} />

        {components.length === 1 && (
          <p className="text-mini text-white/45">One source: the dials below edit it; add a second to blend.</p>
        )}
        {diag?.diagnostics?.missing?.length > 0 && (
          <p className="text-mini text-amber-300/80">{diag.diagnostics.missing.length} blended source(s) are no longer in the gallery and are skipped.</p>
        )}

        <div className="space-y-2 border-t border-white/5 pt-2">
          {DIALS.map((dial) => {
            const entry = dirInfo.dials?.[dial.key];
            const badge = scoreBadge(entry);
            const missing = dirInfo.available && !available.has(dial.key);
            const eff = applied[dial.key];
            return (
              <div key={dial.key} className={missing ? 'pointer-events-none opacity-40' : ''} aria-disabled={missing || undefined}>
                <Slider label={`${dial.label}${dial.unit ? ` (${dial.unit})` : ''}`} info={dial.info}
                  value={Number(recipe.dials?.[dial.key]) || 0} min={dial.min} max={dial.max} step={dial.step}
                  onChange={(v) => setDial(dial.key, v)}
                  modified={Math.abs(Number(recipe.dials?.[dial.key]) || 0) > 1e-9}
                  onReset={() => setDial(dial.key, 0)} />
                <div className="-mt-1 flex justify-between text-nano text-white/40">
                  <span title={entry?.render_verdict || ''}>{missing ? (entry?.render_verdict ? 'render-tested: no controllable effect on the swapped face' : 'no fitted direction') : badge ? <span className={badge.weak ? 'text-amber-300/80' : ''}>fit {badge.text}{badge.weak ? ' · weak' : ''}</span> : ''}</span>
                  {clamped && eff != null && <span className="text-amber-300/80">guard → {dial.key === 'age' ? `${eff.toFixed(1)} yr` : eff.toFixed(2)}</span>}
                </div>
              </div>
            );
          })}
          <Slider label="Identity guard (min cosine)" value={Number(recipe.min_cosine) || 0.8} min={0.5} max={0.99} step={0.01}
            info="The edited vector never drops below this cosine to the blended identity; past it the whole offset is scaled back uniformly (direction kept)."
            onChange={(v) => update({ min_cosine: v })}
            modified={Math.abs((Number(recipe.min_cosine) || 0.8) - 0.8) > 1e-9}
            onReset={() => update({ min_cosine: 0.8 })} />
          {diag?.diagnostics?.cosine_to_anchor != null && (
            <div className="text-nano tabular-nums text-white/45">
              result vs blended identity: cos {diag.diagnostics.cosine_to_anchor.toFixed(3)}{clamped ? ' (clamped by guard)' : ''}
            </div>
          )}
          {dirInfo.reason && <p className="text-mini text-amber-300/80">{dirInfo.reason}</p>}
        </div>

        <div className="flex justify-end">
          <Button size="sm" variant="secondary" onClick={() => onChange({ ...EMPTY_BLEND, dials: { ...EMPTY_BLEND.dials } })}>Reset blender</Button>
        </div>
      </div>
    </Section>
  );
}
