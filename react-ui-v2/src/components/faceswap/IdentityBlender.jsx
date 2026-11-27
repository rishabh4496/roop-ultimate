import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Section, Slider, Toggle, Button } from '../ui';
import { useApi } from '../../useApi';
import {
  BLEND_MODES, BLEND_MODE_LABELS, EMPTY_BLEND, MAX_BLEND_SOURCES, PRIVACY_COSINE_DEFAULT, normalizeBlend,
} from './identityBlend';
import { frameIndexOf } from './identityKeyframes';
import { logFailure } from '../../failureLog';

// Identity Blender: latent blend of up to five source identities on the ArcFace
// hypersphere (Slerp / spherical mean, or the older normalised sum), attribute offsets along fitted directions,
// optional de-identification against reference sources, and a readout of how far apart the sources sit.
// Identity KEYFRAMES (the blend as a function of the frame) are edited on the timeline's Identity Keyframe Track;
// they ride in this same recipe. The maths and its caveats live in app/roop/identity_algebra.py; this panel only
// edits the recipe. The recipe travels inside every preview/swap payload (so the preview
// cache keys on it and a queued job freezes it) and is also POSTed here for the
// diagnostics readout (share of each source, identity-guard clamping).

const MAX_SLOTS = MAX_BLEND_SOURCES;
const MODE_HINT = {
  slerp: 'A share is the fraction of the great-circle arc between the identities (exact Slerp for two, the spherical mean for 3-5).',
  linear: 'The normalised weighted sum of the vectors: the same path as Slerp for two sources, but not constant speed along it.',
};

// Only the dials identity_render_validation.json proves writable through the swapper (BLEND_DIALS in
// identityBlend.js). Age, Femininity <-> Masculinity and Expression were rendered at t = ±0.4 / ±0.75 on 65
// paired faces and moved the swapped face non-monotonically or not at all (age: both signs read OLDER; sex: no
// signed response; expression: null, the swapper takes it from the target), so they were removed here rather
// than shipped as sliders that look live. The backend still refuses them (identity_algebra.dial_steps).
// `jawline` is the "Feature Dominance" direction (68-landmark jaw squareness): ±1 on the dial is a 0.75 step
// along the direction (`step_per_dial`), and the identity guard below (default cosine 0.80) scales it back.
const DIALS = [
  { key: 'jawline', label: 'Feature Dominance (Jaw)', min: -1, max: 1, step: 0.05,
    info: 'Facial-structure weight: the direction that tracks jaw squareness. ±1 is a 0.75 step along it; the identity guard below pulls the result back toward the blended identity. Measured on hyperswap: monotone but weak (jaw ratio −0.011…+0.009 over ±0.75, for −0.11…−0.15 identity cosine).' },
];

const pct = (v) => `${Math.round((v || 0) * 100)}%`;

function scoreBadge(entry) {
  if (!entry || entry.score == null || Number.isNaN(entry.score)) return null;
  const s = Number(entry.score);
  // Scale-free held-out scores: Pearson r for continuous labels, AUC for sex.
  const weak = entry.metric === 'auc' ? s < 0.65 : s < 0.2;
  return { text: `${entry.metric === 'auc' ? 'AUC' : 'r'} ${s.toFixed(2)}`, weak };
}

export default function IdentityBlender({
  sourceFaces = [], sourceFacesInfo = [], value, onChange, onUploadSource, notify, playheadFrame = 1,
}) {
  const { postJSON } = useApi();
  const recipe = normalizeBlend(value || EMPTY_BLEND);
  const [diag, setDiag] = useState(null);
  const [exportName, setExportName] = useState('Blended identity');
  const [exporting, setExporting] = useState(false);
  const [busySlot, setBusySlot] = useState(-1);
  const [pickSlot, setPickSlot] = useState(-1);
  const fileRef = useRef(null);
  const pendingSlot = useRef(-1);

  const infoById = useMemo(() => {
    const m = new Map();
    sourceFacesInfo.forEach((info, index) => { if (info?.id) m.set(info.id, { ...info, index }); });
    return m;
  }, [sourceFacesInfo]);

  const update = (patch) => onChange(normalizeBlend({ ...recipe, ...patch }));
  const setComponents = (components) => update({ components: components.slice(0, MAX_SLOTS) });
  const setDial = (key, v) => update({ dials: { ...recipe.dials, [key]: v } });
  const setDeid = (patch) => update({ deidentify: { ...recipe.deidentify, ...patch } });
  const toggleReference = (id) => {
    const refs = recipe.deidentify.references;
    setDeid({ references: refs.includes(id) ? refs.filter((r) => r !== id) : [...refs, id].slice(0, MAX_BLEND_SOURCES) });
  };

  // Diagnostics: debounced, and re-read whenever the recipe or gallery changes.
  const sig = JSON.stringify(recipe) + '|' + sourceFacesInfo.map((i) => i?.id).join(',');
  useEffect(() => {
    const t = setTimeout(() => {
      postJSON('/api/identity/blend', recipe, { abortOnUnmount: true }).then(setDiag).catch(logFailure('Updating the identity blend', { toast: false }));
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
  const nameOf = (id) => infoById.get(id)?.name || 'Not in gallery';
  const pair = diag?.diagnostics?.pairwise_cosine;
  const pairIds = diag?.diagnostics?.pairwise_ids || [];
  const deidDiag = diag?.diagnostics?.deidentify;
  const kfDiag = diag?.diagnostics?.keyframes?.smoothness;
  // A keyframed preset is exported at the playhead's frame (the identity the preview is showing right now).
  const keyed = recipe.keyframes.length > 0 && components.length >= 2;

  const exportFsz = async () => {
    setExporting(true);
    try {
      const res = await postJSON('/api/identity/export_fsz', {
        recipe: { ...recipe, enabled: true }, frame_index: keyed ? frameIndexOf(playheadFrame) : null, name: exportName,
      });
      notify?.(`Saved "${res.saved}.fsz" to the faceset library${keyed ? ` (the identity at frame ${playheadFrame})` : ''}`, 'success');
    } catch (err) {
      notify?.(err?.message || 'Could not export the blend', 'error');
    } finally { setExporting(false); }
  };
  const clamped = diag?.diagnostics?.clamped;
  const applied = diag?.diagnostics?.applied || {};

  return (
    <Section title="Identity Blender" icon={null} collapsible defaultOpen={recipe.enabled}>
      <div className="space-y-3" data-testid="identity-blender">
        <Toggle label="Blend identities & shift attributes" checked={!!recipe.enabled}
          info="Replaces the swap identity with normalize(Σ wᵢ·zᵢ) over the sources below, then offsets it along fitted attribute directions. Applies to faces assigned to ANY blended source; with one source, the dials edit whichever source a face was assigned."
          onChange={(v) => update({ enabled: v })} />

        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Blend mode">
          <span className="text-mini text-white/60">Mode</span>
          {BLEND_MODES.map((m) => (
            <button key={m} type="button" aria-pressed={recipe.mode === m} data-testid={`blend-mode-${m}`}
              onClick={() => update({ mode: m })} title={MODE_HINT[m]}
              className={`rounded-md border px-2 py-0.5 text-mini font-semibold transition-colors ${
                recipe.mode === m ? 'border-[var(--accent)] bg-[var(--accent)]/15 text-white' : 'border-white/10 text-white/50 hover:border-white/25'}`}>
              {BLEND_MODE_LABELS[m]}
            </button>
          ))}
        </div>

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
                    <div className="text-nano tabular-nums text-muted">
                      share {pct(shareOf(comp))}{d?.cosine_to_blend != null ? ` · cos ${d.cosine_to_blend.toFixed(2)}` : ''}
                    </div>
                  </div>
                  <button type="button" aria-label="Remove from blend" className="text-muted hover:text-white"
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
        <input ref={fileRef} type="file" accept="image/*" className="sr-only" aria-label="Choose a photo for this identity slot"
          onChange={(e) => { uploadInto(pendingSlot.current, Array.from(e.target.files || [])); e.target.value = ''; }} />

        {components.length === 1 && (
          <p className="text-mini text-muted">One source: the dials below edit it; add a second to blend.</p>
        )}
        {diag?.diagnostics?.missing?.length > 0 && (
          <p className="text-mini text-amber-300/80">{diag.diagnostics.missing.length} blended source(s) are no longer in the gallery and are skipped.</p>
        )}

        {pair && pair.length >= 2 && (
          <div data-testid="identity-pairwise" className="space-y-1">
            <div className="text-nano font-semibold uppercase tracking-[0.14em] text-muted">Pairwise cosine</div>
            <table className="w-full border-separate border-spacing-0.5 text-center text-nano tabular-nums">
              <thead>
                <tr>
                  <th aria-hidden="true" />
                  {pairIds.map((id, j) => (
                    <th key={id} scope="col" title={nameOf(id)} className="font-semibold text-white/60">{j + 1}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pair.map((row, i) => (
                  <tr key={pairIds[i] || i}>
                    <th scope="row" title={nameOf(pairIds[i])} className="font-semibold text-white/60">{i + 1}</th>
                    {row.map((c, j) => (
                      <td key={j} title={`${nameOf(pairIds[i])} vs ${nameOf(pairIds[j])}`}
                        className={`rounded ${i === j ? 'text-white/25' : 'bg-white/[0.04] text-white/80'}`}
                        style={i === j ? undefined : { boxShadow: `inset 0 -2px 0 rgba(56,189,248,${Math.max(0, Math.min(1, c)).toFixed(2)})` }}>
                        {i === j ? '·' : c.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="m-0 text-nano text-muted">1 = the same identity, about 0 = strangers. Numbers follow the slot order above.</p>
          </div>
        )}

        <div className="space-y-2 border-t border-white/5 pt-2">
          {DIALS.map((dial) => {
            const entry = dirInfo.dials?.[dial.key];
            const badge = scoreBadge(entry);
            const missing = dirInfo.available && !available.has(dial.key);
            const eff = applied[dial.key];
            return (
              <div key={dial.key} className={missing ? 'pointer-events-none opacity-40' : ''} aria-disabled={missing || undefined}>
                <Slider label={dial.label} info={dial.info}
                  value={Number(recipe.dials?.[dial.key]) || 0} min={dial.min} max={dial.max} step={dial.step}
                  onChange={(v) => setDial(dial.key, v)}
                  modified={Math.abs(Number(recipe.dials?.[dial.key]) || 0) > 1e-9}
                  onReset={() => setDial(dial.key, 0)} />
                <div className="-mt-1 flex justify-between text-nano text-muted">
                  <span title={entry?.render_verdict || ''}>{missing ? (entry?.render_verdict ? 'render-tested: no controllable effect on the swapped face' : 'no fitted direction') : badge ? <span className={badge.weak ? 'text-amber-300/80' : ''}>fit {badge.text}{badge.weak ? ' · weak' : ''}</span> : ''}</span>
                  {clamped && eff != null && <span className="text-amber-300/80">guard → {eff.toFixed(2)}</span>}
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
            <div className="text-nano tabular-nums text-muted">
              result vs blended identity: cos {diag.diagnostics.cosine_to_anchor.toFixed(3)}{clamped ? ' (clamped by guard)' : ''}
            </div>
          )}
          {dirInfo.reason && <p className="text-mini text-amber-300/80">{dirInfo.reason}</p>}
        </div>

        <div className="space-y-2 border-t border-white/5 pt-2" data-testid="identity-deid">
          <Toggle label="De-identify against reference sources" checked={recipe.deidentify.enabled}
            info="Moves the final identity along the great circle toward the point orthogonal to the references, only as far as needed for its cosine to each of them to be at or below the threshold. The bound holds on the vector the swapper receives; what the swapped face does with it is measured, not guaranteed."
            onChange={(v) => setDeid({ enabled: v })} />
          {recipe.deidentify.enabled && (
            <>
              <div className="flex flex-wrap gap-1.5" role="group" aria-label="De-identification references">
                {sourceFacesInfo.length === 0 && <span className="text-mini text-muted">No sources in the gallery.</span>}
                {sourceFacesInfo.map((i, idx) => {
                  const on = recipe.deidentify.references.includes(i?.id);
                  return (
                    <button key={i?.id || idx} type="button" title={`${i?.name || 'Source'} as a reference`}
                      aria-pressed={on} data-testid={`deid-ref-${idx}`} onClick={() => toggleReference(i?.id)}
                      className={`h-10 w-10 overflow-hidden rounded-md border ${on ? 'border-[var(--accent)] ring-1 ring-[var(--accent)]' : 'border-white/20 opacity-70 hover:opacity-100'}`}>
                      {sourceFaces[idx] && <img src={sourceFaces[idx]} alt={i?.name || ''} className="h-full w-full object-cover" />}
                    </button>
                  );
                })}
              </div>
              <Slider label="Privacy threshold (max cosine to references)" value={recipe.deidentify.max_cosine}
                min={0} max={0.95} step={0.01}
                info="The identity's cosine to every reference ends at or below this. About 0.0-0.2 is how far strangers sit from each other; the same person is well above 0.3."
                onChange={(v) => setDeid({ max_cosine: v })}
                modified={Math.abs(recipe.deidentify.max_cosine - PRIVACY_COSINE_DEFAULT) > 1e-9}
                onReset={() => setDeid({ max_cosine: PRIVACY_COSINE_DEFAULT })} />
              {recipe.deidentify.references.length === 0 && (
                <p className="m-0 text-mini text-muted">Pick at least one reference above.</p>
              )}
              {deidDiag && (
                <div className="text-nano tabular-nums text-muted" data-testid="deid-readout">
                  cosine to references {deidDiag.cosine_before.map((c) => c.toFixed(2)).join(' / ')} → {deidDiag.cosine_after.map((c) => c.toFixed(2)).join(' / ')}
                  {deidDiag.applied ? ` · ${deidDiag.cosine_to_primary.toFixed(2)} cosine to the un-edited identity` : ''}
                  {deidDiag.reason === 'already_below' ? ' · already below the threshold, nothing moved' : ''}
                  {deidDiag.reason === 'degenerate' ? ' · the identity lies inside the references\' span: it cannot be separated from them' : ''}
                  {deidDiag.reason === 'floor' ? ' · stopped at the closeness floor, threshold not met' : ''}
                </div>
              )}
            </>
          )}
        </div>

        {recipe.keyframes.length > 0 && (
          <p className="m-0 text-mini text-muted" data-testid="identity-keyframe-summary">
            {recipe.keyframes.length} identity keyframe{recipe.keyframes.length === 1 ? '' : 's'} on this clip, edited on the timeline.
            {kfDiag ? ` Largest step ${((kfDiag.max_step * 180) / Math.PI).toFixed(2)}° per frame.` : ''}
            {components.length < 2 ? ' Add a second source: with one, the keyframes do nothing.' : ''}
          </p>
        )}

        <div className="flex flex-wrap items-end justify-between gap-2 border-t border-white/5 pt-2">
          <div className="flex items-end gap-2">
            <label className="block">
              <span className="text-nano font-semibold uppercase tracking-wider text-muted">Preset name</span>
              <input value={exportName} onChange={(e) => setExportName(e.target.value)} maxLength={80} aria-label="Exported preset name"
                className="mt-1 block w-40 rounded-lg px-2 py-1 glass-input text-white text-mini focus:outline-none" />
            </label>
            <Button size="sm" variant="secondary" onClick={exportFsz}
              disabled={exporting || components.length === 0 || !exportName.trim()} data-testid="identity-export-fsz"
              title={keyed ? `Saves the composite identity at frame ${playheadFrame} as a .fsz preset` : 'Saves the composite identity as a .fsz preset in the faceset library'}>
              {exporting ? 'Saving…' : keyed ? `Export identity at frame ${playheadFrame}` : 'Export as .fsz'}
            </Button>
          </div>
          <Button size="sm" variant="secondary"
            onClick={() => onChange(normalizeBlend({ ...EMPTY_BLEND, keyframes: recipe.keyframes }))}>Reset blender</Button>
        </div>
      </div>
    </Section>
  );
}
