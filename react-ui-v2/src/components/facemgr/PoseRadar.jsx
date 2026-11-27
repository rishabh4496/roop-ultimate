import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { getJSON, postJSON } from '../../api';
import { Button } from '../ui';
import { confirmDialog } from '../confirm';

// Face Bank Quality & Pose Radar.
//
// Left: a five-axis radar of how many faces the set holds in each pose bucket (frontal, left / right
// profile, high / low angle) - a lopsided polygon is a set the swap will struggle with at the angles it
// is thin on. Right: every face as a dot on the yaw x pitch plane, coloured by sharpness, so a blurry
// cluster or a missing quadrant is visible at a glance. Below: the one-click Auto-Prune.
//
// The buckets, the blur rule and the duplicate rule all live on the server (roop/identity_lab.py); this
// draws what it reports and never re-derives a verdict. Auto-Prune is a two-step: the server prices the
// removal first (dry run) and the person confirms the numbers before anything is dropped.

const AXES = ['frontal', 'left_profile', 'right_profile', 'high_angle', 'low_angle'];
const LABEL = { frontal: 'Frontal', left_profile: 'Left profile', right_profile: 'Right profile', high_angle: 'High angle', low_angle: 'Low angle' };

const CX = 110;
const CY = 108;
const R = 74;
const axisPoint = (i, frac) => {
  const a = -Math.PI / 2 + (i * 2 * Math.PI) / AXES.length;
  return [CX + Math.cos(a) * R * frac, CY + Math.sin(a) * R * frac];
};
const pts = (list) => list.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');

// Sharpness 0..1 -> red (soft) .. green (sharp).
const sharpColor = (s) => `hsl(${Math.round(Math.max(0, Math.min(1, s)) * 120)} 70% 52%)`;

const SC_W = 220;
const SC_H = 130;
const scX = (yaw) => ((Math.max(-90, Math.min(90, yaw)) + 90) / 180) * (SC_W - 20) + 10;
const scY = (pitch) => ((Math.max(-45, Math.min(45, pitch)) + 45) / 90) * (SC_H - 20) + 10;

export default function PoseRadar({ refreshKey, selected, onSelect, onPruned, notify, busy = false }) {
  const [radar, setRadar] = useState(null);
  const [pruning, setPruning] = useState(false);

  const load = useCallback(async () => {
    try { setRadar(await getJSON('/api/facemgr/radar')); } catch { setRadar(null); }
  }, []);
  useEffect(() => { load(); }, [load, refreshKey]);

  const counts = useMemo(() => AXES.map((a) => radar?.buckets?.[a]?.count || 0), [radar]);
  const peak = Math.max(1, ...counts);
  const poly = useMemo(() => pts(counts.map((c, i) => axisPoint(i, c / peak))), [counts, peak]);
  const total = radar?.total || 0;

  const autoPrune = async () => {
    setPruning(true);
    try {
      const dry = await postJSON('/api/facemgr/auto_prune', { dry_run: true });
      const plan = dry.plan;
      const blurry = plan.remove.filter((d) => d.reason === 'blurry').length;
      const dup = plan.remove.filter((d) => d.reason === 'duplicate').length;
      const kept = plan.kept_for_coverage.length;
      if (!plan.remove.length) {
        notify?.(kept ? `Nothing to drop (${kept} soft face${kept === 1 ? ' is' : 's are'} the only ${kept === 1 ? 'one' : 'ones'} at ${kept === 1 ? 'its angle' : 'their angles'}, so kept)` : 'No blurry or duplicate faces found', 'info');
        return;
      }
      const ok = await confirmDialog({
        title: `Remove ${plan.remove.length} of ${total} faces?`,
        message: [
          blurry ? `${blurry} blurry (sharpness under ${Math.round(plan.thresholds.blur_below * 100)}%)` : null,
          dup ? `${dup} near-duplicate of a better shot at the same angle` : null,
          kept ? `${kept} soft face${kept === 1 ? '' : 's'} kept because ${kept === 1 ? 'it is' : 'they are'} the only example of that angle` : null,
        ].filter(Boolean).join(' · '),
        confirmLabel: 'Auto-prune',
        danger: true,
      });
      if (!ok) return;
      const res = await postJSON('/api/facemgr/auto_prune', {});
      onPruned?.(res);
      notify?.(`Removed ${res.removed} face${res.removed === 1 ? '' : 's'} (${blurry} blurry, ${dup} duplicate)`);
    } catch (e) { notify?.(e.message, 'error'); }
    finally { setPruning(false); }
  };

  const preview = radar?.prune_preview;
  const pending = preview ? preview.remove.length : 0;

  if (!total) {
    return <div className="text-mini text-muted py-3" data-testid="pose-radar-empty">Add faces to see which angles the set covers.</div>;
  }

  return (
    <div className="space-y-3" data-testid="pose-radar">
      <div className="flex flex-wrap items-start gap-4">
        <figure className="m-0">
          <svg viewBox="0 0 220 216" width="220" height="216" role="img"
               aria-label={`Pose coverage: ${AXES.map((a, i) => `${LABEL[a]} ${counts[i]}`).join(', ')}`}>
            {[1 / 3, 2 / 3, 1].map((f) => (
              <polygon key={f} points={pts(AXES.map((_, i) => axisPoint(i, f)))} fill="none" stroke="currentColor" strokeOpacity={f === 1 ? 0.22 : 0.1} />
            ))}
            {AXES.map((_, i) => {
              const [x, y] = axisPoint(i, 1);
              return <line key={i} x1={CX} y1={CY} x2={x} y2={y} stroke="currentColor" strokeOpacity="0.12" />;
            })}
            <polygon points={poly} fill="var(--accent)" fillOpacity="0.28" stroke="var(--accent)" strokeWidth="1.6" />
            {counts.map((c, i) => {
              const [x, y] = axisPoint(i, c / peak);
              return <circle key={i} cx={x} cy={y} r="3" fill={c ? 'var(--accent)' : 'currentColor'} fillOpacity={c ? 1 : 0.4} />;
            })}
            {AXES.map((a, i) => {
              const [x, y] = axisPoint(i, 1.2);
              const empty = counts[i] === 0;
              return (
                <text key={a} x={x} y={y} textAnchor="middle" dominantBaseline="middle" fontSize="9.5"
                      fill={empty ? '#fbbf24' : 'currentColor'} fillOpacity={empty ? 1 : 0.75}>
                  {LABEL[a]} · {counts[i]}
                </text>
              );
            })}
          </svg>
        </figure>

        <figure className="m-0">
          <svg viewBox={`0 0 ${SC_W} ${SC_H}`} width={SC_W} height={SC_H} role="img"
               aria-label="Every face by yaw and pitch, coloured by sharpness">
            <rect x="10" y="10" width={SC_W - 20} height={SC_H - 20} fill="none" stroke="currentColor" strokeOpacity="0.2" />
            {/* the bucket edges: |yaw| 40 and |pitch| 20 */}
            {[-40, 40].map((yaw) => <line key={yaw} x1={scX(yaw)} x2={scX(yaw)} y1="10" y2={SC_H - 10} stroke="currentColor" strokeOpacity="0.25" strokeDasharray="3 3" />)}
            {[-20, 20].map((p) => <line key={p} x1="10" x2={SC_W - 10} y1={scY(p)} y2={scY(p)} stroke="currentColor" strokeOpacity="0.25" strokeDasharray="3 3" />)}
            {(radar.faces || []).map((f) => (
              <circle key={f.index} cx={scX(f.yaw)} cy={scY(f.pitch)} r={selected === f.index ? 6 : 4.2}
                      fill={sharpColor(f.sharpness)} stroke={selected === f.index ? '#fff' : 'rgba(0,0,0,0.5)'}
                      strokeWidth={selected === f.index ? 2 : 1} style={{ cursor: onSelect ? 'pointer' : 'default' }}
                      onClick={() => onSelect?.(f.index)}>
                <title>{`Face ${f.index + 1}: yaw ${f.yaw}°, pitch ${f.pitch}°, sharpness ${Math.round(f.sharpness * 100)}%`}</title>
              </circle>
            ))}
            <text x="12" y={SC_H - 1} fontSize="8.5" fill="currentColor" fillOpacity="0.55">← left profile</text>
            <text x={SC_W - 12} y={SC_H - 1} fontSize="8.5" textAnchor="end" fill="currentColor" fillOpacity="0.55">right profile →</text>
          </svg>
          <figcaption className="text-nano text-muted mt-0.5">yaw × pitch · <span style={{ color: sharpColor(0.1) }}>soft</span> → <span style={{ color: sharpColor(0.95) }}>sharp</span></figcaption>
        </figure>
      </div>

      {radar.missing?.length > 0 && (
        <p className="text-mini text-amber-300/90 m-0" data-testid="pose-missing">
          No example of: {radar.missing.map((m) => LABEL[m]).join(', ')}. Swaps at those angles fall back to the nearest face you do have.
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="secondary" onClick={autoPrune} disabled={busy || pruning} data-testid="auto-prune"
                title="Drops faces that are blurry for this set, and near-identical repeats of the same angle. A soft face that is the only one at its angle is kept.">
          {pruning ? 'Checking…' : 'Auto-Prune Blurry & Duplicate Outliers'}
        </Button>
        <span className="text-mini text-muted" data-testid="prune-preview">
          {pending ? `would drop ${pending} of ${total}` : 'nothing to drop'}
        </span>
      </div>
    </div>
  );
}
