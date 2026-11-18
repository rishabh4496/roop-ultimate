import React, { useRef, useState } from 'react';
import { Slider } from '../ui';
import { useTimelineTracks } from './timeline/useTimelineData';
import {
  OVERRIDE_PILLS, effectiveValue, formatPill, hasOverride, overrideCount, overridesApply,
  poseBadge, presencePercent, routeLines,
} from './routingModel';

// ── The face routing dock ───────────────────────────────────────────────────────────────────────────────────────────
//
// Detected target faces on the left, the source faces they become on the right, one line per route:
//
//   [1][face] Front   62%  ●────────────╮        ╭──● [face] Source face 1
//   [2][face] Left    41%  ●────────────┴────────╯
//   [3][face] Front   18%  ✕ skip                    [face] Source face 2
//
// Each target row carries its rank, thumbnail, head-pose badge (the pose it was captured at) and how much of the clip the
// person is on screen (the Face Bank scan: "—" until the clip has been scanned, never a made-up 0%). Click a target then a
// source - or drag a source onto a target - to route it; every route goes through the SAME `onMap` the people list below uses,
// so the dock and the list can never disagree about who becomes whom.
//
// The selected person's override pills sit under the board: erode, feather, likeness, LAB strength and enhancer blend FOR THAT
// FACE ONLY (the backend applies them per face: app/roop/person_overrides.py). A pill reads the ordinary setting until it is
// overridden; scrub it to override, click its MOD mark (or right-click the bar) to go back to the ordinary value. They apply
// in "Selected face" / "Selected people" only - the other modes have no per-person identity at the paste - and the dock says so
// instead of letting a value sit there that nothing reads.
//
// Row geometry is fixed (ROW_H + GAP), so the route lines are computed, not measured: no layout effect, nothing to go stale.
//
// KEYBOARD. The target and source rows are mouse affordances (tabindex -1: every extra tab stop on Face Swap counts against
// e2e/tab-stops.spec.js). The keyboard route to the same decisions is the people list below this dock - arrow keys pick the
// person, and each person has a source select. The override pills are ONE tab stop with Left/Right between them.

const ROW_H = 64;
const GAP = 6;
const LINE_W = 40;
const rowY = (i) => i * (ROW_H + GAP) + ROW_H / 2;

function PresenceBadge({ pct }) {
  if (pct === null) return <span className="routing-presence" title="Scan faces on the timeline to see how much of the clip each person is in">—</span>;
  return <span className="routing-presence" title={`On screen in ${pct.toFixed(1)}% of the clip's frames`} data-testid="routing-presence">{Math.round(pct)}%</span>;
}

export default function FaceRoutingDock({
  people, sources, selectedId, onSelect, onMap,
  overrides, onOverride, onClearOverride, onClearPerson,
  settings, swapModeLabel, onUseSelectedPeople,
  targetIndex, targetMediaId, maxFrames, skip = -1,
}) {
  const [activePill, setActivePill] = useState(null);
  const [rove, setRove] = useState(0);
  const pillRefs = useRef([]);
  const [dropOn, setDropOn] = useState(null);
  const { tracks } = useTimelineTracks({ targetIndex, targetKey: targetMediaId, maxFrames: maxFrames || 0 });

  if (!people.length) return null;
  const presenceFor = (id) => presencePercent(tracks.presence.find((r) => r.id === id), maxFrames);
  const selected = people.find((p) => p.id === selectedId) || people[0];
  const rows = people.map((p, i) => ({ id: p.id, y: rowY(i), sourceIndex: p.sourceIndex, color: p.color }));
  const lines = routeLines(rows, sources.map((_, i) => ({ y: rowY(i) })), 0, LINE_W);
  const height = Math.max(people.length, sources.length) * (ROW_H + GAP) - GAP;
  const routedTo = new Map();                       // source index -> the colours of the people routed to it
  people.forEach((p) => { if (p.sourceIndex >= 0) routedTo.set(p.sourceIndex, [...(routedTo.get(p.sourceIndex) || []), p.color]); });
  const applies = overridesApply(swapModeLabel);
  const anyOverride = people.some((p) => hasOverride(overrides, p.id));
  const pill = OVERRIDE_PILLS.find((x) => x.key === activePill) || null;

  return (
    <section aria-label="Face routing" data-testid="face-routing-dock" className="routing-dock">
      <div className="routing-dock__board" style={{ gridTemplateColumns: `minmax(0,1fr) ${LINE_W}px minmax(0,1fr)` }}>
        {/* targets */}
        <ul className="routing-col" aria-label="Target faces" style={{ height }}>
          {people.map((p) => {
            const sel = p.id === selected.id;
            return (
              <li key={p.id} style={{ height: ROW_H }}>
                <button
                  type="button"
                  data-testid={`routing-target-${p.rank}`}
                  aria-pressed={sel}
                  tabIndex={-1}
                  aria-label={`Target ${p.rank}: ${p.name}`}
                  onClick={() => onSelect(p)}
                  onDragOver={sources.length ? (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'link'; setDropOn(p.id); } : undefined}
                  onDragLeave={() => setDropOn((d) => (d === p.id ? null : d))}
                  onDrop={sources.length ? (e) => {
                    e.preventDefault(); setDropOn(null);
                    const raw = e.dataTransfer.getData('text/roop-source');
                    if (raw !== '') onMap(p.id, parseInt(raw, 10));
                  } : undefined}
                  className={`routing-node ${sel ? 'is-selected' : ''} ${dropOn === p.id ? 'is-drop' : ''}`}
                  style={{ '--node': p.color }}
                >
                  <span className="routing-line1">
                    <span className="routing-thumbwrap">
                      <img src={p.thumb} alt="" className="routing-thumb" />
                      <span className="routing-rank" aria-hidden="true">{p.rank}</span>
                    </span>
                    <span className="routing-name" title={p.name}>{p.name}</span>
                    <PresenceBadge pct={presenceFor(p.id)} />
                  </span>
                  <span className="routing-line2">
                    <span className="routing-pose" title={p.pose} data-testid="routing-pose">{poseBadge(p.pose)}</span>
                    {p.sourceIndex === skip && <span className="routing-skip" title="This person is not swapped">skip</span>}
                  </span>
                  {hasOverride(overrides, p.id) && (
                    <span className="routing-ovr-dot" title={`${overrideCount(overrides, p.id)} per-face override${overrideCount(overrides, p.id) === 1 ? '' : 's'}`} data-testid="routing-ovr-dot" />
                  )}
                </button>
              </li>
            );
          })}
        </ul>

        {/* the route lines */}
        <svg className="routing-lines" width={LINE_W} height={height} viewBox={`0 0 ${LINE_W} ${height}`} aria-hidden="true" data-testid="routing-lines">
          {lines.map((l) => {
            const p = people.find((x) => x.id === l.id);
            const sel = p.id === selected.id;
            return (
              <g key={l.id} data-testid={`routing-line-${p.rank}`} data-source={l.sourceIndex}>
                <path d={l.d} fill="none" stroke={p.color} strokeWidth={sel ? 2.4 : 1.4} opacity={sel ? 1 : 0.7}
                  style={sel ? { filter: `drop-shadow(0 0 3px ${p.color})` } : undefined} />
              </g>
            );
          })}
        </svg>

        {/* sources */}
        <ul className="routing-col" aria-label="Source faces" style={{ height }}>
          {sources.map((s) => {
            const colours = routedTo.get(s.index) || [];
            return (
              <li key={s.index} style={{ height: ROW_H }}>
                <button
                  type="button"
                  data-testid={`routing-source-${s.index}`}
                  aria-label={`Route ${selected.name} to ${s.name}`}
                  tabIndex={-1}
                  draggable
                  onDragStart={(e) => { e.dataTransfer.setData('text/roop-source', String(s.index)); e.dataTransfer.effectAllowed = 'link'; }}
                  onClick={() => onMap(selected.id, s.index)}
                  className={`routing-node routing-node--source ${colours.length ? 'is-routed' : ''}`}
                  style={{ '--node': colours[0] || 'transparent' }}
                >
                  <span className="routing-line1">
                    <span className="routing-thumbwrap"><img src={s.thumb} alt="" className="routing-thumb" /></span>
                    <span className="routing-count">{colours.length ? `×${colours.length}` : '—'}</span>
                  </span>
                  <span className="routing-line2"><span className="routing-sourcename" title={s.name}>{s.name}</span></span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      {/* per-face overrides for the selected person */}
      <div className="routing-overrides" data-testid="routing-overrides" role="group" aria-label={`Per-face overrides for ${selected.name}`}>
        <div className="routing-overrides__head">
          <span className="routing-overrides__title">Overrides · {selected.name}</span>
          {hasOverride(overrides, selected.id) && (
            <button type="button" className="routing-link" onClick={() => { onClearPerson(selected.id); setActivePill(null); }}>Reset face</button>
          )}
        </div>
        <div className="routing-pills" role="toolbar" tabIndex={-1} aria-label="Override parameters"
          onKeyDown={(e) => {
            const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
            if (!d) return;
            e.preventDefault();
            const next = (rove + d + OVERRIDE_PILLS.length) % OVERRIDE_PILLS.length;
            setRove(next);
            pillRefs.current[next]?.focus();
          }}>
          {OVERRIDE_PILLS.map((x, pi) => {
            const on = overrides?.[selected.id]?.[x.key] !== undefined;
            return (
              <button
                key={x.key}
                ref={(el) => { pillRefs.current[pi] = el; }}
                type="button"
                tabIndex={pi === rove ? 0 : -1}
                onFocus={() => setRove(pi)}
                data-testid={`routing-pill-${x.key}`}
                data-overridden={on ? 'true' : 'false'}
                aria-pressed={activePill === x.key}
                title={x.title}
                onClick={() => setActivePill((cur) => (cur === x.key ? null : x.key))}
                className={`routing-pill ${on ? 'is-on' : ''}`}
              >
                <span>{x.label}</span>
                <span className="tabular-nums">{formatPill(x, effectiveValue(x, overrides, selected.id, settings))}</span>
              </button>
            );
          })}
        </div>
        {pill && (
          <Slider
            key={`${selected.id}:${pill.key}`}
            label={`${pill.label} · ${selected.name}`}
            info={pill.title}
            min={pill.min} max={pill.max} step={pill.step} percent={!!pill.percent}
            value={effectiveValue(pill, overrides, selected.id, settings)}
            modified={overrides?.[selected.id]?.[pill.key] !== undefined}
            onReset={() => onClearOverride(selected.id, pill.key)}
            onChange={(v) => onOverride(selected.id, pill.key, v)}
            testId={`routing-editor-${pill.key}`}
          />
        )}
        {!applies && (anyOverride || pill) && (
          <div role="status" data-testid="routing-mode-note" className="routing-note">
            <span>Per-face overrides apply in “Selected face” and “Selected people”. The current mode is “{swapModeLabel}”, so these values are saved but not used.</span>
            {onUseSelectedPeople && <button type="button" className="routing-link" onClick={onUseSelectedPeople}>Use Selected people</button>}
          </div>
        )}
        {applies && overrides?.[selected.id]?.restore_identity_weight !== undefined && (
          <div className="routing-note routing-note--quiet">Likeness is read by CodeFormer and UltraMax only.</div>
        )}
      </div>
    </section>
  );
}
