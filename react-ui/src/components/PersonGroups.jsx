import React, { useState, useRef } from 'react';
import { postJSON } from '../api';
import { PERSON_COLORS } from './constants';
import { confirmDialog } from './confirm';
import { Icon } from '../icons';

// Coarse pose buckets we consider "primary coverage" for a person. Anything the
// backend labels (e.g. "Left Profile + Up Tilt") is matched against these by
// substring so combos still count toward the base angle.
const PRIMARY_POSES = ['Front', 'Left Profile', 'Right Profile', 'Up Tilt', 'Down Tilt'];

// Compact pose "compass": center = Front, and L/R/Up/Down points around it.
// Covered directions glow in the person's color; uncovered are faint rings.
function PoseCompass({ covered, color }) {
  const pts = [
    { key: 'Front', cx: 30, cy: 30, r: 5 },
    { key: 'Left Profile', cx: 8, cy: 30, r: 4 },
    { key: 'Right Profile', cx: 52, cy: 30, r: 4 },
    { key: 'Up Tilt', cx: 30, cy: 8, r: 4 },
    { key: 'Down Tilt', cx: 30, cy: 52, r: 4 },
  ];
  return (
    <svg viewBox="0 0 60 60" className="w-14 h-14 shrink-0">
      <line x1="30" y1="8" x2="30" y2="52" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
      <line x1="8" y1="30" x2="52" y2="30" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
      <circle cx="30" cy="30" r="22" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
      {pts.map((pt) => {
        const on = covered.has(pt.key);
        return (
          <circle key={pt.key} cx={pt.cx} cy={pt.cy} r={pt.r}
            fill={on ? color : 'transparent'}
            stroke={on ? color : 'rgba(255,255,255,0.2)'}
            strokeWidth={on ? 0 : 1.4}
            style={on ? { filter: `drop-shadow(0 0 3px ${color})` } : undefined} />
        );
      })}
    </svg>
  );
}

// Group target-face indices by their person rank, preserving rank order.
function groupByPerson(groups) {
  const map = new Map();
  groups.forEach((rank, i) => {
    if (!map.has(rank)) map.set(rank, []);
    map.get(rank).push(i);
  });
  return Array.from(map.entries()).sort((a, b) => a[0] - b[0]); // [rank, [indices]]
}

export default function PersonGroups({
  targetFaces, targetGroups, targetNames, targetFacesInfo,
  selTargetFace, setSelTargetFace,
  sourceFaces, faceMapping, setFaceMapping,
  frame, selTarget,
  setTargetFaces, setTargetGroups, setTargetNames, setTargetFacesInfo,
  notify, clearPreviewCache,
}) {
  const [expanded, setExpanded] = useState({});      // rank -> bool override
  const [editingRank, setEditingRank] = useState(null);
  const [editValue, setEditValue] = useState('');
  const [dropTarget, setDropTarget] = useState(null); // rank being hovered in a drag
  const [busy, setBusy] = useState(false);
  const [harvesting, setHarvesting] = useState(null); // rank being auto-harvested
  const [scanning, setScanning] = useState(false);    // whole-clip auto-capture
  const containerRef = useRef(null);

  const people = groupByPerson(targetGroups.slice(0, targetFaces.length));
  const selRank = targetGroups[selTargetFace];

  // Push the four parallel arrays back to the parent from an API payload.
  const applyPayload = (res) => {
    if (res.target_faces) setTargetFaces(res.target_faces);
    if (res.target_groups) setTargetGroups(res.target_groups);
    if (res.target_names !== undefined && setTargetNames) setTargetNames(res.target_names || []);
    if (res.target_faces_info !== undefined && setTargetFacesInfo) setTargetFacesInfo(res.target_faces_info || []);
    if (clearPreviewCache) clearPreviewCache();
  };

  const isExpanded = (rank) => (rank in expanded ? expanded[rank] : rank === selRank);
  const toggleExpand = (rank) => setExpanded((e) => ({ ...e, [rank]: !isExpanded(rank) }));

  const nameFor = (rank) => (targetNames && targetNames[rank]) || '';
  const labelFor = (rank) => nameFor(rank) || `Person ${rank + 1}`;

  const call = async (path, body, okMsg) => {
    setBusy(true);
    try {
      const res = await postJSON(path, body);
      applyPayload(res);
      if (res && res.message && !res.count) {
        notify(res.message, 'warning');
      } else if (okMsg) {
        notify(okMsg);
      }
      return res;
    } catch (e) {
      notify(e.message, 'error');
    } finally {
      setBusy(false);
    }
  };

  const removeAngle = async (i) => {
    const res = await call('/api/target/remove_face', { index: i });
    if (res && selTargetFace >= (res.target_faces?.length || 0)) {
      setSelTargetFace(Math.max(0, (res.target_faces?.length || 1) - 1));
    }
  };

  const addAngle = (rank) => call('/api/target/add_angle', { person: rank, index: selTarget, frame },
    `Captured a new angle for ${labelFor(rank)}`);

  // Scan the whole video and auto-capture this person at many poses, filling
  // their angle bank so identity survives turns without manual capturing.
  // The backend does a coarse pass plus a targeted refine pass over the moments
  // where the pose changed, so the toast reports what it actually looked at —
  // "0 new angles" after 40 frames means something very different from after 400.
  const autoAngles = async (rank) => {
    setHarvesting(rank);
    try {
      const res = await postJSON('/api/target/auto_angles', { person: rank, index: selTarget });
      applyPayload(res);
      const detail = res.scanned
        ? ` — scanned ${res.scanned} frames in ${res.seconds}s, ${res.bins} pose bin${res.bins === 1 ? '' : 's'} covered`
        : '';
      if (res.count) {
        notify(`Auto-captured ${res.count} new angle${res.count === 1 ? '' : 's'} for ${labelFor(rank)}${detail}`);
        // A wrong angle looks like any other thumbnail, and its cost lands much
        // later as the wrong person being swapped — every match takes the
        // minimum over a person's angles, so one bad entry speaks for all of
        // them. The backend ranks the ones furthest from the original capture;
        // saying how many turns "check every thumbnail" into "check these two".
        // Not an accusation: a true extreme profile lands here too.
        const n = res.review?.length || 0;
        if (n) {
          notify(
            `${n} of them sit far from your original capture — worth checking the thumbnails for ${labelFor(rank)}, and removing any that aren't them.`,
            'warning',
          );
        }
      } else {
        // Distinguish "nobody matched" from "everybody who matched was too
        // blurred or too small to bank", which the count alone cannot.
        const why = Object.entries(res.rejected || {})
          .map(([reason, count]) => `${count} ${reason}`)
          .join(', ');
        notify((res.message || 'No new angles found') + detail
               + (why ? ` (turned away: ${why})` : ''), 'warning');
      }
    } catch (e) {
      notify(e.message, 'error');
    } finally {
      setHarvesting(null);
    }
  };

  const autoCluster = async () => {
    const res = await call('/api/target/autocluster', {});
    if (res) { setExpanded({}); notify(`Grouped into ${res.people} ${res.people === 1 ? 'person' : 'people'}`); }
  };

  // Scan the clip, find everyone in it, and capture each person from THEIR own
  // most-frontal frame. Preferred over capturing by hand: picking the frame
  // yourself means seeding identity from whatever you happen to be looking at,
  // and on footage where people interact the frames that obviously show two
  // boxes are the frames where they touch — the worst possible reference.
  const autoCapture = async () => {
    if (targetFaces.length && !(await confirmDialog({
      title: 'Replace captured people?',
      message: 'Auto-capture scans the clip and captures everyone from scratch, replacing the people you already have.',
      confirmLabel: 'Scan and replace',
    }))) return;
    setScanning(true);
    try {
      const res = await postJSON('/api/target/auto_capture', { index: selTarget, replace: true });
      applyPayload(res);
      if (!res.count) {
        notify(res.message || 'Auto-capture found nobody in this clip', 'warning');
        return;
      }
      setExpanded({});
      setSelTargetFace(0);
      // `separation` is the distance between the captured people, and it is the
      // number that decides whether any later identity decision can work: two
      // people captured 0.12 apart are one identity as far as the matcher is
      // concerned, and every swap after that is a coin flip. Surfaced rather
      // than buried because the failure it predicts (nothing swaps, or the
      // wrong face swaps) shows up much later and looks like a different bug.
      const sep = res.separation;
      const quality = sep == null ? '' :
        sep >= 0.7 ? ` — identities are clearly distinct (${sep.toFixed(2)})`
        : sep >= 0.4 ? ` — identities are usable but close (${sep.toFixed(2)})`
        : ` — WARNING: identities are only ${sep.toFixed(2)} apart and will be mixed up`;
      // The angle count is the number that predicts whether the render will
      // hold: measured on a full clip, one face per person left 27.5% of faces
      // un-swapped where the same capture plus harvested angles left 15.8%.
      const angles = res.angles_added
        ? `, plus ${res.angles_added} harvested angle${res.angles_added === 1 ? '' : 's'}`
        : '';
      notify(
        `Captured ${res.count} ${res.count === 1 ? 'person' : 'people'}${angles} from ${res.scanned} scanned frames${quality}`,
        sep != null && sep < 0.4 ? 'warning' : 'success',
      );
      (res.notes || []).forEach((n) => { if (/WARNING|never clearly apart|overlap/.test(n)) notify(n, 'warning'); });
    } catch (e) {
      notify(e.message, 'error');
    } finally {
      setScanning(false);
    }
  };

  // Remove every captured person/angle from the layout (keeps target media).
  const clearAllFaces = async () => {
    if (!targetFaces.length) return;
    if (!(await confirmDialog({ title: 'Clear all faces?', message: 'Remove all captured target faces? This clears every person in this layout.', confirmLabel: 'Clear all', danger: true }))) return;
    const res = await call('/api/target/clear_faces', {}, 'Cleared all target faces');
    if (res) {
      setExpanded({});
      setSelTargetFace(0);
      if (setFaceMapping) setFaceMapping({});
    }
  };

  // Move a single angle to another person (or a brand-new one via `newPerson`).
  const reassign = (faceIdx, targetRank) => {
    const g = [...targetGroups];
    g[faceIdx] = targetRank;
    setTargetGroups(g);
    postJSON('/api/target/group', { groups: g }).then(applyPayload).catch((e) => notify(e.message, 'error'));
  };

  const commitName = (rank) => {
    setEditingRank(null);
    const name = editValue.trim();
    if (name === nameFor(rank)) return;
    call('/api/target/name', { person: rank, name });
  };

  const setMapping = (rank, val) => {
    setFaceMapping((prev) => ({ ...prev, [rank]: val }));
    if (clearPreviewCache) clearPreviewCache();
  };

  // Scoped arrow-key nav: move the selected person up/down. stopPropagation so
  // the global handler (which uses arrows to step video frames) never fires.
  const onKeyDown = (e) => {
    if (e.target !== containerRef.current) return;
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    e.preventDefault();
    e.stopPropagation();
    const ranks = people.map(([r]) => r);
    const cur = ranks.indexOf(selRank);
    const nextRank = ranks[Math.min(ranks.length - 1, Math.max(0, cur + (e.key === 'ArrowDown' ? 1 : -1)))];
    const firstFace = targetGroups.indexOf(nextRank);
    if (firstFace >= 0) setSelTargetFace(firstFace);
  };

  const otherRanks = people.map(([r]) => r);
  const nextNewRank = (Math.max(-1, ...otherRanks) + 1);

  if (targetFaces.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-white/10 bg-black/10 p-4 text-center space-y-2.5 select-none">
        <div className="flex justify-center opacity-40"><Icon.faces size={22} /></div>
        <div className="text-xs text-white/50 font-semibold">No people captured yet</div>
        <button type="button" disabled={scanning} onClick={autoCapture}
          title="Scan the clip, find everyone in it, and capture each person from their own clearest frame"
          className="mx-auto px-3 py-1.5 rounded-lg text-mini font-bold bg-[var(--accent)]/15 border border-[var(--accent)]/40 text-[var(--accent)] hover:bg-[var(--accent)]/25 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
          {scanning ? 'Scanning the clip…' : 'Auto-capture people'}
        </button>
        <div className="text-mini text-white/45 leading-relaxed">
          Recommended: finds everyone, captures each from their clearest frame, then harvests
          the angles they turn through. Takes a couple of minutes and is worth it — a single
          reference face only matches poses near it, and a person in motion spends most of the
          clip somewhere else.
        </div>
        <div className="text-mini text-white/30 leading-relaxed">
          Or scrub to a clear frame and use <span className="text-white/45 font-bold">“Face from frame”</span>.
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      onKeyDown={onKeyDown}
      className="space-y-3 outline-none focus:ring-1 focus:ring-[var(--accent)]/30 rounded-xl"
    >
      {/* Toolbar */}
      <div className="flex items-center justify-between gap-2">
        <span className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45">
          {people.length} {people.length === 1 ? 'person' : 'people'} · {targetFaces.length} {targetFaces.length === 1 ? 'angle' : 'angles'}
        </span>
        <div className="flex gap-1.5">
          <button type="button" disabled={busy || scanning} onClick={autoCapture}
            title="Re-scan the clip and capture everyone from their own clearest frame, replacing the current people"
            className="px-2 py-1 rounded-lg text-micro font-bold bg-[var(--accent)]/10 border border-[var(--accent)]/30 text-[var(--accent)] hover:bg-[var(--accent)]/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
            {scanning ? 'Scanning…' : 'Auto-capture'}
          </button>
          <button type="button" disabled={busy} onClick={autoCluster}
            title="Group every captured face by identity automatically"
            className="px-2 py-1 rounded-lg text-micro font-bold bg-[var(--accent)]/10 border border-[var(--accent)]/30 text-[var(--accent)] hover:bg-[var(--accent)]/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
            Auto-group
          </button>
          <button type="button" disabled={busy || !targetFaces.length} onClick={clearAllFaces}
            title="Remove all captured target faces from this layout"
            className="px-2 py-1 rounded-lg text-micro font-bold bg-white/[0.03] border border-white/10 text-white/50 hover:text-white/80 hover:border-white/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
            Reset
          </button>
        </div>
      </div>

      {sourceFaces.length > 0 && (
        <div className="text-micro text-white/45 flex items-center gap-1.5 select-none">
          <span className="h-1 w-1 rounded-full bg-[var(--accent)]" />
          Drag a source face onto a person, or use the dropdown, to choose who they become.
        </div>
      )}

      {people.map(([rank, indices]) => {
        const color = PERSON_COLORS[rank % PERSON_COLORS.length];
        const open = isExpanded(rank);
        const isSel = rank === selRank;
        const currentMap = faceMapping[rank] !== undefined ? faceMapping[rank] : rank;
        const mapValid = currentMap >= 0 && currentMap < sourceFaces.length;

        // Pose coverage for this person.
        const poses = indices.map((i) => (targetFacesInfo && targetFacesInfo[i]?.pose) || 'Front');
        const covered = new Set();
        poses.forEach((p) => PRIMARY_POSES.forEach((pp) => { if (p.includes(pp)) covered.add(pp); }));
        const missing = ['Front', 'Left Profile', 'Right Profile'].filter((pp) => !covered.has(pp));

        return (
          <div
            key={rank}
            onDragOver={sourceFaces.length ? (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'link'; setDropTarget(rank); } : undefined}
            onDragLeave={() => setDropTarget((d) => (d === rank ? null : d))}
            onDrop={sourceFaces.length ? (e) => {
              e.preventDefault();
              setDropTarget(null);
              const raw = e.dataTransfer.getData('text/roop-source');
              if (raw !== '') setMapping(rank, parseInt(raw, 10));
            } : undefined}
            className={`rounded-xl border transition-all overflow-hidden ${dropTarget === rank ? 'border-[var(--accent)] shadow-[0_0_0_2px_var(--accent-glow)]' : isSel ? 'border-white/20 bg-white/[0.02]' : 'border-white/5 bg-black/25 hover:border-white/10'}`}
            style={isSel ? { boxShadow: `inset 3px 0 0 ${color}` } : { boxShadow: `inset 3px 0 0 ${color}55` }}
          >
            {/* Header */}
            <div className="flex items-center gap-2 px-3 py-2.5 cursor-pointer" onClick={() => { setSelTargetFace(indices[0]); }}>
              <button type="button" onClick={(e) => { e.stopPropagation(); toggleExpand(rank); }}
                aria-label={`${open ? 'Collapse' : 'Expand'} ${labelFor(rank)}`}
                aria-expanded={open}
                className="text-white/40 hover:text-white/80 transition-transform shrink-0" style={{ transform: open ? 'rotate(90deg)' : 'none' }}><Icon.expand size={13} /></button>
              <img src={targetFaces[indices[0]]} alt="" className="w-8 h-8 rounded-lg object-cover shrink-0 border" style={{ borderColor: color }} />
              <div className="flex-1 min-w-0" onClick={(e) => e.stopPropagation()}>
                {editingRank === rank ? (
                  <input
                    autoFocus
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onBlur={() => commitName(rank)}
                    onKeyDown={(e) => { if (e.key === 'Enter') commitName(rank); if (e.key === 'Escape') setEditingRank(null); }}
                    placeholder={`Person ${rank + 1}`}
                    className="w-full px-2 py-1 rounded-md glass-input text-white text-xs font-bold focus:outline-none"
                  />
                ) : (
                  <div className="flex items-center gap-1.5">
                    <span className="font-extrabold text-sm truncate" style={{ color }}>{labelFor(rank)}</span>
                    <button type="button" title="Rename person"
                      aria-label={`Rename ${labelFor(rank)}`}
                      onClick={() => { setEditingRank(rank); setEditValue(nameFor(rank)); }}
                      className="text-white/25 hover:text-white/70 shrink-0"><Icon.rename size={11} /></button>
                  </div>
                )}
                <div className="text-micro text-white/45 font-medium">{indices.length} {indices.length === 1 ? 'angle' : 'angles'}</div>
              </div>

              {/* Mapping dropdown */}
              {sourceFaces.length > 0 && (
                <select
                  onClick={(e) => e.stopPropagation()}
                  value={currentMap}
                  onChange={(e) => setMapping(rank, parseInt(e.target.value, 10))}
                  title="Which source face this person becomes"
                  className={`px-2 py-1 rounded-lg glass-input text-white text-mini font-bold focus:outline-none cursor-pointer max-w-[120px] shrink-0 ${mapValid ? '' : 'text-white/50'}`}
                >
                  <option value={-1} className="bg-[#121420]">Skip</option>
                  {sourceFaces.map((_, sfIdx) => (
                    <option key={sfIdx} value={sfIdx} className="bg-[#121420]">Face {sfIdx + 1}</option>
                  ))}
                </select>
              )}
            </div>

            {/* Body */}
            {open && (
              <div className="px-3 pb-3 space-y-3 border-t border-white/5 pt-3">
                {/* Angle strip */}
                <div className="flex flex-wrap gap-2">
                  {indices.map((i) => {
                    const pose = (targetFacesInfo && targetFacesInfo[i]?.pose) || 'Front';
                    const sel = i === selTargetFace;
                    return (
                      <div key={i} className="relative group/angle">
                        <button type="button" onClick={() => setSelTargetFace(i)}
                          className={`block rounded-lg overflow-hidden border-2 transition-all ${sel ? 'scale-105' : 'opacity-80 hover:opacity-100'}`}
                          style={{ borderColor: sel ? color : 'transparent' }}>
                          <img src={targetFaces[i]} alt={pose} className="w-14 h-14 object-cover" />
                        </button>
                        {/* hover-enlarge popover */}
                        <div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover/angle:block z-40">
                          <div className="p-1.5 rounded-xl bg-black/95 border border-white/10 shadow-2xl flex flex-col items-center gap-1">
                            <img src={targetFaces[i]} alt="" className="w-28 h-28 object-cover rounded-lg" />
                            <span className="text-micro font-bold text-[var(--accent)] whitespace-nowrap">{pose}</span>
                          </div>
                          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 w-2 h-2 rotate-45 bg-black/95 border-b border-r border-white/10" />
                        </div>
                        {/* pose tag */}
                        <span className="absolute bottom-0.5 left-0.5 right-0.5 text-center text-nano font-black text-white bg-black/70 rounded px-0.5 truncate leading-tight pointer-events-none">{pose}</span>
                        {/* per-angle delete */}
                        <button type="button" title="Remove this angle"
                          aria-label={`Remove the ${pose} angle`}
                          onClick={(e) => { e.stopPropagation(); removeAngle(i); }}
                          className="absolute -top-1.5 -right-1.5 h-5 w-5 rounded-full bg-black/80 text-white/70 opacity-0 group-hover/angle:opacity-100 focus-visible:opacity-100 hover:bg-[var(--accent-hover)] hover:text-white transition-all flex items-center justify-center border border-white/10"><Icon.close size={11} /></button>
                        {/* reassign to another person */}
                        {people.length > 1 && (
                          <select
                            value={rank}
                            onChange={(e) => reassign(i, parseInt(e.target.value, 10))}
                            title="Move this angle to another person"
                            className="mt-1 w-14 px-1 py-0.5 rounded-md glass-input text-white/70 text-nano focus:outline-none cursor-pointer">
                            {otherRanks.map((r) => (
                              <option key={r} value={r} className="bg-[#121420]">→ {labelFor(r)}</option>
                            ))}
                            <option value={nextNewRank} className="bg-[#121420]">→ New</option>
                          </select>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Coverage meter — pose compass + missing-angle hints */}
                <div className="flex items-center gap-3">
                  <PoseCompass covered={covered} color={color} />
                  <div className="min-w-0">
                    <div className="text-nano font-semibold uppercase tracking-[0.14em] text-white/45 mb-1">
                      Angle coverage · {covered.size}/5
                    </div>
                    {missing.length === 0 ? (
                      <span className="text-micro font-bold" style={{ color }}>✓ Well covered</span>
                    ) : (
                      <div className="flex flex-wrap gap-1">
                        {missing.map((pp) => (
                          <span key={pp} title={`No ${pp} angle captured yet — add one for steadier swaps`}
                            className="px-1.5 py-0.5 rounded-md text-nano font-semibold border border-dashed border-white/15 text-white/45">
                            + {pp.replace(' Profile', '')}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Auto-capture angles from the whole video (fills coverage). */}
                <button type="button" disabled={busy || harvesting !== null} onClick={() => autoAngles(rank)}
                  title="Scan the whole video and automatically capture this person at many angles — fills pose coverage so their identity survives turns/profiles without hand-capturing frames. Wrong grabs can be removed with the ✕ on each angle."
                  className="w-full py-1.5 rounded-lg text-mini font-bold bg-[var(--accent)]/12 border border-[var(--accent)]/35 text-[var(--accent)] hover:bg-[var(--accent)]/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2">
                  {harvesting === rank
                    ? (<><span className="h-3 w-3 rounded-full border-2 border-[var(--accent)]/40 border-t-[var(--accent)] animate-spin" /> Scanning video for angles…</>)
                    : 'Auto-capture angles (scan whole video)'}
                </button>

                {/* Grab angle at current frame */}
                <button type="button" disabled={busy} onClick={() => addAngle(rank)}
                  className="w-full py-1.5 rounded-lg text-mini font-bold bg-white/[0.03] border border-white/10 text-white/70 hover:border-[var(--accent)]/40 hover:text-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
                  Capture this person’s angle at frame {frame}
                </button>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
