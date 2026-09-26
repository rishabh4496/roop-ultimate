import React, { useState, useRef } from 'react';
import { postJSON } from '../api';
import { PERSON_COLORS } from './constants';
import { confirmDialog } from './confirm';
import { Icon } from '../icons';
import { targetPersonRecords, normalizeSourceIndex, SKIP } from './faceswap/faceMapping';

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

// Stable person ids are the row keys.  displayRank is derived solely for the
// label/color; deleting/reordering an angle cannot rename another row.
function groupByPerson(groups, personIds) {
  return targetPersonRecords({ targetGroups: groups, targetPersonIds: personIds })
    .map((record) => [record.targetPersonId, record.faceIndices, record.displayRank]);
}

export default function PersonGroups({
  targetFaces, targetGroups, targetNames, targetFacesInfo,
  targetPersonIds, targetReferenceFaceIds, selectedTargetPersonId,
  setSelectedTargetPersonId, _selectedReferenceFaceId, setSelectedReferenceFaceId,
  setTargetPersonIds, setTargetReferenceFaceIds,
  selTargetFace, setSelTargetFace,
  sourceFaces, sourceFacesInfo, faceSelection, setFaceSelection, selectedSource, faceMapping, setFaceMapping,
  frame, selTarget, targetMediaId,
  setTargetFaces, setTargetGroups, setTargetNames, setTargetFacesInfo,
  notify, clearPreviewCache,
  applyTargetContext,
  // Stage 14: the versioned write path for target-scoped identity state.
  // When present, every selection/mapping change is persisted through it so
  // the backend (and therefore a reload) knows the selected person, not just
  // the array position of an angle.
  commitTargetContext,
}) {
  // Select a person by stable id, optionally on one of its angles. The click
  // used to change React state only, so a reload restored whatever angle index
  // the backend last saw — usually the previous person.
  const choosePerson = (personId, faceIndex) => {
    setSelTargetFace(faceIndex);
    if (setSelectedTargetPersonId) setSelectedTargetPersonId(personId);
    const referenceId = Array.isArray(targetReferenceFaceIds)
      ? targetReferenceFaceIds[faceIndex] : undefined;
    if (setSelectedReferenceFaceId && referenceId !== undefined) {
      setSelectedReferenceFaceId(referenceId);
    }
    if (commitTargetContext) {
      commitTargetContext({
        selected_target_person_id: personId,
        ...(referenceId ? { selected_reference_face_id: referenceId } : {}),
        selected_target_face_index: faceIndex,
      });
    }
  };
  const [expanded, setExpanded] = useState({});      // target_person_id -> bool override
  const [editingRank, setEditingRank] = useState(null);
  const [editValue, setEditValue] = useState('');
  const [dropTarget, setDropTarget] = useState(null); // rank being hovered in a drag
  const [busy, setBusy] = useState(false);
  const [harvesting, setHarvesting] = useState(null); // rank being auto-harvested
  const [scanning, setScanning] = useState(false);    // whole-clip auto-capture
  const containerRef = useRef(null);

  const normTargetGroups = (Array.isArray(targetGroups) ? targetGroups : []).map(g => Array.isArray(g) ? (g[0] ?? 0) : g);
  const people = groupByPerson(normTargetGroups.slice(0, targetFaces.length), targetPersonIds);
  const selRank = selectedTargetPersonId
    || targetPersonIds?.[selTargetFace]
    || people[0]?.[0];

  // Push the parallel arrays back to the parent from an API payload.
  const applyPayload = (res) => {
    if (!res) return;
    if (applyTargetContext) {
      applyTargetContext(res, res.target_media_id || targetMediaId);
    }
    if (res.target_faces) setTargetFaces(res.target_faces);
    if (res.target_groups) {
      const flat = res.target_groups.map((g) => Array.isArray(g) ? (g[0] ?? 0) : (typeof g === 'number' ? g : parseInt(g, 10) || 0));
      setTargetGroups(flat);
    }
    if (res.target_names !== undefined && setTargetNames) setTargetNames(res.target_names || []);
    if (res.target_faces_info !== undefined && setTargetFacesInfo) setTargetFacesInfo(res.target_faces_info || []);
    if (res.target_person_source_mapping !== undefined && setFaceMapping) {
      setFaceMapping(res.target_person_source_mapping || {});
    } else if (res.face_mapping && !Array.isArray(res.face_mapping) && setFaceMapping) {
      setFaceMapping(res.face_mapping || {});
    }
    if (res.target_person_ids && setTargetPersonIds) setTargetPersonIds(res.target_person_ids);
    if (res.target_reference_face_ids && setTargetReferenceFaceIds) setTargetReferenceFaceIds(res.target_reference_face_ids);
    if (res.selected_target_person_id && setSelectedTargetPersonId) setSelectedTargetPersonId(res.selected_target_person_id);
    if (setSelectedReferenceFaceId) {
      const refId = res.selected_reference_face_id
        ?? (res.target_reference_face_ids?.[res.selected_target_face_index ?? 0] || null);
      if (refId !== undefined) {
        setSelectedReferenceFaceId(refId);
      }
    }
    if (clearPreviewCache) clearPreviewCache();
  };

  const isExpanded = (rank) => (rank in expanded ? expanded[rank] : rank === selRank);
  const toggleExpand = (rank) => setExpanded((e) => ({ ...e, [rank]: !isExpanded(rank) }));

  const displayRankOf = (personId) => people.find(([id]) => id === personId)?.[2] ?? 0;
  const nameFor = (personId) => {
    const rank = displayRankOf(personId);
    return (targetNames && targetNames[rank]) || '';
  };
  const labelFor = (personId) => nameFor(personId) || `Person ${displayRankOf(personId) + 1}`;

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
    const res = await call('/api/target/remove_face', { index: i, target_media_id: targetMediaId });
    if (res && selTargetFace >= (res.target_faces?.length || 0)) {
      setSelTargetFace(Math.max(0, (res.target_faces?.length || 1) - 1));
    }
  };

  const addAngle = (targetPersonId) => call('/api/target/add_angle', {
    target_person_id: targetPersonId, index: selTarget, frame, target_media_id: targetMediaId,
  },
    `Captured a new angle for ${labelFor(targetPersonId)}`);

  // Scan the whole video and auto-capture this person at many poses, filling
  // their angle bank so identity survives turns without manual capturing.
  // The backend does a coarse pass plus a targeted refine pass over the moments
  // where the pose changed, so the toast reports what it actually looked at —
  // "0 new angles" after 40 frames means something very different from after 400.
  const autoAngles = async (targetPersonId) => {
    const rank = targetPersonId; // legacy local name; value is a stable id
    setHarvesting(targetPersonId);
    try {
      const res = await postJSON('/api/target/auto_angles', {
        target_person_id: targetPersonId, index: selTarget, target_media_id: targetMediaId,
      });
      applyPayload(res);
      const personIndices = (res.target_person_ids || []).reduce((acc, id, idx) => {
        if (id === targetPersonId) acc.push(idx);
        return acc;
      }, []);
      if (personIndices.length > 0) {
        choosePerson(targetPersonId, personIndices[0]);
      }
      const detail = res.scanned
        ? ` — scanned ${res.scanned} frames in ${res.seconds}s, ${res.bins} pose bin${res.bins === 1 ? '' : 's'} covered`
        : '';
      if (res.count) {
        notify(`Auto-captured ${res.count} new angle${res.count === 1 ? '' : 's'} for ${labelFor(targetPersonId)}${detail}`);
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

  const scanFaceBank = async () => {
    if (targetFaces.length && !(await confirmDialog({
      title: 'Scan Face Bank?',
      message: 'Face Bank will scan the clip across scenes, extract 512-d normalized embeddings, and cluster all unique people (DBSCAN/Agglomerative) into distinct character thumbnails.',
      confirmLabel: 'Scan and cluster',
    }))) return;
    setScanning(true);
    try {
      const res = await postJSON('/api/target/face_bank', {
        index: selTarget, clustering_method: 'dbscan', target_media_id: targetMediaId, apply: true,
      });
      applyPayload(res);
      if (!res.count) {
        notify(res.message || 'Face Bank found no faces in this clip', 'warning');
        return;
      }
      setExpanded({});
      setSelTargetFace(0);
      notify(`Face Bank: Discovered ${res.characters?.length || res.count} unique character(s)`, 'success');
    } catch (e) {
      notify(e.message, 'error');
    } finally {
      setScanning(false);
    }
  };

  const autoCluster = async () => {
    const res = await call('/api/target/autocluster', { target_media_id: targetMediaId });
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
      const res = await postJSON('/api/target/auto_capture', {
        index: selTarget, replace: true, target_media_id: targetMediaId,
      });
      applyPayload(res);
      if (!res.count) {
        notify(res.message || 'Auto-capture found nobody in this clip', 'warning');
        return;
      }
      setExpanded({});
      setSelTargetFace(0);
      const capturedPeople = new Set(
        (res.target_person_ids || []).filter((id) => id !== null && id !== undefined && id !== ''),
      );
      if (capturedPeople.size > 1 && setFaceSelection) {
        setFaceSelection('Selected people');
      }
      const firstPersonId = res.target_person_ids?.[0];
      const firstRefId = res.target_reference_face_ids?.[0];
      if (firstPersonId) {
        if (setSelectedTargetPersonId) setSelectedTargetPersonId(firstPersonId);
        if (setSelectedReferenceFaceId && firstRefId) setSelectedReferenceFaceId(firstRefId);
        if (commitTargetContext) {
          commitTargetContext({
            selected_target_person_id: firstPersonId,
            ...(firstRefId ? { selected_reference_face_id: firstRefId } : {}),
            selected_target_face_index: 0,
          });
        }
      }
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
    const res = await call('/api/target/clear_faces', {
      target_media_id: targetMediaId,
    }, 'Cleared all target faces');
    if (res) {
      setExpanded({});
      setSelTargetFace(0);
      if (setFaceMapping) setFaceMapping({});
    }
  };

  // Move a single angle to another person (or a brand-new one via `newPerson`).
  const reassign = (faceIdx, targetPersonId) => {
    const ids = [...(targetPersonIds || [])];
    ids[faceIdx] = targetPersonId;
    if (setTargetPersonIds) setTargetPersonIds(ids);
    postJSON('/api/target/group', {
      target_person_ids: ids, target_media_id: targetMediaId,
    }).then(applyPayload).catch((e) => notify(e.message, 'error'));
  };

  const commitName = (targetPersonId) => {
    setEditingRank(null);
    const name = editValue.trim();
    if (name === nameFor(targetPersonId)) return;
    call('/api/target/name', {
      target_person_id: targetPersonId, name, target_media_id: targetMediaId,
    });
  };

  const sourceIdentityAt = (index) => sourceFacesInfo?.[index]?.id
    || (sourceFaces[index] ? `memory-slot-${index}` : null);

  const setMapping = (targetPersonId, val) => {
    const sourceId = Number(val) >= 0 ? sourceIdentityAt(Number(val)) : null;
    const next = { ...(faceMapping || {}) };
    // In "Selected face" the highlighted person is mapped IMPLICITLY (to the
    // gallery-selected source) and never appears in `faceMapping`. Mapping a
    // SECOND person from that state used to produce {P2: src2} only: the row
    // for P1 kept showing "Face 1" (implicit), the mode stayed "Selected
    // face", and the render swapped exactly one person -- whichever was
    // highlighted. Materialise the implicit entry first, so the mapping the
    // user can SEE is the mapping that runs.
    if (faceSelection === 'Selected face' && selRank && selRank !== targetPersonId
        && !Object.prototype.hasOwnProperty.call(next, selRank)) {
      const implicit = sourceIdentityAt(Number(selectedSource));
      if (implicit) next[selRank] = implicit;
    }
    if (sourceId) next[targetPersonId] = sourceId;
    else delete next[targetPersonId];
    setFaceMapping(next);
    // Two or more people bound to sources is a multi-person job. "Selected
    // face" swaps ONE person by definition, so leaving it there silently drops
    // every other mapping the user just made. Switch, and say so.
    const mapped = Object.keys(next).filter((k) => next[k]);
    if (faceSelection === 'Selected face' && mapped.length >= 2 && setFaceSelection) {
      setFaceSelection('Selected people');
      if (notify) notify(`${mapped.length} people mapped — switched to "Selected people" so all of them swap`, 'info');
    }
    // Mapping is UI-owned, but it is still target-specific state. Persist it
    // immediately so a reload cannot reconstruct B from A's last mapping.
    // Through the versioned path when the parent provides it, so a preview
    // built BEFORE this change cannot land afterwards and overwrite it.
    if (commitTargetContext) {
      commitTargetContext({ face_mapping: next, target_person_source_mapping: next });
    } else {
      postJSON('/api/target/context', {
        target_media_id: targetMediaId,
        face_mapping: next,
        target_person_source_mapping: next,
      }).catch((e) => notify(e.message, 'error'));
    }
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
    const firstFace = people.find(([id]) => id === nextRank)?.[1]?.[0];
    if (firstFace >= 0) choosePerson(nextRank, firstFace);
  };

  const otherRanks = people.map(([r]) => r);

  if (targetFaces.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-white/10 bg-black/10 p-4 text-center space-y-2.5 select-none">
        <div className="flex justify-center opacity-40"><Icon.faces size={22} /></div>
        <div className="text-xs text-white/50 font-semibold">No people captured yet</div>
        <div className="flex justify-center gap-2">
          <button type="button" disabled={scanning} onClick={scanFaceBank}
            title="Scan clip into Face Bank, cluster unique faces with DBSCAN, and build distinct character thumbnails"
            className="px-3 py-1.5 rounded-lg text-mini font-bold bg-[var(--accent)]/20 border border-[var(--accent)]/50 text-[var(--accent)] hover:bg-[var(--accent)]/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
            {scanning ? 'Scanning Face Bank…' : 'Scan Face Bank (Clustering)'}
          </button>
          <button type="button" disabled={scanning} onClick={autoCapture}
            title="Scan the clip, find everyone in it, and capture each person from their own clearest frame"
            className="px-3 py-1.5 rounded-lg text-mini font-bold bg-white/[0.05] border border-white/10 text-white/70 hover:bg-white/10 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
            Auto-capture
          </button>
        </div>
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
          <button type="button" disabled={busy || scanning} onClick={scanFaceBank}
            title="Scan video into Face Bank: extract 512-d embeddings and cluster all unique people (DBSCAN/Agglomerative)"
            className="px-2 py-1 rounded-lg text-micro font-bold bg-[var(--accent)]/20 border border-[var(--accent)]/50 text-[var(--accent)] hover:bg-[var(--accent)]/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
            {scanning ? 'Scanning…' : 'Face Bank'}
          </button>
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

      {people.map(([rank, indices, displayRank]) => {
        const color = PERSON_COLORS[displayRank % PERSON_COLORS.length];
        const open = isExpanded(rank);
        const isSel = rank === selRank;
        // Same helper the swap payload is built from, so the row can never show
        // a source the backend will not actually use.
        const rawStable = faceMapping?.[rank];
        const mappedByIdentity = rawStable == null ? SKIP
          : sourceFacesInfo?.findIndex((info, index) =>
            String(info?.id || `memory-slot-${index}`) === String(rawStable));
        const mappedById = mappedByIdentity >= 0 ? mappedByIdentity
          : normalizeSourceIndex(rawStable, sourceFaces.length);
        const safeMap = mappedById >= 0 ? mappedById
          : (faceSelection === 'Selected face' && rank === selRank
            ? normalizeSourceIndex(selectedSource, sourceFaces.length)
            : SKIP);
        const mapValid = safeMap >= 0 && safeMap < sourceFaces.length;

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
            <div className="flex items-center gap-2 px-3 py-2.5 cursor-pointer" onClick={() => choosePerson(rank, indices[0])}>
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
                    placeholder={`Person ${displayRank + 1}`}
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
                  value={safeMap}
                  onChange={(e) => setMapping(rank, parseInt(e.target.value, 10))}
                  title="Which source face this person becomes"
                  className={`px-2 py-1 rounded-lg glass-input text-white text-mini font-bold focus:outline-none cursor-pointer max-w-[120px] shrink-0 ${mapValid ? '' : 'text-white/50'}`}
                >
                  <option value={-1} className="bg-[#121420]">Ignore / Skip</option>
                  {sourceFaces.map((_, sfIdx) => (
                    <option key={sfIdx} value={sfIdx} className="bg-[#121420]">Source Face {sfIdx + 1}</option>
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
                        <button type="button" onClick={() => choosePerson(rank, i)}
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
                            onChange={(e) => reassign(i, e.target.value)}
                            title="Move this angle to another person"
                            className="mt-1 w-14 px-1 py-0.5 rounded-md glass-input text-white/70 text-nano focus:outline-none cursor-pointer">
                            {otherRanks.map((r) => (
                              <option key={r} value={r} className="bg-[#121420]">→ {labelFor(r)}</option>
                            ))}
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
