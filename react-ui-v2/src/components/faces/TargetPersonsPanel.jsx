import React, { useMemo, useState } from 'react';

const PRIMARY_POSES = ['Front', 'Left Profile', 'Right Profile', 'Up Tilt', 'Down Tilt'];

/**
 * Compact Pose Compass Indicator.
 * Highlights covered pitch and yaw angle buckets in glowing accent tones.
 */
function PoseCompass({ covered = new Set(), color = '#38bdf8' }) {
  const pts = [
    { key: 'Front', cx: 20, cy: 20, r: 3.5 },
    { key: 'Left Profile', cx: 6, cy: 20, r: 2.8 },
    { key: 'Right Profile', cx: 34, cy: 20, r: 2.8 },
    { key: 'Up Tilt', cx: 20, cy: 6, r: 2.8 },
    { key: 'Down Tilt', cx: 20, cy: 34, r: 2.8 },
  ];

  return (
    <svg viewBox="0 0 40 40" className="w-10 h-10 shrink-0">
      <line x1="20" y1="6" x2="20" y2="34" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
      <line x1="6" y1="20" x2="34" y2="20" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
      <circle cx="20" cy="20" r="14" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
      {pts.map((pt) => {
        const on = covered.has(pt.key);
        return (
          <circle
            key={pt.key}
            cx={pt.cx}
            cy={pt.cy}
            r={pt.r}
            fill={on ? color : 'transparent'}
            stroke={on ? color : 'rgba(255,255,255,0.2)'}
            strokeWidth={on ? 0 : 1}
            style={on ? { filter: `drop-shadow(0 0 2px ${color})` } : undefined}
          />
        );
      })}
    </svg>
  );
}

function groupByPerson(groups = [], facesCount = 0) {
  const map = new Map();
  const slice = groups.slice(0, facesCount);
  slice.forEach((rank, i) => {
    if (!map.has(rank)) map.set(rank, []);
    map.get(rank).push(i);
  });
  return Array.from(map.entries()).sort((a, b) => a[0] - b[0]);
}

/**
 * Pro-Workstation Target Person Groups & Angle Banking Panel for React UI 2.0.
 */
export function TargetPersonsPanel({
  targetFaces = [],
  targetGroups = [],
  targetNames = [],
  targetFacesInfo = [],
  selectedTargetFace = 0,
  onSelectTargetFace,
  sourceFaces = [],
  faceMapping = {},
  onChangeMapping,
  onAutoCapture,
  onAutoAngles,
  onAutocluster,
  onAddAngle,
  onRemoveFace,
  onRenamePerson,
  busy = false,
  className = '',
}) {
  const [editingRank, setEditingRank] = useState(null);
  const [editValue, setEditValue] = useState('');

  const people = useMemo(
    () => groupByPerson(targetGroups, targetFaces.length),
    [targetGroups, targetFaces.length],
  );

  const handleStartRename = (rank, currentName) => {
    setEditingRank(rank);
    setEditValue(currentName || `Person ${rank + 1}`);
  };

  const handleCommitRename = (rank) => {
    if (editingRank !== rank) return;
    const name = editValue.trim();
    if (onRenamePerson) onRenamePerson(rank, name);
    setEditingRank(null);
  };

  return (
    <div className={`flex flex-col gap-2.5 p-3 rounded-xl bg-[#0b0d13] border border-white/10 ${className}`}>
      {/* Header & Global Harvesting Actions */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-purple-400">
            Target Identities
          </span>
          <span className="px-1.5 py-0.5 rounded-full bg-white/10 text-[10px] font-mono text-white/70">
            {people.length} Person{people.length === 1 ? '' : 's'}
          </span>
        </div>

        <div className="flex items-center gap-1">
          {onAutocluster && (
            <button
              type="button"
              onClick={onAutocluster}
              title="Auto-Cluster Banked Angles by Identity"
              disabled={busy || targetFaces.length === 0}
              className="px-2 py-0.5 rounded bg-white/5 hover:bg-white/10 border border-white/10 text-[10px] font-medium text-white/80 transition-colors disabled:opacity-40"
            >
              Cluster
            </button>
          )}

          {onAutoCapture && (
            <button
              type="button"
              onClick={onAutoCapture}
              title="Scan Video and Auto-Capture All Detected Faces"
              disabled={busy}
              className="px-2 py-0.5 rounded bg-purple-500/20 hover:bg-purple-500/30 border border-purple-500/40 text-[10px] font-bold text-purple-300 transition-colors disabled:opacity-40"
            >
              Auto-Capture
            </button>
          )}
        </div>
      </div>

      {/* People Groups List */}
      {people.length > 0 ? (
        <div className="flex flex-col gap-2 max-h-[340px] overflow-y-auto pr-1">
          {people.map(([rank, faceIndices]) => {
            const personName = targetNames[rank] || `Person ${rank + 1}`;
            const isEditing = editingRank === rank;
            const mappedSourceIndex = faceMapping[rank];

            // Compute covered poses for this person
            const coveredPoses = new Set();
            faceIndices.forEach((fIdx) => {
              const info = targetFacesInfo[fIdx];
              const poseLabel = info?.pose || '';
              PRIMARY_POSES.forEach((p) => {
                if (poseLabel.toLowerCase().includes(p.toLowerCase())) coveredPoses.add(p);
              });
            });

            return (
              <div
                key={`person-group-${rank}`}
                className="flex flex-col gap-2 p-2.5 rounded-lg bg-white/[0.02] border border-white/10 hover:border-white/20 transition-colors"
              >
                {/* Person Header & Source Face Mapping Select */}
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5 flex-1 min-w-0">
                    <span className="w-2 h-2 rounded-full bg-purple-400 shrink-0" />
                    {isEditing ? (
                      <input
                        type="text"
                        value={editValue}
                        autoFocus
                        onChange={(e) => setEditValue(e.target.value)}
                        onBlur={() => handleCommitRename(rank)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleCommitRename(rank);
                          if (e.key === 'Escape') setEditingRank(null);
                        }}
                        className="px-1.5 py-0.5 rounded bg-black/60 border border-[var(--accent-primary,#38bdf8)] text-xs text-white outline-none w-full"
                      />
                    ) : (
                      <span
                        onClick={() => handleStartRename(rank, personName)}
                        title="Click to rename person"
                        className="text-xs font-semibold text-white/90 truncate cursor-pointer hover:underline"
                      >
                        {personName}
                      </span>
                    )}
                  </div>

                  {/* Mapping Dropdown: Target Person -> Source Face */}
                  <div className="flex items-center gap-1 shrink-0">
                    <span className="text-[10px] font-mono text-white/40">Swap to:</span>
                    <select
                      value={mappedSourceIndex !== undefined ? mappedSourceIndex : rank}
                      onChange={(e) => onChangeMapping?.(rank, Number(e.target.value))}
                      className="px-1.5 py-0.5 rounded bg-black/80 border border-white/15 text-[10px] font-mono text-white outline-none cursor-pointer hover:border-[var(--accent-primary,#38bdf8)]"
                    >
                      <option value={-1}>Do Not Swap</option>
                      {sourceFaces.map((_, sIdx) => (
                        <option key={`src-opt-${sIdx}`} value={sIdx}>
                          Source #{sIdx + 1}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Angle Thumbnails Gallery + Pose Compass */}
                <div className="flex items-center gap-2">
                  {/* Compass */}
                  <div title="Pitch/Yaw Angle Coverage Compass">
                    <PoseCompass covered={coveredPoses} color="#a855f7" />
                  </div>

                  {/* Banked Face Angle Thumbnails */}
                  <div className="flex items-center gap-1.5 overflow-x-auto flex-1 py-1">
                    {faceIndices.map((faceIdx) => {
                      const isSelected = selectedTargetFace === faceIdx;
                      const thumbSrc = targetFaces[faceIdx] || '';
                      const info = targetFacesInfo[faceIdx];

                      return (
                        <div
                          key={`target-angle-${faceIdx}`}
                          onClick={() => onSelectTargetFace?.(faceIdx)}
                          title={`${info?.pose || 'Angle'} (Frame ${info?.frame || 1})`}
                          className={`group relative shrink-0 w-12 aspect-square rounded-md overflow-hidden p-0.5 border cursor-pointer select-none transition-all ${
                            isSelected
                              ? 'border-purple-400 shadow-[0_0_8px_rgba(168,85,247,0.5)]'
                              : 'border-white/10 hover:border-white/30'
                          }`}
                        >
                          <img
                            src={thumbSrc}
                            alt={`Angle #${faceIdx + 1}`}
                            className="w-full h-full object-cover rounded pointer-events-none"
                          />

                          {/* Delete Angle Button */}
                          {onRemoveFace && faceIndices.length > 1 && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                onRemoveFace(faceIdx);
                              }}
                              title="Remove angle"
                              className="absolute top-0.5 right-0.5 w-3.5 h-3.5 rounded bg-black/80 text-[8px] text-red-400 opacity-0 group-hover:opacity-100 flex items-center justify-center hover:bg-red-500 hover:text-white"
                            >
                              ✕
                            </button>
                          )}
                        </div>
                      );
                    })}

                    {/* Add Angle for this Person */}
                    {onAddAngle && (
                      <button
                        type="button"
                        onClick={() => onAddAngle(rank)}
                        title={`Capture current frame angle for ${personName}`}
                        className="shrink-0 w-12 aspect-square rounded-md border border-dashed border-white/20 hover:border-purple-400 text-white/40 hover:text-purple-300 flex items-center justify-center text-sm transition-colors"
                      >
                        +
                      </button>
                    )}
                  </div>
                </div>

                {/* Harvesting Footer Actions */}
                {onAutoAngles && (
                  <div className="flex justify-end pt-1 border-t border-white/5">
                    <button
                      type="button"
                      onClick={() => onAutoAngles(rank)}
                      disabled={busy}
                      title={`Scan video and auto-harvest missing angles for ${personName}`}
                      className="px-2 py-0.5 rounded bg-white/5 hover:bg-white/10 text-[10px] font-mono text-purple-300 flex items-center gap-1 transition-colors"
                    >
                      <span>⚡</span>
                      <span>Auto-Harvest Angles</span>
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        /* Empty State */
        <div className="flex flex-col items-center justify-center gap-1.5 p-4 rounded-lg border border-dashed border-white/15 text-center select-none">
          <span className="text-xl text-white/30">👤</span>
          <div className="space-y-0.5">
            <span className="text-[11px] font-semibold text-white/70">No Target Faces Banked</span>
            <p className="text-[10px] text-white/40">
              Click a face box in preview or run Auto-Capture to detect identities.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
