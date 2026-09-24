import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  Users, UserCheck, ShieldAlert, Sparkles, Sliders, Search, Trash2,
  RefreshCw, X, Clock, Film
} from 'lucide-react';
import { PERSON_COLORS } from '../constants';
import { cacheCrop, getCachedCrop } from './faceBankDb';
import { assignSource, removeSource, normalizeOverrides } from './mappingOps';

/**
 * Format a frame number into SMPTE timecode (HH:MM:SS:FF).
 */
function formatSmpte(frameIndex, fps = 25) {
  const f = Math.max(0, Number(frameIndex) || 0);
  const safeFps = Math.max(1, Number(fps) || 25);
  const totalSeconds = Math.floor(f / safeFps);
  const frames = Math.floor(f % safeFps);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}:${pad(frames)}`;
}

/**
 * Helper to get qualitative text feedback for cosine threshold slider.
 */
function getCosineFeedback(threshold) {
  const t = Number(threshold);
  if (t < 0.40) {
    return { label: 'Permissive (High Recall)', desc: 'Matches extreme angles; slight risk of identity cross-contamination.', color: 'text-amber-400' };
  }
  if (t <= 0.65) {
    return { label: 'Balanced (Recommended)', desc: 'High accuracy with good tolerance for side-profiles and lighting.', color: 'text-emerald-400' };
  }
  return { label: 'Strict (High Precision)', desc: 'Minimizes false positives; may miss steep profiles or motion blur.', color: 'text-sky-400' };
}

/**
 * Cached Thumbnail Component. Loads from IndexedDB or caches newly seen crops.
 */
function CachedFaceCrop({ cropKey, src, alt, className = '' }) {
  const [cachedUrl, setCachedUrl] = useState(src);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    let isMounted = true;
    let createdUrl = null;
    if (!cropKey) {
      setCachedUrl(src);
      return;
    }

    getCachedCrop(cropKey)
      .then((record) => {
        if (!isMounted) return;
        if (record && record.data) {
          if (typeof record.data === 'string') {
            setCachedUrl(record.data);
          } else if (record.data instanceof Blob) {
            createdUrl = URL.createObjectURL(record.data);
            setCachedUrl(createdUrl);
          }
        } else if (src) {
          // If not in cache but src is provided, cache it in IndexedDB
          setCachedUrl(src);
          cacheCrop(cropKey, src, { cachedAt: Date.now() }).catch(() => {});
        }
      })
      .catch(() => {
        if (isMounted) setCachedUrl(src);
      });

    return () => {
      isMounted = false;
      if (createdUrl) {
        URL.revokeObjectURL(createdUrl);
      }
    };
  }, [cropKey, src]);

  if (loadError || !cachedUrl) {
    return (
      <div className={`flex items-center justify-center bg-white/5 text-white/30 ${className}`}>
        <Users size={20} aria-hidden="true" />
      </div>
    );
  }

  return (
    <img
      src={cachedUrl}
      alt={alt || 'Face portrait'}
      className={`object-cover ${className}`}
      onError={() => setLoadError(true)}
      loading="lazy"
    />
  );
}

/**
 * Interactive Face Bank Router component.
 */
export default function FaceBankRouter({
  clusters = [],
  sourceLibrary = [],
  mapping = {},
  onMappingChange,
  parameterOverrides = {},
  onParametersChange,
  onFindInTimeline,
  onScanVideo,
  isScanning = false,
  fps = 25,
  className = '',
}) {
  // ── HOOK DECLARATIONS AT TOP ──────────────────────────────────────────────
  const [selectedClusterId, setSelectedClusterId] = useState(null);
  const [activeDragSource, setActiveDragSource] = useState(null);
  const [dragOverClusterId, setDragOverClusterId] = useState(null);
  const [pointerDragPos, setPointerDragPos] = useState(null);
  const [selectedSourceId, setSelectedSourceId] = useState(null);
  const [filterQuery, setFilterQuery] = useState('');
  const [minAppearances, setMinAppearances] = useState(1);
  const clusterCardRefs = useRef(new Map());

  // Currently selected cluster for the inspector drawer
  const selectedCluster = useMemo(() => {
    if (!selectedClusterId) return null;
    return clusters.find((c) => String(c.person_id || c.id) === String(selectedClusterId)) || null;
  }, [clusters, selectedClusterId]);

  // Normalized active overrides for the selected cluster
  const currentOverrides = useMemo(() => {
    if (!selectedCluster) return normalizeOverrides();
    const key = String(selectedCluster.person_id || selectedCluster.id);
    return normalizeOverrides(parameterOverrides[key]);
  }, [selectedCluster, parameterOverrides]);

  // Filtered clusters
  const filteredClusters = useMemo(() => {
    return clusters.filter((c) => {
      const name = (c.name || `Actor ${c.display_rank ?? ''}`).toLowerCase();
      const count = Number(c.cluster_size || c.count || 1);
      const matchesQuery = !filterQuery || name.includes(filterQuery.toLowerCase());
      const matchesCount = count >= minAppearances;
      return matchesQuery && matchesCount;
    });
  }, [clusters, filterQuery, minAppearances]);

  // Helper to get mapped sources for a cluster (supports 1-to-many and many-to-1)
  const getMappedSourcesForCluster = useCallback((clusterId) => {
    const raw = mapping[clusterId];
    if (raw === undefined || raw === null || raw === -1) return [];
    const sourceIds = Array.isArray(raw) ? raw : [raw];
    return sourceIds
      .map((srcId) => sourceLibrary.find((s, idx) => String(s.id ?? idx) === String(srcId) || idx === srcId))
      .filter(Boolean);
  }, [mapping, sourceLibrary]);

  // ── DRAG AND DROP HANDLERS (HTML5 + Touch/Pointer) ───────────────────────
  const handleDragStart = (e, sourceItem, sourceIdx) => {
    const payload = {
      sourceId: sourceItem.id ?? sourceIdx,
      sourceIndex: sourceIdx,
      sourceName: sourceItem.name || `Source #${sourceIdx + 1}`,
      sourceThumbnail: sourceItem.thumbnail,
    };
    setActiveDragSource(payload);
    if (e.dataTransfer) {
      e.dataTransfer.setData('application/json', JSON.stringify(payload));
      e.dataTransfer.effectAllowed = 'copyMove';
    }
  };

  const handleDragEnd = () => {
    setActiveDragSource(null);
    setDragOverClusterId(null);
    setPointerDragPos(null);
  };

  // Pointer drag for tablet/touch devices
  const handlePointerDownSource = (e, sourceItem, sourceIdx) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    const payload = {
      sourceId: sourceItem.id ?? sourceIdx,
      sourceIndex: sourceIdx,
      sourceName: sourceItem.name || `Source #${sourceIdx + 1}`,
      sourceThumbnail: sourceItem.thumbnail,
    };
    setActiveDragSource(payload);
    setPointerDragPos({ x: e.clientX, y: e.clientY });

    const handlePointerMove = (moveEvt) => {
      setPointerDragPos({ x: moveEvt.clientX, y: moveEvt.clientY });

      // Find element under pointer
      const elem = document.elementFromPoint(moveEvt.clientX, moveEvt.clientY);
      if (elem) {
        const clusterCard = elem.closest('[data-cluster-id]');
        if (clusterCard) {
          setDragOverClusterId(clusterCard.getAttribute('data-cluster-id'));
          return;
        }
      }
      setDragOverClusterId(null);
    };

    const handlePointerUp = (upEvt) => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);

      const elem = document.elementFromPoint(upEvt.clientX, upEvt.clientY);
      if (elem) {
        const clusterCard = elem.closest('[data-cluster-id]');
        if (clusterCard) {
          const targetClusterId = clusterCard.getAttribute('data-cluster-id');
          if (targetClusterId) {
            assignSourceToCluster(targetClusterId, payload.sourceId);
          }
        }
      }

      setActiveDragSource(null);
      setDragOverClusterId(null);
      setPointerDragPos(null);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
  };

  // Assign a source face to a target cluster
  const assignSourceToCluster = (clusterId, sourceId, append = false) => {
    if (!onMappingChange) return;
    onMappingChange(assignSource(mapping, clusterId, sourceId, append));
  };

  // Remove a specific source assignment from a target cluster
  const removeSourceFromCluster = (clusterId, sourceIdToRemove) => {
    if (!onMappingChange) return;
    if (mapping[String(clusterId)] === undefined) return;
    onMappingChange(removeSource(mapping, clusterId, sourceIdToRemove));
  };

  // HTML5 drop handler on cluster card
  const handleCardDrop = (e, clusterId) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOverClusterId(null);

    let sourceId = activeDragSource?.sourceId;
    if (!sourceId && e.dataTransfer) {
      try {
        const data = JSON.parse(e.dataTransfer.getData('application/json'));
        sourceId = data.sourceId ?? data.sourceIndex;
      } catch {
        // Ignored
      }
    }

    if (sourceId !== undefined && sourceId !== null) {
      assignSourceToCluster(clusterId, sourceId, e.shiftKey);
    }
    setActiveDragSource(null);
  };

  // Quick auto map (1-to-1 sequential)
  const handleQuickAutoMap = () => {
    if (!onMappingChange || !sourceLibrary.length || !clusters.length) return;
    const newMap = { ...mapping };
    clusters.forEach((c, idx) => {
      const cId = String(c.person_id || c.id || idx);
      const src = sourceLibrary[idx % sourceLibrary.length];
      newMap[cId] = src.id ?? (idx % sourceLibrary.length);
    });
    onMappingChange(newMap);
  };

  // Clear all mappings
  const handleClearAllMappings = () => {
    if (!onMappingChange) return;
    const newMap = {};
    clusters.forEach((c, idx) => {
      const cId = String(c.person_id || c.id || idx);
      newMap[cId] = -1;
    });
    onMappingChange(newMap);
  };

  // Update parameter overrides for a cluster
  const updateOverride = (field, value) => {
    if (!selectedCluster || !onParametersChange) return;
    const key = String(selectedCluster.person_id || selectedCluster.id);
    const updated = {
      ...currentOverrides,
      [field]: value,
    };
    onParametersChange(key, updated);
  };

  // Handle Find in Timeline
  const handleFindInTimelineClick = () => {
    if (!selectedCluster) return;
    const clusterId = selectedCluster.person_id || selectedCluster.id;
    const bestFrame = selectedCluster.best_frame ?? 0;
    const intervals = selectedCluster.timeline_intervals || [[bestFrame, bestFrame + 25]];
    if (onFindInTimeline) {
      onFindInTimeline(clusterId, intervals);
    }
  };

  return (
    <div className={`flex flex-col h-full bg-[#0A0B11] text-[#F4F5F7] overflow-hidden select-none ${className}`}>
      {/* ── HEADER / STATS TOOLBAR ────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b border-white/10 bg-white/[0.02]">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <Users size={18} aria-hidden="true" />
          </div>
          <div>
            <h2 className="text-compact font-semibold tracking-wide flex items-center gap-2">
              Face Bank Identity Router
              <span className="text-micro font-medium px-2 py-0.5 rounded-full bg-white/10 text-white/80">
                {clusters.length} Identities
              </span>
            </h2>
            <p className="text-mini text-white/50">
              Drag source portraits to map target identities (supports 1-to-many &amp; many-to-1)
            </p>
          </div>
        </div>

        {/* Global Action Buttons */}
        <div className="flex items-center gap-2">
          {onScanVideo && (
            <button
              type="button"
              onClick={onScanVideo}
              disabled={isScanning}
              aria-label="Scan video for faces"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-white/5 hover:bg-white/10 text-white/90 border border-white/10 text-mini font-medium transition-colors disabled:opacity-50"
            >
              <RefreshCw size={13} className={isScanning ? 'animate-spin' : ''} aria-hidden="true" />
              <span>{isScanning ? 'Scanning Video...' : 'Scan Video'}</span>
            </button>
          )}

          <button
            type="button"
            onClick={handleQuickAutoMap}
            disabled={!clusters.length || !sourceLibrary.length}
            aria-label="Auto-map source faces 1 to 1"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/30 text-mini font-medium transition-colors disabled:opacity-40"
          >
            <Sparkles size={13} aria-hidden="true" />
            <span>Auto-Map (1:1)</span>
          </button>

          <button
            type="button"
            onClick={handleClearAllMappings}
            disabled={!clusters.length}
            aria-label="Clear all face mappings"
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-white/5 hover:bg-white/10 text-white/60 hover:text-white/90 border border-white/10 text-mini font-medium transition-colors disabled:opacity-40"
          >
            <Trash2 size={13} aria-hidden="true" />
            <span>Clear Mappings</span>
          </button>
        </div>
      </div>

      {/* ── TOP SECTION: SOURCE LIBRARY (DRAGGABLE PALETTE) ───────────────── */}
      <div className="px-4 py-2.5 border-b border-white/10 bg-white/[0.015]">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="text-micro font-semibold uppercase tracking-wider text-white/50">Source Library</span>
            <span className="text-nano px-1.5 py-0.5 rounded bg-white/10 text-white/70">
              {sourceLibrary.length} Faces
            </span>
          </div>
          <span className="text-nano text-white/40">
            Tip: Click to select, or drag directly onto target cluster cards below
          </span>
        </div>

        {sourceLibrary.length === 0 ? (
          <div className="flex items-center justify-center p-4 rounded-lg border border-dashed border-white/10 text-mini text-white/40 bg-white/[0.01]">
            <span>No source faces loaded. Import portrait photos in the Source Gallery above.</span>
          </div>
        ) : (
          <div className="flex items-center gap-3 overflow-x-auto pb-1 scrollbar-thin">
            {sourceLibrary.map((srcItem, sIdx) => {
              const srcId = srcItem.id ?? sIdx;
              const isSelected = String(selectedSourceId) === String(srcId);
              const isDragging = activeDragSource?.sourceId === srcId;

              return (
                <div
                  key={`src-${srcId}`}
                  draggable
                  onDragStart={(e) => handleDragStart(e, srcItem, sIdx)}
                  onDragEnd={handleDragEnd}
                  onPointerDown={(e) => handlePointerDownSource(e, srcItem, sIdx)}
                  onClick={() => setSelectedSourceId(isSelected ? null : srcId)}
                  className={`group relative flex items-center gap-2.5 p-1.5 pr-3 rounded-lg border transition-all cursor-grab active:cursor-grabbing shrink-0 ${
                    isSelected
                      ? 'bg-rose-500/20 border-rose-500 shadow-md shadow-rose-500/10'
                      : isDragging
                      ? 'opacity-40 border-dashed border-rose-400'
                      : 'bg-white/5 hover:bg-white/10 border-white/10 hover:border-white/20'
                  }`}
                >
                  <div className="relative w-10 h-10 rounded-md overflow-hidden bg-black/40 border border-white/15">
                    <CachedFaceCrop
                      cropKey={`source_${srcId}`}
                      src={srcItem.thumbnail}
                      alt={srcItem.name || `Source #${sIdx + 1}`}
                      className="w-full h-full"
                    />
                    <div className="absolute bottom-0 right-0 px-1 rounded-tl bg-black/80 text-nano font-mono text-white/90">
                      #{sIdx + 1}
                    </div>
                  </div>

                  <div className="flex flex-col text-left">
                    <span className="text-micro font-medium text-white/90 truncate max-w-[90px]">
                      {srcItem.name || `Source ${sIdx + 1}`}
                    </span>
                    <span className="text-nano text-white/40">Drag to assign</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ── MAIN CONTENT AREA: CLUSTER TRAY + INSPECTOR DRAWER ─────────────── */}
      <div className="flex-1 flex overflow-hidden">
        {/* LEFT / CENTER: CLUSTER TRAY */}
        <div className="flex-1 flex flex-col min-w-0 overflow-y-auto p-4 scrollbar-thin">
          {/* Filter Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-white/40" aria-hidden="true" />
                <input
                  type="text"
                  placeholder="Filter identities..."
                  value={filterQuery}
                  onChange={(e) => setFilterQuery(e.target.value)}
                  className="pl-8 pr-3 py-1 rounded-md bg-white/5 border border-white/10 text-mini placeholder:text-white/30 focus:outline-none focus:border-rose-500/50 w-44"
                />
              </div>

              <div className="flex items-center gap-1.5 text-mini text-white/60 ml-2">
                <span className="text-micro">Min Appearances:</span>
                <input
                  type="number"
                  min="1"
                  max="1000"
                  value={minAppearances}
                  onChange={(e) => setMinAppearances(Math.max(1, parseInt(e.target.value, 10) || 1))}
                  className="w-14 px-2 py-0.5 rounded bg-white/5 border border-white/10 text-mini text-center focus:outline-none"
                />
              </div>
            </div>

            <div className="text-nano text-white/40">
              Showing {filteredClusters.length} of {clusters.length} detected identities
            </div>
          </div>

          {/* Grid of Cluster Cards */}
          {filteredClusters.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center p-12 text-center border border-dashed border-white/10 rounded-xl bg-white/[0.01]">
              <Users size={36} className="text-white/20 mb-3" aria-hidden="true" />
              <h3 className="text-plain font-medium text-white/70 mb-1">No Face Clusters Detected</h3>
              <p className="text-mini text-white/40 max-w-sm">
                Click &quot;Scan Video&quot; to perform an automated face detection and DBSCAN clustering pass across the target clip.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-3.5">
              {filteredClusters.map((cluster, cIdx) => {
                const clusterId = String(cluster.person_id || cluster.id || cIdx);
                const displayRank = cluster.display_rank ?? cIdx;
                const accentColor = PERSON_COLORS[displayRank % PERSON_COLORS.length] || '#E94560';
                const isSelected = selectedClusterId === clusterId;
                const isDropTarget = dragOverClusterId === clusterId;
                const mappedSources = getMappedSourcesForCluster(clusterId);
                const overrides = parameterOverrides[clusterId] || {};
                const action = overrides.action || 'swap';
                const appearances = Number(cluster.cluster_size || cluster.count || 1);
                const firstFrame = cluster.best_frame ?? 0;
                const confidence = Math.round((Number(cluster.det_score || 0.95)) * 100);
                const offAxis = Math.round(Number(cluster.off_axis || 0));

                return (
                  <div
                    key={`cluster-${clusterId}`}
                    data-cluster-id={clusterId}
                    ref={(el) => {
                      if (el) clusterCardRefs.current.set(clusterId, el);
                      else clusterCardRefs.current.delete(clusterId);
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      if (dragOverClusterId !== clusterId) setDragOverClusterId(clusterId);
                    }}
                    onDragLeave={() => {
                      if (dragOverClusterId === clusterId) setDragOverClusterId(null);
                    }}
                    onDrop={(e) => handleCardDrop(e, clusterId)}
                    onClick={() => {
                      if (selectedSourceId !== null) {
                        assignSourceToCluster(clusterId, selectedSourceId);
                      } else {
                        setSelectedClusterId(isSelected ? null : clusterId);
                      }
                    }}
                    className={`group relative flex flex-col rounded-xl border transition-all overflow-hidden cursor-pointer ${
                      isSelected
                        ? 'border-2 shadow-lg bg-white/[0.06]'
                        : isDropTarget
                        ? 'border-dashed border-rose-400 bg-rose-500/10 scale-[1.02]'
                        : 'border-white/10 hover:border-white/25 bg-white/[0.03] hover:bg-white/[0.05]'
                    }`}
                    style={{
                      borderColor: isSelected ? accentColor : undefined,
                      boxShadow: isSelected ? `0 0 16px ${accentColor}25` : undefined,
                    }}
                  >
                    {/* Top Identity Color Accent Strip */}
                    <div className="h-1 w-full" style={{ backgroundColor: accentColor }} />

                    {/* Card Body */}
                    <div className="p-3 flex flex-col gap-2.5">
                      {/* Portrait & Core Stats Row */}
                      <div className="flex gap-2.5 items-start">
                        {/* High-res cropped portrait thumbnail */}
                        <div className="relative w-16 h-16 rounded-lg overflow-hidden shrink-0 bg-black/50 border border-white/15 shadow-inner">
                          <CachedFaceCrop
                            cropKey={`cluster_${clusterId}`}
                            src={cluster.thumbnail}
                            alt={cluster.name || `Actor ${displayRank + 1}`}
                            className="w-full h-full"
                          />
                          <div
                            className="absolute top-1 left-1 px-1 rounded text-nano font-bold text-white shadow-sm"
                            style={{ backgroundColor: accentColor }}
                          >
                            #{displayRank + 1}
                          </div>
                        </div>

                        {/* Metadata Details */}
                        <div className="flex-1 min-w-0 flex flex-col justify-between py-0.5">
                          <div className="flex items-center justify-between gap-1">
                            <span className="text-compact font-semibold text-white/95 truncate">
                              {cluster.name || `Actor ${displayRank + 1}`}
                            </span>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedClusterId(isSelected ? null : clusterId);
                              }}
                              aria-label={`Inspect Actor ${displayRank + 1}`}
                              className="p-1 rounded text-white/40 hover:text-white/90 hover:bg-white/10 transition-colors"
                            >
                              <Sliders size={13} aria-hidden="true" />
                            </button>
                          </div>

                          {/* Appearance Frequency */}
                          <div className="flex items-center gap-1.5 text-mini text-white/70">
                            <Film size={11} className="text-white/40" aria-hidden="true" />
                            <span>{appearances} frames</span>
                          </div>

                          {/* First Detected Timestamp */}
                          <div className="flex items-center gap-1.5 text-micro font-mono text-white/50">
                            <Clock size={10} className="text-white/40" aria-hidden="true" />
                            <span>{formatSmpte(firstFrame, fps)} (F#{firstFrame})</span>
                          </div>
                        </div>
                      </div>

                      {/* Confidence & Pose Angle Bar */}
                      <div className="flex items-center justify-between px-2 py-1 rounded bg-white/[0.025] border border-white/5 text-nano">
                        <div className="flex items-center gap-1">
                          <span className="text-white/40">Confidence:</span>
                          <span className={`font-semibold ${confidence >= 85 ? 'text-emerald-400' : confidence >= 70 ? 'text-amber-400' : 'text-rose-400'}`}>
                            {confidence}%
                          </span>
                        </div>
                        <div className="flex items-center gap-1 text-white/40">
                          <span>Pose:</span>
                          <span className="font-mono text-white/70">{offAxis}&deg; off-axis</span>
                        </div>
                      </div>

                      {/* Action Mode & Assigned Source Mapping Badge */}
                      <div className="pt-1 border-t border-white/5">
                        {action === 'keep' ? (
                          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 text-micro font-medium">
                            <UserCheck size={12} aria-hidden="true" />
                            <span>Keep Original (Bypassed)</span>
                          </div>
                        ) : action === 'censor' ? (
                          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20 text-micro font-medium">
                            <ShieldAlert size={12} aria-hidden="true" />
                            <span>Blur / Censor Face</span>
                          </div>
                        ) : mappedSources.length > 0 ? (
                          <div className="flex flex-col gap-1">
                            <div className="flex items-center justify-between text-nano text-white/40">
                              <span>Mapped Sources ({mappedSources.length})</span>
                              <span className="text-nano text-white/30">Drag to replace</span>
                            </div>
                            <div className="flex flex-wrap gap-1">
                              {mappedSources.map((s, idx) => (
                                <div
                                  key={`mapped-${clusterId}-${s.id ?? idx}`}
                                  className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-rose-500/15 border border-rose-500/30 text-rose-200 text-micro font-medium group/chip"
                                >
                                  <span className="truncate max-w-[70px]">{s.name || `Source #${idx + 1}`}</span>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      removeSourceFromCluster(clusterId, s.id ?? idx);
                                    }}
                                    aria-label={`Unmap ${s.name || 'Source'}`}
                                    className="hover:text-white p-0.5"
                                  >
                                    <X size={10} aria-hidden="true" />
                                  </button>
                                </div>
                              ))}
                            </div>
                          </div>
                        ) : (
                          <div
                            className={`flex items-center justify-center p-2 rounded-lg border text-micro transition-colors ${
                              isDropTarget
                                ? 'bg-rose-500/20 border-rose-400 text-rose-200 font-semibold'
                                : 'border-dashed border-white/15 text-white/40 hover:text-white/70 hover:border-white/25'
                            }`}
                          >
                            <span>{isDropTarget ? 'Drop Source Here' : '+ Drop Source to Swap'}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* ── RIGHT: CLUSTER INSPECTOR DRAWER ──────────────────────────────── */}
        {selectedCluster && (
          <div className="w-80 md:w-96 border-l border-white/10 bg-[#0E1017] flex flex-col shrink-0 animate-in slide-in-from-right duration-200 overflow-y-auto scrollbar-thin">
            {/* Drawer Header */}
            <div className="flex items-center justify-between p-4 border-b border-white/10 bg-white/[0.02]">
              <div className="flex items-center gap-2.5">
                <div
                  className="w-3.5 h-3.5 rounded-full"
                  style={{
                    backgroundColor: PERSON_COLORS[(selectedCluster.display_rank ?? 0) % PERSON_COLORS.length] || '#E94560',
                  }}
                />
                <h3 className="text-plain font-semibold text-white/95">
                  Cluster Inspector
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setSelectedClusterId(null)}
                aria-label="Close inspector drawer"
                className="p-1 rounded-md text-white/40 hover:text-white hover:bg-white/10 transition-colors"
              >
                <X size={16} aria-hidden="true" />
              </button>
            </div>

            <div className="p-4 flex flex-col gap-5 flex-1">
              {/* Actor Identity Profile Card */}
              <div className="flex gap-3 p-3 rounded-xl bg-white/[0.03] border border-white/10">
                <div className="w-16 h-16 rounded-lg overflow-hidden shrink-0 bg-black/60 border border-white/15">
                  <CachedFaceCrop
                    cropKey={`cluster_${selectedCluster.person_id || selectedCluster.id}`}
                    src={selectedCluster.thumbnail}
                    alt={selectedCluster.name || 'Actor'}
                    className="w-full h-full"
                  />
                </div>
                <div className="flex-1 min-w-0 flex flex-col justify-center">
                  <span className="text-lead font-semibold text-white/95 truncate">
                    {selectedCluster.name || `Actor ${(selectedCluster.display_rank ?? 0) + 1}`}
                  </span>
                  <span className="text-mini text-white/50">
                    ID: {selectedCluster.person_id || selectedCluster.id}
                  </span>
                  <div className="flex items-center gap-2 mt-1 text-nano text-white/40">
                    <span>{selectedCluster.cluster_size || selectedCluster.count || 1} occurrences</span>
                    <span>&bull;</span>
                    <span>Frame {selectedCluster.best_frame ?? 0}</span>
                  </div>
                </div>
              </div>

              {/* Action Mode Toggle */}
              <div className="flex flex-col gap-1.5">
                <label className="text-micro font-semibold uppercase tracking-wider text-white/60">
                  Target Behavior
                </label>
                <div className="grid grid-cols-3 gap-1.5 p-1 rounded-lg bg-black/40 border border-white/10">
                  <button
                    type="button"
                    onClick={() => updateOverride('action', 'swap')}
                    aria-label="Swap face mode"
                    className={`px-2 py-1.5 rounded-md text-mini font-medium transition-all ${
                      currentOverrides.action === 'swap'
                        ? 'bg-rose-500 text-white shadow-sm'
                        : 'text-white/60 hover:text-white/90'
                    }`}
                  >
                    Swap Face
                  </button>

                  <button
                    type="button"
                    onClick={() => updateOverride('action', 'keep')}
                    aria-label="Keep original face mode"
                    className={`px-2 py-1.5 rounded-md text-mini font-medium transition-all ${
                      currentOverrides.action === 'keep'
                        ? 'bg-emerald-600 text-white shadow-sm'
                        : 'text-white/60 hover:text-white/90'
                    }`}
                  >
                    Keep Original
                  </button>

                  <button
                    type="button"
                    onClick={() => updateOverride('action', 'censor')}
                    aria-label="Blur or censor face mode"
                    className={`px-2 py-1.5 rounded-md text-mini font-medium transition-all ${
                      currentOverrides.action === 'censor'
                        ? 'bg-amber-600 text-white shadow-sm'
                        : 'text-white/60 hover:text-white/90'
                    }`}
                  >
                    Blur / Censor
                  </button>
                </div>
              </div>

              {/* Cosine Similarity Threshold Slider */}
              <div className="flex flex-col gap-2 p-3 rounded-xl bg-white/[0.02] border border-white/10">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className="text-mini font-medium text-white/90">Cosine Match Threshold</span>
                    <span className="text-nano px-1.5 py-0.5 rounded bg-white/10 font-mono text-white/80">
                      {currentOverrides.cosineThreshold.toFixed(2)}
                    </span>
                  </div>
                  <span className={`text-nano font-medium ${getCosineFeedback(currentOverrides.cosineThreshold).color}`}>
                    {getCosineFeedback(currentOverrides.cosineThreshold).label}
                  </span>
                </div>

                <input
                  type="range"
                  min="0.30"
                  max="0.85"
                  step="0.01"
                  value={currentOverrides.cosineThreshold}
                  onChange={(e) => updateOverride('cosineThreshold', parseFloat(e.target.value))}
                  className="w-full accent-rose-500 cursor-pointer"
                />

                <div className="flex justify-between text-nano font-mono text-white/40">
                  <span>0.30 (Permissive)</span>
                  <span>0.60 (Default)</span>
                  <span>0.85 (Strict)</span>
                </div>

                <p className="text-nano text-white/50 leading-relaxed">
                  {getCosineFeedback(currentOverrides.cosineThreshold).desc}
                </p>
              </div>

              {/* Mask Erosion / Dilation Offset Slider */}
              <div className="flex flex-col gap-2 p-3 rounded-xl bg-white/[0.02] border border-white/10">
                <div className="flex items-center justify-between">
                  <span className="text-mini font-medium text-white/90">Custom Mask Erosion / Dilation</span>
                  <span className="text-nano px-1.5 py-0.5 rounded bg-white/10 font-mono text-white/80">
                    {currentOverrides.maskOffset > 0 ? `+${currentOverrides.maskOffset}` : currentOverrides.maskOffset}px
                  </span>
                </div>

                <input
                  type="range"
                  min="-20"
                  max="20"
                  step="1"
                  value={currentOverrides.maskOffset}
                  onChange={(e) => updateOverride('maskOffset', parseInt(e.target.value, 10))}
                  className="w-full accent-rose-500 cursor-pointer"
                />

                <div className="flex justify-between text-nano font-mono text-white/40">
                  <span>-20px (Erode / Tighter)</span>
                  <span>0px (Neutral)</span>
                  <span>+20px (Dilate / Softer)</span>
                </div>

                <p className="text-nano text-white/50 leading-relaxed">
                  Negative values contract the boundary inwards to prevent fringe hair or occluder artefacts. Positive values expand outward for softer skin integration.
                </p>
              </div>

              {/* Assigned Source Identities for this Actor */}
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <span className="text-micro font-semibold uppercase tracking-wider text-white/60">
                    Assigned Source Identity
                  </span>
                  <span className="text-nano text-white/40">
                    {getMappedSourcesForCluster(selectedCluster.person_id || selectedCluster.id).length} assigned
                  </span>
                </div>

                {getMappedSourcesForCluster(selectedCluster.person_id || selectedCluster.id).length === 0 ? (
                  <div className="p-3 rounded-lg border border-dashed border-white/10 text-mini text-white/40 text-center bg-white/[0.01]">
                    No source assigned. Drag a portrait from the Source Library above.
                  </div>
                ) : (
                  <div className="flex flex-col gap-1.5">
                    {getMappedSourcesForCluster(selectedCluster.person_id || selectedCluster.id).map((s, idx) => (
                      <div
                        key={`inspector-mapped-${s.id ?? idx}`}
                        className="flex items-center justify-between p-2 rounded-lg bg-white/5 border border-white/10"
                      >
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 rounded overflow-hidden bg-black/50 border border-white/15">
                            <CachedFaceCrop
                              cropKey={`source_${s.id ?? idx}`}
                              src={s.thumbnail}
                              alt={s.name}
                              className="w-full h-full"
                            />
                          </div>
                          <span className="text-mini font-medium text-white/90">
                            {s.name || `Source #${idx + 1}`}
                          </span>
                        </div>
                        <button
                          type="button"
                          onClick={() => removeSourceFromCluster(selectedCluster.person_id || selectedCluster.id, s.id ?? idx)}
                          aria-label={`Remove ${s.name || 'Source'} from this cluster`}
                          className="p-1 rounded text-white/40 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                        >
                          <Trash2 size={13} aria-hidden="true" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="pt-2 flex flex-col gap-2 mt-auto">
                <button
                  type="button"
                  onClick={handleFindInTimelineClick}
                  aria-label="Find this actor in timeline"
                  className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 border border-rose-500/40 text-mini font-medium transition-all shadow-sm"
                >
                  <Search size={14} aria-hidden="true" />
                  <span>Find in Timeline</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Floating Pointer Drag Ghost (Touch/Pointer Drag Overlay) */}
      {pointerDragPos && activeDragSource && (
        <div
          className="fixed pointer-events-none z-50 transform -translate-x-1/2 -translate-y-1/2 flex items-center gap-2 p-1.5 pr-2.5 rounded-lg bg-rose-500 text-white shadow-2xl border border-white/30 text-mini font-semibold"
          style={{ left: `${pointerDragPos.x}px`, top: `${pointerDragPos.y}px` }}
        >
          <div className="w-6 h-6 rounded overflow-hidden bg-black/40">
            <img src={activeDragSource.sourceThumbnail} alt="" className="w-full h-full object-cover" />
          </div>
          <span>{activeDragSource.sourceName}</span>
        </div>
      )}
    </div>
  );
}
