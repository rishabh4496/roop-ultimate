import React, { useMemo } from 'react';
import { trackingAdapter } from '../../adapters/trackingAdapter';

/**
 * High-precision SVG and HTML tracking overlay.
 * Maps native pixel detections to layout percentages without drift during viewport resize.
 */
export function TrackingOverlay({
  faces = [],
  kps = [],
  pose = [],
  personIds = [],
  selectedFaceIndex = null,
  faceMapping = {},
  imgDim = null,
  showDetections = true,
  _showTracking = true,
  showLandmarks = true,
  showLabels = true,
  overlayOpacity = 1.0,
  onSelectPerson,
}) {
  if (!imgDim || !imgDim.w || !imgDim.h) return null;

  // 1. Render Interactive HTML Bounding Boxes
  const boundingBoxes = useMemo(() => {
    if (!showDetections || !faces || !faces.length) return null;
    const isClickable = typeof onSelectPerson === 'function';

    return faces.map((bbox, i) => {
      const style = trackingAdapter.boxToPercentStyle(bbox, imgDim);
      if (!style) return null;

      const pId = personIds[i] ?? i;
      const isSelected = selectedFaceIndex === i || selectedFaceIndex === pId;
      const mappedSource = faceMapping[pId];
      const hasMapping = mappedSource !== undefined;

      // Status tone calculation
      const roleBadge = hasMapping ? `Target P${pId + 1} -> Src #${mappedSource + 1}` : `Person ${pId + 1}`;
      const borderColor = isSelected
        ? 'var(--accent-primary, #38bdf8)'
        : hasMapping
        ? '#a855f7'
        : 'rgba(251, 191, 36, 0.85)';

      const glowClass = isSelected
        ? 'shadow-[0_0_12px_rgba(56,189,248,0.5)] ring-2 ring-[var(--accent-primary,#38bdf8)]'
        : 'shadow-[0_0_8px_rgba(0,0,0,0.6)]';

      return (
        <div
          key={`bbox-${i}`}
          className={`absolute transition-all duration-150 rounded-sm z-20 ${
            isClickable
              ? 'pointer-events-auto cursor-pointer hover:bg-white/10 group/box'
              : 'pointer-events-none'
          } ${glowClass}`}
          style={{
            ...style,
            borderColor,
            borderWidth: isSelected ? '2px' : '1.5px',
            borderStyle: 'solid',
            opacity: overlayOpacity,
          }}
          title={isClickable ? `Click to capture/select Person ${pId + 1}` : undefined}
          onClick={
            isClickable
              ? (e) => {
                  e.stopPropagation();
                  onSelectPerson(i, pId);
                }
              : undefined
          }
        >
          {/* Corner Reticle Accents for Pro Workstation Look */}
          <div className="absolute -top-1 -left-1 w-2 h-2 border-t-2 border-l-2 border-inherit" />
          <div className="absolute -top-1 -right-1 w-2 h-2 border-t-2 border-r-2 border-inherit" />
          <div className="absolute -bottom-1 -left-1 w-2 h-2 border-b-2 border-l-2 border-inherit" />
          <div className="absolute -bottom-1 -right-1 w-2 h-2 border-b-2 border-r-2 border-inherit" />

          {/* Identification Chip */}
          {showLabels && (
            <div className="absolute -top-6 left-0 flex items-center gap-1 px-1.5 py-0.5 rounded bg-[#090a0f]/90 border border-white/15 backdrop-blur-md whitespace-nowrap shadow-lg pointer-events-none">
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{ backgroundColor: borderColor }}
              />
              <span className="text-[10px] font-mono font-bold tracking-tight text-white">
                {roleBadge}
              </span>
            </div>
          )}

          {/* Click to Add Indicator */}
          {isClickable && (
            <div className="absolute left-1/2 -translate-x-1/2 -bottom-6 opacity-0 group-hover/box:opacity-100 transition-opacity duration-150 px-2 py-0.5 rounded bg-black/90 border border-[var(--accent-primary,#38bdf8)]/40 text-[10px] font-semibold text-[var(--accent-primary,#38bdf8)] whitespace-nowrap pointer-events-none shadow-xl">
              + Select / Capture
            </div>
          )}
        </div>
      );
    });
  }, [
    faces,
    imgDim,
    personIds,
    selectedFaceIndex,
    faceMapping,
    showDetections,
    showLabels,
    overlayOpacity,
    onSelectPerson,
  ]);

  // 2. Render SVG ArcFace Landmarks & 3D Pose Vectors
  const landmarksOverlay = useMemo(() => {
    if (!showLandmarks || !kps || !kps.length) return null;

    const r = Math.max(imgDim.w, imgDim.h) / 320;
    const NAMES = ['eyeL', 'eyeR', 'nose', 'mouthL', 'mouthR'];
    const COLORS = ['#38bdf8', '#38bdf8', '#fbbf24', '#f472b6', '#f472b6'];

    return (
      <svg
        className="absolute inset-0 w-full h-full z-20 pointer-events-none"
        viewBox={`0 0 ${imgDim.w} ${imgDim.h}`}
        preserveAspectRatio="none"
        aria-hidden="true"
        style={{ opacity: overlayOpacity }}
      >
        {kps.map((k, i) => {
          if (!k || k.length < 5) return null;
          const [eyeL, eyeR, nose, mL, mR] = k;
          const p = pose?.[i];
          const poseText = trackingAdapter.formatPose(p);

          return (
            <g key={`landmarks-${i}`}>
              {/* Inter-ocular Eye Axis */}
              <line
                x1={eyeL[0]}
                y1={eyeL[1]}
                x2={eyeR[0]}
                y2={eyeR[1]}
                stroke="#38bdf8"
                strokeWidth={r * 0.5}
                opacity="0.8"
              />

              {/* Mouth Plane Axis */}
              <line
                x1={mL[0]}
                y1={mL[1]}
                x2={mR[0]}
                y2={mR[1]}
                stroke="#f472b6"
                strokeWidth={r * 0.5}
                opacity="0.8"
              />

              {/* Crop Rotation Symmetry Axis (Eye midpoint to Mouth midpoint) */}
              <line
                x1={(eyeL[0] + eyeR[0]) / 2}
                y1={(eyeL[1] + eyeR[1]) / 2}
                x2={(mL[0] + mR[0]) / 2}
                y2={(mL[1] + mR[1]) / 2}
                stroke="#a3e635"
                strokeWidth={r * 0.4}
                opacity="0.65"
                strokeDasharray={`${r * 2} ${r * 1.5}`}
              />

              {/* 5 Landmark Circles */}
              {k.map(([x, y], j) => (
                <circle
                  key={`pt-${j}`}
                  cx={x}
                  cy={y}
                  r={r}
                  fill={COLORS[j]}
                  stroke="#000"
                  strokeWidth={r * 0.3}
                  opacity="0.95"
                >
                  <title>{NAMES[j]}</title>
                </circle>
              ))}

              {/* 3D Head Pose Degree Vector Readout */}
              {poseText && (
                <text
                  x={nose[0]}
                  y={nose[1] - r * 4}
                  fill="#ffffff"
                  fontSize={Math.max(10, r * 4.5)}
                  fontFamily="monospace"
                  fontWeight="bold"
                  textAnchor="middle"
                  style={{ paintOrder: 'stroke' }}
                  stroke="#000000"
                  strokeWidth={r * 1.2}
                >
                  {poseText}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    );
  }, [showLandmarks, kps, pose, imgDim, overlayOpacity]);

  return (
    <>
      {boundingBoxes}
      {landmarksOverlay}
    </>
  );
}
