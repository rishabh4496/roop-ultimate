// The 9-cell angle matrix: one card per pose bin.
//
// Layout (3x3): the centre column is Looking up / Frontal / Looking down; each
// side column steps away from the centre row by row: quarter, half profile,
// profile. So the whole left column is the head turned to the viewer's left.
//
// IMAGES. Each crop is a URL on /api/angle-scan/image/<cache_key>/bin_N.jpg,
// served by the app's range responder (HTTP 206, ETag, If-Range, 304). The
// cache key covers the exact selection, so a URL never changes content: the
// browser can keep it, and a changed pick is a new URL rather than a stale one.
//
// JUMP moves Face Swap's timeline to the card's frame (1-based there; the
// server's frame_idx is the 0-based decoder index). RETAKE puts the card in
// assign mode: scrub the timeline, then Assign pins that frame to the bin.
import React from 'react';
import {
  BIN_BY_NAME, GRID_ORDER, fmtAngle, qualityTier, toTimelineFrame,
  type AngleBinName, type BinEntry, type PortfolioMap, type QualityTier,
} from './angleHudModel';

const TIER_STYLE: Record<QualityTier, string> = {
  Optimal: 'bg-cyan-400/15 text-cyan-200 border-cyan-300/40',
  Acceptable: 'bg-sky-400/10 text-sky-200 border-sky-300/30',
  Poor: 'bg-amber-400/15 text-amber-200 border-amber-300/40',
};

const STATUS_CHIP: Record<string, { text: string; cls: string }> = {
  nearest: { text: 'NEAR', cls: 'bg-amber-400/20 text-amber-200' },
  override: { text: 'MANUAL', cls: 'bg-violet-400/20 text-violet-200' },
};

export interface AngleMatrixGalleryProps {
  portfolio: PortfolioMap;
  selectedBin: AngleBinName | null;
  overrideBin: AngleBinName | null;
  currentFrame: number;                     // Face Swap timeline, 1-based
  busy: boolean;
  onSelect: (bin: AngleBinName) => void;
  onJump: (timelineFrame: number) => void;
  onBeginOverride: (bin: AngleBinName) => void;
  onAssign: (bin: AngleBinName, timelineFrame: number) => void;
  onCancelOverride: () => void;
  onClearOverride: (bin: AngleBinName) => void;
}

function Telemetry({ entry }: { entry: BinEntry }) {
  return (
    <div className="absolute inset-x-0 bottom-0 px-1 py-0.5 bg-slate-950/80 font-mono text-[9px] leading-tight text-cyan-100/90 tabular-nums">
      <div className="flex justify-between gap-1">
        <span title="Yaw (+ = turned toward the viewer's right)">Y{fmtAngle(entry.yaw)}</span>
        <span title="Pitch (+ = looking up)">P{fmtAngle(entry.pitch_up)}</span>
      </div>
      <div className="flex justify-between gap-1 text-cyan-100/60">
        <span title="Sharpness (Laplacian variance, 160 px crop)">S{entry.sharpness != null ? Math.round(entry.sharpness) : '—'}</span>
        <span title="Composite quality score">Q{entry.composite_score != null ? entry.composite_score.toFixed(2) : '—'}</span>
      </div>
    </div>
  );
}

export default function AngleMatrixGallery(props: AngleMatrixGalleryProps) {
  const {
    portfolio, selectedBin, overrideBin, currentFrame, busy,
    onSelect, onJump, onBeginOverride, onAssign, onCancelOverride, onClearOverride,
  } = props;
  return (
    <div className="grid grid-cols-3 gap-1.5" role="list" aria-label="Angle matrix" data-testid="angle-matrix">
      {GRID_ORDER.map((name) => {
        const spec = BIN_BY_NAME[name];
        const entry = portfolio[name];
        const tier = qualityTier(entry);
        const isSel = selectedBin === name;
        const assigning = overrideBin === name;
        const chip = entry ? STATUS_CHIP[entry.status] : undefined;
        const border = assigning
          ? 'border-violet-300/70'
          : entry
            ? (isSel ? 'border-cyan-300/80' : 'border-cyan-400/20 hover:border-cyan-300/50')
            : 'border-dashed border-amber-300/45';
        return (
          <div key={name} role="listitem" data-bin={name} data-status={entry ? entry.status : 'missing'}
               className={`rounded-lg border bg-slate-900/70 overflow-hidden flex flex-col ${border}`}>
            <button type="button" onClick={() => onSelect(name)} aria-pressed={isSel}
                    aria-label={`${spec.label}${entry ? `, frame ${toTimelineFrame(entry.frame_idx ?? 0)}` : ', missing'}`}
                    className="relative aspect-square w-full bg-slate-950 text-left">
              {entry?.url ? (
                <img src={entry.url} alt={`${spec.label} reference`} loading="lazy" decoding="async"
                     draggable={false} className="absolute inset-0 w-full h-full object-cover" />
              ) : (
                <div className="absolute inset-0 grid place-items-center text-center px-1">
                  <span className="font-mono text-[9px] uppercase tracking-wider text-amber-200/70">
                    {entry ? (entry.export_error || 'no crop') : 'no frame'}
                  </span>
                </div>
              )}
              <span className="absolute top-0.5 left-0.5 px-1 rounded bg-slate-950/80 font-mono text-[9px] font-semibold text-slate-200/90">
                {spec.short}
              </span>
              {tier && (
                <span className={`absolute top-0.5 right-0.5 px-1 rounded border font-mono text-[8px] uppercase ${TIER_STYLE[tier]}`}>
                  {tier}
                </span>
              )}
              {chip && (
                <span className={`absolute top-4 right-0.5 px-1 rounded font-mono text-[8px] ${chip.cls}`}
                      title={entry?.status === 'nearest'
                        ? `Borrowed from ${entry.distance_deg.toFixed(1)}° outside this bin`
                        : 'Picked by hand'}>
                  {chip.text}
                </span>
              )}
              {entry && <Telemetry entry={entry} />}
            </button>

            {assigning ? (
              <div className="flex flex-col gap-0.5 p-1">
                <button type="button" disabled={busy} onClick={() => onAssign(name, currentFrame)}
                        className="w-full rounded bg-violet-500/25 hover:bg-violet-500/40 text-violet-100 font-mono text-[9px] py-0.5 disabled:opacity-50">
                  Assign frame {currentFrame}
                </button>
                <button type="button" onClick={onCancelOverride}
                        className="w-full rounded text-slate-400 hover:text-slate-200 font-mono text-[9px]">
                  cancel
                </button>
              </div>
            ) : (
              <div className="flex">
                <button type="button" disabled={!entry || entry.frame_idx == null}
                        onClick={() => entry?.frame_idx != null && onJump(toTimelineFrame(entry.frame_idx))}
                        title={entry?.frame_idx != null ? `Jump to frame ${toTimelineFrame(entry.frame_idx)}` : 'No frame'}
                        className="flex-1 font-mono text-[9px] py-1 text-cyan-200/80 hover:text-cyan-100 hover:bg-cyan-400/10 disabled:opacity-30 disabled:hover:bg-transparent">
                  JUMP
                </button>
                {entry?.status === 'override' ? (
                  <button type="button" disabled={busy} onClick={() => onClearOverride(name)}
                          title="Remove the manual pick and use the automatic one"
                          className="flex-1 font-mono text-[9px] py-1 text-violet-200/80 hover:text-violet-100 hover:bg-violet-400/10 border-l border-white/5 disabled:opacity-40">
                    AUTO
                  </button>
                ) : (
                  <button type="button" disabled={busy} onClick={() => onBeginOverride(name)}
                          title="Scrub the timeline to a better frame and assign it to this bin"
                          className="flex-1 font-mono text-[9px] py-1 text-slate-300/80 hover:text-white hover:bg-white/5 border-l border-white/5 disabled:opacity-40">
                    RETAKE
                  </button>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
