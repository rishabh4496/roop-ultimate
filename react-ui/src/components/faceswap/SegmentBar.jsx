import React, { useState } from 'react';
import { Button } from '../ui';

// The segment strip under the timeline.
//
// Segments exist because a run has exactly one trim range (it lives on the
// target entry, not in the swap payload), so rendering three shots out of a
// long clip is three runs. This turns that into one gesture: mark the ranges,
// queue them in a batch, and — once they are done — join the pieces back into a
// single file with a stream copy.

const fmt = (f, fps) => {
  const total = Math.max(0, f) / (fps || 25);
  const m = Math.floor(total / 60);
  const s = Math.floor(total % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
};

export default function SegmentBar({
  segments,
  fps = 25,
  maxFrames = 1,
  startFrame = 1,
  endFrame = 1,
  onAdd,
  onRemove,
  onClear,
  onJump,
  onQueueAll,
  onJoin,
  joinable = 0,          // how many finished segment jobs are waiting to be joined
  coveredFrames = 0,
  busy = false,
}) {
  const [joining, setJoining] = useState(false);
  const spansWholeClip = startFrame <= 1 && endFrame >= maxFrames;

  const join = async () => {
    setJoining(true);
    try { await onJoin(); } finally { setJoining(false); }
  };

  return (
    <div className="mt-2 rounded-xl border border-white/10 bg-black/20 px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-micro font-bold uppercase tracking-wider text-white/45 mr-1">
          Segments
        </span>

        <Button
          size="sm"
          variant="secondary"
          onClick={() => onAdd(startFrame, endFrame)}
          disabled={busy || spansWholeClip}
          title={spansWholeClip
            ? 'Set In/Out to a shorter range first — a segment covering the whole clip is just a normal run'
            : `Add frames ${startFrame}–${endFrame} as a segment`}
        >
          Add In/Out as segment
        </Button>

        {segments.length > 0 && (
          <>
            <Button size="sm" variant="secondary" onClick={onQueueAll} disabled={busy}>
              ▶ Queue {segments.length} segment{segments.length === 1 ? '' : 's'}
            </Button>
            {joinable >= 2 && (
              <Button size="sm" variant="primary" onClick={join} disabled={joining}>
                {joining ? 'Joining…' : `Join ${joinable} rendered segments`}
              </Button>
            )}
            <Button size="sm" variant="ghost" className="text-red-400" onClick={onClear} disabled={busy}>
              Clear
            </Button>
          </>
        )}
      </div>

      {segments.length === 0 ? (
        <p className="text-mini text-white/45 mt-1.5 mb-0">
          Mark In/Out on the timeline and add it here to swap only the parts you need.
          Each segment renders as its own queued job; join them afterwards for one file.
        </p>
      ) : (
        <>
          <ul className="flex flex-wrap gap-1.5 mt-2 list-none m-0 p-0">
            {segments.map((s, i) => (
              <li key={s.id}>
                <span className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--accent)]/30 bg-[var(--accent)]/10 pl-2 pr-1 py-1 text-micro font-mono text-white/80">
                  <button
                    type="button"
                    onClick={() => onJump(s)}
                    className="hover:text-[var(--accent)] transition-colors"
                    title={`Set In/Out to frames ${s.start}–${s.end} and jump there`}
                    aria-label={`Segment ${i + 1}: frames ${s.start} to ${s.end}. Click to select.`}
                  >
                    {fmt(s.start, fps)}–{fmt(s.end, fps)}
                    <span className="opacity-45 ml-1">{s.end - s.start + 1}f</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onRemove(s.id)}
                    className="px-1 text-white/35 hover:text-red-400 transition-colors"
                    title="Remove this segment"
                    aria-label={`Remove segment ${i + 1}`}
                  >
                    ✕
                  </button>
                </span>
              </li>
            ))}
          </ul>
          <p className="text-micro text-white/45 mt-1.5 mb-0 tabular-nums">
            {coveredFrames.toLocaleString()} of {maxFrames.toLocaleString()} frames
            {maxFrames > 0 && ` (${Math.round((coveredFrames / maxFrames) * 100)}% of the clip)`}
            {' — the rest is not rendered.'}
          </p>
        </>
      )}
    </div>
  );
}
