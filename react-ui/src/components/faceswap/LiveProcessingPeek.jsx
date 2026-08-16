import React from 'react';
import { Icon } from '../../icons';

/**
 * LiveProcessingPeek
 * The image shown inside the processing box while a job runs.
 *
 * Two different things can be on screen here, and the badge says which:
 *
 *  - LIVE — the frame the pipeline most recently finished, republished about
 *    twice a second (`/api/live_frame`, keyed on `liveSeq` from /api/progress
 *    so the browser refetches exactly when there is a newer one). This is the
 *    render actually moving.
 *  - PREVIEW STILL — the fallback before the first live frame arrives, or with
 *    ROOP_LIVE_PREVIEW=0: the last preview rendered for the frame the timeline
 *    is parked on. It does NOT advance, so it must not claim to.
 */
export default function LiveProcessingPeek({
  previewSrc,
  rawUrl,
  liveSrc = '',
  frame = 1,
  maxFrames = 1,
  progressDesc = '',
  paused = false,
}) {
  // /api/live_frame answers 204 when nothing has been published yet, and a
  // reset between one poll and the fetch it triggered can land exactly there —
  // which would leave a broken-image icon in the box. Fall back to the still
  // for that seq instead, and retry naturally when the next seq arrives.
  const [failedSeq, setFailedSeq] = React.useState('');
  const isLive = !!liveSrc && liveSrc !== failedSeq;
  const activeImage = (isLive && liveSrc) || previewSrc || rawUrl;

  return (
    <div className="relative group overflow-hidden rounded-2xl border border-white/15 bg-black/60 shadow-xl backdrop-blur-md transition-all duration-300">
      {/* Media Frame Container */}
      <div className="relative aspect-video w-full flex items-center justify-center overflow-hidden bg-neutral-950">
        {activeImage ? (
          <img
            src={activeImage}
            alt={isLive ? 'Latest processed frame' : 'Preview still'}
            onError={() => { if (isLive) setFailedSeq(liveSrc); }}
            className="h-full w-full object-contain transition-all duration-300 transform-gpu"
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-neutral-500">
            <div className="h-8 w-8 rounded-full border-2 border-white/10 border-t-indigo-500 animate-spin" />
            <span className="text-xs">Waiting for the first processed frame…</span>
          </div>
        )}

        {/* Live Status Overlay Badges */}
        <div className="absolute top-3 left-3 flex items-center gap-2 z-10">
          <span className={`px-2.5 py-1 rounded-full text-micro font-bold uppercase tracking-wider backdrop-blur-md border ${
            paused
              ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
              : isLive
                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                : 'bg-white/10 text-neutral-300 border-white/20'
          } inline-flex items-center gap-1.5`}>
            {paused
              ? <><Icon.pause size={11} /> PAUSED</>
              : isLive
                ? <><Icon.live size={11} /> LIVE</>
                : <><Icon.still size={11} /> PREVIEW STILL</>}
          </span>

          {!isLive && maxFrames > 1 && (
            <span
              className="px-2.5 py-1 rounded-full text-micro font-mono font-bold bg-black/70 backdrop-blur-md border border-white/15 text-white"
              title="The frame this preview was rendered for — not the frame the render has reached"
            >
              of frame {frame} / {maxFrames}
            </span>
          )}
        </div>

        {/* Bottom Description Pill */}
        {progressDesc && (
          <div className="absolute bottom-3 left-3 right-3 z-10">
            <div className="px-3 py-1.5 rounded-xl bg-black/80 backdrop-blur-md border border-white/10 text-xs font-mono text-neutral-200 truncate">
              <span className="text-indigo-400 font-bold mr-2">STATE:</span>
              <span>{progressDesc}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
