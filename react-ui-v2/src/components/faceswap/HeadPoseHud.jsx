import React, { useEffect, useState } from 'react';
import { useApi } from '../../useApi';
import { reportFailure } from '../../failureLog';
import { subscribeThrottled, useThrottledSelector } from '../../store/liveSubscribe';
import { useTelemetryStore } from '../../store/telemetryStore';
import { CONVENTION, hudModel } from './headPoseModel';

// Live yaw / pitch / roll readouts for the heads in the frame being rendered.
//
// WHERE THE NUMBERS COME FROM. Not from a model run for display: `GET /api/trajectory/pose` answers from the
// Stage 2 pre-pass's smoothed trajectory cache (roop/trajectory_cache.py) - the same keypoints the swap aligns
// with, with the pose solved from THEM. The first frame of a job has none: the cache exists once the pre-pass has
// finished, and until then the HUD says so instead of showing zeros.
//
// COST. It lives in the Processing tab's live panels and in Face Swap's collapsible diagnostics section, so it is
// mounted (and polling) only while one of those is open. It follows `current_frame` off the telemetry store at no
// more than 2 Hz, makes ONE small request per change of frame, and re-renders ONLY ITSELF and only when the answer
// differs from the last one (the store is read directly, not through a state hook, so a frame number alone costs
// no render).

const DIAL_TONE = { yaw: 'text-emerald-400', pitch: 'text-orange-400', roll: 'text-blue-300' };

function Dial({ dial }) {
  return (
    <div className="space-y-1" data-dial={dial.name}>
      <div className="flex justify-between items-baseline">
        <span className="text-muted text-micro uppercase font-bold tracking-wider">{dial.label}</span>
        <span className={`${DIAL_TONE[dial.name]} font-bold`}>{dial.text}</span>
      </div>
      {/* The number carries the reading; the bar is a glance (centre = 0). */}
      <div className="relative w-full bg-white/10 h-1.5 rounded-full" aria-hidden="true">
        <div className="absolute left-1/2 top-0 h-full w-px bg-white/30" />
        {dial.pct != null && (
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-2.5 w-1 rounded-full bg-white"
               style={{ left: `${dial.pct}%` }} />
        )}
      </div>
    </div>
  );
}

/** The HUD for an already-computed model (pure: this is what the render check renders). */
export function HeadPoseHudView({ model }) {
  return (
    <section className="space-y-2 text-xs font-mono" aria-label="Head pose" data-testid="head-pose-hud"
             data-status={model.status}>
      <div className="text-micro text-white/55">{model.caption}</div>
      {model.faces.map((face) => (
        <div key={face.track} className="bg-black/25 p-3 rounded-xl border border-white/5 space-y-2.5"
             data-testid="head-pose-face">
          <div className="flex justify-between items-center text-micro">
            <span className="text-white font-semibold">{face.label}</span>
            {face.note && <span className="text-muted">{face.note}</span>}
          </div>
          <div className="grid grid-cols-3 gap-3">
            {face.dials.map((d) => <Dial key={d.name} dial={d} />)}
          </div>
        </div>
      ))}
      {model.status === 'live' && <p className="text-micro text-muted">{CONVENTION}</p>}
    </section>
  );
}

const frameOf = (state) => Math.max(0, Math.floor(Number(state.run.current_frame) || 0));
// The same answer again (or the same failure again) must not cost a render: keep the previous object.
const keep = (next) => (prev) => (JSON.stringify(prev) === JSON.stringify(next) ? prev : next);

export default function HeadPoseHud() {
  const { getJSON } = useApi();
  // A boolean, so a frame counter ticking never re-renders this component (only a job starting or ending does).
  const active = useThrottledSelector((s) => Number(s.run.total_frames) > 0, { hz: 1 });
  const [payload, setPayload] = useState(undefined);

  useEffect(() => {
    if (!active) return undefined;
    let cancelled = false;
    let asked = null;
    const ask = (state) => {
      const frame = frameOf(state);
      if (frame === asked) return;
      asked = frame;
      getJSON(`/api/trajectory/pose?frame=${frame}`, { timeout: 5000 })
        .then((p) => { if (!cancelled) setPayload(keep(p)); })
        .catch((err) => {
          if (cancelled) return;
          setPayload(keep({ available: false, reason: 'The trajectory could not be read.' }));
          reportFailure('Reading the head-pose trajectory', err, { toast: false });
        });
    };
    ask(useTelemetryStore.getState());
    const unsubscribe = subscribeThrottled(ask, 2);
    return () => { cancelled = true; unsubscribe(); };
  }, [active, getJSON]);

  return <HeadPoseHudView model={hudModel(payload, { active })} />;
}
