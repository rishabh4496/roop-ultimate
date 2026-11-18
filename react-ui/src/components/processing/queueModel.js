// Pure view-model for the Render Queue mission control (#/processing).
//
// Everything the cards, the table, the speedometers and the worker rack show is
// derived here from the server's own snapshot (/api/queue, /api/queue/segments).
// Nothing is estimated locally: a missing fps or ETA is `null` and renders as a
// dash, because a speedometer that shows a made-up number is worse than none.

import {
  ACTIVE_STATES, QUEUE_STATE_LABEL, RUNNABLE_STATES, TERMINAL_STATES, jobState,
} from '../faceswap/queueStates';

/**
 * The ten backend states (routes_queue.JOB_STATES), each with one semantic tone.
 * `tone` picks the badge colour family; `motion` says whether the badge pulses.
 * INTERRUPTED and RECOVERABLE are "can be resumed", never styled as failures.
 */
export const STATE_BADGE = {
  QUEUED: { tone: 'muted', motion: 'none', hint: 'Waiting its turn' },
  PREPARING: { tone: 'sky', motion: 'pulse', hint: 'Loading models and inputs' },
  PROCESSING: { tone: 'live', motion: 'pulse', hint: 'Rendering now' },
  PAUSE_REQUESTED: { tone: 'amber', motion: 'pulse', hint: 'Waiting for a safe checkpoint to pause' },
  PAUSED: { tone: 'amber', motion: 'none', hint: 'Paused at a safe checkpoint' },
  COMPLETED: { tone: 'ok', motion: 'none', hint: 'Finished' },
  FAILED: { tone: 'bad', motion: 'none', hint: 'Stopped by an error' },
  CANCELLED: { tone: 'stop', motion: 'none', hint: 'Cancelled by you' },
  INTERRUPTED: { tone: 'resume', motion: 'none', hint: 'The app stopped mid-render; can be picked up again' },
  RECOVERABLE: { tone: 'resume', motion: 'none', hint: 'Has a safe checkpoint; Start resumes it' },
};

/** Tailwind classes per tone, for the badge and the card edge. */
export const TONE_CLASS = {
  muted: { badge: 'text-white/60 bg-white/5 border-white/10', edge: 'border-white/10' },
  sky: { badge: 'text-sky-300 bg-sky-500/10 border-sky-400/30', edge: 'border-sky-400/30' },
  live: { badge: 'text-yellow-300 bg-yellow-500/10 border-yellow-400/40', edge: 'border-yellow-400/50' },
  amber: { badge: 'text-amber-300 bg-amber-500/10 border-amber-400/30', edge: 'border-amber-400/30' },
  ok: { badge: 'text-emerald-300 bg-emerald-500/10 border-emerald-400/30', edge: 'border-emerald-400/25' },
  bad: { badge: 'text-red-300 bg-red-500/10 border-red-400/30', edge: 'border-red-400/35' },
  stop: { badge: 'text-orange-300 bg-orange-500/10 border-orange-400/30', edge: 'border-orange-400/25' },
  resume: { badge: 'text-violet-300 bg-violet-500/10 border-violet-400/30', edge: 'border-violet-400/25' },
};

export const stateBadge = (st) => ({
  state: st,
  label: QUEUE_STATE_LABEL[st] || st,
  ...(STATE_BADGE[st] || { tone: 'muted', motion: 'none', hint: 'Unknown state' }),
});

/** A fraction clamped to 0..1, or null when the server did not send one. */
export function clamp01(v) {
  const n = Number(v);
  if (v === null || v === undefined || !Number.isFinite(n)) return null;
  return Math.min(1, Math.max(0, n));
}

/** Seconds as h:mm:ss / m:ss, or null for "unknown". */
export function fmtEta(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s) || s <= 0) return null;
  const total = Math.round(s);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

/** Frames per second with one decimal below 10, whole numbers above; null for unknown. */
export function fmtFps(fps) {
  const n = Number(fps);
  if (!Number.isFinite(n) || n <= 0) return null;
  return n < 10 ? n.toFixed(1) : String(Math.round(n));
}

/**
 * The top of a speedometer's scale: the next round number above the reading,
 * so the needle always sits inside the dial. Null means there is no reading.
 */
export function gaugeMax(fps) {
  const n = Number(fps);
  if (!Number.isFinite(n) || n <= 0) return null;
  const steps = [5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960];
  const want = n * 1.2;
  return steps.find((s) => s >= want) ?? Math.ceil(want / 100) * 100;
}

/** The dial's fill fraction for a reading against its scale. */
export function gaugeFraction(fps) {
  const max = gaugeMax(fps);
  if (max === null) return 0;
  return Math.min(1, Math.max(0, Number(fps) / max));
}

/**
 * One job as a card: identity, live numbers, what state it is in, and which
 * actions the backend will accept for it. The rules mirror QueuePanel, so a
 * button shown here is one the server will take.
 */
export function jobCard(job, index, { currentId = null, queueRunning = false } = {}) {
  const st = jobState(job);
  const isCurrent = !!currentId && job.id === currentId;
  const isActive = ACTIVE_STATES.includes(st);
  const editable = !isCurrent && !isActive;
  const prog = job.progress || {};
  const fraction = clamp01(prog.fraction);
  const framesDone = Number.isFinite(Number(prog.frames_done)) ? Number(prog.frames_done) : null;
  const framesTotal = Number.isFinite(Number(prog.frames_total)) ? Number(prog.frames_total) : null;
  const fps = Number.isFinite(Number(prog.fps)) && Number(prog.fps) > 0 ? Number(prog.fps) : null;
  const etaS = Number.isFinite(Number(prog.eta_s)) && Number(prog.eta_s) > 0 ? Number(prog.eta_s) : null;
  const hasFrames = job.frame_start != null || job.frame_end != null;
  const isTerminal = TERMINAL_STATES.includes(st);
  return {
    id: job.id,
    index,
    number: index + 1,
    label: job.label || job.target_name || `Job ${index + 1}`,
    target: job.target_name || '',
    source: job.source_name || (job.source_index != null ? `Faceset #${Number(job.source_index) + 1}` : ''),
    swapModel: job.payload?.swap_model || 'inswapper',
    enhancer: job.payload?.enhancer || 'no enhancer',
    frameRange: hasFrames ? `frames ${job.frame_start ?? 1}–${job.frame_end ?? '∞'}` : '',
    state: st,
    badge: stateBadge(st),
    isCurrent,
    isActive,
    isTerminal,
    // Stage progress: the server's fraction for the whole job, and the phase
    // text it reports ("Analyzing", "Encoding"...). Null fraction = no bar.
    fraction,
    pct: fraction === null ? null : Math.round(fraction * 100),
    phase: prog.phase || '',
    framesDone,
    framesTotal,
    fps,
    etaS,
    paused: !!prog.paused,
    error: job.error || '',
    projectId: job.project_id || '',
    // Speedometer reads only while this job is actually producing frames.
    live: isCurrent && queueRunning && !prog.paused,
    canDrag: editable && !queueRunning,
    actions: {
      duplicate: editable,
      // Same rule as QueuePanel: any finished job can be run again.
      retry: editable && isTerminal,
      cancel: isActive || (!isCurrent && !isTerminal),
      cancelNeedsConfirm: isCurrent,
      remove: editable,
      loadCheckpoint: !!job.project_id,
    },
  };
}

/**
 * The reordered id list after dropping `fromId` onto `toId` (before or after it),
 * or null when the move is a no-op or refers to an unknown job.
 */
export function reorderIds(jobs, fromId, toId, after = false) {
  if (!fromId || !toId || fromId === toId) return null;
  const ids = jobs.map((j) => j.id);
  if (!ids.includes(fromId) || !ids.includes(toId)) return null;
  const rest = ids.filter((id) => id !== fromId);
  const at = rest.indexOf(toId);
  rest.splice(after ? at + 1 : at, 0, fromId);
  return rest.every((id, i) => id === ids[i]) ? null : rest;
}

/** Counts for the header strip: how much is running, waiting, finished, or needs a decision. */
export function queueTotals(cards) {
  const t = { total: cards.length, running: 0, waiting: 0, done: 0, attention: 0, resumable: 0 };
  for (const c of cards) {
    if (c.isActive) t.running += 1;
    else if (RUNNABLE_STATES.includes(c.state)) t.waiting += 1;
    if (c.state === 'COMPLETED') t.done += 1;
    if (['FAILED', 'CANCELLED', 'INTERRUPTED'].includes(c.state)) t.attention += 1;
    if (c.state === 'RECOVERABLE' || c.state === 'INTERRUPTED') t.resumable += 1;
  }
  return t;
}

/**
 * Worker rack: one entry per GOP-aligned split (a segment group), with a lane per
 * slice. The backend runs one job at a time on this host, so at most one lane is
 * active; the rack says so instead of inventing parallel lanes.
 */
export function rackGroups(groups) {
  return (groups || []).map((g) => {
    const segs = [...(g.segments || [])].sort((a, b) => a.index - b.index);
    const lanes = segs.map((s) => {
      const st = String(s.state || '').toUpperCase();
      const active = ACTIVE_STATES.includes(st);
      const done = st === 'COMPLETED' && !!s.exists;
      return {
        index: s.index,
        jobId: s.job_id || '',
        state: st,
        badge: stateBadge(st),
        active,
        done,
        pct: clamp01(s.fraction) === null ? null : Math.round(clamp01(s.fraction) * 100),
        frames: s.frames ?? null,
        range: s.start_frame != null ? `${s.start_frame}–${s.end_frame}` : '',
        error: s.error || '',
      };
    });
    const joined = g.joined || {};
    const verify = joined.result?.verify || null;
    let remux = { status: 'unknown', text: 'No joined file yet' };
    if (joined.exists && verify) {
      const okDuration = verify.ok !== false;
      remux = okDuration
        ? { status: 'confirmed', text: joined.result?.source_audio ? 'Joined; source audio remuxed; duration verified' : 'Joined; duration verified (no source audio)' }
        : { status: 'failed', text: `Joined but duration check failed: ${verify.reason || 'see log'}` };
    } else if (joined.exists) {
      remux = { status: 'pending', text: 'Joined file present; not verified' };
    }
    const running = lanes.some((l) => l.active);
    let joinState = 'waiting';
    if (joined.exists) joinState = 'joined';
    else if (g.can_join) joinState = 'ready';
    else if (running) joinState = 'running';
    return {
      group: g.group,
      target: g.target_name || 'Untitled',
      count: g.count ?? lanes.length,
      completed: g.completed ?? lanes.filter((l) => l.done).length,
      lanes,
      running,
      joinState,
      joinHint: g.can_join ? 'All slices finished; ready to join' : (g.join_blocked || 'Waiting for slices'),
      canJoin: !!g.can_join && !joined.exists,
      joinedUrl: joined.exists ? joined.url : '',
      remux,
    };
  });
}
