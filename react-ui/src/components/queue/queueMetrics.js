// Pure helpers behind RenderQueueDrawer, kept out of the JSX so the node check
// (.render-check/telemetry-queue-check.mjs) imports the real code instead of
// testing a copy of it.

/**
 * Rolling ETA over the last `windowFrames` frames. Samples are (time, frame);
 * a frame counter that goes backwards (the next job started) starts a fresh
 * window rather than producing a negative rate.
 */
export class RollingEtaTracker {
  constructor(windowFrames = 100) {
    this.windowFrames = windowFrames;
    this.samples = [];
  }

  reset() {
    this.samples = [];
  }

  addSample(frame, now = performance.now()) {
    const f = Number(frame);
    if (!Number.isFinite(f) || f < 0) return;
    if (this.samples.length && f < this.samples[this.samples.length - 1].frame) {
      this.samples = [];
    }
    this.samples.push({ time: now, frame: f });

    while (this.samples.length > 2 && (f - this.samples[0].frame) > this.windowFrames) {
      this.samples.shift();
    }
  }

  getFps() {
    if (this.samples.length < 2) return null;
    const first = this.samples[0];
    const last = this.samples[this.samples.length - 1];
    const dt = (last.time - first.time) / 1000.0;
    const df = last.frame - first.frame;
    if (dt <= 0 || df <= 0) return null;
    return df / dt;
  }

  getEtaSeconds(totalFrames) {
    const total = Number(totalFrames);
    if (!total || total <= 0 || !this.samples.length) return null;
    const currentFrame = this.samples[this.samples.length - 1].frame;
    const remaining = Math.max(0, total - currentFrame);
    if (remaining === 0) return 0;
    const fps = this.getFps();
    if (!fps || fps <= 0) return null;
    return remaining / fps;
  }
}

/**
 * What a finished queue actually did, from each job's state string.
 * `allSucceeded` is true only when every job COMPLETED; FAILED, CANCELLED and
 * INTERRUPTED are terminal too, and a queue of them is not a success.
 */
export function summarizeQueueOutcome(states) {
  const counts = { completed: 0, failed: 0, cancelled: 0, interrupted: 0 };
  for (const s of states) {
    if (s === 'COMPLETED') counts.completed += 1;
    else if (s === 'FAILED') counts.failed += 1;
    else if (s === 'CANCELLED') counts.cancelled += 1;
    else if (s === 'INTERRUPTED') counts.interrupted += 1;
  }
  const total = states.length;
  const allSucceeded = total > 0 && counts.completed === total;
  let title;
  let body;
  if (allSucceeded) {
    title = 'Render Queue Complete';
    body = `All ${total} job${total === 1 ? '' : 's'} completed.`;
  } else {
    title = counts.completed === 0 ? 'Render Queue Failed' : 'Render Queue Finished With Problems';
    const parts = [`${counts.completed} of ${total} completed`];
    if (counts.failed) parts.push(`${counts.failed} failed`);
    if (counts.cancelled) parts.push(`${counts.cancelled} cancelled`);
    if (counts.interrupted) parts.push(`${counts.interrupted} interrupted`);
    body = `${parts.join(', ')}.`;
  }
  return { ...counts, total, allSucceeded, title, body };
}
