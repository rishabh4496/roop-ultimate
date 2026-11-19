import { useSyncExternalStore } from 'react';
import { TRACE_STAGES, traceStageMs } from '../components/faceswap/flamegraphModel';

const LIMIT = 480;
let samples = [];
let pendingSeekFrame = null;
let traceEpoch = null;
const listeners = new Set();

function emit() {
  for (const listener of listeners) listener();
}

export function recordFrameTrace(frame) {
  const rows = frame?.frame_trace;
  const incomingEpoch = Number(frame?.frame_trace_epoch);
  let reset = false;
  if (Number.isFinite(incomingEpoch)) {
    if (traceEpoch !== null && incomingEpoch !== traceEpoch) {
      samples = [];
      reset = true;
    }
    traceEpoch = incomingEpoch;
  }
  if (!Array.isArray(rows) || !rows.length) {
    if (reset) emit();
    return;
  }
  const lastIncoming = Number(rows[rows.length - 1]?.frame);
  const lastStored = samples[samples.length - 1]?.frame;
  if (Number.isFinite(lastIncoming) && lastStored && lastIncoming < lastStored) {
    samples = [];
    reset = true;
  }

  const byFrame = new Map(samples.map((sample) => [sample.frame, sample]));
  let changed = false;
  for (const row of rows) {
    const frameNumber = Number(row?.frame);
    if (!Number.isFinite(frameNumber) || frameNumber < 1) continue;
    const frameMs = row.frame_ms != null && Number.isFinite(Number(row.frame_ms)) ? Number(row.frame_ms) : null;
    const { stageMs, measuredMs, overAttributedMs } = traceStageMs(row.stage_ms || {}, frameMs);
    if (!Object.values(stageMs).some((value) => value !== null) && row.frame_ms == null) continue;
    const next = {
      frame: Math.round(frameNumber),
      frameMs,
      stageMs,
      // Measured lanes only: the derived `other` lane is the residual, not a measurement.
      totalStageMs: measuredMs,
      overAttributedMs,
    };
    const old = byFrame.get(next.frame);
    if (old && old.frameMs === next.frameMs
        && TRACE_STAGES.every(({ key }) => old.stageMs[key] === next.stageMs[key])) continue;
    byFrame.set(next.frame, next);
    changed = true;
  }
  if (!changed) {
    if (reset) emit();
    return;
  }
  samples = [...byFrame.values()].sort((a, b) => a.frame - b.frame).slice(-LIMIT);
  emit();
}

export function clearFrameTrace() {
  const changed = samples.length > 0 || traceEpoch !== null;
  samples = [];
  traceEpoch = null;
  if (changed) emit();
}

export function requestFrameTraceSeek(frame) {
  const value = Number(frame);
  if (!Number.isFinite(value) || value < 1) return;
  pendingSeekFrame = Math.round(value);
  emit();
  if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
    window.dispatchEvent(new CustomEvent('roop:frame-trace-seek', { detail: pendingSeekFrame }));
  }
}

export function consumeFrameTraceSeek() {
  const frame = pendingSeekFrame;
  pendingSeekFrame = null;
  return frame;
}

export function getFrameTraceSnapshot() {
  return samples;
}

export function subscribeFrameTrace(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useFrameTrace() {
  return useSyncExternalStore(subscribeFrameTrace, getFrameTraceSnapshot, getFrameTraceSnapshot);
}
