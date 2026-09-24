// Pure helpers behind HardwareTelemetryHud, kept out of the JSX so the node
// check (.render-check/telemetry-queue-check.mjs) exercises the real logic.
//
// Rule for everything here: a value the backend did not send is UNKNOWN and
// renders as unknown. No fallback to one GPU's numbers (both a 12 GB 4070 and
// a 6 GB 3060 Laptop run this UI), no estimated split of a total.

export const THERMAL_WARN_C = 80;
export const THERMAL_CRITICAL_C = 86;
// Fraction of the board's own enforced power limit (nvidia-smi power.limit)
// at which the card is treated as power-limited.
export const POWER_LIMIT_FRACTION = 0.95;

const positive = (v) => {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : null;
};

/**
 * Wall-clock ms per output frame, all workers together, from /ws/telemetry's
 * `frame_ms`. NOT a model latency and not divisible into stages; null while
 * the backend has no rate yet.
 */
export function frameTimeMs(run) {
  return positive(run?.frame_ms);
}

/** { used, total, pct } in GB, or null when the total is unknown. */
export function vramUsage(sys) {
  const total = positive(sys?.vram_total);
  if (total === null) return null;
  const used = Math.max(0, Number(sys?.vram_used) || 0);
  return { used, total, pct: Math.min(100, Math.round((used / total) * 100)) };
}

/** null | 'warn' | 'critical' for a GPU temperature in °C. */
export function thermalLevel(tempC) {
  const t = positive(tempC);
  if (t === null) return null;
  if (t >= THERMAL_CRITICAL_C) return 'critical';
  if (t >= THERMAL_WARN_C) return 'warn';
  return null;
}

/**
 * True when draw is at the board's own power limit. Without a reported
 * limit there is no alert: a fixed wattage fits one card and is wrong on
 * every other (190 W can never fire on a ~100 W laptop part).
 */
export function isPowerLimited(powerW, limitW) {
  const p = positive(powerW);
  const l = positive(limitW);
  if (p === null || l === null) return false;
  return p >= l * POWER_LIMIT_FRACTION;
}

/** Map val into [0, 1] over [min, max], clamped. */
export function sparkNorm(val, min, max) {
  return Math.max(0, Math.min(1, (val - min) / Math.max(1e-5, max - min)));
}
