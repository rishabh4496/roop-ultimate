// The terminal dock's pure logic: turning backend log lines into terminal lines, folding a polled
// tail into the dock's map without repeats, and the filter chips. No React, no DOM, so
// .render-check/render-v2fx.mjs drives it directly (the mock backend sends no `log`, so this is
// where the engine half of the terminal is proven).
import { severityOf } from '../components/faceswap/logRing';

export const ENGINE_CAP = 1000;
export const FILTERS = ['ALL', 'ENGINE', 'UI', 'WARN', 'ERROR', 'PERF'];
const LEVEL_OF_SEV = { ERROR: 'error', WARN: 'warn', PERF: 'perf', INFO: 'info' };

/** A backend `_log_lines` entry ({t, msg, seq, level, category}) as a terminal line. */
export function engineLine(l, now = Date.now()) {
  const sev = severityOf(l);
  return {
    id: `en-${l.seq ?? `${l.t}-${l.msg}`}`,
    t: l.t || '',
    msg: String(l.msg ?? ''),
    level: LEVEL_OF_SEV[sev] || 'info',
    // Whole word: the column ellipsizes in CSS (a hard slice printed "performanc").
    src: String(l.category || 'engine').toLowerCase(),
    origin: 'engine',
    seq: Number.isFinite(l.seq) ? l.seq : null,
    at: now,
  };
}

/**
 * Fold a polled tail into `map` (id -> line, insertion-ordered). The backend sends its whole
 * 250-line tail on every poll, so a line arrives many times and must be printed once. Evicts the
 * OLDEST lines past `cap`. Returns how many lines were new.
 */
export function foldEngineLog(map, log, cap = ENGINE_CAP, now = Date.now()) {
  if (!Array.isArray(log) || !log.length) return 0;
  let added = 0;
  for (const l of log) {
    if (!l) continue;
    const line = engineLine(l, now);
    if (map.has(line.id)) continue;
    map.set(line.id, line);
    added += 1;
  }
  if (map.size > cap) {
    const drop = map.size - cap;
    let i = 0;
    for (const k of map.keys()) { if (i++ >= drop) break; map.delete(k); }
  }
  return added;
}

export function matchesFilter(line, filter) {
  switch (filter) {
    case 'ENGINE': return line.origin === 'engine';
    case 'UI': return line.origin === 'ui';
    case 'WARN': return line.level === 'warn';
    case 'ERROR': return line.level === 'error';
    case 'PERF': return line.level === 'perf';
    default: return true;
  }
}
