// The terminal's in-memory log: a fixed-capacity circular buffer, and the severity filter over it.
//
// The backend keeps a rolling 250-line tail (api.py `_log_lines`, maxlen=250) and sends the whole tail on
// every poll. Rendering that array as it arrives meant a long render's history lived only as long as the
// backend's window, and a pathological log (a stage that warns per frame) put thousands of DOM rows on the
// page. The ring keeps up to LOG_RING_CAPACITY lines on the client, de-duplicated by the backend's `seq`, and
// the terminal windows its rendering over a snapshot of it.
//
// Pure (no React, no DOM) so .render-check/render-stage8-logic.mjs can drive 10,000 lines through it.

export const LOG_RING_CAPACITY = 2000;

export const SEVERITIES = ['ALL', 'WARN', 'ERROR', 'PERF'];

const ERROR_RE = /error|fail|abort/i;
// A throughput / timing line. The backend files these under PERFORMANCE / PROFILE (runtime_state.classify_log);
// the pattern catches the same lines when the category is missing (an older backend, a hand-built entry).
const PERF_RE = /\bfps\b|faces\/s|throughput|ms\/frame|bottleneck|stage timing/i;

/**
 * One severity per line, most severe first: ERROR > WARN > PERF > INFO. A line is never in two filters, so
 * the chip counts add up and "ERROR" never shows a warning.
 */
export function severityOf(line) {
  if (!line) return 'INFO';
  if (line.level === 'ERROR' || line.category === 'ERRORS' || ERROR_RE.test(line.msg || '') || /\[BUG-DETECTED\]|\[CRASH AUTOPSY\]|\[CUSTODY FAILED\]/i.test(line.msg || '')) return 'ERROR';
  if (line.level === 'WARNING' || line.category === 'WARNINGS' || String(line.msg || '').startsWith('⚠') || /\[WARN\]/i.test(line.msg || '')) return 'WARN';
  if (line.category === 'PERFORMANCE' || line.category === 'PROFILE' || PERF_RE.test(line.msg || '') || /\[PERF-WARN\]/i.test(line.msg || '')) return 'PERF';
  return 'INFO';
}

/** Lines matching a severity chip. ALL is the identity, returned without copying. */
export function bySeverity(lines, severity) {
  if (!severity || severity === 'ALL') return lines;
  return lines.filter((l) => l.sev === severity);
}

export function createLogRing(capacity = LOG_RING_CAPACITY) {
  if (!Number.isInteger(capacity) || capacity < 1) throw new RangeError('log ring capacity must be a positive integer');
  let buf = new Array(capacity);
  let head = 0;            // index of the OLDEST entry
  let size = 0;
  let lastSeq = -1;
  let version = 0;
  let cached = null;       // snapshot() result for `version`
  let cachedVersion = -1;
  const counts = { WARN: 0, ERROR: 0, PERF: 0, INFO: 0 };

  const clear = () => {
    buf = new Array(capacity);
    head = 0; size = 0; lastSeq = -1;
    counts.WARN = 0; counts.ERROR = 0; counts.PERF = 0; counts.INFO = 0;
    version += 1;
  };

  const newest = () => (size ? buf[(head + size - 1) % capacity] : null);

  const push = (entry) => {
    if (size < capacity) {
      buf[(head + size) % capacity] = entry;
      size += 1;
    } else {
      counts[buf[head].sev] -= 1;      // the overwritten line leaves the counts too
      buf[head] = entry;
      head = (head + 1) % capacity;
    }
    counts[entry.sev] += 1;
  };

  /**
   * Fold one poll's tail into the ring. Returns true when anything changed.
   *
   * `seq` is the backend's per-run counter: it restarts at 0 when a new run begins. A tail whose newest seq is
   * BEHIND the ring's is a new run; so is a tail that re-sends the ring's newest seq with different text (a
   * restarted run that has already climbed past the old high-water mark before the first poll). Either way
   * the old run's lines are dropped rather than interleaved with the new run's.
   */
  const ingest = (lines) => {
    if (!Array.isArray(lines) || lines.length === 0) {
      if (size && Array.isArray(lines) && lines.length === 0 && lastSeq >= 0) { clear(); return true; }
      return false;
    }
    const tail = lines[lines.length - 1];
    const last = newest();
    let restarted = false;
    if (last) {
      if (typeof tail.seq === 'number' && tail.seq < lastSeq) restarted = true;
      else {
        const same = lines.find((l) => l.seq === lastSeq);
        if (same && same.msg !== last.msg) restarted = true;
      }
    }
    if (restarted) clear();
    let changed = restarted;
    for (const l of lines) {
      if (typeof l.seq !== 'number' || l.seq <= lastSeq) continue;
      push({ ...l, sev: severityOf(l) });
      lastSeq = l.seq;
      changed = true;
    }
    if (changed) version += 1;
    return changed;
  };

  /** Oldest -> newest. The same array instance until the ring next changes, so it is safe as a memo key. */
  const snapshot = () => {
    if (cachedVersion === version && cached) return cached;
    const out = new Array(size);
    for (let i = 0; i < size; i += 1) out[i] = buf[(head + i) % capacity];
    cached = out;
    cachedVersion = version;
    return out;
  };

  return {
    ingest,
    snapshot,
    clear,
    get size() { return size; },
    get capacity() { return capacity; },
    get version() { return version; },
    get lastSeq() { return lastSeq; },
    counts: () => ({ ...counts }),
  };
}
