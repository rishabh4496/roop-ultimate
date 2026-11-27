// The UI half of the terminal dock's feed: what the CLIENT did (navigation, toasts, connection
// changes, request failures), as terminal lines. The engine half is the backend's own rolling log
// (/api/progress `log`), which the dock merges in by `seq`.
//
// A tiny pub/sub with a bounded buffer, so a line pushed before the dock mounts (the boot
// sequence runs first) is still there when it opens. Pure: no React, no DOM.

const CAP = 400;
let seq = 0;
let lines = [];
const listeners = new Set();

/** level: 'info' | 'ok' | 'warn' | 'error' | 'cmd' | 'perf'. `src` names the subsystem ("ui", "nav", "net"...). */
export function termLog(msg, level = 'info', src = 'ui') {
  const d = new Date();
  const t = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
  const line = { id: `ui-${++seq}`, t, msg: String(msg), level, src, origin: 'ui', at: Date.now() };
  lines = lines.length >= CAP ? [...lines.slice(lines.length - CAP + 1), line] : [...lines, line];
  listeners.forEach((fn) => { try { fn(lines); } catch { /* a listener's failure is its own */ } });
  return line;
}

export const termLines = () => lines;

export function onTermLines(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function clearTermLines() {
  lines = [];
  listeners.forEach((fn) => fn(lines));
}

/** Map a toast type onto a terminal level. */
export const levelOfToast = (type) => ({ error: 'error', warning: 'warn', warn: 'warn', info: 'info', success: 'ok' }[type] || 'info');
