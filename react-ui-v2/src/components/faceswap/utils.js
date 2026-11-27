// Small pure helpers shared across the Face Swap UI.

export const num = (v, d) => (v === undefined || v === null || v === '' ? d : Number(v));

export const fmtTime = (ms) => {
  const s = Math.max(0, Math.round(ms / 1000));
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}m ${String(s % 60).padStart(2, '0')}s` : `${s}s`;
};

// Short ascending 3-note chime via WebAudio (no asset, CSP-safe).
export const playChime = () => {
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    const ctx = new AC();
    const now = ctx.currentTime;
    [523.25, 659.25, 783.99].forEach((f, i) => {
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.type = 'sine';
      o.frequency.value = f;
      o.connect(g);
      g.connect(ctx.destination);
      const t = now + i * 0.11;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(0.12, t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0008, t + 0.34);
      o.start(t);
      o.stop(t + 0.36);
    });
    setTimeout(() => ctx.close().catch(() => {}), 1400);
  } catch { /* audio not available */ }
};

// The counterpart for a run that did NOT finish: two descending notes on a
// lower, duller waveform. It has to be audible from another room and instantly
// distinguishable from the success chime without being listened to — the whole
// job of these sounds is to be understood while you are looking elsewhere.
export const playFailTone = () => {
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    const ctx = new AC();
    const now = ctx.currentTime;
    [415.30, 311.13].forEach((f, i) => {
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.type = 'triangle';
      o.frequency.value = f;
      o.connect(g);
      g.connect(ctx.destination);
      const t = now + i * 0.19;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(0.11, t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0008, t + 0.42);
      o.start(t);
      o.stop(t + 0.44);
    });
    setTimeout(() => ctx.close().catch(() => {}), 1400);
  } catch { /* audio not available */ }
};

export const notifyDesktop = (title, body) => {
  try {
    if (!('Notification' in window)) return;
    if (document.visibilityState === 'visible') return; // only notify when tab is backgrounded
    if (Notification.permission === 'granted') new Notification(title, { body, silent: true });
  } catch { /* ignore */ }
};
