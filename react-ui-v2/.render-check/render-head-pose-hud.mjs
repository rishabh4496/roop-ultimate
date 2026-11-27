/**
 * The head-pose HUD (live yaw / pitch / roll), driven by `GET /api/trajectory/pose`.
 *
 *   * model: angles print with a sign and one decimal, a missing angle prints a dash (never 0.0 degrees); bar markers
 *     clamp to their bar and are absent for a missing angle; a face that was not detected says so in WORDS;
 *   * states: an idle app, a pending first answer, "no trajectory yet" (with the server's reason), "no face near this
 *     frame" and a live frame are five different things and read as five different things;
 *   * markup: a labelled region, the number carries the reading (the bar is aria-hidden), the sign convention is
 *     printed whenever angles are;
 *   * the container asks for ONE small thing, with an explicit short deadline, follows the telemetry store at <= 2 Hz
 *     and owns no timer of its own, and a failed request becomes "not available", never a stale or zero reading.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

globalThis.window = globalThis;
// src/api.js reads `window.location` at module scope (the container imports it through useApi).
globalThis.location = { protocol: 'http:', host: '127.0.0.1:42003', origin: 'http://127.0.0.1:42003', hash: '' };
globalThis.window.location = globalThis.location;

const here = dirname(fileURLToPath(import.meta.url));
const src = (rel) => readFileSync(join(here, '..', 'src', rel), 'utf8');

const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const M = await import('../src/components/faceswap/headPoseModel.js');
const { HeadPoseHudView } = await import('../src/components/faceswap/HeadPoseHud.jsx');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const html = (el) => renderToStaticMarkup(el).replace(/<!-- -->/g, '');
const text = (h) => h.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
const view = (payload, opts) => html(React.createElement(HeadPoseHudView, { model: M.hudModel(payload, opts) }));

const face = (track, yaw, pitch, roll, flags = 0) => ({ track, yaw, pitch, roll, flags, bbox: [0, 0, 10, 10], scale: 1 });
const live = (faces, extra = {}) => ({ available: true, frame: 130, requested: 30, frame_start: 100, faces, ...extra });

console.log('── the model ──────────────────────────────────────────────');
ok('angles carry a sign and one decimal', M.fmtAngle(12.34) === '+12.3°' && M.fmtAngle(-7.26) === '-7.3°', `${M.fmtAngle(12.34)} ${M.fmtAngle(-7.26)}`);
ok('zero has no sign, and a negative that rounds to zero is not printed as -0.0',
  M.fmtAngle(0) === '0.0°' && M.fmtAngle(-0.04) === '0.0°', `${M.fmtAngle(0)} ${M.fmtAngle(-0.04)}`);
ok('a missing angle is a dash, never 0.0 degrees', M.fmtAngle(null) === '—' && M.fmtAngle(undefined) === '—' && M.fmtAngle(NaN) === '—');
ok('a marker sits at the centre for 0 and at the ends for the full range',
  M.markerPct(0, 90) === 50 && M.markerPct(90, 90) === 100 && M.markerPct(-90, 90) === 0);
ok('a marker clamps to its bar', M.markerPct(500, 90) === 100 && M.markerPct(-500, 90) === 0);
ok('a missing angle has no marker', M.markerPct(null, 90) === null && M.markerPct(NaN, 90) === null);
ok('flags become words: gap-filled / flow-tracked / predicted, and a detection says nothing',
  M.noteOf(0) === '' && M.noteOf(1) === 'gap-filled' && M.noteOf(3) === 'flow-tracked' && M.noteOf(5) === 'predicted');

console.log('── the five states ────────────────────────────────────────');
ok('idle app: nothing to ask for', M.hudModel(undefined, { active: false }).status === 'idle');
ok('active, no answer yet: waiting', M.hudModel(undefined).status === 'waiting');
{
  const m = M.hudModel({ available: false, reason: 'no trajectory yet (the pre-pass has not finished)' });
  ok('the server says there is none: "none", with its reason', m.status === 'none' && /pre-pass has not finished/.test(m.caption), m.caption);
}
ok('available but nobody near this frame: empty', M.hudModel(live([], { requested: 30 })).status === 'empty'
  && /frame 30/.test(M.hudModel(live([], { requested: 30 })).caption));
{
  const m = M.hudModel(live([face(0, 12.3, -4.5, 1.2), face(2, -40, 20, 0, 2)]));
  ok('live: one row per face, dials in yaw / pitch / roll order',
    m.status === 'live' && m.faces.length === 2 && m.faces[0].dials.map((d) => d.name).join() === 'yaw,pitch,roll', JSON.stringify(m.faces[0]));
  ok('the caption names the frame and how far it is from the one asked for (130 is 0 from 100 + 30)',
    m.caption === 'Frame 130', m.caption);
  ok('a flow-tracked face says so', m.faces[1].note === 'flow-tracked');
}
ok('a frame that is not the one asked for says how far', /\(\+3 from now\)/.test(M.hudModel(live([face(0, 1, 2, 3)], { frame: 133 })).caption));
ok('a null payload is "none", not a crash', M.hudModel(null).status === 'none');

console.log('── the markup ─────────────────────────────────────────────');
{
  const h = view(live([face(0, 12.3, -4.5, 1.2), face(2, null, 20, 0, 1)]));
  const t = text(h);
  ok('a labelled region for assistive tech', /<section[^>]*aria-label="Head pose"[^>]*data-testid="head-pose-hud"/.test(h), h.slice(0, 200));
  ok('every number is printed as text', /\+12\.3°/.test(t) && /-4\.5°/.test(t) && /\+1\.2°/.test(t) && /\+20\.0°/.test(t), t);
  ok('a missing angle prints a dash', /Yaw —/.test(t), t);
  ok('the gap-filled face says so in words', /gap-filled/.test(t));
  ok('the bars are hidden from assistive tech (the number carries the reading)',
    (h.match(/aria-hidden="true"/g) || []).length === 6, String((h.match(/aria-hidden="true"/g) || []).length));
  ok('the sign convention is printed beside live angles', /\+yaw: nose toward your right/.test(t));
  ok('a missing angle has no marker', (h.match(/-translate-x-1\/2 h-2\.5/g) || []).length === 5, String((h.match(/-translate-x-1\/2 h-2\.5/g) || []).length));
}
{
  const h = view(undefined, { active: false });
  ok('idle prints the caption and no convention line', /Appears while a job runs/.test(text(h)) && !/\+yaw/.test(text(h)));
  ok('every state names itself for tests', /data-status="idle"/.test(h) && /data-status="live"/.test(view(live([face(0, 1, 2, 3)]))));
}

console.log('── the container ──────────────────────────────────────────');
{
  const code = src('components/faceswap/HeadPoseHud.jsx').replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
  ok('one request, to the trajectory route, with an explicit deadline',
    (code.match(/getJSON\(/g) || []).length === 1 && /getJSON\(`\/api\/trajectory\/pose\?frame=\$\{[^}]*\}`,\s*\{\s*timeout:\s*5000\s*\}\)/.test(code));
  ok('it follows the telemetry store through a 2 Hz throttled subscription, and owns no timer',
    /subscribeThrottled\(ask,\s*2\)/.test(code) && /useTelemetryStore\.getState\(\)/.test(code)
    && !/\b(setInterval|setTimeout)\b/.test(code));
  ok('a frame number alone costs no render: the component reads a boolean, not the frame',
    /useThrottledSelector\(\(s\) => Number\(s\.run\.total_frames\) > 0/.test(code)
    && !/useThrottledSelector\(\(s\) => s\.run\.current_frame/.test(code));
  ok('the same answer (or failure) again keeps the previous state, so it renders nothing',
    /const keep = \(next\) => \(prev\) => \(JSON\.stringify\(prev\) === JSON\.stringify\(next\) \? prev : next\)/.test(code)
    && (code.match(/setPayload\(keep\(/g) || []).length === 2);
  ok('a request is only made while a job has frames (an idle app asks nothing)', /if \(!active\) return undefined;/.test(code));
  ok('a stale answer cannot land after the frame moved on or the HUD unmounted',
    /\.then\(\(p\) => \{ if \(!cancelled\) setPayload\(keep\(p\)\); \}\)/.test(code)
    && /if \(cancelled\) return;/.test(code) && /return \(\) => \{ cancelled = true; unsubscribe\(\); \}/.test(code));
  ok('a failed read becomes "not available" and is reported, not left as the last reading',
    /available:\s*false/.test(code) && /reportFailure\(/.test(code));
}

console.log(failures === 0 ? '\nall head-pose-hud checks passed' : `\n${failures} head-pose-hud check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
