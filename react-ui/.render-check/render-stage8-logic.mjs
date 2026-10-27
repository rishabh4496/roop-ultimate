/**
 * Stage 8's pure logic (plain node, no browser): the terminal's circular log buffer and severity filter, and the
 * compare divider's transform geometry.
 *
 *   node .render-check/run-module.mjs render-stage8-logic.mjs
 *
 * The ring is where a quiet bug would live: a wrap-around off by one, a count that drifts when a line is overwritten,
 * a new run's lines interleaved with the last run's. Each is checked against a plain-array model.
 */
import {
  LOG_RING_CAPACITY, SEVERITIES, bySeverity, createLogRing, severityOf,
} from '../src/components/faceswap/logRing.js';
import { dividerTransform } from '../src/components/faceswap/wipeTransform.js';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const line = (seq, msg, extra = {}) => ({ seq, msg, t: '00:00:00', part: 0, category: 'PROCESSING', level: 'INFO', ...extra });
const MSGS = [
  ['ERROR', { level: 'ERROR', category: 'ERRORS', msg: 'CUDA out of memory' }],
  ['ERROR', { msg: 'swap failed on frame 12' }],
  ['WARN', { level: 'WARNING', category: 'WARNINGS', msg: 'fallback to CPU' }],
  ['WARN', { msg: '⚠ low VRAM headroom' }],
  ['PERF', { category: 'PERFORMANCE', msg: 'stage 3 settled' }],
  ['PERF', { msg: 'throughput 21.4 faces/s' }],
  ['INFO', { msg: 'Loading models' }],
];

console.log('severity');
{
  ok('the four chips are ALL / WARN / ERROR / PERF', eq(SEVERITIES, ['ALL', 'WARN', 'ERROR', 'PERF']));
  for (const [want, l] of MSGS) ok(`${want}: ${l.msg}`, severityOf(line(1, l.msg, l)) === want, severityOf(line(1, l.msg, l)));
  ok('ERROR outranks WARN and PERF on one line', severityOf(line(1, 'warning: fps fell, render failed', { level: 'WARNING' })) === 'ERROR');
  ok('a missing line is INFO, not a throw', severityOf(null) === 'INFO' && severityOf({}) === 'INFO');
}

console.log('ring: capacity and order');
{
  ok('capacity is 2,000', LOG_RING_CAPACITY === 2000);
  ok('a non-positive capacity is refused', (() => { try { createLogRing(0); return false; } catch (e) { return e instanceof RangeError; } })());
  const r = createLogRing(5);
  r.ingest([1, 2, 3].map((n) => line(n, `m${n}`)));
  ok('under capacity: all lines, oldest first', eq(r.snapshot().map((l) => l.seq), [1, 2, 3]));
  r.ingest([4, 5, 6, 7].map((n) => line(n, `m${n}`)));
  ok('over capacity: the OLDEST are dropped, order kept', eq(r.snapshot().map((l) => l.seq), [3, 4, 5, 6, 7]) && r.size === 5);
  ok('lastSeq follows the newest', r.lastSeq === 7);
}

console.log('ring: 10,000 lines');
{
  const r = createLogRing();
  const all = Array.from({ length: 10_000 }, (_, i) => {
    const [, m] = MSGS[i % MSGS.length];
    return line(i + 1, `${m.msg} #${i}`, m);
  });
  // The backend's real shape: a rolling 250-line tail, resent whole on every poll.
  for (let end = 250; end <= all.length; end += 50) r.ingest(all.slice(end - 250, end));
  const snap = r.snapshot();
  ok('holds exactly 2,000', snap.length === 2000 && r.size === 2000, String(snap.length));
  ok('newest is line 10,000, oldest is 8,001', snap.at(-1).seq === 10_000 && snap[0].seq === 8001, `${snap[0].seq}..${snap.at(-1).seq}`);
  ok('contiguous (no gap, no duplicate)', snap.every((l, i) => i === 0 || l.seq === snap[i - 1].seq + 1));
  const c = r.counts();
  const model = { WARN: 0, ERROR: 0, PERF: 0, INFO: 0 };
  snap.forEach((l) => { model[severityOf(l)] += 1; });
  ok('counts match a recount of the window (overwrites un-count)', eq(c, model), `${JSON.stringify(c)} vs ${JSON.stringify(model)}`);
  ok('the severity partition is exact', c.WARN + c.ERROR + c.PERF + c.INFO === 2000);
  ok('each chip returns only its own lines', ['WARN', 'ERROR', 'PERF'].every((s) => bySeverity(snap, s).every((l) => l.sev === s)
    && bySeverity(snap, s).length === c[s]));
  ok('ALL is the same array, uncopied', bySeverity(snap, 'ALL') === snap);
}

console.log('ring: polling behaviour');
{
  const r = createLogRing(100);
  const tail = Array.from({ length: 30 }, (_, i) => line(i + 1, `a${i + 1}`));
  ok('first poll changes the ring', r.ingest(tail) === true);
  const v = r.version;
  const s1 = r.snapshot();
  ok('the same tail again changes nothing (idempotent by seq)', r.ingest(tail) === false && r.version === v);
  ok('and the snapshot is the SAME array (safe as a memo key)', r.snapshot() === s1);
  ok('an overlapping tail adds only the new lines', r.ingest([...tail.slice(10), line(31, 'a31'), line(32, 'a32')]) && r.size === 32);

  r.ingest([line(1, 'b1'), line(2, 'b2')]);
  ok('a lower seq is a NEW RUN: the old lines are dropped, not interleaved',
    eq(r.snapshot().map((l) => l.msg), ['b1', 'b2']));

  const r2 = createLogRing(100);
  r2.ingest([line(1, 'old1'), line(2, 'old2'), line(3, 'old3')]);
  r2.ingest([line(2, 'NEW2'), line(3, 'NEW3'), line(4, 'NEW4')]);
  ok('same seq, different text is a new run too (it climbed past the old mark before the first poll)',
    eq(r2.snapshot().map((l) => l.msg), ['NEW2', 'NEW3', 'NEW4']));

  const r3 = createLogRing(100);
  r3.ingest([line(1, 'x')]);
  ok('an empty tail after content clears (the backend cleared its log)', r3.ingest([]) === true && r3.size === 0);
  ok('an empty tail on an empty ring is a no-op', r3.ingest([]) === false && r3.ingest(undefined) === false);
}

console.log('ring: cost');
{
  const r = createLogRing();
  const t0 = performance.now();
  for (let end = 250; end <= 20_000; end += 25) r.ingest(Array.from({ length: 250 }, (_, i) => line(end - 249 + i, `L${end - 249 + i}`)));
  r.snapshot();
  const ms = performance.now() - t0;
  console.log(`        (800 polls of a 250-line tail, 20,000 lines: ${ms.toFixed(1)} ms total)`);
  ok('ingesting is cheap enough to do in render (<1 s for 800 polls)', ms < 1000, `${ms.toFixed(0)} ms`);
  const snap = r.snapshot();
  const t1 = performance.now();
  for (let i = 0; i < 100; i += 1) bySeverity(snap, 'ERROR');
  const per = (performance.now() - t1) / 100;
  console.log(`        (severity filter over 2,000 lines: ${per.toFixed(3)} ms)`);
  ok('a severity filter pass is far inside a 2 ms interaction budget', per < 2, `${per.toFixed(3)} ms`);
}

console.log('compare divider transform');
{
  ok('vertical curtain at 30% -> translate 70% of the stage (== the old left: 70%)',
    dividerTransform(30, 0, false, null) === 'translate3d(70%, 0, 0)');
  ok('horizontal curtain at 30% -> translateY 70%', dividerTransform(30, 0, true, null) === 'translate3d(0, 70%, 0)');
  ok('endpoints', dividerTransform(0, 0, false, null) === 'translate3d(100%, 0, 0)' && dividerTransform(100, 0, false, null) === 'translate3d(0%, 0, 0)');
  // The rotated wipe must land where `left: calc(50% + d*cos%)`, `top: calc(50% + d*sin%)` did.
  const host = { offsetWidth: 800, offsetHeight: 450 };
  for (const [pct, angle] of [[25, 45], [75, 135], [50, 90], [10, 270], [90, 10]]) {
    const d = 100 - pct - 50;
    const rad = (angle * Math.PI) / 180;
    const oldDx = ((d * Math.cos(rad)) / 100) * host.offsetWidth;     // old left offset from 50%, in px
    const oldDy = ((d * Math.sin(rad)) / 100) * host.offsetHeight;    // old top offset from 50%, in px
    const m = /translate3d\(([-\d.e]+)px, ([-\d.e]+)px, 0\) translate\(-50%, -50%\) rotate\(([-\d.]+)deg\)/.exec(dividerTransform(pct, angle, false, host));
    ok(`rotated ${angle} deg at ${pct}% matches the old left/top (${oldDx.toFixed(1)}, ${oldDy.toFixed(1)} px)`,
      !!m && Math.abs(Number(m[1]) - oldDx) < 1e-6 && Math.abs(Number(m[2]) - oldDy) < 1e-6 && Number(m[3]) === angle + 90);
  }
  ok('a not-yet-mounted host does not throw or NaN', !dividerTransform(40, 30, false, null).includes('NaN'));
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? '\nstage 8 logic: all checks passed' : `\nstage 8 logic: ${failures} FAILED`);
