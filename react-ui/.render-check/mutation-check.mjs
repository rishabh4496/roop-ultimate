/**
 * Mutation check: prove the render harness actually detects the regressions it
 * claims to guard, by reintroducing each bug and confirming the check goes red.
 *
 * A green test suite is worth exactly as much as its ability to go red. This
 * reverts one fix at a time, runs the harness, and asserts a FAILURE — the
 * inverse of a normal test. Anything that stays green with its fix removed is
 * reported here as an untested claim rather than quietly counted as coverage.
 *
 * Every mutation is applied to a copy and the original is restored in a
 * `finally`, so an interrupted run cannot leave a mutated source behind.
 */
import { readFileSync, writeFileSync, copyFileSync, unlinkSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { execFileSync } from 'node:child_process';
import process from 'node:process';

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, '..', 'src');

const MUTATIONS = [
  {
    name: 'progress clamp removed (the full-ring bug)',
    file: join(src, 'components', 'Processing.jsx'),
    // The LIVE clamp (useLiveRun) — the one the ring and bars are drawn from.
    from: 'const rawProg = Number(run.progress);\n  const prog = Number.isFinite(rawProg) ? Math.min(1, Math.max(0, rawProg)) : 0;',
    to: 'const prog = Number(run.progress) || 0;',
  },
  {
    name: 'ETA no longer suppressed while paused',
    file: join(src, 'components', 'Processing.jsx'),
    from: 'const etaMs = !paused && !pauseRequested && !stopping ? etaMsOf(run, elapsedMs) : 0;',
    to: 'const etaMs = etaMsOf(run, elapsedMs);',
  },
  {
    name: 'pipeline rail back to whole-run progress',
    file: join(src, 'components', 'Processing.jsx'),
    from: 'style={{ width: `${Math.max(6, stageFrac * 100)}%`, boxShadow: \'0 0 10px var(--accent-glow)\' }}',
    to: 'style={{ width: `${Math.max(6, prog * 100)}%`, boxShadow: \'0 0 10px var(--accent-glow)\' }}',
  },
  {
    name: 'ARIA removed from the progress bar',
    file: join(src, 'components', 'Processing.jsx'),
    from: '        role="progressbar"',
    to: '',
  },
  {
    name: 'models panel reads settings instead of the runtime snapshot',
    file: join(src, 'components', 'faceswap', 'RunModelsPanel.jsx'),
    from: 'const swapper = pick(model.swap_model, runtime?.model, p.swap_model);',
    to: 'const swapper = pick(p.swap_model);',
  },
  {
    name: 'diagnostics ignores the pushed frame counters',
    file: join(src, 'components', 'faceswap', 'DiagnosticsPanel.jsx'),
    from: '    if (counted) {\n      return { ...counted, left: Math.max(0, counted.total - counted.done) };\n    }',
    to: '',
  },
];

const run = () => {
  try {
    execFileSync(process.execPath, [join(here, 'run.mjs')], { stdio: 'pipe' });
    return 0;
  } catch (e) {
    return e.status ?? 1;
  }
};

console.log('\nBaseline (all fixes in place) — expect PASS');
const baseline = run();
console.log(`  exit=${baseline}  ${baseline === 0 ? 'PASS' : 'UNEXPECTED FAILURE'}`);
if (baseline !== 0) {
  console.log('\nBaseline is red; mutation results would be meaningless. Stopping.');
  process.exit(1);
}

let undetected = 0;
console.log('\nMutations — each should make the harness go RED');
for (const m of MUTATIONS) {
  const backup = `${m.file}.mutbak`;
  copyFileSync(m.file, backup);
  try {
    const original = readFileSync(m.file, 'utf8');
    const target = original.includes('\r\n') ? m.from.replace(/\r?\n/g, '\r\n') : m.from.replace(/\r\n/g, '\n');
    if (!original.includes(target)) {
      console.log(`  SKIP  ${m.name}\n          anchor not found; the harness cannot speak to this`);
      undetected += 1;
      continue;
    }
    writeFileSync(m.file, original.replace(target, m.to));
    const code = run();
    if (code !== 0) {
      console.log(`  CAUGHT   ${m.name}`);
    } else {
      console.log(`  MISSED   ${m.name}  <-- this fix is NOT covered by any check`);
      undetected += 1;
    }
  } finally {
    copyFileSync(backup, m.file);
    unlinkSync(backup);
  }
}

console.log(`\n${undetected === 0 ? 'EVERY FIX IS COVERED' : `${undetected} fix(es) uncovered`}: `
  + `${MUTATIONS.length - undetected}/${MUTATIONS.length} mutations detected\n`);
process.exit(undetected === 0 ? 0 : 1);
