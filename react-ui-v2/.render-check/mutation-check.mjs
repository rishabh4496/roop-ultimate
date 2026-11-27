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
    file: join(src, 'store', 'telemetryStore.js'),
    from: 'const done = total === null ? null : Math.min(total, Math.max(0, counts.done));',
    to: 'const done = total === null ? null : counts.done;',
  },
  {
    name: 'ETA no longer suppressed while paused',
    file: join(src, 'components', 'Processing.jsx'),
    from: 'const etaMs = !paused && !pauseRequested && !stopping ? run.etaMs : 0;',
    to: 'const etaMs = run.etaMs;',
  },
  {
    name: 'render bar back to the backend percentage',
    file: join(src, 'store', 'telemetryStore.js'),
    from: 'fraction: total === null ? null : done / total,',
    to: 'fraction: total === null ? null : run.progress,',
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
