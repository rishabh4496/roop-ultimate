/**
 * Self-test for audit-truncation.mjs.
 *
 * An auditor that reports zero findings is worthless unless it is known to fire
 * on a real leak. These fixtures pin both directions: the shapes that must be
 * reported, and the shapes that must NOT be (the false positives that caused
 * no-op churn on the first pass).
 *
 * The POSITIVE fixtures are the real RunHistory bug; the NEGATIVE fixtures are
 * the three sites measured in Chromium as already contained.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const AUDIT = path.resolve('scripts/audit-truncation.mjs');

const FIXTURES = [
  {
    name: 'flex-1 wrapper, truncating child capped only by max-w-full',
    expect: 'leak',
    jsx: `export default () => (
  <div className="flex items-end gap-2 h-24 pt-2">
    <div className="flex-1 flex flex-col items-center gap-1 group relative">
      <span className="text-nano font-mono truncate max-w-full">{r.fps}</span>
    </div>
  </div>
);`,
  },
  {
    name: 'same wrapper, sealed with min-w-0',
    expect: 'clean',
    jsx: `export default () => (
  <div className="flex items-end gap-2 h-24 pt-2">
    <div className="flex-1 min-w-0 flex flex-col items-center gap-1 group relative">
      <span className="text-nano font-mono truncate max-w-full">{r.fps}</span>
    </div>
  </div>
);`,
  },
  {
    name: 'truncating element IS the flex item (overflow:hidden floors it at 0)',
    expect: 'clean',
    jsx: `export default () => (
  <div className="flex items-baseline justify-between gap-2">
    <span className="text-white/30">{k}</span>
    <span className="truncate text-right font-semibold">{v}</span>
  </div>
);`,
  },
  {
    name: 'grid cell wrapper (Tailwind floors tracks at minmax(0,1fr))',
    expect: 'clean',
    jsx: `export default () => (
  <div className="grid grid-cols-2 gap-1.5">
    <div className="px-2 py-1 flex items-center justify-between">
      <span className="truncate">{label}</span><span className="font-mono">{value}</span>
    </div>
  </div>
);`,
  },
  {
    name: 'wrapper with an explicit max-w-[90px] on the text',
    expect: 'clean',
    jsx: `export default () => (
  <div className="flex items-center gap-2">
    <div className="text-micro">
      <span className="font-semibold block truncate max-w-[90px]">{name}</span>
    </div>
  </div>
);`,
  },
  {
    name: 'flex-col wrapper: item is stretched, cannot widen the row',
    expect: 'clean',
    jsx: `export default () => (
  <div className="flex flex-col gap-2">
    <div className="p-2">
      <span className="truncate">{name}</span>
    </div>
  </div>
);`,
  },
];

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'trunc-audit-'));
let failures = 0;

for (const f of FIXTURES) {
  const dir = fs.mkdtempSync(path.join(tmp, 'case-'));
  fs.writeFileSync(path.join(dir, 'Fixture.jsx'), f.jsx);

  let out = '';
  let code = 0;
  try {
    out = execFileSync(process.execPath, [AUDIT, dir], { encoding: 'utf8' });
  } catch (e) {
    out = e.stdout ?? '';
    code = e.status ?? 1;
  }

  const reported = code !== 0;
  const want = f.expect === 'leak';
  const ok = reported === want;
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  expect=${f.expect.padEnd(5)} got=${reported ? 'leak ' : 'clean'}  ${f.name}`);
  if (!ok) console.log(out.split('\n').map((l) => '        ' + l).join('\n'));
}

fs.rmSync(tmp, { recursive: true, force: true });
console.log(`\n${FIXTURES.length - failures}/${FIXTURES.length} fixtures behaved as specified`);
process.exitCode = failures ? 1 : 0;
