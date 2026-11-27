// The offline-bundle scan must be able to FAIL: feed it bundles with each kind of violation and a clean one.
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { scan } from './verify-offline-bundle.mjs';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? ` -- ${detail}` : ''}`);
};

function bundle(files) {
  const root = mkdtempSync(join(tmpdir(), 'offline-scan-'));
  mkdirSync(join(root, 'assets'));
  for (const [name, body] of Object.entries(files)) writeFileSync(join(root, name), body);
  return root;
}
const run = (files) => { const root = bundle(files); try { return scan(root); } finally { rmSync(root, { recursive: true, force: true }); } };

const clean = run({
  'index.html': '<link rel="stylesheet" href="/assets/a.css"><script type="module" src="/assets/a.js"></script>',
  'assets/a.css': '@font-face{src:url(/assets/f.woff2)} /*! tailwindcss v4 | MIT | https://tailwindcss.com */',
  'assets/a.js': 'const ns="http://www.w3.org/2000/svg";throw new Error("see https://react.dev/errors/31")',
});
ok('a clean bundle (relative assets, namespace + doc text only) passes', clean.problems.length === 0 && clean.scanned === 3, clean.problems.join(' | '));

ok('a Google Fonts <link> fails', run({ 'index.html': '<link href="https://fonts.googleapis.com/css2?family=Inter" rel="stylesheet">' }).problems.length >= 1);
ok('a CDN <script src> fails', run({ 'index.html': '<script src="https://cdn.example.com/x.js"></script>' }).problems.length >= 1);
ok('a protocol-relative <script src> fails', run({ 'index.html': '<script src="//cdn.example.com/x.js"></script>' }).problems.some((p) => p.includes('script src')));
ok('a CSS @import of a remote sheet fails', run({ 'assets/a.css': '@import url("https://fonts.googleapis.com/css?family=X");' }).problems.some((p) => p.includes('@import')));
ok('a CSS url() to a remote font fails', run({ 'assets/a.css': '@font-face{src:url(https://fonts.gstatic.com/x.woff2)}' }).problems.some((p) => p.includes('url()')));
ok('a remote URL inside JS fails even outside a fetch position', run({ 'assets/a.js': 'fetch("https://api.example.com/v1")' }).problems.some((p) => p.includes('api.example.com')));
ok('a localhost URL is still an external URL (the app is same-origin)', run({ 'assets/a.js': 'const u="http://127.0.0.1:8001/api"' }).problems.length === 1);

console.log(failures === 0 ? '\nall offline-bundle scan checks passed' : `\n${failures} offline-bundle scan check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
