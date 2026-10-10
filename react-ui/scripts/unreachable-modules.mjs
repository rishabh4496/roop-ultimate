#!/usr/bin/env node
/**
 * Unreachable-module scan: every source module under src/ must be reachable
 * from the app's entry point, or the build is not telling the truth about what
 * ships.
 *
 *   node scripts/unreachable-modules.mjs          # exit 1 if anything is unreachable
 *   node scripts/unreachable-modules.mjs --json   # machine-readable report
 *
 * Why this exists. Six component folders (studio, preview, timeline, facebank,
 * queue, telemetry: ~6,800 lines) sat in src/ for two weeks with no importer.
 * Each had its own `.render-check` script and each script PASSED, so the suite
 * was green while none of that code could ever run in the app -- the project's
 * recurring failure of something reporting success while not running. Lint,
 * the build and the tests cannot see it: Vite only bundles what is imported, and
 * a test that imports a module directly proves nothing about whether the app does.
 *
 * What "reachable" means here: the transitive closure, from src/main.jsx, of
 *   - static imports and re-exports                  import x from './a'
 *   - dynamic imports (the lazy tab panels)          import('./Tab')
 *   - Web Workers and other URL assets Vite bundles  new URL('./x.worker.js', import.meta.url)
 * Anything else under src/ is reported. There is deliberately no allowlist: a
 * module that is wanted should be imported, and one that is not should be deleted.
 */
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, dirname, resolve, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import process from 'node:process';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(here, '..', 'src');
const ENTRY = join(SRC, 'main.jsx');
const EXTS = ['.jsx', '.js', '.tsx', '.ts', '.mjs'];

const walk = (dir) => readdirSync(dir).flatMap((name) => {
  const p = join(dir, name);
  if (statSync(p).isDirectory()) return walk(p);
  return EXTS.some((e) => p.endsWith(e)) && !p.endsWith('.d.ts') ? [p] : [];
});

const resolveSpec = (from, spec) => {
  if (!spec.startsWith('.')) return null;            // a package, not a source module
  const base = resolve(dirname(from), spec);
  const candidates = [base, ...EXTS.map((e) => base + e), ...EXTS.map((e) => join(base, `index${e}`))];
  return candidates.find((c) => existsSync(c) && statSync(c).isFile()) || null;
};

// import/export ... from '...', side-effect imports, dynamic import('...'), and
// new URL('...', import.meta.url). Comments are stripped first so a path that is
// only MENTIONED in prose does not make a module look reachable.
const stripComments = (src) => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:'"`\\])\/\/.*$/gm, '$1');
const PATTERNS = [
  /\b(?:import|export)\s[^'"`;]*?\bfrom\s*['"]([^'"]+)['"]/g,
  /\bimport\s*['"]([^'"]+)['"]/g,
  /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
  /new\s+URL\(\s*['"]([^'"]+)['"]\s*,\s*import\.meta\.url\s*\)/g,
];

export function scan() {
  const files = walk(SRC);
  const edges = new Map(files.map((f) => {
    const src = stripComments(readFileSync(f, 'utf8'));
    const out = new Set();
    for (const re of PATTERNS) {
      for (const m of src.matchAll(re)) {
        const target = resolveSpec(f, m[1]);
        if (target) out.add(target);
      }
    }
    return [f, out];
  }));

  const reachable = new Set([ENTRY]);
  const stack = [ENTRY];
  while (stack.length) {
    for (const next of edges.get(stack.pop()) || []) {
      if (!reachable.has(next)) { reachable.add(next); stack.push(next); }
    }
  }

  const rel = (f) => relative(SRC, f).split(sep).join('/');
  const lineCount = (f) => readFileSync(f, 'utf8').split('\n').length;
  const unreachable = files.filter((f) => !reachable.has(f)).map((f) => ({ file: rel(f), lines: lineCount(f) }));
  // `reachable` also holds non-source targets (index.css), so count source modules only.
  return { entry: rel(ENTRY), files: files.length, reachable: files.filter((f) => reachable.has(f)).length, unreachable };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (!existsSync(ENTRY)) {
    console.error(`unreachable-modules: entry ${ENTRY} not found`);
    process.exit(2);
  }
  const r = scan();
  if (process.argv.includes('--json')) {
    console.log(JSON.stringify(r, null, 2));
  } else if (r.unreachable.length === 0) {
    console.log(`unreachable-modules: 0 unreachable (${r.reachable} of ${r.files} modules reachable from ${r.entry})`);
  } else {
    const total = r.unreachable.reduce((n, u) => n + u.lines, 0);
    console.error(`unreachable-modules: ${r.unreachable.length} module(s), ${total} lines, not reachable from ${r.entry}:`);
    for (const u of r.unreachable) console.error(`  ${String(u.lines).padStart(5)}  src/${u.file}`);
    console.error('Import it from the app or delete it; there is no allowlist.');
  }
  process.exit(r.unreachable.length ? 1 : 0);
}
