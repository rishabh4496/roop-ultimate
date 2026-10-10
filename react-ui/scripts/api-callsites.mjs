#!/usr/bin/env node
/**
 * API call-site scan: every request the UI makes states whether it may outrun the 15 s
 * default deadline, and what it says matches src/longRunning.js.
 *
 *   node scripts/api-callsites.mjs          # exit 1 on any violation
 *   node scripts/api-callsites.mjs --json   # machine-readable report (all sites + violations)
 *
 * api.js gives every getJSON / postJSON / postFile(s) call 15 s unless it passes
 * `timeout: 0`. Which endpoints deserve `timeout: 0` is decided ONCE, in longRunning.js,
 * with a reason each. This enumerates every call site in src/ and checks, per site:
 *
 *   missing-opt-out     the endpoint is listed as long-running, but the call has no
 *                       `timeout: 0` (it will be cut off at 15 s: a 4 GB upload, a cold
 *                       TensorRT build). THE FAILURE THIS EXISTS TO CATCH.
 *   unlisted-opt-out    the call passes `timeout: 0` for an endpoint that is not listed.
 *                       Either it is long-running (add it, with the reason) or it is not
 *                       (drop the opt-out: an unexplained "wait forever" is exactly the
 *                       bug the default removes).
 *   dynamic-no-timeout  the path is computed (`act(path)`, `/api/projects/${id}/${verb}`),
 *                       so it cannot be classified from the call. Say what you mean:
 *                       `timeout: timeoutFor('POST', path)` (the registry decides at run
 *                       time) or an explicit `timeout:`.
 *   stale-entry         a longRunning.js entry matches no call site at all.
 *
 * Raw `fetch()` (streaming frames, workers, blob downloads) is outside api.js and not
 * covered here; each of those has its own AbortSignal.
 */
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, dirname, resolve, relative, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import process from 'node:process';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(here, '..', 'src');
const SKIP = new Set(['api.js', 'useApi.js']);          // the client itself
export const FNS = { getJSON: { method: 'GET', opts: 1 }, postJSON: { method: 'POST', opts: 2 }, postFiles: { method: 'POST', opts: 3 }, postFile: { method: 'POST', opts: 3 } };

const walk = (dir) => readdirSync(dir).flatMap((n) => {
  const p = join(dir, n);
  if (statSync(p).isDirectory()) return walk(p);
  return /\.(jsx?|tsx?)$/.test(p) && !p.endsWith('.d.ts') ? [p] : [];
});

// Blank out comments but keep every character position (so line numbers stay true). A
// `//` is a comment only after whitespace or at line start: `${proto}//${host}` and
// `http://x` are code.
export function blankComments(src) {
  const blank = (s) => s.replace(/[^\n]/g, ' ');
  return src
    .replace(/\/\*[\s\S]*?\*\//g, blank)
    .replace(/(^|\s)\/\/[^\n]*/g, (m, lead) => lead + blank(m.slice(lead.length)));
}

// 1 where a character is INSIDE a string or template-literal text (not in a `${...}` code
// expression), so `'postJSON("/api/x")'` is not mistaken for a call. A '...' or "..." string
// cannot span lines, so one that never closes (an apostrophe in JSX text: `It's here`)
// resyncs at the newline instead of swallowing the rest of the file.
export function stringMask(src) {
  const mask = new Uint8Array(src.length);
  const saved = [];                  // code brace-depth to restore after a template's `${...}`
  let mode = 'code'; let q = null; let depth = 0;
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if (mode === 'str') {
      if (c === '\n') { mode = 'code'; continue; }
      mask[i] = 1;
      if (c === '\\') mask[++i] = 1;
      else if (c === q) mode = 'code';
    } else if (mode === 'tpl') {
      mask[i] = 1;
      if (c === '\\') mask[++i] = 1;
      else if (c === '`') mode = 'code';
      else if (c === '$' && src[i + 1] === '{') { mask[++i] = 1; saved.push(depth); depth = 0; mode = 'code'; }
    } else if (c === '"' || c === "'") { mode = 'str'; q = c; mask[i] = 1; }
    else if (c === '`') { mode = 'tpl'; mask[i] = 1; }
    else if (c === '{') depth++;
    else if (c === '}') {
      if (depth === 0 && saved.length) { depth = saved.pop(); mode = 'tpl'; mask[i] = 1; }
      else depth--;
    }
  }
  return mask;
}

// The balanced text of the argument list that opens right after `start`, and its end.
export function argList(src, start) {
  let depth = 1; let q = null; let i = start;
  for (; i < src.length && depth > 0; i++) {
    const c = src[i];
    if (q) { if (c === '\\') i++; else if (c === q) q = null; continue; }
    if (c === '"' || c === "'" || c === '`') q = c;
    else if (c === '(' || c === '{' || c === '[') depth++;
    else if (c === ')' || c === '}' || c === ']') depth--;
  }
  return src.slice(start, i - 1);
}

export function splitTop(args) {
  const out = []; let depth = 0; let q = null; let cur = '';
  for (let i = 0; i < args.length; i++) {
    const c = args[i];
    if (q) { cur += c; if (c === '\\') { cur += args[++i] ?? ''; } else if (c === q) q = null; continue; }
    if (c === '"' || c === "'" || c === '`') { q = c; cur += c; continue; }
    if (c === '(' || c === '{' || c === '[') depth++;
    if (c === ')' || c === '}' || c === ']') depth--;
    if (c === ',' && depth === 0) { out.push(cur.trim()); cur = ''; continue; }
    cur += c;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

// Turn the first argument into { kind: 'literal'|'dynamic', path }. A template whose only
// interpolation trails a clean path (`/api/update/check${query}`) is that path; one whose
// interpolation is a path segment (`/api/projects/${id}/${verb}`) is dynamic.
function resolvePath(expr) {
  const m = expr.match(/^(['"`])([\s\S]*)\1$/);
  if (!m) return { kind: 'dynamic', path: null };
  let text = m[2].split('?')[0];
  if (m[1] === '`') {
    const at = text.indexOf('${');
    if (at >= 0) {
      const prefix = text.slice(0, at);
      if (prefix.startsWith('/') && !prefix.endsWith('/') && /\$\{[\s\S]*\}$/.test(text) && text.indexOf('${', at + 2) < 0) text = prefix;
      else text = text.replace(/\$\{[^}]*\}/g, '{}');
    }
  }
  if (!text.startsWith('/')) return { kind: 'dynamic', path: null };
  text = text.replace(/\/+$/, '');
  return { kind: text.includes('{}') ? 'dynamic' : 'literal', path: text };
}

// Two paths could be the same request if they agree segment by segment, `{}` matching anything.
const couldMatch = (a, b) => {
  const x = a.split('/'); const y = b.split('/');
  return x.length === y.length && x.every((s, i) => s === y[i] || s === '{}' || y[i] === '{}');
};

export async function scan({ srcDir = SRC, registryPath = join(srcDir, 'longRunning.js') } = {}) {
  const { LONG_RUNNING, longRunningKey } = await import(pathToFileURL(registryPath).href);
  const sites = [];
  const referenced = new Set();      // every '/api/...' string literal anywhere in src (wrappers call with literals)
  for (const file of walk(srcDir)) {
    const rel = relative(srcDir, file).split(sep).join('/');
    if (SKIP.has(rel)) continue;
    const src = blankComments(readFileSync(file, 'utf8'));
    // '/api/x', `/api/x/${id}/y?q=1`: templated segments become `{}`, the query goes.
    for (const lit of src.matchAll(/['"`](\/api\/[^'"`\s]*)/g)) {
      referenced.add(lit[1].replace(/\$\{[^}]*\}/g, '{}').split('?')[0].replace(/\/+$/, ''));
    }
    const inString = stringMask(src);
    for (const m of src.matchAll(/(?<![\w$.])(getJSON|postJSON|postFiles|postFile)\s*\(/g)) {
      if (inString[m.index]) continue;                                            // text, not a call
      const before = src.slice(Math.max(0, m.index - 10), m.index);
      if (/\b(function|const|let|var|export)\s+$/.test(before)) continue;          // a declaration
      const args = splitTop(argList(src, m.index + m[0].length));
      const spec = FNS[m[1]];
      const resolved = resolvePath(args[0] || '');
      const opts = args[spec.opts] || '';
      sites.push({
        file: rel,
        line: src.slice(0, m.index).split('\n').length,
        fn: m[1],
        method: spec.method,
        path: resolved.path,
        kind: resolved.kind,
        optOut: /\btimeout\s*:\s*0\b(?![.\d])/.test(opts),
        timeoutFor: /\btimeoutFor\s*\(/.test(opts),
        anyTimeout: /\btimeout\s*:/.test(opts),
      });
    }
  }

  const violations = [];
  const hit = new Set();
  for (const s of sites) {
    if (s.kind === 'literal') {
      const key = longRunningKey(s.method, s.path);
      if (key) hit.add(key);
      if (key && !s.optOut) violations.push({ ...s, rule: 'missing-opt-out', key, why: LONG_RUNNING[key] });
      else if (!key && s.optOut) violations.push({ ...s, rule: 'unlisted-opt-out' });
    } else {
      if (!s.timeoutFor && !s.anyTimeout) violations.push({ ...s, rule: 'dynamic-no-timeout' });
      if (s.path) for (const key of Object.keys(LONG_RUNNING)) {
        const [method, p] = key.split(' ');
        if (method === s.method && couldMatch(s.path, p)) hit.add(key);
      }
    }
  }
  for (const key of Object.keys(LONG_RUNNING)) {
    const [, p] = key.split(' ');
    if (!hit.has(key) && [...referenced].some((r) => couldMatch(r, p))) hit.add(key);
    if (!hit.has(key)) violations.push({ rule: 'stale-entry', key, file: 'longRunning.js', line: 0 });
  }
  return { sites, violations, registry: Object.keys(LONG_RUNNING).length };
}

const describe = (v) => {
  const at = `${v.file}:${v.line}`;
  switch (v.rule) {
    case 'missing-opt-out': return `${at}  ${v.fn}('${v.path}') is long-running (${v.why}) but has no \`timeout: 0\``;
    case 'unlisted-opt-out': return `${at}  ${v.fn}('${v.path}') passes \`timeout: 0\` but ${v.method} ${v.path} is not in longRunning.js`;
    case 'dynamic-no-timeout': return `${at}  ${v.fn}(${v.path ? `'${v.path}'` : '<computed path>'}) cannot be classified: pass \`timeout: timeoutFor('${v.method}', path)\``;
    case 'stale-entry': return `longRunning.js  '${v.key}' matches no call site`;
    default: return JSON.stringify(v);
  }
};

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (!existsSync(SRC)) { console.error(`api-callsites: ${SRC} not found`); process.exit(2); }
  const r = await scan();
  if (process.argv.includes('--json')) {
    console.log(JSON.stringify(r, null, 2));
  } else if (r.violations.length === 0) {
    const opt = r.sites.filter((s) => s.optOut).length;
    console.log(`api-callsites: ${r.sites.length} call sites OK (${opt} explicit \`timeout: 0\`, ${r.registry} long-running endpoints listed)`);
  } else {
    console.error(`api-callsites: ${r.violations.length} violation(s) in ${r.sites.length} call sites:`);
    for (const v of r.violations) console.error(`  [${v.rule}] ${describe(v)}`);
  }
  process.exit(r.violations.length ? 1 : 0);
}
