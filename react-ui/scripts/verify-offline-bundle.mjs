// A local app must look right and say so with the network cable pulled. This scans the PRODUCTION bundle (dist/) for
// anything that would reach another machine:
//
//   1. a FETCH position - <link href>, <script src>, <img src>, <iframe src>, CSS `url(...)` / `@import` - that names
//      an http(s) origin. There is no allowlist for these: a stylesheet, script, font or image comes from dist/ or
//      it does not ship.
//   2. any other http(s) URL anywhere in a text asset, whose host must be one of the few that are text, not requests:
//        www.w3.org        XML namespace identifiers (svg / xlink / xhtml): compared as strings, never fetched
//        react.dev         the link printed inside React's own error messages
//        tailwindcss.com   the licence banner
//      Adding a host here is a decision: say why it is text.
//
// It reads files and makes no request, so it runs anywhere `npm run build` ran. The runtime counterpart is
// e2e/offline.spec.js, which loads every tab with all foreign origins aborted.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, '..', 'dist');

export const TEXT_HOSTS = new Set(['www.w3.org', 'react.dev', 'tailwindcss.com']);
const TEXT_EXT = new Set(['.html', '.js', '.mjs', '.css', '.json', '.svg', '.map', '.txt', '.webmanifest']);

const URL_RE = /https?:\/\/[A-Za-z0-9._~:/?#@!$&'()*+,;=%[\]-]+/g;
// An http(s) origin in a position the browser fetches.
const FETCH_RES = [
  [/<link\b[^>]*\bhref\s*=\s*["']?\s*(?:https?:)?\/\/[^\s"'>]+/gi, '<link href>'],
  [/<script\b[^>]*\bsrc\s*=\s*["']?\s*(?:https?:)?\/\/[^\s"'>]+/gi, '<script src>'],
  [/<(?:img|iframe|source|video|audio|embed)\b[^>]*\bsrc\s*=\s*["']?\s*(?:https?:)?\/\/[^\s"'>]+/gi, 'media/frame src'],
  [/@import\s+(?:url\()?\s*["']?(?:https?:)?\/\/[^\s"')]+/gi, 'CSS @import'],
  [/url\(\s*["']?(?:https?:)?\/\/[^\s"')]+/gi, 'CSS url()'],
];

function* files(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* files(p);
    else if (TEXT_EXT.has(extname(name).toLowerCase())) yield p;
  }
}

export function scan(root = dist) {
  const problems = [];
  const hosts = new Map();
  let scanned = 0;
  for (const file of files(root)) {
    scanned += 1;
    const text = readFileSync(file, 'utf8');
    const rel = relative(root, file).replace(/\\/g, '/');
    for (const [re, what] of FETCH_RES) {
      for (const m of text.matchAll(re)) problems.push(`${rel}: ${what} reaches out: ${m[0].slice(0, 120)}`);
    }
    for (const m of text.matchAll(URL_RE)) {
      let host;
      try { host = new URL(m[0]).hostname; } catch { host = m[0]; }
      hosts.set(host, (hosts.get(host) || 0) + 1);
      if (!TEXT_HOSTS.has(host)) problems.push(`${rel}: external URL to ${host}: ${m[0].slice(0, 120)}`);
    }
  }
  return { scanned, hosts, problems };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  let result;
  try {
    result = scan();
  } catch (err) {
    console.error(`verify-offline-bundle: cannot read dist/ (${err.message}) - run npm run build first`);
    process.exit(2);
  }
  if (result.scanned === 0) {
    console.error('verify-offline-bundle: dist/ holds no text assets - nothing was checked');
    process.exit(2);
  }
  const seen = [...result.hosts].map(([h, n]) => `${h} x${n}`).join(', ') || 'none';
  if (result.problems.length) {
    console.error(`verify-offline-bundle: ${result.problems.length} problem(s) in ${result.scanned} files`);
    result.problems.forEach((p) => console.error(`  ${p}`));
    process.exit(1);
  }
  console.log(`verify-offline-bundle: ${result.scanned} files, no external URL outside the text-only hosts (${seen})`);
}
