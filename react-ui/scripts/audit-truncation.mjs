/**
 * Layout leak audit for filename / metadata displays.
 *
 * WHAT ACTUALLY CAUSES A LEAK
 * ---------------------------
 * The common advice is "put min-w-0 on every flex child that truncates". That
 * is too broad, and following it produces no-op churn. The real rule, per CSS
 * Flexbox 4.5 (automatic minimum size) and verified in Chromium against this
 * project's own compiled CSS:
 *
 *   * A flex item's automatic minimum size is content-based ONLY while its
 *     computed `overflow` is `visible`. Tailwind's `truncate` expands to
 *     `overflow:hidden`, so an element that truncates AND is itself the flex
 *     item is already floored at 0. min-w-0 there changes nothing.
 *
 *   * Tailwind v4 compiles `grid-cols-N` to repeat(N, minmax(0,1fr)), so grid
 *     items are already floored at 0 too. min-w-0 on them changes nothing.
 *
 *   * The leak is real when an INTERMEDIATE wrapper sits between the flex
 *     container and the truncating text, and that wrapper keeps
 *     overflow:visible with no width bound. It takes its content's max-content
 *     width as its minimum, refuses to shrink, and widens the column. This is
 *     the only shape that needs min-w-0.
 *
 * Run `node scripts/probe-minwidth-rule.mjs` to re-derive these facts in a real
 * browser if the Tailwind version or CSS changes.
 *
 * The scanner tokenises JSX tags with a string/template/brace-aware scanner so
 * multi-line and self-closing tags nest correctly, then walks the true ancestor
 * chain for each truncating element.
 */
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(process.argv[2] ?? 'src');

function walk(dir, acc = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, acc);
    else if (/\.jsx$/.test(entry.name)) acc.push(full);
  }
  return acc;
}

const VOID_TAGS = new Set(['img', 'input', 'br', 'hr', 'source', 'track', 'meta', 'link', 'area', 'base', 'col', 'embed', 'param', 'wbr']);

/** Extract JSX tags in source order: {name, kind, cls, line, text}. */
function scanTags(src) {
  const tags = [];
  const lineAt = (idx) => src.slice(0, idx).split('\n').length;

  for (let i = 0; i < src.length; i++) {
    if (src[i] !== '<') continue;
    const m = src.slice(i + 1).match(/^(\/?)([A-Za-z][\w.]*)/);
    if (!m) continue;

    let depth = 0;
    let quote = null;
    let end = -1;
    for (let j = i + 1; j < src.length; j++) {
      const c = src[j];
      if (quote) {
        if (c === '\\') { j++; continue; }
        if (c === quote) quote = null;
        continue;
      }
      if (c === '"' || c === "'" || c === '`') { quote = c; continue; }
      if (c === '{') depth++;
      else if (c === '}') depth--;
      else if (c === '>' && depth === 0) { end = j; break; }
      else if (c === '<' && depth === 0) break;
    }
    if (end === -1) continue;

    const body = src.slice(i, end + 1);
    const isClose = m[1] === '/';
    const selfClosed = /\/>$/.test(body.trimEnd());
    const name = m[2];
    const clsMatch = body.match(/className=(?:"([^"]*)"|\{`([^`]*)`\}|\{([\s\S]*?)\}\s*(?=\s[\w-]+=|\s*\/?>))/);

    tags.push({
      name,
      kind: isClose ? 'close' : (selfClosed || VOID_TAGS.has(name) ? 'self' : 'open'),
      cls: clsMatch ? (clsMatch[1] ?? clsMatch[2] ?? clsMatch[3] ?? '') : '',
      line: lineAt(i),
      text: body.replace(/\s+/g, ' ').slice(0, 120),
    });
    i = end;
  }
  return tags;
}

const isFlex = (cls) => /(?:^|[\s"'`])(?:flex|inline-flex)(?:$|[\s"'`])/.test(cls);
const isGrid = (cls) => /(?:^|[\s"'`])grid(?:$|[\s"'`])/.test(cls);
// Row direction is where horizontal intrinsic width bites. `flex-col` items are
// stretched to the container width instead, so they cannot widen it.
const isFlexRow = (cls) => isFlex(cls) && !/(?:^|[\s:])flex-col\b/.test(cls);
const hasMinW0 = (cls) => /\bmin-w-0\b/.test(cls);
const outOfFlow = (cls) => /\b(?:absolute|fixed)\b/.test(cls);
// `overflow:hidden` in any form floors the automatic minimum size at 0.
const clipsOverflow = (cls) => /\b(?:truncate|overflow-hidden|overflow-x-auto|overflow-x-hidden|overflow-auto|overflow-scroll|line-clamp-\d)\b/.test(cls);

// A definite width cap. `max-w-full`/`max-w-none` are percentage or no-op and do
// NOT bound the intrinsic max-content contribution, so they are excluded.
//
// Every alternative is anchored with (?<![\w-]) rather than \b: `-` counts as a
// word boundary, so a bare \bw-full\b also matches INSIDE `max-w-full` and would
// wrongly treat the percentage case as a cap. That exact bug hid a real leak.
const hasWidthCap = (cls) =>
  /(?<![\w-])max-w-\[[^\]]+\]/.test(cls) ||
  /(?<![\w-])max-w-(?!full(?![\w-])|none(?![\w-]))[\w.]+/.test(cls) ||
  /(?<![\w-])w-\[[^\]]+\]/.test(cls) ||
  /(?<![\w-])(?:w-\d|w-px|w-full)(?![\w-])/.test(cls) ||
  /(?<![\w-])basis-0(?![\w-])/.test(cls);

const sealsChain = (cls) => hasMinW0(cls) || hasWidthCap(cls) || clipsOverflow(cls) || outOfFlow(cls);
const truncates = (cls) => /\b(?:truncate|line-clamp-\d)\b/.test(cls);

const leaks = [];

for (const file of walk(ROOT)) {
  const rel = path.relative(process.cwd(), file);
  const stack = [];

  for (const tag of scanTags(fs.readFileSync(file, 'utf8'))) {
    if (tag.kind === 'close') { stack.pop(); continue; }

    if (truncates(tag.cls)) {
      // If the truncating element itself is width-capped or out of flow, its
      // max-content contribution is already bounded and no ancestor can be
      // widened by it. (`max-w-full` is a percentage and would NOT qualify;
      // hasWidthCap excludes it deliberately.)
      if (hasWidthCap(tag.cls) || outOfFlow(tag.cls)) {
        if (tag.kind === 'open') stack.push(tag);
        continue;
      }

      // Walk up looking for an intermediate wrapper that is a flex item of a
      // row-flex parent and is NOT sealed. Grid parents are skipped: Tailwind
      // floors grid tracks at minmax(0,1fr) already.
      for (let d = stack.length - 1; d >= 0; d--) {
        const anc = stack[d];
        if (outOfFlow(anc.cls)) break;   // out of flow: cannot widen an ancestor
        if (sealsChain(anc.cls)) break;  // chain already sealed
        const parent = stack[d - 1];
        if (!parent) break;
        if (isGrid(parent.cls)) break;   // grid tracks are floored at 0
        if (!isFlexRow(parent.cls)) continue;
        leaks.push({
          file: rel,
          line: tag.line,
          child: tag.text,
          ancestor: `<${anc.name}> line ${anc.line} class="${anc.cls.slice(0, 80)}"`,
          parent: `<${parent.name}> line ${parent.line} class="${parent.cls.slice(0, 60)}"`,
        });
        break;
      }
    }

    if (tag.kind === 'open') stack.push(tag);
  }
}

console.log(`=== unsealed flex wrapper between a row-flex container and truncating text: ${leaks.length} ===\n`);
for (const l of leaks) {
  console.log(`${l.file}:${l.line}`);
  console.log(`    text    : ${l.child}`);
  console.log(`    wrapper : ${l.ancestor}   <- needs min-w-0`);
  console.log(`    in flex : ${l.parent}\n`);
}
process.exitCode = leaks.length > 0 ? 1 : 0;
