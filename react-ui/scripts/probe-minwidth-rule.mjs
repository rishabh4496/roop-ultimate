/**
 * Establish the ACTUAL rule for when min-w-0 is required, empirically.
 *
 * The folklore is "always add min-w-0 to flex children that truncate". The CSS
 * spec is narrower: a flex item's automatic minimum size is its content-based
 * minimum size ONLY when its computed `overflow` is `visible`. If the item sets
 * overflow:hidden (which `truncate` does), its automatic minimum size is 0 and
 * min-w-0 is redundant.
 *
 * Tailwind also compiles `grid-cols-N` to repeat(N, minmax(0, 1fr)), so grid
 * items are already floored at 0 and cannot be widened by their content.
 *
 * These probes decide which of those claims hold in a real engine, so the audit
 * flags only genuine leaks.
 */
import { createRequire } from 'node:module';

// PLAYWRIGHT_REQUIRE_FROM: a directory whose node_modules holds playwright-core
// (e.g. an npx cache entry); defaults to this package. CHROME_PATH: the
// Chromium executable to drive.
const REQ = createRequire(process.env.PLAYWRIGHT_REQUIRE_FROM || import.meta.url);
const { chromium } = REQ('playwright-core');
const CHROME = process.env.CHROME_PATH;
if (!CHROME) throw new Error('set CHROME_PATH to a Chromium executable');

const LONG = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA';

const BASE = `
  .frame{width:200px;font:12px monospace}
  .row{display:flex;gap:4px}
  .trunc{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .minw0{min-width:0}
  .g2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px}
  .g2auto{display:grid;grid-template-columns:repeat(2,1fr);gap:4px}
`;

const PROBES = [
  {
    q: 'flex item WITH truncate, no min-w-0 -> does it shrink?',
    html: `<div class="frame"><div class="row" id="c"><span class="trunc" id="p">${LONG}</span></div></div>`,
  },
  {
    q: 'flex item WITH truncate + min-w-0 -> shrink?',
    html: `<div class="frame"><div class="row" id="c"><span class="trunc minw0" id="p">${LONG}</span></div></div>`,
  },
  {
    q: 'flex item overflow:visible wrapping a truncating child, no min-w-0',
    html: `<div class="frame"><div class="row" id="c"><div id="p"><span class="trunc">${LONG}</span></div></div></div>`,
  },
  {
    q: 'flex item overflow:visible wrapping truncating child, WITH min-w-0',
    html: `<div class="frame"><div class="row" id="c"><div class="minw0" id="p"><span class="trunc">${LONG}</span></div></div></div>`,
  },
  {
    q: 'grid minmax(0,1fr) cell, overflow:visible wrapper, no min-w-0',
    html: `<div class="frame"><div class="g2" id="c"><div id="p"><span class="trunc">${LONG}</span></div><div>b</div></div></div>`,
  },
  {
    q: 'grid 1fr (auto min) cell, overflow:visible wrapper, no min-w-0',
    html: `<div class="frame"><div class="g2auto" id="c"><div id="p"><span class="trunc">${LONG}</span></div><div>b</div></div></div>`,
  },
];

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage();

for (const probe of PROBES) {
  await page.setContent(`<!doctype html><style>${BASE}</style>${probe.html}`);
  const r = await page.evaluate(() => {
    const c = document.getElementById('c');
    const p = document.getElementById('p');
    return { containerW: c.getBoundingClientRect().width, probeW: p.getBoundingClientRect().width };
  });
  const contained = r.containerW <= 201;
  console.log(`${contained ? 'CONTAINED ' : 'LEAKS     '} container=${r.containerW.toFixed(0)}px probe=${r.probeW.toFixed(0)}px  ${probe.q}`);
}

await browser.close();
