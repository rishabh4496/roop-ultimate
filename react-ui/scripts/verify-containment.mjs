/**
 * Measure each of the four audited sites with its REAL markup and the REAL
 * production CSS, before vs after, and report whether the fix changes anything.
 *
 * Method note: the container must NOT clip, or the measurement is meaningless.
 * A parent with `overflow:hidden` hides the leak instead of reporting it, which
 * is exactly the mistake that made an earlier run of this check show "no
 * overflow" for cases that do overflow. Here the frame is a fixed width with
 * overflow VISIBLE, and we compare each element's right edge against the
 * frame's right edge.
 *
 * Per CSS flexbox §4.5, a flex item's automatic minimum size applies only when
 * its computed `overflow` is `visible`. `truncate` sets overflow:hidden, so a
 * truncating element that is ITSELF the flex item already floors at 0 and needs
 * no min-w-0. The fix only matters for an intermediate wrapper that stays
 * overflow:visible. These measurements decide which of the four sites is which.
 */
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const REQ = createRequire('G:/pinokio/cache/npm_config_cache/_npx/e41f203b7505f1fb/');
const { chromium } = REQ('playwright-core');
const CHROME = 'C:/Users/rishr/AppData/Local/ms-playwright/chromium-1234/chrome-win64/chrome.exe';
const DIST = 'G:/pinokio/api/roop-ultimate/react-ui/dist/assets';

const CSS = fs.readFileSync(path.join(DIST, fs.readdirSync(DIST).find((f) => f.endsWith('.css'))), 'utf8');

const LONGVAL = 'NVIDIA GeForce RTX 4070 / TensorRT 10.4.0 fp16 mixed precision';
const LONGLABEL = 'Adaptive enhancer profile quality tier selection';
const LONGNAME = 'this_is_an_extremely_long_generated_filename_00001.mp4';

const CASES = [
  {
    name: 'DiagnosticsPanel: run-config value',
    width: 320,
    before: `<div class="px-3 py-2.5"><div class="mt-2 space-y-1">
      <div class="flex items-baseline justify-between gap-2 font-mono text-micro">
        <span class="text-white/30">execution_provider</span>
        <span class="truncate text-right font-semibold text-white/60" data-m>${LONGVAL}</span>
      </div></div></div>`,
    after: `<div class="px-3 py-2.5"><div class="mt-2 space-y-1">
      <div class="flex items-baseline justify-between gap-2 font-mono text-micro">
        <span class="shrink-0 text-white/30">execution_provider</span>
        <span class="min-w-0 truncate text-right font-semibold text-white/60" data-m>${LONGVAL}</span>
      </div></div></div>`,
  },
  {
    name: 'QualityProfilesModal: settings grid cell label',
    width: 360,
    before: `<div class="grid grid-cols-2 gap-1.5">
      <div class="px-2 py-1 rounded text-nano flex items-center justify-between bg-white/5" data-m>
        <span class="truncate">${LONGLABEL}</span><span class="font-mono font-bold">BALANCED</span>
      </div><div class="px-2 py-1 rounded text-nano flex items-center justify-between bg-white/5">
        <span class="truncate">Mask</span><span class="font-mono font-bold">XSeg</span></div></div>`,
    after: `<div class="grid grid-cols-2 gap-1.5">
      <div class="min-w-0 px-2 py-1 rounded text-nano flex items-center justify-between bg-white/5" data-m>
        <span class="min-w-0 truncate">${LONGLABEL}</span><span class="shrink-0 font-mono font-bold">BALANCED</span>
      </div><div class="min-w-0 px-2 py-1 rounded text-nano flex items-center justify-between bg-white/5">
        <span class="min-w-0 truncate">Mask</span><span class="shrink-0 font-mono font-bold">XSeg</span></div></div>`,
  },
  {
    name: 'RunHistory: FPS bar column',
    width: 300,
    before: `<div class="flex items-end gap-2 h-24 pt-2">${Array.from({ length: 5 }, (_, i) =>
      `<div class="flex-1 flex flex-col items-center gap-1 group relative"${i === 0 ? ' data-m' : ''}>
        <div class="w-full bg-emerald-500/30 rounded-t" style="height:60%"></div>
        <span class="text-nano font-mono text-white/40 truncate max-w-full">${i === 0 ? LONGNAME : '24'}</span>
      </div>`).join('')}</div>`,
    after: `<div class="flex items-end gap-2 h-24 pt-2">${Array.from({ length: 5 }, (_, i) =>
      `<div class="flex-1 min-w-0 flex flex-col items-center gap-1 group relative"${i === 0 ? ' data-m' : ''}>
        <div class="w-full bg-emerald-500/30 rounded-t" style="height:60%"></div>
        <span class="text-nano font-mono text-white/40 truncate max-w-full">${i === 0 ? LONGNAME : '24'}</span>
      </div>`).join('')}</div>`,
  },
];

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

const run = async (width, body) => {
  await page.setContent(
    `<!doctype html><html><head><meta charset="utf-8"><style>${CSS}</style>
     <style>html,body{margin:0;background:#09090b}
     /* overflow VISIBLE: we want to observe the leak, not clip it away */
     #frame{width:${width}px;overflow:visible}</style></head>
     <body><div id="frame">${body}</div></body></html>`,
    { waitUntil: 'load' },
  );
  await page.evaluate(() => document.fonts.ready);
  return page.evaluate(() => {
    const frame = document.getElementById('frame');
    const fr = frame.getBoundingClientRect();
    const el = document.querySelector('[data-m]');
    const er = el.getBoundingClientRect();
    // widest right edge anywhere in the subtree: catches a child spilling out
    let maxRight = fr.left;
    frame.querySelectorAll('*').forEach((n) => {
      const r = n.getBoundingClientRect();
      if (r.width > 0) maxRight = Math.max(maxRight, r.right);
    });
    return {
      frameW: Math.round(fr.width),
      measuredW: Math.round(er.width),
      spillPx: Math.round(maxRight - fr.right),
    };
  });
};

let changed = 0;
for (const c of CASES) {
  const b = await run(c.width, c.before);
  const a = await run(c.width, c.after);
  const verdict = b.spillPx > 1 && a.spillPx <= 1 ? 'FIX REQUIRED (leak closed)'
    : b.spillPx <= 1 && a.spillPx <= 1 ? 'already contained before the change (fix is a no-op)'
    : 'STILL LEAKS';
  if (b.spillPx > 1) changed++;
  console.log(`\n${c.name}  [frame ${c.width}px]`);
  console.log(`   before: element ${b.measuredW}px, spills ${b.spillPx}px past the frame`);
  console.log(`   after : element ${a.measuredW}px, spills ${a.spillPx}px past the frame`);
  console.log(`   => ${verdict}`);
}

await browser.close();
console.log(`\n${changed}/${CASES.length} sites had a real, measurable leak.`);
