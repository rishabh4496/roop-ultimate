import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const ALLOWLIST_PATH = join(here, 'allowlist.json');

// Tab ids as App.jsx's ALL_TABS names them. `processing` is transient: it is in
// the nav only while a run exists, but its hash route always renders.
export const TABS = [
  { id: 'home', label: 'Home' },
  { id: 'faceswap', label: 'Face Swap' },
  { id: 'batch', label: 'Batch Matrix' },
  { id: 'processing', label: 'Processing' },
  { id: 'facemgr', label: 'Face Manager' },
  { id: 'extras', label: 'Editor' },
  { id: 'gallery', label: 'Outputs' },
  { id: 'history', label: 'History' },
  { id: 'settings', label: 'Settings' },
];

// 1x1 transparent PNG -- a valid image, so the UI's happy path runs.
export const PNG_1X1 = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
  'base64',
);

export const loadAllowlist = () => JSON.parse(readFileSync(ALLOWLIST_PATH, 'utf8'));

// Everything that finds a known failure goes through here so that "update the
// baseline" is one switch (`npm run test:e2e:baseline`) and the file stays the
// single record of what is currently broken.
export const UPDATE_BASELINE = process.env.E2E_UPDATE_BASELINE === '1';

export function saveAllowlist(mutate) {
  const data = loadAllowlist();
  mutate(data);
  writeFileSync(ALLOWLIST_PATH, `${JSON.stringify(data, null, 2)}\n`);
}

const SETTLE_MS = 1500;

// The app is "up" once the shell has bootstrapped (/api/meta + /api/settings
// answered, so the nav exists and a tab is current) and the lazy tab chunk has
// replaced its spinner.
export async function openTab(page, id) {
  await page.goto(`/#/${id}`);
  await page.locator('nav[aria-label="Main"] button[aria-current="page"]').waitFor();
  await page.locator('.deferred-fallback').waitFor({ state: 'detached' });
  // Let route-level fetches (state, settings panels) land so the first read of
  // the DOM is the loaded view, not its skeleton. NOT `networkidle`: Face Swap
  // never reaches it (that is exactly what idle-requests.spec.js measures), so
  // waiting for quiet would turn a finding into a 60 s timeout.
  await page.waitForTimeout(SETTLE_MS);
}

/**
 * Open Settings on a given tab ('providers' | 'cpu' | 'paths' | 'diagnostics' | 'general') by seeding the
 * value the app itself persists (store/workspaceStore.js, key roop_workspace), before the page loads.
 * Settings is five tabs behind one hash; specs about a control name the tab it lives on instead of
 * clicking through, and still go through the real route.
 */
export const startOnSettingsTab = (page, id) => page.addInitScript((tab) => {
  try { window.localStorage.setItem('roop_workspace', JSON.stringify({ settingsTab: tab })); } catch { /* storage blocked */ }
}, id);

// Views that are not a tab of their own but a state a tab can be in, and that
// have their own markup. The Batch Matrix tab is four different forms behind one
// hash: auditing only its default left Strategies 2-4 (16 critical axe
// violations) unseen.
export const STRATEGY_BUTTONS = '[class~="lg:grid-cols-4"] > button';
export const STATES = [
  { id: 'batch/grouped', tab: 'batch', strategy: 1 },
  { id: 'batch/matrix', tab: 'batch', strategy: 2 },
  { id: 'batch/recipes', tab: 'batch', strategy: 3 },
  // Settings is five tabs behind one hash; auditing only the default left four of them unseen.
  { id: 'settings/cpu', tab: 'settings', settingsTab: /^CPU/ },
  { id: 'settings/paths', tab: 'settings', settingsTab: /^Paths/ },
  { id: 'settings/diagnostics', tab: 'settings', settingsTab: /^Runtime Diagnostics/ },
  { id: 'settings/general', tab: 'settings', settingsTab: /^Appearance/ },
];

export async function openState(page, state) {
  await openTab(page, state.tab);
  if (state.settingsTab) await page.getByRole('tab', { name: state.settingsTab }).click();
  else await page.locator(STRATEGY_BUTTONS).nth(state.strategy).click();
  await page.waitForTimeout(600);
}

/** Every view the a11y and contrast audits cover: the 9 tabs, then the states. */
export const VIEWS = [
  ...TABS.map((t) => ({ id: t.id, open: (page) => openTab(page, t.id) })),
  ...STATES.map((s) => ({ id: s.id, open: (page) => openState(page, s) })),
];

// A solid-colour PNG of any size, written by hand (zlib + CRC32), for tests that need a frame with
// real dimensions: the mock's placeholders are SVGs, which never decode into a canvas frame.
import { deflateSync } from 'node:zlib';

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
const crc32 = (buf) => {
  let c = 0xffffffff;
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};
const chunk = (type, data) => {
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
};

/** `pixel(x, y) -> [r, g, b]` (default: a diagonal gradient so a mis-mapped picture is visible). */
export function makePng(w, h, pixel = (x, y) => [x % 256, y % 256, 128]) {
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    const row = y * (w * 3 + 1);
    raw[row] = 0;
    for (let x = 0; x < w; x++) {
      const [r, g, b] = pixel(x, y);
      raw[row + 1 + x * 3] = r; raw[row + 2 + x * 3] = g; raw[row + 3 + x * 3] = b;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0)),
  ]);
}
