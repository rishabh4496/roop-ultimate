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
  await page.locator('header nav button[aria-current="page"]').waitFor();
  await page.locator('.deferred-fallback').waitFor({ state: 'detached' });
  // Let route-level fetches (state, settings panels) land so the first read of
  // the DOM is the loaded view, not its skeleton. NOT `networkidle`: Face Swap
  // never reaches it (that is exactly what idle-requests.spec.js measures), so
  // waiting for quiet would turn a finding into a 60 s timeout.
  await page.waitForTimeout(SETTLE_MS);
}
