// The suite serves dist/ with `vite preview`. A missing or STALE build would
// test code that is no longer in src/, and a stale one passes quietly -- so
// refuse both here instead of letting a green run mean "the old bundle is fine".
import { existsSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';

const uiRoot = join(dirname(fileURLToPath(import.meta.url)), '..');

function newestMtime(dir) {
  let newest = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    newest = Math.max(newest, entry.isDirectory() ? newestMtime(full) : statSync(full).mtimeMs);
  }
  return newest;
}

export default function globalSetup() {
  const index = join(uiRoot, 'dist', 'index.html');
  if (!existsSync(index)) {
    throw new Error('react-ui/dist is missing. Run `npm run build` first (`npm run check` does).');
  }
  const built = statSync(index).mtimeMs;
  const sources = [join(uiRoot, 'src'), join(uiRoot, 'public')].filter(existsSync);
  const edited = Math.max(
    statSync(join(uiRoot, 'index.html')).mtimeMs,
    ...sources.map(newestMtime),
  );
  if (edited > built) {
    throw new Error('react-ui/dist is older than src/. Run `npm run build` so the e2e suite tests the code as it is.');
  }
}
