// The suite serves dist/ with `vite preview`. A missing or STALE build would test code that is no
// longer in src/, and a stale one passes quietly, so refuse both. 2.0 also bundles 1.0's logic
// modules (the @engine alias = react-ui/src) and the settings schema, so an edit there makes the
// build stale too.
import { existsSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';

const uiRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = join(uiRoot, '..');

function newestMtime(path) {
  if (!existsSync(path)) return 0;
  const st = statSync(path);
  if (!st.isDirectory()) return st.mtimeMs;
  let newest = 0;
  for (const entry of readdirSync(path, { withFileTypes: true })) {
    newest = Math.max(newest, newestMtime(join(path, entry.name)));
  }
  return newest;
}

export default function globalSetup() {
  const index = join(uiRoot, 'dist', 'index.html');
  if (!existsSync(index)) {
    throw new Error('react-ui-v2/dist is missing. Run `npm run build` in react-ui-v2 first.');
  }
  const built = statSync(index).mtimeMs;
  const inputs = [
    join(uiRoot, 'index.html'),
    join(uiRoot, 'src'),
    join(uiRoot, 'public'),
    join(repoRoot, 'react-ui', 'src'),
    join(repoRoot, 'app', 'settings.schema.json'),
  ];
  const stale = inputs.filter((p) => newestMtime(p) > built);
  if (stale.length) {
    throw new Error(`react-ui-v2/dist is older than ${stale.join(', ')}. Run \`npm run build\` in react-ui-v2 so the suite tests the code as it is.`);
  }
}
