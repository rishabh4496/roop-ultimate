// Re-record e2e/allowlist.json from what the UI does RIGHT NOW. Run it after
// fixing something (to ratchet the file down) or after deciding that a new
// failure is acceptable (to record it) -- and read the diff before committing,
// because whatever is in that file stops failing the suite.
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const uiRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const cli = join(uiRoot, 'node_modules', '@playwright', 'test', 'cli.js');

const run = spawnSync(process.execPath, [cli, 'test', '-c', 'e2e/playwright.config.js'], {
  cwd: uiRoot,
  stdio: 'inherit',
  env: { ...process.env, E2E_UPDATE_BASELINE: '1' },
});
process.exit(run.status ?? 1);
