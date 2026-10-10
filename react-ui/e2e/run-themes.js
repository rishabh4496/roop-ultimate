// Runs the opt-in theme-contrast sweep (contrast-themes.spec.js).
//   node e2e/run-themes.js            check against the recorded baselines
//   node e2e/run-themes.js --update   re-record them
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const uiRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const cli = join(uiRoot, 'node_modules', '@playwright', 'test', 'cli.js');
const update = process.argv.includes('--update');

const run = spawnSync(process.execPath, [cli, 'test', '-c', 'e2e/playwright.config.js', 'contrast-themes'], {
  cwd: uiRoot,
  stdio: 'inherit',
  env: { ...process.env, E2E_THEMES: '1', ...(update ? { E2E_UPDATE_BASELINE: '1' } : {}) },
});
process.exit(run.status ?? 1);
