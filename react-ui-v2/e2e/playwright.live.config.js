// React UI 2.0 against the REAL engine: `npm run test:e2e:live` in react-ui-v2.
//
// Not part of `npm run test:e2e`. It needs a running, IDLE backend (no render, no queue) that
// serves the built 2.0 at /v2/, with a target selected, a target person picked and a source in the
// cast. It renders twice on the GPU (a stopped partial and a complete 30-frame render, both kept in
// the output folder and the second in History), adds a queue job and removes it again, and puts the
// target's In/Out back as it found them.
//
//   ROOP_V2_LIVE_URL   the backend (default http://127.0.0.1:42003)
import { defineConfig, devices } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const BASE = (process.env.ROOP_V2_LIVE_URL || 'http://127.0.0.1:42003').replace(/\/$/, '');

export default defineConfig({
  testDir: here,
  testMatch: 'live.spec.js',
  outputDir: join(here, 'test-results'),
  reporter: [['list']],
  workers: 1,
  retries: 0,
  // A cold engine builds its TensorRT engines on the first render.
  timeout: 30 * 60_000,
  expect: { timeout: 20_000 },
  use: {
    baseURL: `${BASE}/v2/`,
    ...devices['Desktop Chrome'],
    viewport: { width: 1600, height: 960 },
    reducedMotion: 'reduce',
    trace: 'retain-on-failure',
  },
});
