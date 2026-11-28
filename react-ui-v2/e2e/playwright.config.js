// React UI 2.0's browser tests. They run against a PRODUCTION build of 2.0 served by `vite preview`
// (base /v2/), with 1.0's mock server (react-ui/mock-server) standing in for app/api.py: no GPU, no
// Python, no running app needed.
//
//   browser -> vite preview :PREVIEW_PORT/v2/ --(proxy /api, /ws, /outputs)--> mock :MOCK_PORT
//
// The proxy is vite.config.js's own (on when ROOP_API_PORT is set), so the wiring under test is
// the one `npm run preview` has.
//
// Where the mock's answer differs from the engine's, fixtures.js answers with the ENGINE's shape
// (read from app/routes_*.py) so a test checks what 2.0 does with what the engine really sends.
//
// live.spec.js drives the REAL engine and is not part of this run: see playwright.live.config.js.
import { defineConfig, devices } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const uiRoot = join(here, '..');
const repoRoot = join(uiRoot, '..');

// Off 1.0's 4310/4311 so both suites can run at once, and off the usual dev/backend ports so a
// running session is never mistaken for the fixture.
export const PREVIEW_PORT = Number(process.env.E2E_V2_PREVIEW_PORT) || 4410;
export const MOCK_PORT = Number(process.env.E2E_V2_MOCK_PORT) || 4411;

// `node <entry>` rather than `npx`: npx would download a missing tool instead of failing.
const node = `"${process.execPath}"`;

export default defineConfig({
  testDir: here,
  testMatch: '*.spec.js',
  testIgnore: 'live.spec.js',
  globalSetup: join(here, 'global-setup.js'),
  outputDir: join(here, 'test-results'),
  reporter: [['list'], ['html', { open: 'never', outputFolder: join(here, 'playwright-report') }]],
  // One worker: the mock is a single stateful process (the cast, the queue, a render in flight).
  workers: 1,
  fullyParallel: false,
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: `http://127.0.0.1:${PREVIEW_PORT}/v2/`,
    ...devices['Desktop Chrome'],
    viewport: { width: 1600, height: 960 },
    reducedMotion: 'reduce',
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      // NODE_ENV=production: the mock serves only its API (no second Vite in middleware mode).
      command: `${node} ${join(repoRoot, 'node_modules', 'tsx', 'dist', 'cli.mjs')} react-ui/mock-server/server.ts`,
      cwd: repoRoot,
      env: { NODE_ENV: 'production', PORT: String(MOCK_PORT) },
      url: `http://127.0.0.1:${MOCK_PORT}/api/meta`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      command: `${node} ${join(uiRoot, 'node_modules', 'vite', 'bin', 'vite.js')} preview --host 127.0.0.1 --port ${PREVIEW_PORT} --strictPort`,
      cwd: uiRoot,
      env: { ROOP_API_PORT: String(MOCK_PORT) },
      url: `http://127.0.0.1:${PREVIEW_PORT}/v2/`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
