// End-to-end checks for the React UI: axe, idle-request budget, nav visibility
// and tab-stop count. They run against a PRODUCTION build served by
// `vite preview`, with the mock server (react-ui/mock-server) standing in for
// app/api.py -- no GPU, no Python, no running app needed.
//
//   browser -> vite preview :PREVIEW_PORT --(proxy /api, /ws)--> mock :MOCK_PORT
//
// The proxy is vite.config.js's own (it turns on when ROOP_API_PORT is set), so
// the wiring under test is the one `npm run preview` has.
//
// The mock serves SVG placeholders and invented telemetry. These tests say
// nothing about the real backend; they check what the SHELL does with whatever
// it is given.
import { defineConfig, devices } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const uiRoot = join(here, '..');
const repoRoot = join(uiRoot, '..');

// Off the usual 3000/5173/8001 so a running dev session or backend is never
// mistaken for the fixture (or killed by it).
export const PREVIEW_PORT = Number(process.env.E2E_PREVIEW_PORT) || 4310;
export const MOCK_PORT = Number(process.env.E2E_MOCK_PORT) || 4311;

// Direct `node <entry>` rather than `npx`: npx would download a missing tool
// instead of failing, and the first run would then depend on the network.
const node = `"${process.execPath}"`;

export default defineConfig({
  testDir: here,
  testMatch: '*.spec.js',
  globalSetup: join(here, 'global-setup.js'),
  outputDir: join(here, 'test-results'),
  reporter: [['list'], ['html', { open: 'never', outputFolder: join(here, 'playwright-report') }]],
  // One worker: the mock is a single stateful process and the idle-request
  // tests count requests, so a second page hitting it would be noise.
  workers: 1,
  fullyParallel: false,
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: `http://127.0.0.1:${PREVIEW_PORT}`,
    ...devices['Desktop Chrome'],
    // framer-motion honours this through <MotionConfig reducedMotion="user">,
    // so axe never samples a half-faded colour.
    reducedMotion: 'reduce',
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      // NODE_ENV=production makes the mock serve dist/ statically instead of
      // starting a second Vite in middleware mode; we only want its /api here.
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
      url: `http://127.0.0.1:${PREVIEW_PORT}/`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
