/**
 * Runner for render-error-boundary.mjs: same shape as run.mjs (Vite's own SSR
 * pipeline, so the JSX and aliases are the ones the production build uses, and a
 * result reported through a global because Vite's SSR runner swallows
 * process.exit).
 */
import { createServer } from 'vite';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import process from 'node:process';

const here = dirname(fileURLToPath(import.meta.url));

const server = await createServer({
  root: join(here, '..'),
  configFile: join(here, '..', 'vite.config.js'),
  server: { middlewareMode: true, hmr: false },
  appType: 'custom',
  logLevel: 'error',
});

let code = 1;
try {
  await server.ssrLoadModule(join(here, 'render-error-boundary.mjs'));
  const failures = globalThis.__RENDER_CHECK_FAILURES__;
  if (failures === undefined) {
    console.error('error-boundary check did not report a result (module did not finish)');
  } else {
    code = failures === 0 ? 0 : 1;
  }
} catch (err) {
  console.error(err);
} finally {
  await server.close();
}
process.exit(code);
