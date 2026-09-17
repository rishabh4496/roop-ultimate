/**
 * Runner: transform the JSX through Vite's own SSR pipeline, then execute.
 *
 * Using Vite rather than a standalone esbuild/babel step is deliberate — it
 * resolves aliases, CSS imports and the exact plugin chain the production build
 * uses, so what gets executed here is what actually ships. A separate transform
 * could disagree with the build and give a green run for code the app never
 * sees.
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
  await server.ssrLoadModule(join(here, 'render-processing.mjs'));
  // The module reports through a global. It cannot use process.exit: Vite's SSR
  // runner swallows it, which made a FAILING run exit 0 — caught by deliberately
  // reverting the progress clamp and watching the harness report two failures
  // and then succeed anyway.
  const failures = globalThis.__RENDER_CHECK_FAILURES__;
  if (failures === undefined) {
    console.error('render check did not report a result (module did not finish)');
    code = 1;
  } else {
    code = failures === 0 ? 0 : 1;
  }
} catch (err) {
  console.error(err);
  code = 1;
} finally {
  await server.close();
}
process.exit(code);
