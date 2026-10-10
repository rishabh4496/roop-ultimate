/**
 * Runner for any check module that needs Vite's resolver (extensionless imports, JSX, the
 * aliases the production build uses): `node .render-check/run-module.mjs render-api-client.mjs`.
 *
 * The module reports its result through `globalThis.__RENDER_CHECK_FAILURES__` (Vite's SSR
 * runner swallows process.exit, which once made a FAILING run exit 0).
 */
import { createServer } from 'vite';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import process from 'node:process';

const here = dirname(fileURLToPath(import.meta.url));
const target = process.argv[2];
if (!target) {
  console.error('usage: node .render-check/run-module.mjs <module.mjs>');
  process.exit(2);
}

const server = await createServer({
  root: join(here, '..'),
  configFile: join(here, '..', 'vite.config.js'),
  server: { middlewareMode: true, hmr: false },
  appType: 'custom',
  logLevel: 'error',
});

let code = 1;
try {
  await server.ssrLoadModule(join(here, target));
  const failures = globalThis.__RENDER_CHECK_FAILURES__;
  if (failures === undefined) console.error(`${target} did not report a result (module did not finish)`);
  else code = failures === 0 ? 0 : 1;
} catch (err) {
  console.error(err);
} finally {
  await server.close();
}
process.exit(code);
