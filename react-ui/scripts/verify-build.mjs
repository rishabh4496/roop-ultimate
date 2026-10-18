#!/usr/bin/env node
/**
 * Build preconditions and postconditions for the launcher scripts.
 *
 *   node scripts/verify-build.mjs vite   # before `npm run build`: the vite executable is installed
 *   node scripts/verify-build.mjs dist   # after it: dist/index.html exists and is not empty
 *
 * Why this exists. install.js, update.js and start_react.js run `npm ci` and
 * `npm run build` in a Pinokio shell, and Pinokio does not stop a script because
 * a command inside a shell exited non-zero. When `npm ci` failed, the next line
 * ran anyway and the only message was the shell's own "'vite' is not recognized",
 * three screens below the real cause. These checks name the missing file and
 * print the marker that the launchers' `on: [{ break: true }]` patterns match, so
 * a missing dependency or an absent build stops the script at the step that
 * caused it instead of being committed as a finished install.
 *
 * Checks the file system rather than asking npm, so it is the same on Windows
 * (vite.cmd), macOS and Linux (vite).
 */
import { existsSync, statSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import process from 'node:process';

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(here, '..');
const BIN = join(ROOT, 'node_modules', '.bin');

const fail = (why) => {
  console.error(`[build-verify] FAIL: ${why}`);
  process.exit(1);
};

const checks = {
  vite() {
    const exe = process.platform === 'win32' ? 'vite.cmd' : 'vite';
    if (!existsSync(join(BIN, exe))) {
      fail(`node_modules/.bin/${exe} is missing; the dependency install did not complete`);
    }
    // `.bin/vite` is a shim; the package itself is what `npm run build` executes.
    if (!existsSync(join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js'))) {
      fail('node_modules/vite/bin/vite.js is missing; the vite package is incomplete');
    }
    console.log('[build-verify] vite installed');
  },
  dist() {
    const index = join(ROOT, 'dist', 'index.html');
    if (!existsSync(index) || statSync(index).size === 0) {
      fail('dist/index.html is missing or empty; the build did not produce a UI');
    }
    console.log('[build-verify] dist/index.html present');
  },
};

const check = checks[process.argv[2]];
if (!check) {
  fail(`unknown check ${JSON.stringify(process.argv[2])}; expected one of ${Object.keys(checks).join(', ')}`);
}
check();
