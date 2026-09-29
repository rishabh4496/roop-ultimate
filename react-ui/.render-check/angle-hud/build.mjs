// Build the Angle HUD harness page into argv[2] (an output directory).
// Run from react-ui/ so Tailwind's source detection sees src/.
import { build } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import process from 'node:process';

const here = dirname(fileURLToPath(import.meta.url));
const outDir = process.argv[2] ? resolve(process.argv[2]) : join(here, 'dist');

await build({
  root: here,
  base: './',
  configFile: false,
  logLevel: 'warn',
  plugins: [react()],
  css: { postcss: join(here, '..', '..') },
  build: { outDir, emptyOutDir: true },
});
console.log(`angle-hud harness built -> ${outDir}`);
