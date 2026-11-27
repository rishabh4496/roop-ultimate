import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const ENGINE = resolve(here, '../react-ui/src')

const apiPort = process.env.ROOP_API_PORT || '8001'
const port = process.env.PORT ? Number(process.env.PORT) : undefined

// `npm run dev` only: in production the FastAPI backend serves dist/ at /v2/ itself (same origin,
// no proxy). The dev server needs the proxy so /api and the /ws sockets reach a running backend.
const useProxy = Boolean(process.env.ROOP_STANDALONE_DEV || process.env.ROOP_API_PORT)
const proxy = useProxy ? {
  '/api': { target: `http://127.0.0.1:${apiPort}`, changeOrigin: true },
  '/ws': { target: `ws://127.0.0.1:${apiPort}`, ws: true, changeOrigin: true },
  '/outputs': { target: `http://127.0.0.1:${apiPort}`, changeOrigin: true },
} : undefined

export default defineConfig({
  // Served by app/api.py under /v2/ beside React UI 1.0 at "/". Routing is hash-based, so only
  // asset URLs need the prefix and Vite rewrites them all from `base`.
  base: '/v2/',
  plugins: [react()],
  resolve: {
    alias: {
      // React UI 2.0 has its own screens and components - none of 1.0's visuals - but it talks to the
      // engine through 1.0's LOGIC modules: the API client, the telemetry/queue hooks and the pure
      // selection/mapping builders. Importing them (not copying) means the engine contract has one
      // source: the first React UI 2.0 re-derived it and shipped unable to create a job.
      // docs/UI_V2_VISUAL_AUDIT.md lists every module imported this way; a test checks none is visual.
      '@engine': ENGINE,
      // The generated settings schema (tools/gen_settings.py from app/settings.py): the Settings
      // screen renders every UI setting from it, so a setting added to the engine appears here.
      '@schema': resolve(here, '../app/settings.schema.json'),
    },
    // 1.0's modules sit outside this root; without dedupe their `react` / `zustand` imports resolve
    // to react-ui/node_modules and the app would run two Reacts (hooks then throw).
    dedupe: ['react', 'react-dom', 'zustand'],
  },
  server: {
    port,
    proxy,
    fs: { allow: [here, ENGINE, resolve(here, '../app/settings.schema.json')] },
  },
  preview: { port, proxy },
})
