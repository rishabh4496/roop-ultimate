import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiPort = process.env.ROOP_API_PORT || '8001'
const port = process.env.PORT ? Number(process.env.PORT) : undefined

// The proxy below is for `npm run dev` ONLY.
//
// In production there is no Vite server and no proxy at all: the launcher runs
// `vite build`, and the FastAPI backend serves react-ui/dist itself (see the
// SPA fallback at the bottom of app/api.py). That makes the UI same-origin
// with the API, so `window.location.origin` in src/api.js already points at
// the backend and /ws/telemetry upgrades directly against it.
//
// That change exists because running `vite preview` as a second server put a
// Node toolchain on the RUNTIME path: `vite preview` refuses to start when
// dist/ is missing, so any build failure on another machine (Vite 8 needs Node
// ^20.19 || >=22.12, and rolldown ships per-platform binaries) took down the
// server rather than just the build, and the user got a Vite error instead of
// the app.
//
// The dev server still needs the proxy, because there the UI is served by Vite
// on its own port and /api would otherwise hit the dev server and 404 -- which
// the UI reports as "Cannot reach backend" while the backend is perfectly
// healthy. `/ws` needs its OWN entry: proxy rules match by path prefix so
// '/api' does not cover it, and `ws: true` is required for Vite to forward the
// HTTP Upgrade handshake at all.
// The proxy below is only needed when running Vite as a standalone dev server
// alongside a separate Python backend on ROOP_API_PORT. When running inside
// server.ts, Express handles /api directly.
const useProxy = Boolean(process.env.ROOP_STANDALONE_DEV || process.env.ROOP_API_PORT)
const proxy = useProxy ? {
  '/api': {
    target: `http://127.0.0.1:${apiPort}`,
    changeOrigin: true,
  },
  '/ws': {
    target: `ws://127.0.0.1:${apiPort}`,
    ws: true,
    changeOrigin: true,
  },
} : undefined

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Vite doesn't read PORT on its own; wire it up so a dev session can be
    // given a specific port (falls back to Vite's default 5173 when unset).
    port,
    proxy,
  },
  // Kept so `npm run preview` remains usable for hand-checking a build against
  // a running backend. It is NOT how the app is served.
  preview: {
    port,
    proxy,
  },
})
