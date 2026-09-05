import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiPort = process.env.ROOP_API_PORT || '8001'
const port = process.env.PORT ? Number(process.env.PORT) : undefined

// Both servers need the SAME `/api` proxy. The launcher serves the built app
// through `vite preview` (see start_react.js for why), and `preview` does not
// inherit anything from `server` — a proxy defined only under `server` leaves
// every fetch in the built app hitting the static server itself and 404ing,
// which surfaces as "Cannot reach backend" while the backend is perfectly fine.
const proxy = {
  '/api': {
    target: `http://127.0.0.1:${apiPort}`,
    changeOrigin: true,
  },
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // The Pinokio launcher (start_react.js) assigns a dedicated dev-server port
    // via PORT; Vite doesn't read PORT on its own, so wire it up here (falls
    // back to Vite's default 5173 when unset).
    port,
    proxy,
  },
  preview: {
    port,
    proxy,
  },
})
