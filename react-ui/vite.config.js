import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiPort = process.env.ROOP_API_PORT || '8001'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // The Pinokio launcher (start_react.js) assigns a dedicated dev-server port
    // via PORT; Vite doesn't read PORT on its own, so wire it up here (falls
    // back to Vite's default 5173 when unset).
    port: process.env.PORT ? Number(process.env.PORT) : undefined,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${apiPort}`,
        changeOrigin: true,
      }
    }
  }
})
