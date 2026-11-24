import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built output is served by the FastAPI app (backend/app/main.py mounts
// frontend/dist at /). During development, /api is proxied to the backend so
// the dev server and production behave identically.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    assetsInlineLimit: 0,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
