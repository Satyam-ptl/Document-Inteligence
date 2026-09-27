import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // Backend runs on :8000. Avoids CORS in dev; prod build can use
      // VITE_API_BASE_URL directly instead of the proxy.
      '/api': 'http://localhost:8000',
    },
  },
})
