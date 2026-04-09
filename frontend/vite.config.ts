import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // This allows calling /predict in dev without CORS issues
      // if you prefer to proxy instead of using the full URL.
      // Uncomment below if needed:
      // '/api': { target: 'http://localhost:8000', rewrite: (path) => path.replace(/^\/api/, '') }
    }
  }
})
