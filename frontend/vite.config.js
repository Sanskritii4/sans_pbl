import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/network': 'http://localhost:8000',
      '/train': 'http://localhost:8000',
      '/route': 'http://localhost:8000',
      '/static-route': 'http://localhost:8000',
      '/metrics': 'http://localhost:8000',
      '/agent': 'http://localhost:8000',
    },
  },
})
