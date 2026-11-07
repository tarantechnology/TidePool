/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Backend routes already start with /api, so the prefix stays.
    proxy: {
      '/api': { target: process.env.BACKEND_URL || 'http://localhost:8000', changeOrigin: true },
    },
  },
  test: {
    environment: 'node',
  },
})
