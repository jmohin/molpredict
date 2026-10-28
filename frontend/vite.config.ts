import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Ketcher expects Node's process.env (and `global`, shimmed in index.html)...
  define: { 'process.env': {} },
  // ...and Node's `events`; the trailing slash forces the npm polyfill over Vite's node-builtin stub.
  resolve: { alias: { events: 'events/' } },
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
