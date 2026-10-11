import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend always talks to the backend via same-origin "/api".
// In dev, Vite proxies "/api" to the FastAPI server on port 8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
