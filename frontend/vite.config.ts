/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

/**
 * Vite config for the RAGOps console.
 *
 * The dev server proxies `/api` to the FastAPI backend so the browser never
 * issues a cross-origin request. CORS is therefore a non-issue in development
 * and the `X-API-Key` header flows through untouched. In production, set
 * `VITE_API_URL` to the absolute API origin (or leave it empty when the console
 * is served behind the same origin as the API).
 *
 * Vitest shares this file via `test`, so component tests resolve the exact same
 * path aliases and plugins as the dev server.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  const apiTarget = env.VITE_API_PROXY_TARGET ?? "http://localhost:8000";

  return {
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      port: 5173,
      strictPort: true,
      proxy: {
        "/api": {
          target: apiTarget,
          changeOrigin: true,
        },
      },
    },
    preview: {
      port: 4173,
    },
    build: {
      outDir: "dist",
      sourcemap: mode !== "production",
    },
    test: {
      environment: "jsdom",
      globals: true,
      setupFiles: ["./src/test/setup.ts"],
      css: false,
      include: ["src/**/*.{test,spec}.{ts,tsx}"],
    },
  };
});
