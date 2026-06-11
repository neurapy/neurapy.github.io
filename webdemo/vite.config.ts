import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  publicDir: "public",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: false,
    watch: {
      ignored: ["**/public/data/**"],
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
    strictPort: false,
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
