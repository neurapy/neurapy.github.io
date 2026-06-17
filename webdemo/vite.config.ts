import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize, resolve, sep } from "node:path";
import { defineConfig, type Plugin } from "vite";

const tinyDataRoot = resolve("test-results", "tiny-data");
const tinyDataPrefix = "/fixtures/tiny-data/";

function contentTypeFor(path: string): string {
  const ext = extname(path).toLowerCase();
  if (ext === ".json") return "application/json; charset=utf-8";
  if (ext === ".svg") return "image/svg+xml";
  if (ext === ".png") return "image/png";
  return "application/octet-stream";
}

function tinyDataFixturePlugin(): Plugin {
  return {
    name: "pinnfluence-tiny-data-fixtures",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const pathname = request.url ? new URL(request.url, "http://localhost").pathname : "";
        if (!pathname.startsWith(tinyDataPrefix)) {
          next();
          return;
        }

        let relative: string;
        try {
          relative = decodeURIComponent(pathname.slice(tinyDataPrefix.length));
        } catch {
          response.statusCode = 400;
          response.end("Bad request");
          return;
        }

        const filePath = normalize(join(tinyDataRoot, relative));
        if (filePath !== tinyDataRoot && !filePath.startsWith(`${tinyDataRoot}${sep}`)) {
          response.statusCode = 403;
          response.end("Forbidden");
          return;
        }
        if (!existsSync(filePath) || !statSync(filePath).isFile()) {
          response.statusCode = 404;
          response.end("Not found");
          return;
        }

        response.setHeader("Content-Type", contentTypeFor(filePath));
        createReadStream(filePath).pipe(response);
      });
    },
  };
}

export default defineConfig({
  base: "./",
  publicDir: "public",
  plugins: [tinyDataFixturePlugin()],
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
