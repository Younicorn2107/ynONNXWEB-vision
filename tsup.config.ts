import { defineConfig } from "tsup";

// Consumed through a bundler: onnxruntime and the core stay external, so the
// host resolves them once. No `clean` here: the build script cleans first.
export default defineConfig([
  {
    entry: {
      index: "src/index.ts",
      core: "src/core.ts",
    },
    format: ["esm"],
    dts: true,
    treeshake: true,
    external: ["onnxruntime-web", "@youni/onnxweb-core"],
  },
]);
