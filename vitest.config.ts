import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Node, not jsdom: everything under tests/ is pure logic or runs the real
    // onnxruntime-web on its WebAssembly backend. Anything needing `caches`,
    // `navigator.gpu` or a Worker belongs in the browser suite of the demo,
    // where those are real rather than mocked into agreeing with us.
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
