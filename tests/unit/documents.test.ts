import { describe, expect, it } from "vitest";

import { readCodes, readingOrder, stretchedGrey } from "../../src/core.js";

const block = (x: number, y: number, name: string) => ({ rect: { x, y, w: 10, h: 10 }, lineHeight: 20, name });

describe("documents", () => {
  it("reads blocks top to bottom, then left to right within a row", () => {
    const order = readingOrder([block(300, 105, "c"), block(10, 400, "d"), block(200, 98, "b"), block(10, 102, "a")]).map((b) => b.name);
    expect(order).toEqual(["a", "b", "c", "d"]);
  });

  it("stretches grey levels and doubles the size", () => {
    const img = stretchedGrey({ data: Uint8Array.of(100, 100, 100, 150, 150, 150), width: 2, height: 1 }, 2);
    expect([img.width, img.height]).toEqual([4, 2]);
    expect(img.data[0]).toBe(0);
    expect(img.data[3 * 3]).toBe(255);
  });

  it("says when the browser cannot read codes", async () => {
    expect(await readCodes({ data: new Uint8Array(3), width: 1, height: 1 }, null)).toBeNull();
  });
});
