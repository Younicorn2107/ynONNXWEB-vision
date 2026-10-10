import { describe, expect, it } from "vitest";

import { TextFollower, TranslationCache, inpaintTelea, parseCharacterDict, type TextBlock } from "../../src/core.js";

describe("scene text pieces", () => {
  it("reads PaddleOCR's character dictionary from inference.yml (quoted, plain, escaped)", () => {
    const yml = "PostProcess:\n  character_dict:\n  - '!'\n  - $\n  - ''''\n  - \"\\\\\"\n  - A\n  name: CTCLabelDecode\n";
    expect(parseCharacterDict(yml)).toEqual(["!", "$", "'", "\\", "A"]);
  });

  it("translates each text once; live answers come from the cache", async () => {
    let calls = 0;
    const cache = new TranslationCache(async (texts) => {
      calls += 1;
      return texts.map((t) => t.toUpperCase());
    });
    expect(cache.get(["hello"])).toEqual([null]);
    await new Promise((r) => setTimeout(r, 0));
    expect(cache.get(["hello", "hello"])).toEqual(["HELLO", "HELLO"]);
    expect(await cache.all(["hello", "world"])).toEqual(["HELLO", "WORLD"]);
    expect(calls).toBe(2);
  });

  it("inpainting fills a hole from its surroundings", () => {
    const w = 20;
    const h = 20;
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i += 1) rgba.set([200, 100, 50, 255], i * 4);
    const mask = new Uint8Array(w * h);
    for (let y = 8; y < 12; y += 1) for (let x = 6; x < 14; x += 1) {
      mask[y * w + x] = 1;
      rgba.set([0, 0, 0, 255], (y * w + x) * 4);
    }
    inpaintTelea(rgba, w, h, mask, { x0: 0, y0: 0, x1: w, y1: h });
    const p = (10 * w + 10) * 4;
    expect([rgba[p], rgba[p + 1], rgba[p + 2]]).toEqual([200, 100, 50]);
  });
});

describe("text followed from frame to frame", () => {
  const w = 160;
  const h = 120;
  const scene = (dx: number, dy: number) => {
    const data = new Uint8Array(w * h * 3);
    for (let y = 0; y < h; y += 1)
      for (let x = 0; x < w; x += 1) {
        const sx = x - dx;
        const sy = y - dy;
        const v = 128 + 60 * Math.sin(sx / 5) * Math.cos(sy / 7) + 50 * Math.sin((sx + 2 * sy) / 11);
        data.fill(Math.max(0, Math.min(255, Math.round(v))), (y * w + x) * 3, (y * w + x) * 3 + 3);
      }
    return { data, width: w, height: h, channels: 3 as const };
  };
  const block: TextBlock = {
    lines: [{ quad: [[40, 40], [100, 40], [100, 60], [40, 60]], text: "EXIT", confidence: 0.99, angle: 0, score: 0.9 }],
    text: "EXIT",
    rect: { x: 40, y: 40, w: 60, h: 20 },
    angle: 0,
    lineHeight: 20,
    color: [255, 255, 255],
    background: [0, 0, 0],
    align: "left",
    bold: false,
  };

  it("moves a block read three frames ago by the camera's motion since, in full-size pixels", () => {
    const follower = new TextFollower();

    [0, 1, 2, 3].forEach((i) => follower.track(scene(2 * i, -i), i, 2));
    const [moved] = follower.place([block], 0);
    expect(moved!.rect.x).toBeCloseTo(40 + 12, 0);
    expect(moved!.rect.y).toBeCloseTo(40 - 6, 0);
    expect(moved!.lines[0]!.quad[2][0]).toBeCloseTo(100 + 12, 0);

    expect(follower.place([block], 3)[0]!.rect.x).toBe(40);
  });

  it("leaves blocks of a frame it never saw where they are", () => {
    const follower = new TextFollower();
    follower.track(scene(0, 0), 10);
    expect(follower.place([block], 4)[0]!.rect).toEqual(block.rect);
  });
});
