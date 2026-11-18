import { describe, expect, it } from "vitest";

import { MODERATORS, readModeration, readModerationPreprocessor } from "../../src/core.js";

describe("moderation", () => {
  it("reads a two-class model into one unsafe probability", () => {
    const [safe, unsafe] = readModeration([2, 0, 0, 3], ["normal", "nsfw"], MODERATORS["falconsai"]!.unsafe, 0.5);
    expect(safe!.label).toBe("normal");
    expect(safe!.flagged).toBe(false);
    expect(safe!.unsafe).toBeCloseTo(1 / (1 + Math.exp(2)), 6);
    expect(unsafe!.flagged).toBe(true);
    expect(unsafe!.scores["nsfw"]).toBeCloseTo(1 / (1 + Math.exp(-3)), 6);
  });

  it("adds the unsafe grades of a graded model", () => {
    const [r] = readModeration([0, 0, 0, 0], ["neutral", "low", "medium", "high"], MODERATORS["freepik"]!.unsafe, 0.5);
    expect(r!.unsafe).toBeCloseTo(0.5, 6);
    expect(r!.flagged).toBe(true);
  });

  it("reads the preprocessor's size, mean and scale", () => {
    const p = readModerationPreprocessor({ size: { height: 448, width: 448 }, image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5], rescale_factor: 1 / 255 });
    expect([p.width, p.height, p.normalize]).toEqual([448, 448, true]);
    expect(readModerationPreprocessor({}).width).toBe(224);
  });
});
