import { describe, expect, it } from "vitest";

import { KeyframeTimeline, type BoundingBox, type PoseResult } from "../../src/core.js";

const box = (x: number, y: number, trackId: number | null, w = 20, h = 40): BoundingBox => ({
  x1: x, y1: y, x2: x + w, y2: y + h, confidence: 0.9, classId: 0, className: "person", trackId, isInterpolated: false,
});

const pose = (boxes: BoundingBox[]): PoseResult => ({
  modelName: "m", task: "pose", source: null, inferenceTimeMs: 5, imageShape: [480, 640], boxes,
  keypoints: boxes.map((b) => ({ box: b, points: [{ x: b.x1 + 10, y: b.y1 + 5, confidence: 0.9, name: "nose" }] })),
});

describe("results for every frame from keyframes", () => {
  it("interpolates tracked boxes and keypoints between two keyframes", () => {
    const t = new KeyframeTimeline();
    t.add(0, pose([box(100, 100, 1)]));
    t.add(10, pose([box(200, 120, 1)]));
    const mid = t.resultAt(5) as PoseResult;
    expect(mid.boxes[0]!.x1).toBeCloseTo(150, 6);
    expect(mid.boxes[0]!.y1).toBeCloseTo(110, 6);
    expect(mid.boxes[0]!.isInterpolated).toBe(true);
    expect(mid.keypoints[0]!.points[0]!.x).toBeCloseTo(160, 6);
    expect((t.resultAt(10) as PoseResult).boxes[0]!.isInterpolated).toBe(false);
  });

  it("a track on one side only shows until halfway", () => {
    const t = new KeyframeTimeline();
    t.add(0, pose([box(100, 100, 1), box(300, 100, 2)]));
    t.add(10, pose([box(110, 100, 1)]));
    expect((t.resultAt(3) as PoseResult).boxes.map((b) => b.trackId)).toEqual([1, 2]);
    expect((t.resultAt(7) as PoseResult).boxes.map((b) => b.trackId)).toEqual([1]);
  });

  it("carries on at a damped speed after the last keyframe; holds when asked; stops after maxFrames", () => {
    const t = new KeyframeTimeline();
    t.add(0, pose([box(100, 100, 1)]));
    t.add(4, pose([box(120, 100, 1)]));
    const one = t.resultAt(5, { damping: 1 }) as PoseResult;
    expect(one.boxes[0]!.x1).toBeCloseTo(125, 6);
    const damped = t.resultAt(24, { damping: 0.5 }) as PoseResult;

    expect(damped.boxes[0]!.x1).toBeLessThanOrEqual(125.0001);
    expect(damped.keypoints[0]!.points[0]!.x - damped.boxes[0]!.x1).toBeCloseTo(10, 6);
    expect((t.resultAt(10, { after: "hold" }) as PoseResult).boxes[0]!.x1).toBe(120);
    expect((t.resultAt(40, { maxFrames: 30 }) as PoseResult).boxes).toHaveLength(0);
  });

  it("does not drift from jitter: a track slower than the deadband stays put", () => {
    const t = new KeyframeTimeline();
    t.add(0, pose([box(100, 100, 1)]));
    t.add(10, pose([box(100.5, 100, 1)]));
    expect((t.resultAt(20, { damping: 1 }) as PoseResult).boxes[0]!.x1).toBeCloseTo(100.5, 6);
  });
});
