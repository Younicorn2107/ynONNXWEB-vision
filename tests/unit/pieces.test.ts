/** The pieces with no Python twin to compare against, checked on their own terms. */

import { describe, expect, it } from "vitest";

import {
  MotionCompensator,
  ObjectTracker,
  decodeDetect,
  decodePose,
  detectTask,
  estimateSimilarity,
  exportResult,
  floatToHalf,
  fromHalf,
  halfToFloat,
  headLayout,
  inputShape,
  linearSumAssignment,
  parseNames,
  preprocess,
  propagate,
  publishedModel,
  rotatedIou,
  toHalf,
  type DetectionResult,
  type LetterboxMeta,
} from "../../src/core.js";

describe("half floats", () => {
  it("round-trips and rounds to nearest even", () => {
    for (const v of [0, 1, -2.5, 0.1, 65504, 6.1e-5, 1e-7, 3.14159]) {
      expect(Math.abs(halfToFloat(floatToHalf(v)) - v)).toBeLessThanOrEqual(Math.abs(v) * 1e-3 + 6e-8);
    }
    expect(floatToHalf(1)).toBe(0x3c00);
    expect(floatToHalf(-2)).toBe(0xc000);
    expect(floatToHalf(70000)).toBe(0x7c00);
    expect(Array.from(fromHalf(toHalf(Float32Array.of(0.5, 2))))).toEqual([0.5, 2]);
  });
});

describe("model metadata", () => {
  it("reads Ultralytics' names and the task from a name", () => {
    expect(parseNames("{0: 'person', 1: 'bicycle', 2: \"it's\"}")).toEqual(["person", "bicycle", "it's"]);
    expect(parseNames("{1: 'b', 0: 'a'}")).toEqual(["a", "b"]);
    expect(parseNames(undefined)).toBeNull();
    expect(detectTask("yolo26n-seg")).toBe("segment");
    expect(detectTask("org/yolov8n-pose:onnx/model.onnx")).toBe("pose");
    expect(detectTask("yolo11s")).toBe("detect");
    expect(publishedModel("yolo26n-pose")?.repo).toBe("onnx-community/yolo26n-pose-ONNX");
    expect(publishedModel("onnx-community/yolo26s-ONNX")?.task).toBe("detect");
    expect(publishedModel("yolov10n")).toBeNull();
    expect(publishedModel("yolo26n")?.precisions[0]).toBe("float16");
  });

  it("tells the head's layout from the graph", () => {
    const io = (inputs: string[], outputs: [string, (number | string)[]][], metadata: Record<string, string> = {}) => ({
      inputs: inputs.map((name) => ({ name, dims: [1, 3, 640, 640] })),
      outputs: outputs.map(([name, dims]) => ({ name, dims })),
      metadata,
    });
    expect(headLayout(io(["pixel_values"], [["logits", ["b", 300, 80]], ["pred_boxes", ["b", 300, 4]]]), 640)).toEqual({ nmsFree: true, normalized: true, split: true, imgsz: 640 });
    expect(headLayout(io(["pixel_values"], [["logits", ["b", 300, 57]]], { end2end: "True" }), 640)).toMatchObject({ nmsFree: true, normalized: true, split: false });
    expect(headLayout(io(["images"], [["output0", [1, 300, 6]]]), 640)).toMatchObject({ nmsFree: true, normalized: false });
    expect(headLayout(io(["images"], [["output0", [1, 84, 8400]]]), 640)).toMatchObject({ nmsFree: false });
    expect(headLayout(io(["images"], [["output0", [1, 300, 38]], ["output1", [1, 32, 160, 160]]], { end2end: "False", args: "{'nms': True}" }), 640).nmsFree).toBe(false);
    expect(headLayout(io(["images"], [["output0", [1, 300, 38]]], { args: "{'batch': 1, 'nms': True}" }), 640).nmsFree).toBe(true);
  });
});

const meta: LetterboxMeta = { originalShape: [1080, 810], scale: 640 / 1080, padTop: 0, padLeft: 80 };

describe("normalized exports (onnx-community)", () => {
  it("logits + pred_boxes: sigmoid, cx cy w h × input size, letterbox undone", () => {
    const logits = new Float32Array(3 * 2).fill(-9);
    logits[0 * 2 + 1] = 2;
    logits[2 * 2 + 0] = 0.5;
    const boxes = Float32Array.of(0.5, 0.5, 0.25, 0.5, 0, 0, 0, 0, 0.25, 0.25, 0.1, 0.1);
    const [r] = decodeDetect(
      [{ data: logits, dims: [1, 3, 2] }, { data: boxes, dims: [1, 3, 4] }],
      [meta],
      { classNames: ["a", "b"], modelName: "m", layout: { nmsFree: true, normalized: true, split: true, imgsz: 640 } },
    );
    expect(r!.boxes.map((b) => b.className)).toEqual(["b", "a"]);
    expect(r!.boxes[0]!.confidence).toBeCloseTo(1 / (1 + Math.exp(-2)), 6);

    expect(r!.boxes[0]!.x1).toBeCloseTo(270, 3);
    expect(r!.boxes[0]!.y2).toBeCloseTo(480 / (640 / 1080), 2);
  });

  it("pose rows: coordinates and keypoint confidences divided by the input size, undone", () => {
    const row = new Float32Array(6 + 51);
    row.set([0.25, 0.25, 0.5, 0.75, 0.9, 0]);
    for (let k = 0; k < 17; k += 1) row.set([0.375, 0.5, 0.95 / 640], 6 + 3 * k);
    const [r] = decodePose([{ data: row, dims: [1, 1, 57] }], [meta], {
      classNames: ["person"],
      modelName: "m",
      layout: { nmsFree: true, normalized: true, split: false, imgsz: 640 },
    });
    const p = r!.keypoints[0]!.points[0]!;
    expect(p.confidence).toBeCloseTo(0.95, 5);
    expect(p.x).toBeCloseTo((240 - 80) / (640 / 1080), 2);
    expect(p.name).toBe("nose");
  });
});

describe("assignment and rotated IoU", () => {
  const brute = (cost: number[][]): number => {
    const rows = cost.length;
    const cols = cost[0]!.length;
    let best = Infinity;
    const walk = (r: number, used: Set<number>, sum: number, left: number) => {
      if (left === 0 || r === rows) {
        if (left === 0) best = Math.min(best, sum);
        return;
      }
      walk(r + 1, used, sum, left);
      for (let c = 0; c < cols; c += 1) {
        if (used.has(c)) continue;
        used.add(c);
        walk(r + 1, used, sum + cost[r]![c]!, left - 1);
        used.delete(c);
      }
    };
    walk(0, new Set(), 0, Math.min(rows, cols));
    return best;
  };
  it("is optimal, square and rectangular", () => {
    let seed = 3;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    for (const [r, c] of [[4, 4], [3, 5], [5, 3], [1, 4], [6, 6]] as const) {
      const cost = Array.from({ length: r }, () => Array.from({ length: c }, rnd));
      const pairs = linearSumAssignment(cost);
      expect(pairs.length).toBe(Math.min(r, c));
      expect(new Set(pairs.map((p) => p[1])).size).toBe(pairs.length);
      const total = pairs.reduce((s, [i, j]) => s + cost[i]![j]!, 0);
      expect(total).toBeCloseTo(brute(cost), 9);
    }
  });

  it("rotated IoU: identical, disjoint, a quarter turn of a square", () => {
    expect(rotatedIou([10, 10, 4, 2, 0.3], [10, 10, 4, 2, 0.3])).toBeCloseTo(1, 9);
    expect(rotatedIou([0, 0, 2, 2, 0], [10, 10, 2, 2, 1])).toBe(0);
    expect(rotatedIou([0, 0, 2, 2, 0], [0, 0, 2, 2, Math.PI / 2])).toBeCloseTo(1, 9);

    expect(rotatedIou([0, 0, 20, 2, Math.PI / 4], [0, 0, 20, 2, -Math.PI / 4])).toBeCloseTo(4 / (40 + 40 - 4), 6);
  });
});

describe("camera motion", () => {
  it("similarity from matched points, outliers rejected", () => {
    const src: [number, number][] = [];
    const dst: [number, number][] = [];
    const a = Math.cos(0.05) * 1.1;
    const b = Math.sin(0.05) * 1.1;
    for (let i = 0; i < 40; i += 1) {
      const x = (i * 37) % 200;
      const y = (i * 53) % 150;
      src.push([x, y]);
      dst.push(i % 9 === 0 ? [x + 40, y - 30] : [a * x - b * y + 5, b * x + a * y - 3]);
    }
    const m = estimateSimilarity(src, dst)!;
    expect(m[0]).toBeCloseTo(a, 6);
    expect(m[3]).toBeCloseTo(b, 6);
    expect(m[2]).toBeCloseTo(5, 4);
    expect(m[5]).toBeCloseTo(-3, 4);
  });

  it("a frame shifted by (6, −4) pixels is seen shifted", () => {
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
    const gmc = new MotionCompensator({ downscale: 1 });
    gmc.estimate(scene(0, 0));
    const m = gmc.estimate(scene(6, -4));
    expect(m[2]).toBeCloseTo(6, 0);
    expect(m[5]).toBeCloseTo(-4, 0);
    expect(m[0]).toBeCloseTo(1, 2);
  });
});

describe("propagation and exports", () => {
  const frame = (x: number): DetectionResult => ({
    modelName: "m",
    task: "detect",
    source: null,
    inferenceTimeMs: 3,
    imageShape: [200, 300],
    boxes: [{ x1: x, y1: 50, x2: x + 40, y2: 130, confidence: 0.9, classId: 0, className: "person", trackId: null, isInterpolated: false }],
  });

  it("a skipped frame continues the motion, without touching the tracker", async () => {
    const tracker = new ObjectTracker("bytetrack");
    for (let t = 0; t < 6; t += 1) await tracker.update(frame(20 + 10 * t));
    const before = tracker.activeTracks();
    const next = propagate(tracker, 7, 5, frame(70));
    expect(next.boxes[0]!.isInterpolated).toBe(true);
    expect(next.boxes[0]!.trackId).toBe(1);
    expect(next.boxes[0]!.x1).toBeGreaterThan(75);
    expect(next.inferenceTimeMs).toBe(0);
    expect(tracker.activeTracks()).toEqual(before);
  });

  it("refuses ReID without a model, and per-image formats on a video", () => {
    expect(() => new ObjectTracker("botsort", { withReid: true })).toThrow(/ReID model/);
    expect(() => exportResult({ source: "v", fps: 0, totalFrames: 0, processedFrames: 0, frameStep: 1, frames: [] }, "txt")).toThrow(/per image/);
    expect(exportResult(frame(10), "csv").split("\r\n")[0]).toBe("source,class_id,class_name,confidence,x1,y1,x2,y2,track_id,is_interpolated");
  });
});

describe("input size", () => {
  const frame = (width: number, height: number) => ({ data: new Uint8Array(width * height * 4).fill(200), width, height, channels: 4 as const });

  it("letterboxes a 16:9 frame to the smallest stride multiples with rect", () => {
    expect(inputShape([frame(1280, 720)], 640)).toEqual([640, 640]);
    expect(inputShape([frame(1280, 720)], 640, true)).toEqual([384, 640]);
    expect(inputShape([frame(810, 1080)], 480, true)).toEqual([480, 384]);
    const { dims, metas } = preprocess([frame(1280, 720)], 640, { rect: true });
    expect(dims).toEqual([1, 3, 384, 640]);
    expect(metas[0]).toMatchObject({ scale: 0.5, padTop: 12, padLeft: 0 });
  });
});
