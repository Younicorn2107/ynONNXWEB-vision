/**
 * The web port against what the Python library computes on the same inputs
 * (`scripts/gen_fixtures.py`): letterbox, NMS, rotated IoU, every decoder, Kalman, tracking, SAHI
 * grid, exports.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  KalmanFilterXYAH,
  ObjectTracker,
  SAHISlicer,
  decodeClassify,
  decodeDetect,
  decodeObb,
  decodePose,
  decodeSegment,
  decodeSemantic,
  nms,
  preprocess,
  recordsOf,
  rotatedIou,
  rotatedNms,
  toVocXml,
  toYoloTxt,
  type DecoderContext,
  type DetectionResult,
  type LetterboxMeta,
} from "../../src/core.js";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const F: any = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "drift.json"), "utf-8"));

const close = (a: number, b: number, tol = 1e-3) => expect(Math.abs(a - b)).toBeLessThanOrEqual(tol);

describe("letterbox and centre crop", () => {
  for (const c of F.preprocess) {
    it(`${c.width}×${c.height} → ${c.imgsz}${c.centerCrop ? " (crop)" : ""}`, () => {
      const image = { data: Uint8Array.from(c.image), width: c.width, height: c.height, channels: 3 as const };
      const { tensor, metas } = preprocess([image], c.imgsz, { centerCrop: c.centerCrop });
      expect(metas[0]).toEqual(c.meta);
      let worst = 0;
      for (let i = 0; i < tensor.length; i += 1) worst = Math.max(worst, Math.abs(Math.round(tensor[i]! * 255) - c.tensorBytes[i]));

      expect(worst).toBe(0);
    });
  }
});

describe("NMS", () => {
  for (const c of F.nms) {
    it(`${c.scores.length} boxes, agnostic ${c.agnostic}`, () => {
      expect(nms(c.boxes, c.scores, c.classes, { iouThreshold: 0.45, confThreshold: 0.25, agnostic: c.agnostic })).toEqual(c.keep);
    });
  }
  it("exact rotated IoU, as OpenCV's polygon intersection", () => {
    for (const c of F.rotatedIou) close(rotatedIou(c.a, c.b), c.iou, 1e-4);
  });
  it("rotated NMS", () => {
    const r = F.rotatedNms;
    expect(rotatedNms(r.boxes, r.scores, r.classes, { iouThreshold: 0.45, confThreshold: 0.25 })).toEqual(r.keep);
  });
});

describe("decoders", () => {
  const D = F.decoders;
  const meta: LetterboxMeta = D.meta;
  const ctx = (names: string[], nmsFree = false): DecoderContext => ({
    classNames: names,
    modelName: "m",
    layout: { nmsFree, normalized: false, split: false, imgsz: 64 },
  });
  const sameBoxes = (got: DetectionResult["boxes"], want: { xyxy: number[]; confidence: number; classId: number }[]) => {
    expect(got.length).toBe(want.length);
    got.forEach((b, i) => {
      [b.x1, b.y1, b.x2, b.y2].forEach((v, k) => close(v, want[i]!.xyxy[k]!));
      close(b.confidence, want[i]!.confidence, 1e-6);
      expect(b.classId).toBe(want[i]!.classId);
    });
  };

  it("standard detection: transpose, best class, NMS, letterbox undone", () => {
    const out = { data: Float32Array.from(D.detect.data), dims: D.detect.dims };
    sameBoxes(decodeDetect([out], [meta], ctx(D.names), { conf: 0.3, iou: 0.5 })[0]!.boxes, D.detect.boxes);
    sameBoxes(decodeDetect([out], [meta], ctx(D.names), { conf: 0.3, iou: 0.5, classes: [1, 3] })[0]!.boxes, D.detectClasses.boxes);
  });

  it("end-to-end rows", () => {
    const out = { data: Float32Array.from(D.endToEnd.data), dims: D.endToEnd.dims };
    sameBoxes(decodeDetect([out], [meta], ctx(D.names, true), { conf: 0.25 })[0]!.boxes, D.endToEnd.boxes);
  });

  it("pose: boxes and 17 keypoints", () => {
    const out = { data: Float32Array.from(D.pose.data), dims: D.pose.dims };
    const r = decodePose([out], [meta], ctx(["person"]), { conf: 0.4, iou: 0.5 })[0]!;
    sameBoxes(r.boxes, D.pose.boxes);
    r.keypoints.forEach((set, i) => set.points.forEach((p, k) => [p.x, p.y, p.confidence].forEach((v, j) => close(v, D.pose.keypoints[i][k][j]))));
    expect(r.keypoints[0]!.points[9]!.name).toBe("left_wrist");
  });

  it("segmentation: boxes and masks from the prototypes", () => {
    const S = D.segment;
    const r = decodeSegment([{ data: Float32Array.from(S.data), dims: S.dims }, { data: Float32Array.from(S.protos), dims: S.protoDims }], [meta], ctx(["a", "b"]), { conf: 0.5, iou: 0.5 })[0]!;
    sameBoxes(r.boxes, S.boxes);
    r.masks.forEach((m, i) => {
      let sum = 0;
      const rows: number[] = [];
      for (let y = 0; y < m.height; y += 1) {
        let any = false;
        for (let x = 0; x < m.width; x += 1) if (m.data[y * m.width + x]) { sum += 1; any = true; }
        if (any) rows.push(y);
      }

      expect(Math.abs(sum - S.masks[i].sum)).toBeLessThanOrEqual(Math.max(3, S.masks[i].sum * 0.01));
      if (rows.length) expect([rows[0], rows[rows.length - 1]]).toEqual(S.masks[i].rows);
    });
  });

  it("classification: softmax only when needed, top-k", () => {
    for (const c of D.classify) {
      const r = decodeClassify([{ data: Float32Array.from(c.data), dims: c.dims }], null, ctx(Array.from({ length: 10 }, (_, i) => `k${i}`)), { topK: 4 });
      r.forEach((res, b) => res.topK.forEach((p, k) => {
        expect(p.classId).toBe(c.topK[b][k][0]);
        close(p.confidence, c.topK[b][k][1], 1e-6);
      }));
    }
  });

  it("oriented boxes: rotated NMS, centres and sizes back to the image", () => {
    const r = decodeObb([{ data: Float32Array.from(D.obb.data), dims: D.obb.dims }], [meta], ctx(["x", "y", "z"]), { conf: 0.3, iou: 0.4 })[0]!;
    expect(r.orientedBoxes.length).toBe(D.obb.boxes.length);
    r.orientedBoxes.forEach((b, i) => {
      const w = D.obb.boxes[i];
      [b.cx, b.cy, b.w, b.h, b.angle, b.confidence].forEach((v, k) => close(v, w[k]));
      expect(b.classId).toBe(w[6]);
    });
  });

  it("semantic: argmax, letterbox cropped, nearest resize", () => {
    const r = decodeSemantic([{ data: Float32Array.from(D.semantic.data), dims: D.semantic.dims }], [meta], ctx(["bg", "a", "b", "c"]))[0]!;
    const counts = [0, 0, 0, 0];
    for (const v of r.mask) counts[v]! += 1;
    expect(counts).toEqual(D.semantic.counts);
    expect(Array.from(r.mask.subarray(0, 200))).toEqual(D.semantic.row0);
    expect(Array.from(r.mask.subarray(r.mask.length - 200))).toEqual(D.semantic.rowLast);
  });
});

describe("tracking", () => {
  it("Kalman predict/update", () => {
    const kf = new KalmanFilterXYAH();
    let s = kf.initiate([50, 60, 0.5, 40]);
    for (let i = 0; i < 3; i += 1) s = kf.predict(s.mean, s.covariance);
    s = kf.update(s.mean, s.covariance, [53, 62, 0.52, 41]);
    s = kf.predict(s.mean, s.covariance);
    s.mean.forEach((v, i) => close(v, F.kalman.mean[i], 1e-9));
    s.covariance.forEach((row, i) => close(row[i]!, F.kalman.covDiag[i], 1e-9));
  });

  it("ByteTrack ids over a sequence (occlusion, weak detections, a newcomer)", async () => {
    const tracker = new ObjectTracker("bytetrack", { minHits: 1 })
    for (const frame of F.tracker) {
      const result: DetectionResult = {
        modelName: "m",
        task: "detect",
        source: null,
        inferenceTimeMs: 0,
        imageShape: [400, 400],
        boxes: frame.boxes.map((b: { xyxy: number[]; confidence: number; classId: number; className: string }) => ({
          x1: b.xyxy[0], y1: b.xyxy[1], x2: b.xyxy[2], y2: b.xyxy[3], confidence: b.confidence, classId: b.classId, className: b.className, trackId: null, isInterpolated: false,
        })),
      };
      const tracked = await tracker.update(result);
      expect(tracked.boxes.map((b) => b.trackId)).toEqual(frame.trackIds);
    }
  });
});

describe("SAHI and exports", () => {
  it("tile grid", () => {
    for (const c of F.sahi) {
      expect(new SAHISlicer({ tileSize: c.tile, tileOverlap: c.overlap }).grid(c.width, c.height).map((t) => [t.xOffset, t.yOffset, t.width, t.height])).toEqual(c.grid);
    }
  });

  it("YOLO labels, VOC XML and records, as Python writes them", () => {
    const E = F.exports;
    const result: DetectionResult = {
      modelName: "m",
      task: "detect",
      source: E.result.source,
      inferenceTimeMs: 0,
      imageShape: E.result.imageShape,
      boxes: E.result.boxes.map((b: { xyxy: number[]; confidence: number; classId: number; className: string }) => ({
        x1: b.xyxy[0], y1: b.xyxy[1], x2: b.xyxy[2], y2: b.xyxy[3], confidence: b.confidence, classId: b.classId, className: b.className, trackId: null, isInterpolated: false,
      })),
    };
    expect(toYoloTxt(result)).toBe(E.txt.replace(/\r\n/g, "\n"));
    expect(toVocXml(result)).toBe(E.xml.replace(/\r\n/g, "\n"));
    expect(recordsOf(result)).toEqual(E.records);
  });
});
