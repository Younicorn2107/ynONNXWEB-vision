/** Raw ONNX outputs → results in the original image, for every task. */

import { COCO_KEYPOINT_NAMES } from "./coco.js";
import { nms, rotatedNms } from "./nms.js";
import { pyRound, toOriginalBox, toOriginalPoint } from "./preprocess.js";
import type {
  BoundingBox,
  ClassificationResult,
  DepthResult,
  DetectionResult,
  KeypointSet,
  LetterboxMeta,
  Mask,
  OBBResult,
  OrientedBox,
  PoseResult,
  SegmentationResult,
  SemanticResult,
} from "./types.js";

/** One output of the graph, as onnxruntime returns it. */
export interface RawOutput {
  data: ArrayLike<number>;
  dims: readonly number[];
}

/** How a model lays out its head: decided once, at load, from the graph. */
export interface HeadLayout {
  /** Rows already deduplicated by the model. */
  nmsFree: boolean;
  /** Coordinates divided by the input size (`onnx-community` exports). */
  normalized: boolean;
  /** Detection split in `logits` (pre-sigmoid) and `pred_boxes`. */
  split: boolean;
  /** The model input side, to undo `normalized`. */
  imgsz: number;
}

export interface DecodeOptions {
  conf?: number;
  iou?: number;
  /** Class ids to keep; all when absent. */
  classes?: readonly number[] | null;
  maxDet?: number;
  agnosticNms?: boolean;
  /** Segmentation: mask probability threshold. */
  maskThreshold?: number;
  /** Classification: predictions kept. */
  topK?: number;
}

export interface DecoderContext {
  classNames: readonly string[];
  modelName: string;
  layout: HeadLayout;
  /** Pose: keypoint names, by index (COCO by default). */
  keypointNames?: readonly string[];
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Image `b` of a `[B, …]` output, as a 2-D `[rows, cols]` view. */
function plane(output: RawOutput, b: number): { data: ArrayLike<number> & { subarray?: unknown }; rows: number; cols: number } {
  const dims = output.dims;
  const rows = dims.length === 3 ? dims[1]! : dims[0]!;
  const cols = dims.length === 3 ? dims[2]! : dims[1]!;
  const size = rows * cols;
  const data = output.data as Float32Array;
  return { data: dims.length === 3 ? data.subarray(b * size, (b + 1) * size) : data, rows, cols };
}

/**
 * `[N, attrs]`, predictions first, whatever the export chose: the larger axis is the predictions'.
 */
export function predictionsFirst(output: RawOutput, b: number): { data: Float32Array; n: number; attrs: number } {
  const { data, rows, cols } = plane(output, b);
  if (rows >= cols) return { data: Float32Array.from(data as ArrayLike<number>), n: rows, attrs: cols };
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) for (let c = 0; c < cols; c += 1) out[c * rows + r] = data[r * cols + c]!;
  return { data: out, n: cols, attrs: rows };
}

function batchOf(output: RawOutput): number {
  return output.dims.length === 3 || output.dims.length === 4 ? output.dims[0]! : 1;
}

function nameOf(names: readonly string[], id: number): string {
  return id >= 0 && id < names.length ? names[id]! : `class${id}`;
}

function makeBox(xyxy: readonly number[], confidence: number, classId: number, names: readonly string[]): BoundingBox {
  return {
    x1: xyxy[0]!,
    y1: xyxy[1]!,
    x2: xyxy[2]!,
    y2: xyxy[3]!,
    confidence,
    classId,
    className: nameOf(names, classId),
    trackId: null,
    isInterpolated: false,
  };
}

/**
 * Kept rows of a decoded image: model-space corner boxes, scores, classes, and each row's extra
 * channels.
 */
export interface Kept {
  boxes: number[][];
  scores: number[];
  classIds: number[];
  extra: Float32Array[];
}

/** Standard head `[N, 4+nc+extraCount]`: best class per row, NMS. */
function keepStandard(output: RawOutput, b: number, extraCount: number, options: DecodeOptions): Kept {
  const { data, rows, cols } = plane(output, b);

  const attrsFirst = rows < cols;
  const n = attrsFirst ? cols : rows;
  const attrs = attrsFirst ? rows : cols;
  const at = attrsFirst ? (i: number, a: number) => data[a * n + i]! : (i: number, a: number) => data[i * attrs + a]!;
  const nc = attrs - 4 - extraCount;
  const best = new Float32Array(n).fill(-Infinity);
  const bestClass = new Int32Array(n);
  if (attrsFirst) {
    for (let c = 0; c < nc; c += 1) {
      const base = (4 + c) * n;
      for (let i = 0; i < n; i += 1) {
        const v = data[base + i]!;
        if (v > best[i]!) {
          best[i] = v;
          bestClass[i] = c;
        }
      }
    }
  } else {
    for (let i = 0; i < n; i += 1) {
      const row = i * attrs + 4;
      for (let c = 0; c < nc; c += 1) {
        const v = data[row + c]!;
        if (v > best[i]!) {
          best[i] = v;
          bestClass[i] = c;
        }
      }
    }
  }

  const conf = options.conf ?? 0.25;
  const candidates: number[] = [];
  for (let i = 0; i < n; i += 1) if (best[i]! >= conf) candidates.push(i);
  const m = candidates.length;
  const xyxy = new Float32Array(m * 4);
  const scores = new Float32Array(m);
  const classIds = new Int32Array(m);
  candidates.forEach((i, k) => {
    const [cx, cy, w, h] = [at(i, 0), at(i, 1), at(i, 2), at(i, 3)];
    xyxy[4 * k] = Math.fround(cx - w / 2);
    xyxy[4 * k + 1] = Math.fround(cy - h / 2);
    xyxy[4 * k + 2] = Math.fround(cx + w / 2);
    xyxy[4 * k + 3] = Math.fround(cy + h / 2);
    scores[k] = best[i]!;
    classIds[k] = bestClass[i]!;
  });
  const keep = nms(xyxy, scores, classIds, {
    iouThreshold: options.iou ?? 0.45,
    confThreshold: conf,
    agnostic: options.agnosticNms ?? false,
    maxDet: options.maxDet ?? 300,
  });
  return {
    boxes: keep.map((k) => Array.from(xyxy.subarray(4 * k, 4 * k + 4))),
    scores: keep.map((k) => scores[k]!),
    classIds: keep.map((k) => classIds[k]!),
    extra: keep.map((k) => Float32Array.from({ length: extraCount }, (_, j) => at(candidates[k]!, 4 + nc + j))),
  };
}

/**
 * At most `maxDet` rows: in the model's order (already best first in an end-to-end head), re-sorted
 * by score only when there are too many.
 */
function topRows<T>(rows: T[], score: (row: T) => number, maxDet: number): T[] {
  if (rows.length <= maxDet) return rows;
  return rows.map((row, i) => ({ row, i })).sort((a, z) => score(z.row) - score(a.row) || a.i - z.i).slice(0, maxDet).map((e) => e.row);
}

/** End-to-end rows `[N, 6+extra]`: x1 y1 x2 y2 score class, then extra; filtered by confidence. */
function keepEndToEnd(output: RawOutput, b: number, layout: HeadLayout, options: DecodeOptions, scaleExtra: boolean): Kept {
  const { data, rows: n, cols } = plane(output, b);
  const k = layout.normalized ? layout.imgsz : 1;
  const conf = options.conf ?? 0.25;
  const rows: number[] = [];
  for (let i = 0; i < n; i += 1) if (data[i * cols + 4]! >= conf) rows.push(i);
  const kept = topRows(rows, (i) => data[i * cols + 4]!, options.maxDet ?? 300);
  return {
    boxes: kept.map((i) => [0, 1, 2, 3].map((c) => Math.fround(data[i * cols + c]! * k))),
    scores: kept.map((i) => data[i * cols + 4]!),
    classIds: kept.map((i) => Math.trunc(data[i * cols + 5]!)),
    extra: kept.map((i) => {
      const row = Float32Array.from({ length: cols - 6 }, (_, c) => data[i * cols + 6 + c]!);
      if (scaleExtra && k !== 1) for (let c = 0; c < row.length; c += 1) row[c] = row[c]! * k;
      return row;
    }),
  };
}

/** `logits [N, nc]` (pre-sigmoid) + `pred_boxes [N, 4]` (normalized cx cy w h). */
function keepSplit(logits: RawOutput, boxes: RawOutput, b: number, layout: HeadLayout, options: DecodeOptions): Kept {
  const { data: scoresRaw, rows: n, cols: nc } = plane(logits, b);
  const { data: box } = plane(boxes, b);
  const conf = options.conf ?? 0.25;
  const found: { i: number; score: number; cls: number }[] = [];
  for (let i = 0; i < n; i += 1) {
    let best = 0;
    let raw = -Infinity;
    for (let c = 0; c < nc; c += 1) {
      const v = scoresRaw[i * nc + c]!;
      if (v > raw) {
        raw = v;
        best = c;
      }
    }
    const score = Math.fround(sigmoid(raw));
    if (score >= conf) found.push({ i, score, cls: best });
  }
  const kept = topRows(found, (f) => f.score, options.maxDet ?? 300);
  const k = layout.normalized ? layout.imgsz : 1;
  return {
    boxes: kept.map(({ i }) => {
      const [cx, cy, w, h] = [box[4 * i]! * k, box[4 * i + 1]! * k, box[4 * i + 2]! * k, box[4 * i + 3]! * k];
      return [Math.fround(cx - w / 2), Math.fround(cy - h / 2), Math.fround(cx + w / 2), Math.fround(cy + h / 2)];
    }),
    scores: kept.map((f) => f.score),
    classIds: kept.map((f) => f.cls),
    extra: kept.map(() => new Float32Array(0)),
  };
}

function filterClasses(kept: Kept, classes: readonly number[] | null | undefined): Kept {
  if (!classes) return kept;
  const allowed = new Set(classes);
  const idx = kept.classIds.map((c, i) => (allowed.has(c) ? i : -1)).filter((i) => i >= 0);
  return {
    boxes: idx.map((i) => kept.boxes[i]!),
    scores: idx.map((i) => kept.scores[i]!),
    classIds: idx.map((i) => kept.classIds[i]!),
    extra: idx.map((i) => kept.extra[i]!),
  };
}

function base(context: DecoderContext, meta: LetterboxMeta) {
  return { modelName: context.modelName, source: null, inferenceTimeMs: 0, imageShape: [...meta.originalShape] as [number, number] };
}

/** Detection, every family. */
export function decodeDetect(outputs: readonly RawOutput[], metas: readonly LetterboxMeta[], context: DecoderContext, options: DecodeOptions = {}): DetectionResult[] {
  const { layout } = context;
  return metas.map((meta, b) => {
    let kept: Kept;
    if (layout.split) kept = keepSplit(outputs[0]!, outputs[1]!, b, layout, options);
    else if (layout.nmsFree) kept = keepEndToEnd(outputs[0]!, b, layout, options, false);
    else kept = keepStandard(outputs[0]!, b, 0, options);
    kept = filterClasses(kept, options.classes);
    return {
      ...base(context, meta),
      task: "detect" as const,
      boxes: kept.boxes.map((xyxy, i) => makeBox(toOriginalBox(xyxy, meta), kept.scores[i]!, kept.classIds[i]!, context.classNames)),
    };
  });
}

/** Pose: person boxes and `K` keypoints `(x, y, confidence)` each. */
export function decodePose(outputs: readonly RawOutput[], metas: readonly LetterboxMeta[], context: DecoderContext, options: DecodeOptions = {}): PoseResult[] {
  const { layout } = context;
  const names = context.keypointNames ?? COCO_KEYPOINT_NAMES;
  const output = outputs[0]!;
  return metas.map((meta, b) => {
    let kept: Kept;
    if (layout.nmsFree) {
      kept = keepEndToEnd(output, b, layout, options, true);
    } else {
      const { rows, cols } = plane(output, b);
      kept = keepStandard(output, b, Math.min(rows, cols) - 5, options);
    }
    const width = kept.extra[0]?.length ?? 0;
    if (width % 3 !== 0) throw new RangeError(`pose head has ${width} keypoint values per row; expected 3 per keypoint`);
    const boxes = kept.boxes.map((xyxy, i) => makeBox(toOriginalBox(xyxy, meta), kept.scores[i]!, 0, context.classNames));
    const keypoints: KeypointSet[] = kept.extra.map((row, i) => ({
      box: boxes[i]!,
      points: Array.from({ length: row.length / 3 }, (_, k) => {
        const [x, y] = toOriginalPoint(row[3 * k]!, row[3 * k + 1]!, meta);
        return { x, y, confidence: row[3 * k + 2]!, name: names[k] ?? null };
      }),
    }));
    return { ...base(context, meta), task: "pose" as const, boxes, keypoints };
  });
}

const MASK_COEFFS = 32;

/** Where the letterboxed content sits in a grid of `mh × mw` over the model input. */
/** The part of a prototype grid `[mh, mw]` the image covers (the letterbox padding left out). */
export function contentRegion(meta: LetterboxMeta, mh: number, mw: number) {
  const [h0, w0] = meta.originalShape;
  const modelH = meta.padTop * 2 + meta.scale * h0;
  const modelW = meta.padLeft * 2 + meta.scale * w0;
  const sh = modelH > 0 ? mh / modelH : 1;
  const sw = modelW > 0 ? mw / modelW : 1;
  return {
    top: Math.trunc(pyRound(meta.padTop * sh)),
    left: Math.trunc(pyRound(meta.padLeft * sw)),
    height: Math.max(1, Math.trunc(pyRound(meta.scale * h0 * sh))),
    width: Math.max(1, Math.trunc(pyRound(meta.scale * w0 * sw))),
  };
}

/** OpenCV's float `INTER_LINEAR` taps for one axis. */
function floatTaps(src: number, dst: number): { i0: Int32Array; i1: Int32Array; f: Float32Array } {
  const scale = src / dst;
  const i0 = new Int32Array(dst);
  const i1 = new Int32Array(dst);
  const f = new Float32Array(dst);
  for (let d = 0; d < dst; d += 1) {
    let x = Math.fround((d + 0.5) * scale - 0.5);
    let s = Math.floor(x);
    x -= s;
    if (s < 0) {
      x = 0;
      s = 0;
    }
    if (s >= src - 1) {
      x = 0;
      s = src - 1;
    }
    i0[d] = s;
    i1[d] = Math.min(s + 1, src - 1);
    f[d] = x;
  }
  return { i0, i1, f };
}

/** Instance segmentation: boxes, then each box's mask from the prototypes. */
export function decodeSegment(outputs: readonly RawOutput[], metas: readonly LetterboxMeta[], context: DecoderContext, options: DecodeOptions = {}): SegmentationResult[] {
  const protos = outputs[1] ?? null;
  return segmentBoxes(outputs, metas, context, options).map(({ result, kept }, b) => {
    const { boxes } = result;
    const meta = metas[b]!;
    const masks: Mask[] = [];
    if (protos && boxes.length > 0) {
      const [c, mh, mw] = protos.dims.length === 4 ? protos.dims.slice(1) : protos.dims;
      const size = c! * mh! * mw!;
      const proto = (protos.data as Float32Array).subarray(protos.dims.length === 4 ? b * size : 0, protos.dims.length === 4 ? (b + 1) * size : size);
      masks.push(...composeMasks(kept, proto, c!, mh!, mw!, boxes, meta, options.maskThreshold ?? 0.5));
    }
    return { ...result, masks };
  });
}

/**
 * Instance segmentation without its masks: each image's boxes, and the mask coefficients of each
 * box (`kept.extra`) for whoever composes the masks — `decodeSegment` on the CPU, `GpuMaskComposer`
 * where the prototypes are.
 */
export function segmentBoxes(outputs: readonly RawOutput[], metas: readonly LetterboxMeta[], context: DecoderContext, options: DecodeOptions = {}): { result: SegmentationResult; kept: Kept }[] {
  const { layout } = context;
  return metas.map((meta, b) => {
    let kept = layout.nmsFree ? keepEndToEnd(outputs[0]!, b, layout, options, false) : keepStandard(outputs[0]!, b, MASK_COEFFS, options);
    kept = filterClasses(kept, options.classes);
    const boxes = kept.boxes.map((xyxy, i) => makeBox(toOriginalBox(xyxy, meta), kept.scores[i]!, kept.classIds[i]!, context.classNames));
    return { result: { ...base(context, meta), task: "segment" as const, boxes, masks: [] }, kept };
  });
}

function composeMasks(kept: Kept, proto: Float32Array, c: number, mh: number, mw: number, boxes: BoundingBox[], meta: LetterboxMeta, threshold: number): Mask[] {
  const [h0, w0] = meta.originalShape;
  const region = contentRegion(meta, mh, mw);
  const xs = floatTaps(region.width, w0);
  const ys = floatTaps(region.height, h0);
  const area = mh * mw;
  return kept.extra.map((coeffs, k) => {
    const box = boxes[k]!;
    const x1 = Math.max(0, pyRound(box.x1));
    const y1 = Math.max(0, pyRound(box.y1));
    const x2 = Math.min(w0, Math.max(x1 + 1, pyRound(box.x2)));
    const y2 = Math.min(h0, Math.max(y1 + 1, pyRound(box.y2)));

    const prob = new Float32Array(region.width * region.height);
    const [gy0, gy1] = [ys.i0[Math.min(y1, h0 - 1)]!, ys.i1[Math.max(0, y2 - 1)]!];
    const [gx0, gx1] = [xs.i0[Math.min(x1, w0 - 1)]!, xs.i1[Math.max(0, x2 - 1)]!];
    for (let y = gy0; y <= gy1; y += 1) {
      for (let x = gx0; x <= gx1; x += 1) {
        const at = Math.min(mh - 1, region.top + y) * mw + Math.min(mw - 1, region.left + x);
        let logit = 0;
        for (let j = 0; j < c; j += 1) logit += coeffs[j]! * proto[j * area + at]!;
        prob[y * region.width + x] = Math.fround(1 / (1 + Math.exp(-logit)));
      }
    }
    const data = new Uint8Array(w0 * h0);

    const line = new Float32Array(region.width);
    for (let y = y1; y < y2; y += 1) {
      const r0 = ys.i0[y]! * region.width;
      const r1 = ys.i1[y]! * region.width;
      const fy = ys.f[y]!;
      for (let g = gx0; g <= gx1; g += 1) line[g] = prob[r0 + g]! * (1 - fy) + prob[r1 + g]! * fy;
      const row = y * w0;
      for (let x = x1; x < x2; x += 1) {
        const fx = xs.f[x]!;
        if (line[xs.i0[x]!]! * (1 - fx) + line[xs.i1[x]!]! * fx >= threshold) data[row + x] = 1;
      }
    }
    return { data, width: w0, height: h0, box };
  });
}

/** Classification: `[B, C]` logits or probabilities → the top `topK`. */
export function decodeClassify(outputs: readonly RawOutput[], metas: readonly LetterboxMeta[] | null, context: DecoderContext, options: DecodeOptions = {}): ClassificationResult[] {
  const output = outputs[0]!;
  const classes = output.dims[output.dims.length - 1]!;
  const batch = output.dims.length >= 2 ? output.dims[0]! : 1;
  const topK = options.topK ?? 5;
  return Array.from({ length: batch }, (_, b) => {
    const logits = Array.from({ length: classes }, (_, c) => output.data[b * classes + c]!);
    const total = logits.reduce((s, v) => s + v, 0);
    let probs = logits;

    if (Math.min(...logits) < 0 || total <= 0 || Math.abs(total - 1) > 1e-3) {
      const max = Math.max(...logits);
      const exp = logits.map((v) => Math.exp(v - max));
      const sum = exp.reduce((s, v) => s + v, 0);
      probs = exp.map((v) => v / sum);
    }
    const order = probs.map((_, i) => i).sort((i, j) => probs[j]! - probs[i]! || j - i).slice(0, topK);
    const meta = metas?.[b];
    return {
      modelName: context.modelName,
      source: null,
      inferenceTimeMs: 0,
      imageShape: meta ? ([...meta.originalShape] as [number, number]) : ([0, 0] as [number, number]),
      task: "classify" as const,
      topK: order.map((i) => ({ classId: i, className: nameOf(context.classNames, i), confidence: probs[i]! })),
    };
  });
}

/**
 * Oriented boxes. Standard `[B, 4+nc+1, N]` (xywh, class scores, angle) with the exact rotated NMS;
 * end-to-end `[B, N, 7]` (cx cy w h, score, class, angle) as it comes.
 */
export function decodeObb(outputs: readonly RawOutput[], metas: readonly LetterboxMeta[], context: DecoderContext, options: DecodeOptions = {}): OBBResult[] {
  const { layout } = context;
  const output = outputs[0]!;
  return metas.map((meta, b) => {
    let rows: { box: number[]; score: number; cls: number; angle: number }[];
    if (layout.nmsFree) {
      const { data, rows: n, cols } = plane(output, b);
      const k = layout.normalized ? layout.imgsz : 1;
      rows = [];
      for (let i = 0; i < n; i += 1) {
        const r = i * cols;
        if (data[r + 4]! < (options.conf ?? 0.25)) continue;
        rows.push({ box: [data[r]! * k, data[r + 1]! * k, data[r + 2]! * k, data[r + 3]! * k], score: data[r + 4]!, cls: Math.trunc(data[r + 5]!), angle: data[r + 6]! });
      }
      rows = topRows(rows, (r) => r.score, options.maxDet ?? 300);
    } else {
      const { data, n, attrs } = predictionsFirst(output, b);
      const nc = attrs - 5;
      if (nc <= 0) throw new RangeError(`OBB head has ${attrs} attributes; expected 4 + nc + 1 (angle)`);
      const boxes5 = new Float32Array(n * 5);
      const scores = new Float32Array(n);
      const classIds = new Int32Array(n);
      for (let i = 0; i < n; i += 1) {
        const r = i * attrs;
        let best = 0;
        let score = -Infinity;
        for (let c = 0; c < nc; c += 1) {
          if (data[r + 4 + c]! > score) {
            score = data[r + 4 + c]!;
            best = c;
          }
        }
        boxes5.set([data[r]!, data[r + 1]!, data[r + 2]!, data[r + 3]!, data[r + 4 + nc]!], 5 * i);
        scores[i] = score;
        classIds[i] = best;
      }
      const keep = rotatedNms(boxes5, scores, classIds, {
        iouThreshold: options.iou ?? 0.45,
        confThreshold: options.conf ?? 0.25,
        agnostic: options.agnosticNms ?? false,
        maxDet: options.maxDet ?? 300,
      });
      rows = keep.map((i) => ({ box: Array.from(boxes5.subarray(5 * i, 5 * i + 4)), score: scores[i]!, cls: classIds[i]!, angle: boxes5[5 * i + 4]! }));
    }
    if (options.classes) {
      const allowed = new Set(options.classes);
      rows = rows.filter((r) => allowed.has(r.cls));
    }
    const orientedBoxes: OrientedBox[] = rows.map((r) => {
      const [cx, cy] = toOriginalPoint(r.box[0]!, r.box[1]!, meta);
      return {
        cx,
        cy,
        w: r.box[2]! / meta.scale,
        h: r.box[3]! / meta.scale,
        angle: r.angle,
        confidence: r.score,
        classId: r.cls,
        className: nameOf(context.classNames, r.cls),
      };
    });
    return { ...base(context, meta), task: "obb" as const, orientedBoxes };
  });
}

/** Whether an output holds integers (a class map) rather than scores. */
function isIntegerOutput(output: RawOutput): boolean {
  const data = output.data as unknown;
  return !(data instanceof Float32Array || data instanceof Float64Array || Array.isArray(data));
}

/**
 * Semantic segmentation → one class id per original pixel. Either `[B, C, h, w]` logits (argmax
 * here), or — what Ultralytics exports — the class map itself, `[B, h, w]` integers, the argmax
 * done in the graph.
 */
export function decodeSemantic(outputs: readonly RawOutput[], metas: readonly LetterboxMeta[], context: DecoderContext): SemanticResult[] {
  const output = outputs[0]!;
  const isMap = output.dims.length === 3 && isIntegerOutput(output);
  const dims = isMap ? [output.dims[0]!, 1, output.dims[1]!, output.dims[2]!] : output.dims.length === 3 ? [1, ...output.dims] : output.dims;
  if (dims.length !== 4) throw new RangeError(`semantic head must be [B, C, h, w] logits or a [B, h, w] class map; got [${output.dims.join(", ")}]`);
  const [, c, mh, mw] = dims as [number, number, number, number];
  const area = mh * mw;
  return metas.map((meta, b) => {
    const classMap = new Int32Array(area);
    const offset = b * c * area;
    if (isMap) {
      for (let p = 0; p < area; p += 1) classMap[p] = Number(output.data[offset + p]!);
    } else {
      for (let p = 0; p < area; p += 1) {
        let best = 0;
        let value = -Infinity;
        for (let k = 0; k < c; k += 1) {
          const v = output.data[offset + k * area + p]!;
          if (v > value) {
            value = v;
            best = k;
          }
        }
        classMap[p] = best;
      }
    }
    const [h0, w0] = meta.originalShape;
    const region = contentRegion(meta, mh, mw);

    const mask = new Int32Array(h0 * w0);
    const fy = 1 / (h0 / region.height);
    const fx = 1 / (w0 / region.width);
    for (let y = 0; y < h0; y += 1) {
      const sy = Math.min(Math.floor(y * fy), region.height - 1);
      const row = Math.min(mh - 1, region.top + sy) * mw;
      for (let x = 0; x < w0; x += 1) {
        const sx = Math.min(Math.floor(x * fx), region.width - 1);
        mask[y * w0 + x] = classMap[row + Math.min(mw - 1, region.left + sx)]!;
      }
    }
    return { ...base(context, meta), task: "semantic" as const, mask, classNames: [...context.classNames] };
  });
}

/**
 * Depth: `[B, 1, h, w]` (or `[B, h, w]`) over the letterboxed input → one value per original pixel:
 * the padding cropped, then bilinear resize.
 */
export function decodeDepth(outputs: readonly RawOutput[], metas: readonly LetterboxMeta[], context: DecoderContext): DepthResult[] {
  const output = outputs[0]!;
  const dims = output.dims;
  const [mh, mw] = [dims[dims.length - 2]!, dims[dims.length - 1]!];
  const area = mh * mw;
  return metas.map((meta, b) => {
    const map = output.data as Float32Array;
    const offset = b * area;
    const [h0, w0] = meta.originalShape;
    const region = contentRegion(meta, mh, mw);
    const xs = floatTaps(region.width, w0);
    const ys = floatTaps(region.height, h0);
    const depth = new Float32Array(h0 * w0);
    let min = Infinity;
    let max = -Infinity;
    const at = (y: number, x: number) => map[offset + Math.min(mh - 1, region.top + y) * mw + Math.min(mw - 1, region.left + x)]!;
    for (let y = 0; y < h0; y += 1) {
      const fy = ys.f[y]!;
      for (let x = 0; x < w0; x += 1) {
        const fx = xs.f[x]!;
        const top = at(ys.i0[y]!, xs.i0[x]!) * (1 - fx) + at(ys.i0[y]!, xs.i1[x]!) * fx;
        const bottom = at(ys.i1[y]!, xs.i0[x]!) * (1 - fx) + at(ys.i1[y]!, xs.i1[x]!) * fx;
        const v = top * (1 - fy) + bottom * fy;
        depth[y * w0 + x] = v;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    return { ...base(context, meta), task: "depth" as const, depth, min, max };
  });
}

export function batchSize(outputs: readonly RawOutput[]): number {
  return batchOf(outputs[0]!);
}
