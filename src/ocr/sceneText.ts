/**
 * Text in an image: found (DBNet), read (SVTR, CTC), grouped into blocks, translated, erased (Telea
 * inpainting) and written back in place — in the colour, size and angle of the original, fitted to
 * its width.
 */

import { huggingFaceResolver, loadModel, type LoadOptions, type ModelBundle, type ModelResolver, type ModelSession } from "@youni/onnxweb-core";
import { ARGMAX_SUFFIX, InferenceError, MAX_SUFFIX, argmaxHead, readGraphInfo, type TensorFactory } from "@youni/onnxweb-core/core";

import { channelsOf, resizeLinear, type RasterImage } from "../preprocess.js";
import { inpaintTelea } from "./inpaint.js";

export type OcrSize = "tiny" | "small" | "medium";

type Point = [number, number];

export interface TextLine {
  /** Corners: top-left, top-right, bottom-right, bottom-left (image pixels). */
  quad: [Point, Point, Point, Point];
  text: string;
  confidence: number;
  /** Radians, of the top edge. */
  angle: number;
  /** Detection score. */
  score: number;
}

export interface TextBlock {
  lines: TextLine[];
  text: string;
  /** Axis-aligned bounds of the block. */
  rect: { x: number; y: number; w: number; h: number };
  angle: number;
  lineHeight: number;
  /** Estimated text and background colours (RGB). */
  color: [number, number, number];
  background: [number, number, number];
  /** How the lines sit: their left edges aligned, or their centres. */
  align: "left" | "center";
  /** Strokes thick for their height (share of text pixels): drawn bold. */
  bold: boolean;
}

export interface SceneTextOptions extends Omit<LoadOptions, "model" | "resolver" | "computeType"> {
  Tensor: TensorFactory;
  /** `tiny` by default: fast enough for a live camera. */
  detector?: OcrSize;
  recognizer?: OcrSize;
  endpoint?: string;
  /**
   * Longest side the detector sees (a multiple of 32). By default the image's own, up to 4000, as
   * PaddleOCR does: shrinking a photo erases its small text. Less is faster.
   */
  detectSide?: number;
  /**
   * Lines read with less confidence than this are dropped: 0.8. Text reads at 0.9 and more; logos
   * and textures mistaken for letters at 0.3–0.8 (and, kept, they would be "translated" and painted
   * over).
   */
  minConfidence?: number;
  /**
   * PaddleOCR's text-line orientation classifier (PP-LCNet, 0° or 180°): a line upside down — a
   * photo taken the wrong way up, a sign seen from below — is turned before it is read, as
   * PaddleOCR's pipeline does with `use_textline_orientation`. `x0_25` (1 MB) by default, `x1_0`
   * the larger one, `false` to read every line as it lies.
   */
  orientation?: "x0_25" | "x1_0" | false;
}

/**
 * WebGPU, then WebAssembly, unless asked otherwise: every image and every line has its own shape,
 * which WebNN compiles anew each time (a page: 23 s on WebNN, 1.5 s on WebGPU, measured
 * 2026-10-08).
 */
const DEFAULT_BACKENDS = ["webgpu", "wasm"] as const;

/** DBNet's post-processing, from the detector's `inference.yml` (PaddleOCR's defaults otherwise). */
interface DetectParams {
  thresh: number;
  boxThresh: number;
  unclipRatio: number;
}

function detectParams(yml: string): DetectParams {
  const read = (key: string, fallback: number) => {
    const m = new RegExp(`^\\s*${key}:\\s*([0-9.]+)`, "m").exec(yml);
    return m ? Number(m[1]) : fallback;
  };
  return { thresh: read("thresh", 0.3), boxThresh: read("box_thresh", 0.6), unclipRatio: read("unclip_ratio", 1.5) };
}

/** A source that shows a repository's `inference.onnx` as its only graph. */
function rootGraph(repo: string, endpoint?: string): ModelResolver {
  const base = huggingFaceResolver({ repo, manifest: false, ...(endpoint ? { endpoint } : {}) });
  return {
    id: `${base.id}#inference`,
    manifestUrls: () => [],
    assetUrl: (key, filename, place) => ({ ...base.assetUrl(key, filename === "model.onnx" ? "inference.onnx" : filename, place), filename }),
    listFiles: async (key, signal) => {
      const listing = await base.listFiles?.(key, signal);
      if (!listing) return null;
      const rename = (files: typeof listing.graphs) => files.map((f) => (f.name === "inference.onnx" ? { ...f, name: "model.onnx" } : f));
      return { graphs: rename(listing.graphs), root: rename(listing.root) };
    },
  };
}

/** The `character_dict` list of a PaddleOCR `inference.yml`. */
export function parseCharacterDict(yml: string): string[] {
  const start = yml.indexOf("character_dict:");
  if (start < 0) return [];
  const out: string[] = [];
  for (const line of yml.slice(start).split(/\r?\n/).slice(1)) {
    const m = /^\s*-\s(.*)$/.exec(line);
    if (!m) {
      if (line.trim() === "") continue;
      break;
    }
    let v = m[1]!;
    if (v.startsWith("'") && v.endsWith("'") && v.length >= 2) v = v.slice(1, -1).replace(/''/g, "'");
    else if (v.startsWith('"') && v.endsWith('"') && v.length >= 2) v = JSON.parse(v) as string;
    out.push(v);
  }
  return out;
}

function convexHull(points: Point[]): Point[] {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o: Point, a: Point, b: Point) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Point[] = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2]!, lower[lower.length - 1]!, q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper: Point[] = [];
  for (const q of [...p].reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2]!, upper[upper.length - 1]!, q) <= 0) upper.pop();
    upper.push(q);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

/** Minimum-area rectangle of a hull (rotating edges): centre, size, angle. */
function minAreaRect(hull: Point[]): { cx: number; cy: number; w: number; h: number; angle: number } {
  let best = { cx: 0, cy: 0, w: 0, h: 0, angle: 0, area: Infinity };
  for (let i = 0; i < hull.length; i += 1) {
    const a = hull[i]!;
    const b = hull[(i + 1) % hull.length]!;
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
    for (const [x, y] of hull) {
      const u = x * c + y * s;
      const v = -x * s + y * c;
      minU = Math.min(minU, u); maxU = Math.max(maxU, u);
      minV = Math.min(minV, v); maxV = Math.max(maxV, v);
    }
    const area = (maxU - minU) * (maxV - minV);
    if (area < best.area) {
      const u = (minU + maxU) / 2;
      const v = (minV + maxV) / 2;
      best = { cx: u * c - v * s, cy: u * s + v * c, w: maxU - minU, h: maxV - minV, angle, area };
    }
  }

  let { w, h, angle } = best;
  while (angle > Math.PI / 4) {
    angle -= Math.PI / 2;
    [w, h] = [h, w];
  }
  while (angle <= -Math.PI / 4) {
    angle += Math.PI / 2;
    [w, h] = [h, w];
  }
  return { cx: best.cx, cy: best.cy, w, h, angle };
}

function quadOf(r: { cx: number; cy: number; w: number; h: number; angle: number }): [Point, Point, Point, Point] {
  const c = Math.cos(r.angle);
  const s = Math.sin(r.angle);
  const at = (u: number, v: number): Point => [r.cx + u * c - v * s, r.cy + u * s + v * c];
  return [at(-r.w / 2, -r.h / 2), at(r.w / 2, -r.h / 2), at(r.w / 2, r.h / 2), at(-r.w / 2, r.h / 2)];
}

/** Bilinear sample of an RGBA/RGB image. */
function sample(img: RasterImage, cn: number, x: number, y: number, out: number[]): void {
  const cx = Math.min(Math.max(x, 0), img.width - 1.001);
  const cy = Math.min(Math.max(y, 0), img.height - 1.001);
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const fx = cx - x0;
  const fy = cy - y0;
  const d = img.data;
  const i00 = (y0 * img.width + x0) * cn;
  const i10 = i00 + cn;
  const i01 = i00 + img.width * cn;
  const i11 = i01 + cn;
  for (let k = 0; k < 3; k += 1) out[k] = (d[i00 + k]! * (1 - fx) + d[i10 + k]! * fx) * (1 - fy) + (d[i01 + k]! * (1 - fx) + d[i11 + k]! * fx) * fy;
}

/** A line's crop: its top-left corner and the image step per crop pixel, across and down. */
interface Crop {
  tl: Point;
  ux: [number, number];
  uy: [number, number];
  width: number;
}

/**
 * A crop written into the recognizer's input (BGR planes scaled to [-1, 1]), row `y`, column `x` at
 * `base + y * stride + x`. `sample`'s arithmetic, inlined: the crops of a page are a million
 * samples.
 */
function cropInto(img: RasterImage, cn: number, c: Crop, input: Float32Array, base: number, plane: number, stride: number): void {
  const d = img.data;
  const maxX = img.width - 1.001;
  const maxY = img.height - 1.001;
  const row = img.width * cn;
  for (let y = 0; y < REC_HEIGHT; y += 1) {
    const ox = c.tl[0] + (y + 0.5) * c.uy[0];
    const oy = c.tl[1] + (y + 0.5) * c.uy[1];
    const at = base + y * stride;
    for (let x = 0; x < c.width; x += 1) {
      const cx = Math.min(Math.max(ox + (x + 0.5) * c.ux[0], 0), maxX);
      const cy = Math.min(Math.max(oy + (x + 0.5) * c.ux[1], 0), maxY);
      const x0 = Math.floor(cx);
      const y0 = Math.floor(cy);
      const fx = cx - x0;
      const fy = cy - y0;
      const i00 = y0 * row + x0 * cn;
      const i10 = i00 + cn;
      const i01 = i00 + row;
      const i11 = i01 + cn;
      const r = (d[i00]! * (1 - fx) + d[i10]! * fx) * (1 - fy) + (d[i01]! * (1 - fx) + d[i11]! * fx) * fy;
      const g = (d[i00 + 1]! * (1 - fx) + d[i10 + 1]! * fx) * (1 - fy) + (d[i01 + 1]! * (1 - fx) + d[i11 + 1]! * fx) * fy;
      const b = (d[i00 + 2]! * (1 - fx) + d[i10 + 2]! * fx) * (1 - fy) + (d[i01 + 2]! * (1 - fx) + d[i11 + 2]! * fx) * fy;
      input[at + x] = b / 127.5 - 1;
      input[at + plane + x] = g / 127.5 - 1;
      input[at + 2 * plane + x] = r / 127.5 - 1;
    }
  }
}

/**
 * Luminance of a crop on a coarse grid, centred and scaled: what a line looks like, whatever the
 * light.
 */
function lookOf(img: RasterImage, cn: number, c: Crop): Float32Array {
  const out = new Float32Array(LOOK_COLS * LOOK_ROWS);
  const px = [0, 0, 0];
  let mean = 0;
  for (let j = 0; j < LOOK_ROWS; j += 1)
    for (let i = 0; i < LOOK_COLS; i += 1) {
      const x = ((i + 0.5) * c.width) / LOOK_COLS;
      const y = ((j + 0.5) * REC_HEIGHT) / LOOK_ROWS;
      sample(img, cn, c.tl[0] + x * c.ux[0] + y * c.uy[0], c.tl[1] + x * c.ux[1] + y * c.uy[1], px);
      const v = 0.299 * px[0]! + 0.587 * px[1]! + 0.114 * px[2]!;
      out[j * LOOK_COLS + i] = v;
      mean += v;
    }
  mean /= out.length;
  let spread = 0;
  for (const v of out) spread += (v - mean) ** 2;
  spread = Math.sqrt(spread / out.length) + 1;
  for (let k = 0; k < out.length; k += 1) out[k] = (out[k]! - mean) / spread;
  return out;
}

function lookDistance(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let k = 0; k < a.length; k += 1) sum += Math.abs(a[k]! - b[k]!);
  return sum / a.length;
}

const REPOS: Record<"det" | "rec", Record<OcrSize, string>> = {
  det: { tiny: "PaddlePaddle/PP-OCRv6_tiny_det_onnx", small: "PaddlePaddle/PP-OCRv6_small_det_onnx", medium: "PaddlePaddle/PP-OCRv6_medium_det_onnx" },
  rec: { tiny: "PaddlePaddle/PP-OCRv6_tiny_rec_onnx", small: "PaddlePaddle/PP-OCRv6_small_rec_onnx", medium: "PaddlePaddle/PP-OCRv6_medium_rec_onnx" },
};

const ORIENTATION_REPOS = { x0_25: "PaddlePaddle/PP-LCNet_x0_25_textline_ori_onnx", x1_0: "PaddlePaddle/PP-LCNet_x1_0_textline_ori_onnx" } as const;
/** The orientation classifier's input, from its inference.yml (`ResizeImage: [160, 80]`). */
const ORI_WIDTH = 160;
const ORI_HEIGHT = 80;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const REC_HEIGHT = 48;
/** The recognizer's training width: narrower crops are padded to it, as PaddleOCR does. */
const REC_MIN_WIDTH = 320;
/** Lines read together; sorted by width first, so a batch pads little. */
const REC_BATCH = 8;
/** A crop this much taller than wide is vertical text: turned a quarter, as PaddleOCR does. */
const VERTICAL_RATIO = 1.5;
/** The grid a line's look is taken on, for `reuse`. */
const LOOK_COLS = 48;
const LOOK_ROWS = 8;
/** Looks this close (mean of centred, scaled luminance) are the same line. */
const LOOK_SAME = 0.12;

export interface ReadOptions {
  /**
   * Lines of the previous `read` that come back — same size, same place (within half a line), same
   * look — keep their text instead of being read again: a live camera on a still page re-detects
   * every frame and recognizes nothing new. False by default (each read on its own).
   */
  reuse?: boolean;
  /**
   * With `reuse`: the camera's motion since the previous read's frame (an affine `[a, b, tx, c, d,
   * ty]`, from `TextFollower.motionSince`). The lines remembered are moved by it before they are
   * compared: on a camera panning across a page, every line has moved further than half a line
   * between two reads.
   */
  moved?: readonly [number, number, number, number, number, number];
  /** This read's confidence floor, instead of the reader's `minConfidence`. */
  minConfidence?: number;
}

/** A line as last read, for `reuse`. */
interface Remembered {
  centre: Point;
  width: number;
  height: number;
  look: Float32Array;
  text: string;
  confidence: number;
}

export class SceneTextReader {
  private constructor(
    private readonly det: ModelBundle,
    private readonly rec: ModelBundle,
    private readonly dict: string[],
    private readonly Tensor: TensorFactory,
    readonly detectSide: number,
    private readonly params: DetectParams,
    readonly minConfidence: number,
    private readonly ori: ModelBundle | null,
  ) {}

  static async load(options: SceneTextOptions): Promise<SceneTextReader> {
    const { Tensor, detector = "tiny", recognizer = "tiny", endpoint, detectSide = 4000, minConfidence = 0.8, orientation = "x0_25", backends = DEFAULT_BACKENDS, ...given } = options;
    const rest = { ...given, backends };
    const det = await loadModel({ ...rest, extraFiles: ["inference.yml"], model: { name: REPOS.det[detector], computeType: "float32" }, resolver: rootGraph(REPOS.det[detector], endpoint) });
    let rec: ModelBundle;
    try {
      rec = await loadModel({
        ...rest,
        extraFiles: ["inference.yml"],
        model: { name: REPOS.rec[recognizer], computeType: "float32" },
        resolver: rootGraph(REPOS.rec[recognizer], endpoint),

        rewrite: (graph) => {
          const output = readGraphInfo(graph).outputs[0]?.name;
          return output ? argmaxHead(graph, output).bytes : graph;
        },
      });
    } catch (error) {
      await det.close();
      throw error;
    }
    const dict = parseCharacterDict(rec.assetText("inference.yml") ?? "");
    if (dict.length === 0) {
      await Promise.all([det.close(), rec.close()]);
      throw new InferenceError("the recognition model came without its character dictionary (inference.yml)");
    }
    let ori: ModelBundle | null = null;
    if (orientation) {
      try {
        ori = await loadModel({ ...rest, model: { name: ORIENTATION_REPOS[orientation], computeType: "float32" }, resolver: rootGraph(ORIENTATION_REPOS[orientation], endpoint) });
      } catch (error) {
        await Promise.all([det.close(), rec.close()]);
        throw error;
      }
    }
    return new SceneTextReader(det, rec, dict, Tensor, Math.max(32, Math.round(detectSide / 32) * 32), detectParams(det.assetText("inference.yml") ?? ""), minConfidence, ori);
  }

  get backend(): string {
    return this.det.only.backend;
  }

  #remembered: Remembered[] = [];
  /** Lines the last `read` took from the one before (`reuse`). */
  reused = 0;
  /**
   * Lines the last `read` found and read, then dropped under the confidence floor: what to look at
   * when text is "not found" (an unusual font reads at 0.6–0.8, logos and textures lower).
   */
  dropped: { text: string; confidence: number }[] = [];

  /** Every line of text: its corners, what it says, how sure. */
  async read(image: RasterImage, options: ReadOptions = {}): Promise<TextLine[]> {
    const quads = await this.#detect(image);
    if (quads.length === 0) {
      this.#remembered = [];
      this.reused = 0;
      return [];
    }
    if (options.reuse && options.moved) {
      const m = options.moved;
      const scale = Math.sqrt(Math.abs(m[0] * m[4] - m[1] * m[3])) || 1;
      this.#remembered = this.#remembered.map((r) => ({
        ...r,
        centre: [m[0] * r.centre[0] + m[1] * r.centre[1] + m[2], m[3] * r.centre[0] + m[4] * r.centre[1] + m[5]] as Point,
        width: r.width * scale,
        height: r.height * scale,
      }));
    }
    return this.#recognize(image, quads, options.reuse ?? false, options.minConfidence ?? this.minConfidence);
  }

  async #detect(image: RasterImage): Promise<{ quad: [Point, Point, Point, Point]; score: number; angle: number }[]> {
    const { width: W, height: H } = image;

    const scale = Math.min(this.detectSide / Math.max(W, H), Math.max(1, 64 / Math.min(W, H)));
    const w = Math.max(32, Math.round((W * scale) / 32) * 32);
    const h = Math.max(32, Math.round((H * scale) / 32) * 32);
    const rgb = resizeLinear(image, w, h);
    const plane = w * h;
    const input = new Float32Array(3 * plane);

    for (let i = 0; i < plane; i += 1) {
      for (let c = 0; c < 3; c += 1) input[c * plane + i] = (rgb[3 * i + (2 - c)]! / 255 - MEAN[c]!) / STD[c]!;
    }
    const session: ModelSession = this.det.only;
    const out = await session.run({ [session.inputs[0]!.name]: new this.Tensor("float32", input, [1, 3, h, w]) });
    const prob = (out[session.outputs[0]!.name] as { data: Float32Array }).data;

    const label = new Int32Array(plane).fill(-1);
    const found: { quad: [Point, Point, Point, Point]; score: number; angle: number }[] = [];
    const stack: number[] = [];
    const sx = W / w;
    const sy = H / h;
    const { thresh, boxThresh, unclipRatio } = this.params;
    for (let i = 0; i < plane; i += 1) {
      if (prob[i]! <= thresh || label[i] !== -1) continue;
      const pts: Point[] = [];
      let sum = 0;
      stack.push(i);
      label[i] = found.length;
      while (stack.length > 0) {
        const p = stack.pop()!;
        const x = p % w;
        const y = (p - x) / w;
        pts.push([x, y]);
        sum += prob[p]!;
        for (let dy = -1; dy <= 1; dy += 1)
          for (let dx = -1; dx <= 1; dx += 1) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            const q = ny * w + nx;
            if (label[q] === -1 && prob[q]! > thresh) {
              label[q] = found.length;
              stack.push(q);
            }
          }
      }
      if (pts.length < 4 || sum / pts.length < boxThresh) continue;
      const r = minAreaRect(convexHull(pts.map(([x, y]) => [x + 0.5, y + 0.5] as Point)));
      if (Math.min(r.w, r.h) < 2) continue;

      const d = (r.w * r.h * unclipRatio) / (2 * (r.w + r.h));
      const grown = { ...r, w: r.w + 2 * d, h: r.h + 2 * d };
      if (Math.min(grown.w, grown.h) < 5) continue;
      const quad = quadOf(grown).map(([x, y]) => [x * sx, y * sy] as Point) as [Point, Point, Point, Point];
      found.push({ quad, score: sum / pts.length, angle: Math.atan2(quad[1][1] - quad[0][1], quad[1][0] - quad[0][0]) });
    }

    return found.sort((a, b) => a.quad[0][1] - b.quad[0][1] || a.quad[0][0] - b.quad[0][0]);
  }

  /**
   * Lines the classifier sees upside down get their corners turned half a circle. Vertical
   * candidates are left alone (read both ways already).
   */
  async #orient(image: RasterImage, cn: number, candidates: { box: number; quad: [Point, Point, Point, Point] }[], boxes: number, reuse: boolean): Promise<void> {
    const counts = new Map<number, number>();
    for (const c of candidates) counts.set(c.box, (counts.get(c.box) ?? 0) + 1);
    const asked = candidates
      .map((c, i) => ({ c, i }))
      .filter(({ c }) => counts.get(c.box) === 1 && !(reuse && this.#seenBefore(c.quad)));
    if (asked.length === 0 || boxes === 0) return;
    const session = this.ori!.only;
    const plane = ORI_WIDTH * ORI_HEIGHT;
    const px = [0, 0, 0];
    for (let start = 0; start < asked.length; start += REC_BATCH) {
      const batch = asked.slice(start, start + REC_BATCH);
      const input = new Float32Array(batch.length * 3 * plane);
      batch.forEach(({ c }, b) => {
        const [tl, tr, , bl] = c.quad;
        for (let y = 0; y < ORI_HEIGHT; y += 1)
          for (let x = 0; x < ORI_WIDTH; x += 1) {
            const u = (x + 0.5) / ORI_WIDTH;
            const v = (y + 0.5) / ORI_HEIGHT;
            sample(image, cn, tl[0] + u * (tr[0] - tl[0]) + v * (bl[0] - tl[0]), tl[1] + u * (tr[1] - tl[1]) + v * (bl[1] - tl[1]), px);
            const i = y * ORI_WIDTH + x;
            for (let k = 0; k < 3; k += 1) input[b * 3 * plane + k * plane + i] = (px[2 - k]! / 255 - MEAN[k]!) / STD[k]!;
          }
      });
      const out = await session.run({ [session.inputs[0]!.name]: new this.Tensor("float32", input, [batch.length, 3, ORI_HEIGHT, ORI_WIDTH]) });
      const scores = (out[session.outputs[0]!.name] as { data: Float32Array }).data;
      batch.forEach(({ c }, b) => {
        if (scores[2 * b + 1]! > scores[2 * b]!) {
          const [tl, tr, br, bl] = c.quad;
          c.quad = [br, bl, tl, tr];
        }
      });
    }
  }

  /**
   * A line of the previous read at this place and size (its text will be reused: its orientation is
   * kept).
   */
  #seenBefore(quad: [Point, Point, Point, Point]): boolean {
    const [tl, tr, br, bl] = quad;
    const centre = [(tl[0] + tr[0] + br[0] + bl[0]) / 4, (tl[1] + tr[1] + br[1] + bl[1]) / 4];
    const width = Math.hypot(tr[0] - tl[0], tr[1] - tl[1]);
    const height = Math.hypot(bl[0] - tl[0], bl[1] - tl[1]);
    return this.#remembered.some(
      (r) => Math.abs(r.width - width) <= 0.15 * width && Math.abs(r.height - height) <= 0.2 * height && Math.hypot(r.centre[0] - centre[0]!, r.centre[1] - centre[1]!) <= 0.5 * height,
    );
  }

  async #recognize(image: RasterImage, boxes: { quad: [Point, Point, Point, Point]; score: number; angle: number }[], reuse: boolean, floor: number): Promise<TextLine[]> {
    const cn = channelsOf(image);

    const candidates: { box: number; quad: [Point, Point, Point, Point] }[] = [];
    boxes.forEach(({ quad }, box) => {
      const [tl, tr, br, bl] = quad;
      const vertical = Math.hypot(bl[0] - tl[0], bl[1] - tl[1]) >= VERTICAL_RATIO * Math.hypot(tr[0] - tl[0], tr[1] - tl[1]);
      if (!vertical) candidates.push({ box, quad });
      else candidates.push({ box, quad: [bl, tl, tr, br] }, { box, quad: [tr, br, bl, tl] });
    });

    if (this.ori) await this.#orient(image, cn, candidates, boxes.length, reuse);
    const crops: Crop[] = candidates.map(({ quad }) => {
      const [tl, tr, , bl] = quad;
      const wq = Math.hypot(tr[0] - tl[0], tr[1] - tl[1]);
      const hq = Math.hypot(bl[0] - tl[0], bl[1] - tl[1]);
      const width = Math.max(8, Math.min(1600, Math.ceil((REC_HEIGHT * wq) / Math.max(1, hq))));
      return { tl, ux: [(tr[0] - tl[0]) / width, (tr[1] - tl[1]) / width], uy: [(bl[0] - tl[0]) / REC_HEIGHT, (bl[1] - tl[1]) / REC_HEIGHT], width };
    });
    const read: { text: string; confidence: number }[] = new Array(candidates.length);

    const shapes = candidates.map(({ quad }) => {
      const [tl, tr, br, bl] = quad;
      return {
        centre: [(tl[0] + tr[0] + br[0] + bl[0]) / 4, (tl[1] + tr[1] + br[1] + bl[1]) / 4] as Point,
        width: Math.hypot(tr[0] - tl[0], tr[1] - tl[1]),
        height: Math.hypot(bl[0] - tl[0], bl[1] - tl[1]),
      };
    });
    const looks = crops.map((c) => lookOf(image, cn, c));
    let reused = 0;
    if (reuse) {
      shapes.forEach((s, i) => {
        const same = this.#remembered.find(
          (r) =>
            Math.abs(r.width - s.width) <= 0.15 * s.width &&
            Math.abs(r.height - s.height) <= 0.2 * s.height &&
            Math.hypot(r.centre[0] - s.centre[0], r.centre[1] - s.centre[1]) <= 0.5 * s.height &&
            lookDistance(r.look, looks[i]!) <= LOOK_SAME,
        );
        if (same) {
          read[i] = { text: same.text, confidence: same.confidence };
          reused += 1;
        }
      });
    }
    this.reused = reused;

    const order = crops
      .map((_, i) => i)
      .filter((i) => read[i] === undefined)
      .sort((a, b) => crops[a]!.width - crops[b]!.width);
    const session = this.rec.only;
    const space = this.dict.length + 1;
    for (let start = 0; start < order.length; start += REC_BATCH) {
      const batch = order.slice(start, start + REC_BATCH);

      const maxW = Math.ceil(Math.max(REC_MIN_WIDTH, ...batch.map((i) => crops[i]!.width)) / 160) * 160;
      const plane = REC_HEIGHT * maxW;
      const input = new Float32Array(batch.length * 3 * plane);

      batch.forEach((index, b) => cropInto(image, cn, crops[index]!, input, b * 3 * plane, plane, maxW));
      const out = await session.run({ [session.inputs[0]!.name]: new this.Tensor("float32", input, [batch.length, 3, REC_HEIGHT, maxW]) });
      const { steps, at } = bestPerStep(out);
      batch.forEach((index, b) => {
        let text = "";
        let last = 0;
        let confidence = 0;
        let kept = 0;
        for (let t = 0; t < steps; t += 1) {
          const [best, p] = at(b, t);
          if (best !== 0 && best !== last) {
            text += best === space ? " " : (this.dict[best - 1] ?? "");
            confidence += p;
            kept += 1;
          }
          last = best;
        }
        read[index] = { text: text.trim(), confidence: kept > 0 ? confidence / kept : 0 };
      });
    }
    this.#remembered = shapes.map((s, i) => ({ ...s, look: looks[i]!, text: read[i]!.text, confidence: read[i]!.confidence }));
    const best = new Map<number, number>();
    candidates.forEach((c, i) => {
      const held = best.get(c.box);
      if (held === undefined || read[i]!.confidence > read[held]!.confidence) best.set(c.box, i);
    });
    const lines = boxes.map((box, b) => {
      const i = best.get(b)!;
      const quad = candidates[i]!.quad;
      return { quad, text: read[i]!.text, confidence: read[i]!.confidence, angle: Math.atan2(quad[1][1] - quad[0][1], quad[1][0] - quad[0][0]), score: box.score };
    });
    this.dropped = lines.filter((l) => l.text.length > 0 && l.confidence < floor).map((l) => ({ text: l.text, confidence: l.confidence }));
    return lines.filter((l) => l.text.length > 0 && l.confidence >= floor);
  }

  async close(): Promise<void> {
    await Promise.all([this.det.close(), this.rec.close(), this.ori?.close()]);
  }
}

/**
 * Each step's best class and score: from the graph's argmax head when it has one (`argmaxHead`),
 * else scanned from the full scores.
 */
function bestPerStep(out: Record<string, unknown>): { steps: number; at: (b: number, t: number) => [number, number] } {
  const entries = Object.entries(out);
  const ids = entries.find(([name]) => name.endsWith(ARGMAX_SUFFIX))?.[1] as { data: Float32Array; dims: number[] } | undefined;
  const max = entries.find(([name]) => name.endsWith(MAX_SUFFIX))?.[1] as { data: Float32Array } | undefined;
  if (ids && max) {
    const steps = ids.dims[1]!;
    return { steps, at: (b, t) => [ids.data[b * steps + t]!, max.data[b * steps + t]!] };
  }
  const scores = entries[0]?.[1] as { data: Float32Array; dims: number[] } | undefined;
  if (!scores) throw new InferenceError("the recognizer gave no output");
  const [, steps, classes] = scores.dims as [number, number, number];
  return {
    steps,
    at: (b, t) => {
      const row = (b * steps + t) * classes;
      let best = 0;
      let p = -Infinity;
      for (let k = 0; k < classes; k += 1) {
        const v = scores.data[row + k]!;
        if (v > p) {
          p = v;
          best = k;
        }
      }
      return [best, p];
    },
  };
}

function boundsOf(quad: readonly Point[]) {
  const xs = quad.map((p) => p[0]);
  const ys = quad.map((p) => p[1]);
  return { x: Math.min(...xs), y: Math.min(...ys), w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) };
}

const lineHeight = (l: TextLine) => Math.hypot(l.quad[3][0] - l.quad[0][0], l.quad[3][1] - l.quad[0][1]);

/**
 * Lines stacked closely, of similar height and overlapping horizontally, read as one block (a
 * paragraph, a sign).
 */
export function groupBlocks(lines: readonly TextLine[], image: RasterImage): TextBlock[] {
  const blocks: TextLine[][] = [];
  for (const line of lines) {
    const b = boundsOf(line.quad);
    const h = lineHeight(line);
    const home = blocks.find((block) => {
      const prev = block[block.length - 1]!;
      const pb = boundsOf(prev.quad);
      const ph = lineHeight(prev);
      const gap = b.y - (pb.y + pb.h);
      const overlap = Math.min(b.x + b.w, pb.x + pb.w) - Math.max(b.x, pb.x);
      return Math.abs(h - ph) < 0.35 * Math.max(h, ph) && gap > -0.5 * h && gap < 0.9 * h && overlap > 0.2 * Math.min(b.w, pb.w);
    });
    if (home) home.push(line);
    else blocks.push([line]);
  }
  return blocks.map((ls) => {
    const all = ls.flatMap((l) => l.quad);
    const rect = boundsOf(all);
    const { density, ...colors } = colorsOf(image, ls);
    const lefts = ls.map((l) => boundsOf(l.quad).x);
    const centres = ls.map((l) => boundsOf(l.quad).x + boundsOf(l.quad).w / 2);
    const spread = (a: number[]) => Math.max(...a) - Math.min(...a);
    const align = ls.length > 1 && spread(centres) < spread(lefts) * 0.5 ? "center" : "left";
    return { lines: ls, text: ls.map((l) => l.text).join(" "), rect, angle: ls[0]!.angle, lineHeight: ls.reduce((s, l) => s + lineHeight(l), 0) / ls.length, ...colors, align, bold: density > 0.3 };
  });
}

/**
 * Background: the median of the pixels just outside the lines; text: the mean of those far from it.
 */
function colorsOf(image: RasterImage, lines: readonly TextLine[]): { color: [number, number, number]; background: [number, number, number]; density: number } {
  const cn = channelsOf(image);
  const ring: number[][] = [[], [], []];
  const inside: [number, number, number][] = [];
  const px = [0, 0, 0];
  for (const l of lines) {
    const b = boundsOf(l.quad);
    const step = Math.max(1, Math.floor(Math.min(b.w, b.h) / 12));
    for (let y = Math.floor(b.y); y < b.y + b.h; y += step)
      for (let x = Math.floor(b.x); x < b.x + b.w; x += step) {
        sample(image, cn, x, y, px);
        const edge = x - b.x < step * 2 || b.x + b.w - x < step * 2 || y - b.y < step || b.y + b.h - y < step * 1.5;
        if (edge) for (let k = 0; k < 3; k += 1) ring[k]!.push(px[k]!);
        else inside.push([px[0]!, px[1]!, px[2]!]);
      }
  }
  const median = (a: number[]) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1]! : 128);
  const background: [number, number, number] = [median(ring[0]!), median(ring[1]!), median(ring[2]!)];
  const far = inside.filter((p) => Math.hypot(p[0] - background[0], p[1] - background[1], p[2] - background[2]) > 60);
  const color: [number, number, number] = far.length
    ? [far.reduce((s, p) => s + p[0], 0) / far.length, far.reduce((s, p) => s + p[1], 0) / far.length, far.reduce((s, p) => s + p[2], 0) / far.length]
    : background[0] + background[1] + background[2] > 384 ? [20, 20, 20] : [235, 235, 235];
  return { color, background, density: inside.length ? far.length / inside.length : 0 };
}

type Context2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

export interface RewriteOptions {
  /** Font family of the rewritten text. */
  font?: string;
  /** `inpaint` (Telea, as cv2.inpaint) or `fill` (the background colour: faster). */
  erase?: "inpaint" | "fill";
}

/**
 * The text pixels of a block: inside its lines, far from the background colour, grown by a pixel.
 */
function textMask(image: { data: Uint8ClampedArray; width: number; height: number }, block: TextBlock): { mask: Uint8Array; rect: { x0: number; y0: number; x1: number; y1: number } } {
  const { width: W, height: H, data } = image;
  const mask = new Uint8Array(W * H);
  const [br, bg, bb] = block.background;
  const pad = Math.ceil(block.lineHeight * 0.15) + 2;
  const x0 = Math.max(0, Math.floor(block.rect.x) - pad);
  const y0 = Math.max(0, Math.floor(block.rect.y) - pad);
  const x1 = Math.min(W, Math.ceil(block.rect.x + block.rect.w) + pad);
  const y1 = Math.min(H, Math.ceil(block.rect.y + block.rect.h) + pad);
  for (const line of block.lines) {
    const b = boundsOf(line.quad);
    for (let y = Math.max(0, Math.floor(b.y)); y < Math.min(H, b.y + b.h); y += 1)
      for (let x = Math.max(0, Math.floor(b.x)); x < Math.min(W, b.x + b.w); x += 1) {
        const p = (y * W + x) * 4;
        if (Math.hypot(data[p]! - br, data[p + 1]! - bg, data[p + 2]! - bb) > 45) mask[y * W + x] = 1;
      }
  }

  const grown = mask.slice();
  for (let y = y0; y < y1; y += 1)
    for (let x = x0; x < x1; x += 1) {
      if (!mask[y * W + x]) continue;
      for (let dy = -2; dy <= 2; dy += 1)
        for (let dx = -2; dx <= 2; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && ny >= 0 && nx < W && ny < H) grown[ny * W + nx] = 1;
        }
    }
  return { mask: grown, rect: { x0, y0, x1, y1 } };
}

/** Words wrapped to `width` at font `size`. */
function wrap(ctx: Context2D, text: string, width: number): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let current = "";
  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (ctx.measureText(next).width <= width || !current) current = next;
    else {
      lines.push(current);
      current = word;
    }
  }
  if (current) lines.push(current);
  return lines;
}

/**
 * Erase each block's text and write `translations[i]` in its place, on a canvas that already shows
 * the image (same size).
 */
export function rewriteBlocks(ctx: Context2D, blocks: readonly TextBlock[], translations: readonly (string | null)[], options: RewriteOptions = {}): void {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  const image = ctx.getImageData(0, 0, W, H);
  const erase = options.erase ?? "inpaint";

  const rewritten = (i: number) => {
    const text = translations[i];
    return !!text && text.trim() !== blocks[i]!.text.trim();
  };
  blocks.forEach((block, i) => {
    if (!rewritten(i)) return;
    const { mask, rect } = textMask(image, block);
    if (erase === "inpaint") inpaintTelea(image.data, W, H, mask, rect, 3);
    else {
      for (let y = rect.y0; y < rect.y1; y += 1)
        for (let x = rect.x0; x < rect.x1; x += 1)
          if (mask[y * W + x]) image.data.set(block.background, (y * W + x) * 4);
    }
  });
  ctx.putImageData(image, 0, 0);
  const family = options.font ?? "system-ui, 'Segoe UI', Arial, sans-serif";
  blocks.forEach((block, i) => {
    const text = translations[i];
    if (!text || !rewritten(i)) return;
    const width = Math.max(block.rect.w, block.lineHeight * 2);
    const height = Math.max(block.rect.h, block.lineHeight);

    let size = Math.max(6, block.lineHeight * 0.9);
    let lines: string[] = [];
    for (; size >= 6; size *= 0.92) {
      ctx.font = `${block.bold ? "bold " : ""}${Math.round(size)}px ${family}`;
      lines = wrap(ctx, text, width);
      const widest = Math.max(...lines.map((l) => ctx.measureText(l).width));
      if (lines.length * size * 1.25 <= height * 1.05 && widest <= width * 1.02) break;
    }
    ctx.save();
    const cx = block.rect.x + block.rect.w / 2;
    const cy = block.rect.y + block.rect.h / 2;
    ctx.translate(cx, cy);
    ctx.rotate(block.angle);
    ctx.fillStyle = `rgb(${block.color.map(Math.round).join(",")})`;
    ctx.textBaseline = "middle";
    ctx.textAlign = block.align;
    const x = block.align === "left" ? -block.rect.w / 2 : 0;
    const total = lines.length * size * 1.25;
    lines.forEach((line, k) => ctx.fillText(line, x, -total / 2 + size * 1.25 * (k + 0.5)));
    ctx.restore();
  });
  void W;
  void H;
}

/**
 * Translations kept by source text: `get` answers at once from the cache and starts translating
 * what is new in the background (one batch at a time).
 */
export class TranslationCache {
  readonly #done = new Map<string, string>();
  readonly #pending = new Set<string>();
  #busy = false;

  constructor(private readonly translate: (texts: string[]) => Promise<string[]>) {}

  get(texts: readonly string[]): (string | null)[] {
    const missing = [...new Set(texts.filter((t) => !this.#done.has(t) && !this.#pending.has(t)))];
    for (const t of missing) this.#pending.add(t);
    void this.#drain();
    return texts.map((t) => this.#done.get(t) ?? null);
  }

  /** Translate now (an image, not a live feed). */
  async all(texts: readonly string[]): Promise<string[]> {
    const missing = [...new Set(texts.filter((t) => !this.#done.has(t)))];
    if (missing.length > 0) {
      const out = await this.translate(missing);
      missing.forEach((t, i) => this.#done.set(t, out[i] ?? t));
    }
    return texts.map((t) => this.#done.get(t)!);
  }

  async #drain(): Promise<void> {
    if (this.#busy || this.#pending.size === 0) return;
    this.#busy = true;
    try {
      while (this.#pending.size > 0) {
        const batch = [...this.#pending].slice(0, 8);
        try {
          const out = await this.translate(batch);
          batch.forEach((t, i) => this.#done.set(t, out[i] ?? t));
        } catch {
          batch.forEach((t) => this.#done.set(t, t));
        }
        for (const t of batch) this.#pending.delete(t);
      }
    } finally {
      this.#busy = false;
    }
  }
}
