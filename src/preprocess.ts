/**
 * YOLO image preprocessing: letterbox (ratio kept, gray padding) or centre crop (classification),
 * `[0, 1]` float32, BCHW — and the way back to the original image's pixels.
 */

import type { LetterboxMeta } from "./types.js";

/** Ultralytics' gray. */
export const PAD_COLOR = 114;

/** An RGB or RGBA image, row-major. `ImageData` is one (four channels). */
export interface RasterImage {
  data: Uint8Array | Uint8ClampedArray;
  width: number;
  height: number;
  /** 3 (RGB) or 4 (RGBA, the alpha ignored); read from the length when absent. */
  channels?: 3 | 4;
}

export function channelsOf(image: RasterImage): 3 | 4 {
  if (image.channels) return image.channels;
  return image.data.length === image.width * image.height * 3 ? 3 : 4;
}

/** Python's `round`: halves go to the even neighbour. */
export function pyRound(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

const COEF_BITS = 11;
const COEF_SCALE = 1 << COEF_BITS;

/**
 * Per destination index: the two source indices and their fixed-point weights, as OpenCV computes
 * them. Across columns a tap past the edge is moved onto it with its whole weight; across rows
 * OpenCV keeps the weights and only clamps the rows read (`clampWeights: false`), which rounds the
 * first and last rows differently.
 */
function linearTaps(src: number, dst: number, clampWeights: boolean): { index: Int32Array; weight: Int32Array } {
  const scale = src / dst;
  const index = new Int32Array(dst * 2);
  const weight = new Int32Array(dst * 2);
  for (let d = 0; d < dst; d += 1) {
    let f = Math.fround((d + 0.5) * scale - 0.5);
    let s = Math.floor(f);
    f = Math.fround(f - s);
    if (clampWeights && s < 0) {
      f = 0;
      s = 0;
    }
    if (clampWeights && s >= src - 1) {
      f = 0;
      s = src - 1;
    }
    const w0 = Math.max(-32768, Math.min(32767, Math.round((1 - f) * COEF_SCALE)));
    index[2 * d] = Math.min(Math.max(s, 0), src - 1);
    index[2 * d + 1] = Math.min(Math.max(s + 1, 0), src - 1);
    weight[2 * d] = w0;
    weight[2 * d + 1] = COEF_SCALE - w0;
  }
  return { index, weight };
}

/**
 * Bilinear resize of an 8-bit image to `width × height`, RGB out (the alpha of an RGBA source
 * dropped). OpenCV's `INTER_LINEAR` for `uint8`, rounding included.
 */
export function resizeLinear(image: RasterImage, width: number, height: number): Uint8Array {
  const cn = channelsOf(image);
  const { data, width: sw, height: sh } = image;
  const out = new Uint8Array(width * height * 3);
  const xs = linearTaps(sw, width, true);
  const ys = linearTaps(sh, height, false);

  const rowCache = new Map<number, Int32Array>();
  const hrow = (y: number): Int32Array => {
    let row = rowCache.get(y);
    if (row) return row;
    row = new Int32Array(width * 3);
    const base = y * sw * cn;
    for (let x = 0; x < width; x += 1) {
      const i0 = base + xs.index[2 * x]! * cn;
      const i1 = base + xs.index[2 * x + 1]! * cn;
      const a0 = xs.weight[2 * x]!;
      const a1 = xs.weight[2 * x + 1]!;
      for (let c = 0; c < 3; c += 1) row[x * 3 + c] = data[i0 + c]! * a0 + data[i1 + c]! * a1;
    }
    if (rowCache.size > 4) rowCache.delete(rowCache.keys().next().value!);
    rowCache.set(y, row);
    return row;
  };
  for (let y = 0; y < height; y += 1) {
    const r0 = hrow(ys.index[2 * y]!);
    const r1 = hrow(ys.index[2 * y + 1]!);
    const b0 = ys.weight[2 * y]!;
    const b1 = ys.weight[2 * y + 1]!;
    const o = y * width * 3;
    for (let i = 0; i < width * 3; i += 1) {
      const v = (((b0 * (r0[i]! >> 4)) >> 16) + ((b1 * (r1[i]! >> 4)) >> 16) + 2) >> 2;
      out[o + i] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return out;
}

function rgbOf(image: RasterImage): Uint8Array {
  const cn = channelsOf(image);
  if (cn === 3) return image.data instanceof Uint8Array ? image.data : new Uint8Array(image.data);
  const n = image.width * image.height;
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i += 1) {
    out[3 * i] = image.data[4 * i]!;
    out[3 * i + 1] = image.data[4 * i + 1]!;
    out[3 * i + 2] = image.data[4 * i + 2]!;
  }
  return out;
}

/** The size of `image` once fitted, ratio kept, in an `imgsz × imgsz` square. */
function fitted(image: { width: number; height: number }, imgsz: number): { scale: number; nw: number; nh: number } {
  const scale = Math.min(imgsz / image.height, imgsz / image.width);
  return { scale, nw: Math.max(1, pyRound(image.width * scale)), nh: Math.max(1, pyRound(image.height * scale)) };
}

/** The stride every side of a YOLO input is a multiple of. */
export const STRIDE = 32;

/**
 * The input `[height, width]` for `images`: the `imgsz` square, or with `rect` the smallest stride
 * multiples around the fitted images — a 16:9 frame at 640 goes in as 640 × 384, 40 % fewer pixels
 * for the same boxes (dynamic graphs only).
 */
export function inputShape(images: readonly { width: number; height: number }[], imgsz: number, rect = false): [number, number] {
  if (!rect) return [imgsz, imgsz];
  let height = STRIDE;
  let width = STRIDE;
  for (const image of images) {
    const { nw, nh } = fitted(image, imgsz);
    height = Math.max(height, Math.ceil(nh / STRIDE) * STRIDE);
    width = Math.max(width, Math.ceil(nw / STRIDE) * STRIDE);
  }
  return [height, width];
}

/**
 * The image resized with its ratio kept to fit an `imgsz` square, centred on a gray `height ×
 * width` canvas (the square itself by default).
 */
export function letterbox(image: RasterImage, imgsz: number, shape: [number, number] = [imgsz, imgsz]): { rgb: Uint8Array; meta: LetterboxMeta } {
  const { width: w, height: h } = image;
  const [height, width] = shape;
  const { scale, nw, nh } = fitted(image, imgsz);
  const resized = nw === w && nh === h ? rgbOf(image) : resizeLinear(image, nw, nh);
  const padTop = Math.floor((height - nh) / 2);
  const padLeft = Math.floor((width - nw) / 2);
  const rgb = new Uint8Array(height * width * 3).fill(PAD_COLOR);
  for (let y = 0; y < nh; y += 1) {
    rgb.set(resized.subarray(y * nw * 3, (y + 1) * nw * 3), ((y + padTop) * width + padLeft) * 3);
  }
  return { rgb, meta: { originalShape: [h, w], scale, padTop, padLeft } };
}

/** The short side resized to `imgsz`, then the centre square: how classifiers were trained. */
export function centerCrop(image: RasterImage, imgsz: number): { rgb: Uint8Array; meta: LetterboxMeta } {
  const { width: w, height: h } = image;
  const scale = imgsz / Math.min(h, w);
  const rw = Math.max(imgsz, pyRound(w * scale));
  const rh = Math.max(imgsz, pyRound(h * scale));
  const resized = rw === w && rh === h ? rgbOf(image) : resizeLinear(image, rw, rh);
  const top = Math.floor((rh - imgsz) / 2);
  const left = Math.floor((rw - imgsz) / 2);
  const rgb = new Uint8Array(imgsz * imgsz * 3);
  for (let y = 0; y < imgsz; y += 1) {
    const from = ((y + top) * rw + left) * 3;
    rgb.set(resized.subarray(from, from + imgsz * 3), y * imgsz * 3);
  }
  return { rgb, meta: { originalShape: [h, w], scale, padTop: -top || 0, padLeft: -left || 0 } };
}

/**
 * Images as one BCHW float32 batch in `[0, 1]`, with where each sits in it. `centerCrop` replaces
 * the letterbox for classification; `rect` letterboxes to the smallest stride multiples instead of
 * the square (`inputShape`).
 */
export function preprocess(
  images: readonly RasterImage[],
  imgsz: number,
  options: { centerCrop?: boolean; rect?: boolean } = {},
): { tensor: Float32Array; dims: [number, number, number, number]; metas: LetterboxMeta[] } {
  const [height, width] = options.centerCrop ? [imgsz, imgsz] : inputShape(images, imgsz, options.rect);
  const plane = height * width;
  const tensor = new Float32Array(images.length * 3 * plane);
  const metas: LetterboxMeta[] = [];
  images.forEach((image, b) => {
    const { rgb, meta } = options.centerCrop ? centerCrop(image, imgsz) : letterbox(image, imgsz, [height, width]);
    const base = b * 3 * plane;
    for (let i = 0; i < plane; i += 1) {
      tensor[base + i] = rgb[3 * i]! / 255;
      tensor[base + plane + i] = rgb[3 * i + 1]! / 255;
      tensor[base + 2 * plane + i] = rgb[3 * i + 2]! / 255;
    }
    metas.push(meta);
  });
  return { tensor, dims: [images.length, 3, height, width], metas };
}

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

/** `[x1, y1, x2, y2]` in model space → the original image, clipped to it. */
export function toOriginalBox(box: ArrayLike<number>, meta: LetterboxMeta): [number, number, number, number] {
  const [h, w] = meta.originalShape;
  const s = meta.scale;
  return [
    clamp(Math.fround((box[0]! - meta.padLeft) / s), 0, w),
    clamp(Math.fround((box[1]! - meta.padTop) / s), 0, h),
    clamp(Math.fround((box[2]! - meta.padLeft) / s), 0, w),
    clamp(Math.fround((box[3]! - meta.padTop) / s), 0, h),
  ];
}

/** One `(x, y)` point in model space → the original image, clipped to it. */
export function toOriginalPoint(x: number, y: number, meta: LetterboxMeta): [number, number] {
  const [h, w] = meta.originalShape;
  return [clamp(Math.fround((x - meta.padLeft) / meta.scale), 0, w), clamp(Math.fround((y - meta.padTop) / meta.scale), 0, h)];
}
