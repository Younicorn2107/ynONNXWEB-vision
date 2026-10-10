/** Global motion compensation for a moving camera (BoT-SORT). */

import { channelsOf, resizeLinear, type RasterImage } from "./preprocess.js";

/** A 2 × 3 affine, row-major: `[a, b, tx, c, d, ty]`. */
export type Affine = [number, number, number, number, number, number];

export const IDENTITY: Readonly<Affine> = [1, 0, 0, 0, 1, 0];

interface Gray {
  data: Float32Array;
  width: number;
  height: number;
}

/** OpenCV's RGB → gray weights. */
function toGray(rgb: Uint8Array, width: number, height: number): Gray {
  const data = new Float32Array(width * height);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = Math.round(0.299 * rgb[3 * i]! + 0.587 * rgb[3 * i + 1]! + 0.114 * rgb[3 * i + 2]!);
  }
  return { data, width, height };
}

/** Half size, by 2 × 2 averaging: one pyramid level. */
function halve(g: Gray): Gray {
  const width = Math.max(1, g.width >> 1);
  const height = Math.max(1, g.height >> 1);
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1)
    for (let x = 0; x < width; x += 1) {
      const x0 = Math.min(2 * x, g.width - 1);
      const y0 = Math.min(2 * y, g.height - 1);
      const x1 = Math.min(x0 + 1, g.width - 1);
      const y1 = Math.min(y0 + 1, g.height - 1);
      data[y * width + x] = (g.data[y0 * g.width + x0]! + g.data[y0 * g.width + x1]! + g.data[y1 * g.width + x0]! + g.data[y1 * g.width + x1]!) / 4;
    }
  return { data, width, height };
}

function sample(g: Gray, x: number, y: number): number {
  const cx = Math.min(Math.max(x, 0), g.width - 1.001);
  const cy = Math.min(Math.max(y, 0), g.height - 1.001);
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const fx = cx - x0;
  const fy = cy - y0;
  const i = y0 * g.width + x0;
  const a = g.data[i]! * (1 - fx) + g.data[i + 1]! * fx;
  const b = g.data[i + g.width]! * (1 - fx) + g.data[i + g.width + 1]! * fx;
  return a * (1 - fy) + b * fy;
}

/** Sobel gradients. */
function gradients(g: Gray): { gx: Float32Array; gy: Float32Array } {
  const { width: w, height: h, data } = g;
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  const at = (x: number, y: number) => data[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))]!;
  for (let y = 0; y < h; y += 1)
    for (let x = 0; x < w; x += 1) {
      gx[y * w + x] = at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1);
      gy[y * w + x] = at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1);
    }
  return { gx, gy };
}

/**
 * Shi–Tomasi corners: the smallest eigenvalue of the 3 × 3 structure tensor, local maxima, spaced
 * out.
 */
export function goodFeatures(g: Gray, maxCorners: number, quality: number, minDistance: number): [number, number][] {
  const { width: w, height: h } = g;
  const { gx, gy } = gradients(g);
  const response = new Float32Array(w * h);
  let best = 0;
  for (let y = 1; y < h - 1; y += 1)
    for (let x = 1; x < w - 1; x += 1) {
      let xx = 0;
      let xy = 0;
      let yy = 0;
      for (let dy = -1; dy <= 1; dy += 1)
        for (let dx = -1; dx <= 1; dx += 1) {
          const i = (y + dy) * w + x + dx;
          xx += gx[i]! * gx[i]!;
          xy += gx[i]! * gy[i]!;
          yy += gy[i]! * gy[i]!;
        }
      const half = (xx + yy) / 2;
      const v = half - Math.sqrt(((xx - yy) / 2) ** 2 + xy * xy);
      response[y * w + x] = v;
      if (v > best) best = v;
    }
  if (best <= 0) return [];
  const threshold = best * quality;
  const candidates: number[] = [];
  for (let y = 1; y < h - 1; y += 1)
    for (let x = 1; x < w - 1; x += 1) {
      const i = y * w + x;
      const v = response[i]!;
      if (v < threshold) continue;
      let isMax = true;
      for (let dy = -1; dy <= 1 && isMax; dy += 1) for (let dx = -1; dx <= 1; dx += 1) if (response[i + dy * w + dx]! > v) isMax = false;
      if (isMax) candidates.push(i);
    }
  candidates.sort((a, b) => response[b]! - response[a]! || a - b);
  const kept: [number, number][] = [];
  const min2 = minDistance * minDistance;
  for (const i of candidates) {
    const x = i % w;
    const y = Math.floor(i / w);
    if (kept.some(([kx, ky]) => (kx - x) ** 2 + (ky - y) ** 2 < min2)) continue;
    kept.push([x, y]);
    if (kept.length >= maxCorners) break;
  }
  return kept;
}

/** Pyramidal Lucas–Kanade: where each point of `prev` went in `next`; null when lost. */
export function lucasKanade(prev: Gray, next: Gray, points: readonly [number, number][], window = 21, levels = 3): ([number, number] | null)[] {
  const pyramidA = [prev];
  const pyramidB = [next];
  for (let l = 1; l <= levels; l += 1) {
    if (pyramidA[l - 1]!.width < window || pyramidA[l - 1]!.height < window) break;
    pyramidA.push(halve(pyramidA[l - 1]!));
    pyramidB.push(halve(pyramidB[l - 1]!));
  }

  const grads = pyramidA.map((a) => {
    const { gx, gy } = gradients(a);
    return { gx: { data: gx, width: a.width, height: a.height }, gy: { data: gy, width: a.width, height: a.height } };
  });
  const r = window >> 1;
  const size = window * window;
  const ix = new Float64Array(size);
  const iy = new Float64Array(size);
  const iv = new Float64Array(size);
  return points.map(([px, py]) => {
    let gx = 0;
    let gy = 0;
    for (let l = pyramidA.length - 1; l >= 0; l -= 1) {
      const a = pyramidA[l]!;
      const b = pyramidB[l]!;
      const { gx: dx, gy: dy } = grads[l]!;
      const s = 2 ** l;
      const x = px / s;
      const y = py / s;
      let xx = 0;
      let xy = 0;
      let yy = 0;
      let at = 0;
      for (let wy = -r; wy <= r; wy += 1)
        for (let wx = -r; wx <= r; wx += 1) {
          const sx = x + wx;
          const sy = y + wy;
          const gxv = sample(dx, sx, sy) / 8;
          const gyv = sample(dy, sx, sy) / 8;
          ix[at] = gxv;
          iy[at] = gyv;
          iv[at] = sample(a, sx, sy);
          at += 1;
          xx += gxv * gxv;
          xy += gxv * gyv;
          yy += gyv * gyv;
        }
      const det = xx * yy - xy * xy;
      if (det < 1e-6 * window * window) return null;
      let vx = 0;
      let vy = 0;
      for (let iter = 0; iter < 30; iter += 1) {
        let bx = 0;
        let by = 0;
        let k = 0;
        for (let wy = -r; wy <= r; wy += 1)
          for (let wx = -r; wx <= r; wx += 1) {
            const diff = iv[k]! - sample(b, x + gx + vx + wx, y + gy + vy + wy);
            bx += diff * ix[k]!;
            by += diff * iy[k]!;
            k += 1;
          }
        const ux = (yy * bx - xy * by) / det;
        const uy = (xx * by - xy * bx) / det;
        vx += ux;
        vy += uy;
        if (ux * ux + uy * uy < 1e-4) break;
      }
      gx = l > 0 ? 2 * (gx + vx) : gx + vx;
      gy = l > 0 ? 2 * (gy + vy) : gy + vy;
    }
    const nx = px + gx;
    const ny = py + gy;
    if (!Number.isFinite(nx) || !Number.isFinite(ny) || nx < 0 || ny < 0 || nx > next.width - 1 || ny > next.height - 1) return null;
    return [nx, ny];
  });
}

/** Least-squares similarity `[a −b tx; b a ty]` over point pairs. */
function fitSimilarity(src: readonly [number, number][], dst: readonly [number, number][]): Affine | null {
  const n = src.length;
  if (n < 2) return null;
  let mx = 0, my = 0, nx = 0, ny = 0;
  for (let i = 0; i < n; i += 1) {
    mx += src[i]![0];
    my += src[i]![1];
    nx += dst[i]![0];
    ny += dst[i]![1];
  }
  mx /= n; my /= n; nx /= n; ny /= n;
  let sxx = 0, sab = 0, sba = 0;
  for (let i = 0; i < n; i += 1) {
    const ax = src[i]![0] - mx;
    const ay = src[i]![1] - my;
    const bx = dst[i]![0] - nx;
    const by = dst[i]![1] - ny;
    sxx += ax * ax + ay * ay;
    sab += ax * bx + ay * by;
    sba += ax * by - ay * bx;
  }
  if (sxx < 1e-9) return null;
  const a = sab / sxx;
  const b = sba / sxx;
  return [a, -b, nx - a * mx + b * my, b, a, ny - b * mx - a * my];
}

/**
 * RANSAC over 2-point samples, then least squares on the inliers. Deterministic: a fixed-seed
 * generator.
 */
export function estimateSimilarity(src: readonly [number, number][], dst: readonly [number, number][], threshold = 3, iterations = 500): Affine | null {
  const n = src.length;
  if (n < 2) return null;
  let seed = 0x2545f491;
  const random = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 4294967296;
  };
  const t2 = threshold * threshold;
  const inliersOf = (m: Affine) => {
    const out: number[] = [];
    for (let i = 0; i < n; i += 1) {
      const [x, y] = src[i]!;
      const ex = m[0] * x + m[1] * y + m[2] - dst[i]![0];
      const ey = m[3] * x + m[4] * y + m[5] - dst[i]![1];
      if (ex * ex + ey * ey <= t2) out.push(i);
    }
    return out;
  };
  let best: number[] = [];
  for (let it = 0; it < iterations; it += 1) {
    const i = Math.floor(random() * n);
    let j = Math.floor(random() * (n - 1));
    if (j >= i) j += 1;
    const m = fitSimilarity([src[i]!, src[j]!], [dst[i]!, dst[j]!]);
    if (!m) continue;
    const inliers = inliersOf(m);
    if (inliers.length > best.length) best = inliers;
    if (best.length === n) break;
  }
  if (best.length < 2) return null;
  return fitSimilarity(best.map((i) => src[i]!), best.map((i) => dst[i]!));
}

export interface MotionCompensatorOptions {
  maxFeatures?: number;
  qualityLevel?: number;
  minDistance?: number;
  minMatches?: number;
  /** Integer downscale before estimating (speed). */
  downscale?: number;
  /** Lucas–Kanade's window (21, OpenCV's default) and pyramid levels (3). */
  window?: number;
  levels?: number;
}

/** Frame-to-frame camera motion: the affine mapping the previous frame onto the current one. */
export class MotionCompensator {
  readonly #maxFeatures: number;
  readonly #quality: number;
  readonly #minDistance: number;
  readonly #minMatches: number;
  readonly #downscale: number;
  readonly #window: number;
  readonly #levels: number;
  #prev: Gray | null = null;

  constructor(options: MotionCompensatorOptions = {}) {
    this.#maxFeatures = options.maxFeatures ?? 200;
    this.#quality = options.qualityLevel ?? 0.01;
    this.#minDistance = options.minDistance ?? 8;
    this.#minMatches = options.minMatches ?? 8;
    this.#downscale = Math.max(1, Math.trunc(options.downscale ?? 2));
    this.#window = options.window ?? 21;
    this.#levels = options.levels ?? 3;
  }

  estimate(frame: RasterImage): Affine {
    const d = this.#downscale;
    const width = Math.max(1, Math.floor(frame.width / d));
    const height = Math.max(1, Math.floor(frame.height / d));
    const rgb = d > 1 ? resizeLinear(frame, width, height) : rgbBytes(frame);
    const gray = toGray(rgb, width, height);
    const prev = this.#prev;
    this.#prev = gray;
    if (!prev || prev.width !== gray.width || prev.height !== gray.height) return [...IDENTITY];
    const corners = goodFeatures(prev, this.#maxFeatures, this.#quality, this.#minDistance);
    if (corners.length < this.#minMatches) return [...IDENTITY];
    const moved = lucasKanade(prev, gray, corners, this.#window, this.#levels);
    const src: [number, number][] = [];
    const dst: [number, number][] = [];
    moved.forEach((p, i) => {
      if (p) {
        src.push(corners[i]!);
        dst.push(p);
      }
    });
    if (src.length < this.#minMatches) return [...IDENTITY];
    const m = estimateSimilarity(src, dst);
    if (!m) return [...IDENTITY];

    return [m[0], m[1], m[2] * d, m[3], m[4], m[5] * d];
  }

  /** Forget the previous frame (a new clip or scene). */
  reset(): void {
    this.#prev = null;
  }
}

function rgbBytes(frame: RasterImage): Uint8Array {
  const cn = channelsOf(frame);
  if (cn === 3) return frame.data instanceof Uint8Array ? frame.data : new Uint8Array(frame.data);
  const out = new Uint8Array(frame.width * frame.height * 3);
  for (let i = 0; i < frame.width * frame.height; i += 1) {
    out[3 * i] = frame.data[4 * i]!;
    out[3 * i + 1] = frame.data[4 * i + 1]!;
    out[3 * i + 2] = frame.data[4 * i + 2]!;
  }
  return out;
}

export function transformPoint(m: Readonly<Affine>, x: number, y: number): [number, number] {
  return [m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]];
}

/** The uniform scale of a partial affine. */
export function transformScale(m: Readonly<Affine>): number {
  const det = m[0] * m[4] - m[1] * m[3];
  return det !== 0 ? Math.sqrt(Math.abs(det)) : 1;
}
