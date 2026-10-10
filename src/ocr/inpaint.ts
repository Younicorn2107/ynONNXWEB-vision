/**
 * Telea inpainting (fast marching), as OpenCV's `cv2.inpaint(…, INPAINT_TELEA)`: the masked pixels
 * are filled from the boundary inwards, each from its known neighbours within `radius`, weighted by
 * direction, distance and level.
 */

const KNOWN = 0;
const BAND = 1;
const INSIDE = 2;

class MinHeap {
  readonly #items: { key: number; at: number }[] = [];
  get size(): number {
    return this.#items.length;
  }
  push(key: number, at: number): void {
    const a = this.#items;
    a.push({ key, at });
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p]!.key <= a[i]!.key) break;
      [a[p], a[i]] = [a[i]!, a[p]!];
      i = p;
    }
  }
  pop(): { key: number; at: number } {
    const a = this.#items;
    const top = a[0]!;
    const last = a.pop()!;
    if (a.length > 0) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < a.length && a[l]!.key < a[m]!.key) m = l;
        if (r < a.length && a[r]!.key < a[m]!.key) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i]!, a[m]!];
        i = m;
      }
    }
    return top;
  }
}

/** Solve the eikonal equation at (y, x) from two neighbours (Telea's `solve`). */
function solve(dist: Float32Array, flag: Uint8Array, w: number, i1: number, i2: number): number {
  const d1 = flag[i1] === INSIDE ? 1e6 : dist[i1]!;
  const d2 = flag[i2] === INSIDE ? 1e6 : dist[i2]!;
  const m = Math.min(d1, d2);
  if (flag[i1] !== INSIDE && flag[i2] !== INSIDE) {
    const r = 2 - (d1 - d2) * (d1 - d2);
    if (r > 0) return (d1 + d2 + Math.sqrt(r)) / 2;
  }
  void w;
  return m + 1;
}

/**
 * Fill the pixels where `mask` is 1 (row-major over the whole image) inside the rectangle [x0, x1)
 * × [y0, y1) of `rgba` (width `width`).
 */
export function inpaintTelea(rgba: Uint8ClampedArray, width: number, height: number, mask: Uint8Array, rect: { x0: number; y0: number; x1: number; y1: number }, radius = 3): void {
  const x0 = Math.max(0, rect.x0);
  const y0 = Math.max(0, rect.y0);
  const w = Math.min(width, rect.x1) - x0;
  const h = Math.min(height, rect.y1) - y0;
  if (w <= 2 || h <= 2) return;
  const n = w * h;
  const flag = new Uint8Array(n);
  const dist = new Float32Array(n).fill(1e6);
  const heap = new MinHeap();
  const at = (x: number, y: number) => y * w + x;
  const px = (i: number) => ((y0 + Math.floor(i / w)) * width + x0 + (i % w)) * 4;
  for (let y = 0; y < h; y += 1) for (let x = 0; x < w; x += 1) if (mask[(y0 + y) * width + x0 + x]) flag[at(x, y)] = INSIDE;

  for (let y = 0; y < h; y += 1)
    for (let x = 0; x < w; x += 1) {
      const i = at(x, y);
      if (flag[i] !== INSIDE) continue;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const j = at(nx, ny);
        if (flag[j] === KNOWN) {
          flag[j] = BAND;
          dist[j] = 0;
          heap.push(0, j);
        }
      }
    }
  for (let i = 0; i < n; i += 1) if (flag[i] === KNOWN) dist[i] = 0;
  while (heap.size > 0) {
    const { at: i } = heap.pop();
    if (flag[i] === KNOWN) continue;
    flag[i] = KNOWN;
    const x = i % w;
    const y = Math.floor(i / w);
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const j = at(nx, ny);
      if (flag[j] !== INSIDE) continue;

      const xs = [nx > 0 ? at(nx - 1, ny) : j, nx < w - 1 ? at(nx + 1, ny) : j];
      const ys = [ny > 0 ? at(nx, ny - 1) : j, ny < h - 1 ? at(nx, ny + 1) : j];
      let d = 1e6;
      for (const a of xs) for (const b of ys) d = Math.min(d, solve(dist, flag, w, a, b));
      dist[j] = d;

      let sw = 0;
      const sum = [0, 0, 0];
      const gx = (flag[xs[1]!] !== INSIDE ? dist[xs[1]!]! : dist[j]!) - (flag[xs[0]!] !== INSIDE ? dist[xs[0]!]! : dist[j]!);
      const gy = (flag[ys[1]!] !== INSIDE ? dist[ys[1]!]! : dist[j]!) - (flag[ys[0]!] !== INSIDE ? dist[ys[0]!]! : dist[j]!);
      for (let ky = -radius; ky <= radius; ky += 1)
        for (let kx = -radius; kx <= radius; kx += 1) {
          const qx = nx + kx;
          const qy = ny + ky;
          if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
          const q = at(qx, qy);
          if (flag[q] === INSIDE || (q === j)) continue;
          const r2 = kx * kx + ky * ky;
          if (r2 > radius * radius) continue;
          const dir = Math.abs(-kx * gx - ky * gy) / Math.sqrt(r2) + 1e-6;
          const dst = 1 / r2;
          const lev = 1 / (1 + Math.abs(dist[q]! - dist[j]!));
          const wt = dir * dst * lev;
          const p = px(q);
          sum[0]! += wt * rgba[p]!;
          sum[1]! += wt * rgba[p + 1]!;
          sum[2]! += wt * rgba[p + 2]!;
          sw += wt;
        }
      if (sw > 0) {
        const p = px(j);
        rgba[p] = sum[0]! / sw;
        rgba[p + 1] = sum[1]! / sw;
        rgba[p + 2] = sum[2]! / sw;
      }
      flag[j] = BAND;
      heap.push(d, j);
    }
  }
}
