/**
 * Drawing a result on a canvas — boxes and labels, masks, keypoints and skeletons, oriented boxes,
 * a classification banner, a semantic overlay — and blurring what was found, before an image leaves
 * the page.
 */

import { COCO_SKELETON } from "./coco.js";
import { trackedOf, type TrackHistory } from "./history.js";
import type { BoundingBox, Mask, YOLOResult } from "./types.js";

export interface AnnotationConfig {
  boxThickness?: number;
  /** Label font size, pixels. */
  fontSize?: number;
  maskAlpha?: number;
  /**
   * Smoothing of maps and masks, 0–1: 0 draws them at full resolution, 0.5 with half the points on
   * each side, 1 not at all (boxes and labels stay). Coarser is faster to build and softer to look
   * at.
   */
  smoothing?: number;
  keypointRadius?: number;
  showConf?: boolean;
  showClass?: boolean;
  showTrackId?: boolean;
}

/** 20 distinct colours (RGB), by class id, cycled: Python's palette. */
export const PALETTE: readonly (readonly [number, number, number])[] = [
  [255, 56, 56], [255, 157, 151], [255, 112, 31], [255, 178, 29],
  [207, 210, 49], [72, 249, 10], [146, 204, 23], [61, 219, 134],
  [26, 147, 52], [0, 212, 187], [44, 153, 168], [0, 194, 255],
  [52, 69, 147], [100, 115, 255], [0, 24, 236], [132, 56, 255],
  [82, 0, 133], [203, 56, 255], [255, 149, 200], [255, 55, 199],
];

export function classColor(classId: number): readonly [number, number, number] {
  return PALETTE[((classId % PALETTE.length) + PALETTE.length) % PALETTE.length]!;
}

const css = ([r, g, b]: readonly [number, number, number], alpha = 1) => `rgba(${r}, ${g}, ${b}, ${alpha})`;

type Context2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

function labelOf(box: BoundingBox, config: Required<AnnotationConfig>): string {
  const parts: string[] = [];
  if (config.showClass) parts.push(box.className);
  if (config.showTrackId && box.trackId !== null) parts.push(`#${box.trackId}`);
  if (config.showConf) parts.push(box.confidence.toFixed(2));
  return parts.join(" ");
}

function drawLabel(ctx: Context2D, text: string, x: number, y: number, color: readonly [number, number, number], config: Required<AnnotationConfig>): void {
  ctx.font = `${config.fontSize}px sans-serif`;
  const width = ctx.measureText(text).width + 4;
  const height = config.fontSize + 4;
  const top = Math.max(0, y - height);
  ctx.fillStyle = css(color);
  ctx.fillRect(x, top, width, height);
  ctx.fillStyle = "#fff";
  ctx.textBaseline = "bottom";
  ctx.fillText(text, x + 2, top + height - 2);
}

type Overlay = OffscreenCanvas | HTMLCanvasElement;

const mapCache = new WeakMap<object, { smoothing: number; canvas: Overlay }>();
const maskCache = new WeakMap<object, { smoothing: number; canvas: Overlay; x: number; y: number; w: number; h: number } | null>();

function newCanvas(w: number, h: number): Overlay {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

function depthColor(t: number): [number, number, number] {
  return [
    Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)))),
    Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)))),
    Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)))),
  ];
}

/** A depth or class map, coloured at a resolution reduced by `smoothing`. */
function mapOverlay(result: Extract<YOLOResult, { task: "depth" | "semantic" }>, smoothing: number): Overlay {
  const cached = mapCache.get(result);
  if (cached && cached.smoothing === smoothing) return cached.canvas;
  const [H, W] = result.imageShape;
  const w = Math.max(1, Math.round(W * (1 - smoothing)));
  const h = Math.max(1, Math.round(H * (1 - smoothing)));
  const canvas = newCanvas(w, h);
  const ctx = canvas.getContext("2d") as Context2D;
  const image = ctx.createImageData(w, h);
  const px = image.data;
  const span = result.task === "depth" ? result.max - result.min || 1 : 1;
  for (let y = 0; y < h; y += 1) {
    const sy = Math.min(H - 1, Math.floor(((y + 0.5) * H) / h));
    for (let x = 0; x < w; x += 1) {
      const sx = Math.min(W - 1, Math.floor(((x + 0.5) * W) / w));
      const i = sy * W + sx;
      const o = (y * w + x) * 4;
      if (result.task === "depth") {
        const [r, g, b] = depthColor(1 - (result.depth[i]! - result.min) / span);
        px[o] = r;
        px[o + 1] = g;
        px[o + 2] = b;
        px[o + 3] = 255;
      } else {
        const id = result.mask[i]!;
        if (!id) continue;
        const [r, g, b] = classColor(id);
        px[o] = r;
        px[o + 1] = g;
        px[o + 2] = b;
        px[o + 3] = 255;
      }
    }
  }
  ctx.putImageData(image, 0, 0);
  mapCache.set(result, { smoothing, canvas });
  return canvas;
}

/** One instance mask, coloured over its box only, at a resolution reduced by `smoothing`. */
function maskOverlay(mask: Mask, smoothing: number): { canvas: Overlay; x: number; y: number; w: number; h: number } | null {
  const cached = maskCache.get(mask);
  if (cached !== undefined && (cached === null || cached.smoothing === smoothing)) return cached;
  const x0 = Math.max(0, Math.floor(mask.box.x1));
  const y0 = Math.max(0, Math.floor(mask.box.y1));
  const x1 = Math.min(mask.width, Math.ceil(mask.box.x2));
  const y1 = Math.min(mask.height, Math.ceil(mask.box.y2));
  if (x1 <= x0 || y1 <= y0) {
    maskCache.set(mask, null);
    return null;
  }
  const bw = x1 - x0;
  const bh = y1 - y0;
  const w = Math.max(1, Math.round(bw * (1 - smoothing)));
  const h = Math.max(1, Math.round(bh * (1 - smoothing)));
  const canvas = newCanvas(w, h);
  const ctx = canvas.getContext("2d") as Context2D;
  const image = ctx.createImageData(w, h);
  const [r, g, b] = classColor(mask.box.classId);
  for (let y = 0; y < h; y += 1) {
    const sy = y0 + Math.min(bh - 1, Math.floor(((y + 0.5) * bh) / h));
    for (let x = 0; x < w; x += 1) {
      const sx = x0 + Math.min(bw - 1, Math.floor(((x + 0.5) * bw) / w));
      if (!mask.data[sy * mask.width + sx]) continue;
      image.data.set([r, g, b, 255], (y * w + x) * 4);
    }
  }
  ctx.putImageData(image, 0, 0);
  const out = { smoothing, canvas, x: x0, y: y0, w: bw, h: bh };
  maskCache.set(mask, out);
  return out;
}

function drawOverlay(ctx: Context2D, overlay: Overlay, x: number, y: number, w: number, h: number, alpha: number): void {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(overlay as CanvasImageSource, x, y, w, h);
  ctx.restore();
}

/**
 * Draw `result` over what the canvas already shows (draw the image first). The canvas must be the
 * image's size: results are in its pixels.
 */
export function annotate(ctx: Context2D, result: YOLOResult, options: AnnotationConfig = {}): void {
  const config: Required<AnnotationConfig> = {
    boxThickness: options.boxThickness ?? 2,
    fontSize: options.fontSize ?? 14,
    maskAlpha: options.maskAlpha ?? 0.4,
    smoothing: options.smoothing ?? 0,
    keypointRadius: options.keypointRadius ?? 3,
    showConf: options.showConf ?? true,
    showClass: options.showClass ?? true,
    showTrackId: options.showTrackId ?? true,
  };
  const [height, width] = result.imageShape;
  const smoothing = Math.min(1, Math.max(0, options.smoothing ?? 0));
  if (smoothing < 1 && config.maskAlpha > 0) {
    if (result.task === "depth" || result.task === "semantic") {
      const overlay = mapOverlay(result, smoothing);
      drawOverlay(ctx, overlay, 0, 0, width, height, result.task === "depth" ? Math.max(config.maskAlpha, 0.6) : config.maskAlpha);
    } else if (result.task === "segment") {
      for (const mask of result.masks) {
        const o = maskOverlay(mask, smoothing);
        if (o) drawOverlay(ctx, o.canvas, o.x, o.y, o.w, o.h, config.maskAlpha);
      }
    }
  }
  if ("boxes" in result) {
    for (const box of result.boxes) {
      const color = classColor(box.classId);
      ctx.strokeStyle = css(color);
      ctx.lineWidth = box.isInterpolated ? Math.max(1, config.boxThickness - 1) : config.boxThickness;
      ctx.strokeRect(Math.trunc(box.x1), Math.trunc(box.y1), Math.trunc(box.x2) - Math.trunc(box.x1), Math.trunc(box.y2) - Math.trunc(box.y1));
      const label = labelOf(box, config);
      if (label) drawLabel(ctx, label, Math.trunc(box.x1), Math.trunc(box.y1), color, config);
    }
  }
  if (result.task === "pose") {
    for (const set of result.keypoints) {
      const color = css(classColor(set.box.classId));
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = Math.max(1, config.boxThickness - 1);
      const pts = set.points;
      for (const [a, b] of COCO_SKELETON) {
        const pa = pts[a];
        const pb = pts[b];
        if (!pa || !pb || pa.confidence <= 0.3 || pb.confidence <= 0.3) continue;
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      }
      for (const p of pts) {
        if (p.confidence <= 0.3) continue;
        ctx.beginPath();
        ctx.arc(p.x, p.y, config.keypointRadius, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }
  if (result.task === "obb") {
    for (const box of result.orientedBoxes) {
      const color = classColor(box.classId);
      const c = Math.cos(box.angle);
      const s = Math.sin(box.angle);
      const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([dx, dy]) => [
        box.cx + (dx! * box.w * c) / 2 - (dy! * box.h * s) / 2,
        box.cy + (dx! * box.w * s) / 2 + (dy! * box.h * c) / 2,
      ]);
      ctx.strokeStyle = css(color);
      ctx.lineWidth = config.boxThickness;
      ctx.beginPath();
      corners.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x!, y!) : ctx.lineTo(x!, y!)));
      ctx.closePath();
      ctx.stroke();

      const id = config.showTrackId && box.trackId !== null && box.trackId !== undefined ? `#${box.trackId}` : "";
      const label = [config.showClass ? box.className : "", id, config.showConf ? box.confidence.toFixed(2) : ""].filter(Boolean).join(" ");
      const top = corners.reduce((best, p) => (p[1]! < best[1]! ? p : best));
      if (label) drawLabel(ctx, label, Math.trunc(top[0]!), Math.trunc(top[1]!), color, config);
    }
  }
  if (result.task === "classify") {
    ctx.font = `bold ${config.fontSize + 2}px sans-serif`;
    ctx.textBaseline = "alphabetic";
    result.topK.slice(0, 5).forEach((p, i) => {
      ctx.fillStyle = css(classColor(p.classId));
      ctx.fillText(`${p.className}: ${p.confidence.toFixed(2)}`, 8, 24 + 22 * i);
    });
  }
}

export interface BlurOptions {
  /** Blur radius in pixels; by default a tenth of each region's smaller side, at least 8. */
  radius?: number;
  /** Margin added around each box, as a fraction of its size. */
  padding?: number;
  /** Only these classes (names or ids); every box by default. */
  classes?: readonly (string | number)[];
  /** Fill instead of blur: no trace of the content survives. */
  fill?: string;
}

/**
 * Blur (or fill) every box of `result` — or every masked pixel of a segmentation — on a canvas that
 * shows the image: anonymize before upload. Uses the canvas `filter`, applied to a copy of each
 * region.
 */
export function blurRegions(ctx: Context2D, result: YOLOResult, options: BlurOptions = {}): number {
  if (!("boxes" in result)) return 0;
  const wanted = options.classes ? new Set(options.classes.map(String)) : null;
  const pad = options.padding ?? 0.05;
  const [height, width] = result.imageShape;
  const source = ctx.canvas;
  let count = 0;
  result.boxes.forEach((box, k) => {
    if (wanted && !wanted.has(box.className) && !wanted.has(String(box.classId))) return;
    const bw = box.x2 - box.x1;
    const bh = box.y2 - box.y1;
    const x = Math.max(0, Math.floor(box.x1 - bw * pad));
    const y = Math.max(0, Math.floor(box.y1 - bh * pad));
    const w = Math.min(width, Math.ceil(box.x2 + bw * pad)) - x;
    const h = Math.min(height, Math.ceil(box.y2 + bh * pad)) - y;
    if (w <= 0 || h <= 0) return;
    const mask = result.task === "segment" ? result.masks[k] : undefined;
    ctx.save();
    if (mask) {
      const path = new Path2D();
      for (let row = y; row < y + h; row += 1) {
        let start = -1;
        for (let col = x; col <= x + w; col += 1) {
          const inside = col < x + w && mask.data[row * mask.width + col] === 1;
          if (inside && start < 0) start = col;
          if (!inside && start >= 0) {
            path.rect(start, row, col - start, 1);
            start = -1;
          }
        }
      }
      ctx.clip(path);
    } else {
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();
    }
    if (options.fill) {
      ctx.fillStyle = options.fill;
      ctx.fillRect(x, y, w, h);
    } else {
      const radius = options.radius ?? Math.max(8, Math.round(Math.min(w, h) / 10));
      ctx.filter = `blur(${radius}px)`;

      const m = radius * 2;
      const sx = Math.max(0, x - m);
      const sy = Math.max(0, y - m);
      const sw = Math.min(width, x + w + m) - sx;
      const sh = Math.min(height, y + h + m) - sy;
      const copy = new OffscreenCanvas(sw, sh);
      copy.getContext("2d")!.drawImage(source as CanvasImageSource, sx, sy, sw, sh, 0, 0, sw, sh);
      ctx.drawImage(copy, sx, sy);
    }
    ctx.restore();
    count += 1;
  });
  return count;
}

/**
 * The path of each numbered object of `visible`, from `history`: its centres where the model saw
 * it, joined by lines. Objects no longer in `visible` are not drawn: a trail leaves the screen with
 * its object.
 */
export function drawTrails(ctx: Context2D, history: TrackHistory, visible: YOLOResult | null, options: { lineWidth?: number; maxPoints?: number } = {}): void {
  if (!visible) return;
  const width = options.lineWidth ?? 2;
  const maxPoints = options.maxPoints ?? 300;
  ctx.save();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  for (const box of trackedOf(visible)) {
    if (box.trackId === null) continue;
    const points = history.trail(box.trackId).slice(-maxPoints);
    if (points.length === 0) continue;
    const color = classColor(box.trackId);
    ctx.strokeStyle = css(color, 0.9);
    ctx.fillStyle = css(color, 1);
    ctx.lineWidth = width;
    ctx.beginPath();
    points.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.stroke();
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, width * 1.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}
