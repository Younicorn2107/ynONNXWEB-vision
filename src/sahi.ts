/**
 * SAHI — sliced aided hyper inference: a large image cut into overlapping tiles, each inferred, the
 * boxes moved back and merged by a cross-tile NMS. For small objects in large images (drone,
 * satellite, 4K).
 */

import { SAHIError } from "./errors.js";
import { nms } from "./nms.js";
import { channelsOf, type RasterImage } from "./preprocess.js";
import type { BoundingBox, DetectionResult, TileMeta } from "./types.js";

export interface SAHIOptions {
  /** Tile side in pixels (at least 32). */
  tileSize?: number;
  /** Overlap between neighbouring tiles, in `[0, 1)`. */
  tileOverlap?: number;
  /** IoU of the cross-tile merge. */
  mergeIou?: number;
}

export class SAHISlicer {
  readonly tileSize: number;
  readonly tileOverlap: number;
  readonly mergeIou: number;

  constructor(options: SAHIOptions = {}) {
    this.tileSize = options.tileSize ?? 640;
    this.tileOverlap = options.tileOverlap ?? 0.2;
    this.mergeIou = options.mergeIou ?? 0.5;
    if (!(this.tileOverlap >= 0 && this.tileOverlap < 1)) throw new SAHIError(`tileOverlap must be in [0, 1), got ${this.tileOverlap}`);
    if (this.tileSize < 32) throw new SAHIError(`tileSize must be >= 32, got ${this.tileSize}`);
  }

  /**
   * The tile grid: steps of `tileSize × (1 − overlap)`, the last tile of each row and column moved
   * back against the edge so the whole image is covered. An image smaller than a tile is one tile.
   */
  grid(width: number, height: number): TileMeta[] {
    const step = Math.max(1, Math.trunc(this.tileSize * (1 - this.tileOverlap)));
    const xs = this.#offsets(width, step);
    const ys = this.#offsets(height, step);
    return ys.flatMap((y) =>
      xs.map((x) => ({ xOffset: x, yOffset: y, width: Math.min(this.tileSize, width - x), height: Math.min(this.tileSize, height - y) })),
    );
  }

  /** The image cut into its tiles. */
  slice(image: RasterImage): { tiles: RasterImage[]; metas: TileMeta[] } {
    const metas = this.grid(image.width, image.height);
    const cn = channelsOf(image);
    const tiles = metas.map((m) => {
      const data = new Uint8Array(m.width * m.height * cn);
      for (let y = 0; y < m.height; y += 1) {
        const from = ((m.yOffset + y) * image.width + m.xOffset) * cn;
        data.set(image.data.subarray(from, from + m.width * cn), y * m.width * cn);
      }
      return { data, width: m.width, height: m.height, channels: cn };
    });
    return { tiles, metas };
  }

  /** Each tile's boxes moved back into the image, then one NMS across tiles. */
  merge(results: readonly DetectionResult[], metas: readonly TileMeta[], originalShape: [number, number], mergeIou?: number): DetectionResult {
    if (results.length !== metas.length) throw new SAHIError(`got ${results.length} results for ${metas.length} tiles`);
    const all: BoundingBox[] = [];
    results.forEach((result, t) => {
      const m = metas[t]!;
      for (const b of result.boxes) all.push({ ...b, x1: b.x1 + m.xOffset, y1: b.y1 + m.yOffset, x2: b.x2 + m.xOffset, y2: b.y2 + m.yOffset });
    });
    const template = results[0];
    const merged: DetectionResult = {
      modelName: template?.modelName ?? "",
      task: "detect",
      source: template?.source ?? null,
      inferenceTimeMs: results.reduce((s, r) => s + r.inferenceTimeMs, 0),
      imageShape: originalShape,
      boxes: [],
    };
    if (all.length === 0) return merged;
    const xyxy = Float32Array.from(all.flatMap((b) => [b.x1, b.y1, b.x2, b.y2]));
    const keep = nms(xyxy, all.map((b) => b.confidence), all.map((b) => b.classId), {
      iouThreshold: mergeIou ?? this.mergeIou,
      confThreshold: 0,
      maxDet: all.length,
    });
    merged.boxes = keep.map((i) => all[i]!);
    return merged;
  }

  #offsets(length: number, step: number): number[] {
    if (length <= this.tileSize) return [0];
    const offsets: number[] = [];
    for (let o = 0; o <= length - this.tileSize; o += step) offsets.push(o);
    const last = length - this.tileSize;
    if (offsets[offsets.length - 1] !== last) offsets.push(last);
    return offsets;
  }
}
