/** Appearance embeddings for BoT-SORT with ReID. */

import { loadModel, type LoadOptions, type ModelBundle, type ModelSession } from "@youni/onnxweb-core";
import type { TensorFactory } from "@youni/onnxweb-core/core";

import { TrackerError } from "./errors.js";
import { frameSize as frameDims, isFrameSource, type FrameSource, type ImageInput } from "./gpu.js";
import { channelsOf, resizeLinear } from "./preprocess.js";
import type { BoundingBox } from "./types.js";

export interface ReIDOptions extends LoadOptions {
  Tensor: TensorFactory;
  /** Crop size `[height, width]` when the graph's is dynamic (OSNet's 128 × 64). */
  inputHw?: [number, number];
}

export class ReIDEmbedder {
  private constructor(
    private readonly bundle: ModelBundle,
    private readonly session: ModelSession,
    private readonly Tensor: TensorFactory,
    readonly inputHw: [number, number],
  ) {}

  static async load(options: ReIDOptions): Promise<ReIDEmbedder> {
    const { Tensor, inputHw, ...rest } = options;
    let bundle: ModelBundle;
    try {
      bundle = await loadModel(rest);
    } catch (error) {
      throw new TrackerError(`cannot load the ReID model: ${(error as Error).message}`, { cause: error });
    }
    const session = bundle.only;
    const dims = session.inputs[0]?.dims ?? [];
    const fixed = (d: number | string | undefined) => (typeof d === "number" && d > 0 ? d : null);

    const trained = Number(/\d+/.exec(session.metadata["imgsz"] ?? "")?.[0]) || null;
    const hw: [number, number] = [fixed(dims[2]) ?? inputHw?.[0] ?? trained ?? 128, fixed(dims[3]) ?? inputHw?.[1] ?? trained ?? 64];
    return new ReIDEmbedder(bundle, session, Tensor, hw);
  }

  /** One L2-normalized row per box; a zero row for an empty crop. */
  async embed(image: ImageInput, boxes: readonly BoundingBox[]): Promise<Float32Array[]> {
    if (boxes.length === 0) return [];
    const [th, tw] = this.inputHw;
    const plane = th * tw;
    const batch = new Float32Array(boxes.length * 3 * plane);
    const [width, height] = isFrameSource(image) ? frameDims(image) : [image.width, image.height];
    const canvas = isFrameSource(image) ? new OffscreenCanvas(tw, th) : null;
    const valid = boxes.map((box, n) => {
      const x1 = Math.max(0, Math.trunc(box.x1));
      const y1 = Math.max(0, Math.trunc(box.y1));
      const x2 = Math.min(width, Math.trunc(box.x2));
      const y2 = Math.min(height, Math.trunc(box.y2));
      if (x2 <= x1 || y2 <= y1) return false;
      let rgb: Uint8Array;
      if (canvas && isFrameSource(image)) {
        const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
        ctx.drawImage(image as CanvasImageSource, x1, y1, x2 - x1, y2 - y1, 0, 0, tw, th);
        const rgba = ctx.getImageData(0, 0, tw, th).data;
        rgb = new Uint8Array(plane * 3);
        for (let i = 0; i < plane; i += 1) {
          rgb[3 * i] = rgba[4 * i]!;
          rgb[3 * i + 1] = rgba[4 * i + 1]!;
          rgb[3 * i + 2] = rgba[4 * i + 2]!;
        }
      } else {
        const raster = image as Exclude<ImageInput, FrameSource>;
        const cn = channelsOf(raster);
        const crop = new Uint8Array((x2 - x1) * (y2 - y1) * cn);
        for (let y = y1; y < y2; y += 1) {
          crop.set(raster.data.subarray((y * raster.width + x1) * cn, (y * raster.width + x2) * cn), (y - y1) * (x2 - x1) * cn);
        }
        rgb = resizeLinear({ data: crop, width: x2 - x1, height: y2 - y1, channels: cn }, tw, th);
      }
      const base = n * 3 * plane;
      for (let i = 0; i < plane; i += 1) {
        batch[base + i] = rgb[3 * i]! / 255;
        batch[base + plane + i] = rgb[3 * i + 1]! / 255;
        batch[base + 2 * plane + i] = rgb[3 * i + 2]! / 255;
      }
      return true;
    });
    const name = this.session.inputs[0]!.name;
    const outputs = await this.session.run({ [name]: new this.Tensor("float32", batch, [boxes.length, 3, th, tw]) });
    const out = outputs[this.session.outputs[0]!.name] as { data: Float32Array };
    const dim = out.data.length / boxes.length;
    return boxes.map((_, n) => {
      const row = Float32Array.from(out.data.subarray(n * dim, (n + 1) * dim));
      if (!valid[n]) return row.fill(0);
      let norm = 0;
      for (const v of row) norm += v * v;
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < row.length; i += 1) row[i] = row[i]! / norm;
      return row;
    });
  }

  async close(): Promise<void> {
    await this.bundle.close();
  }
}

/** Cosine distance in `[0, 1]`: `(1 − cos) / 2`, `[tracks][detections]`. */
export function appearanceCost(tracks: readonly Float32Array[], detections: readonly Float32Array[]): number[][] {
  return tracks.map((t) =>
    detections.map((d) => {
      let dot = 0;
      for (let i = 0; i < t.length; i += 1) dot += t[i]! * d[i]!;
      return (1 - dot) / 2;
    }),
  );
}
