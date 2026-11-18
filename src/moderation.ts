/** Visual moderation: is an image unsafe (nudity, sexual content)? */

import { huggingFaceResolver, loadModel, type LoadOptions, type ModelBundle } from "@youni/onnxweb-core";
import type { TensorFactory } from "@youni/onnxweb-core/core";

import { YnONNXYOLOError } from "./errors.js";
import { frameSize, isFrameSource, type ImageInput } from "./gpu.js";
import { channelsOf, resizeLinear, type RasterImage } from "./preprocess.js";

export interface ModeratorEntry {
  repo: string;
  /** Labels whose probabilities add up to "unsafe". */
  unsafe: readonly string[];
  licence: string;
  notes: string;
}

export const MODERATORS: Readonly<Record<string, ModeratorEntry>> = {
  falconsai: { repo: "Rekib23r/nsfw_image_detection-ONNX", unsafe: ["nsfw"], licence: "Apache-2.0", notes: "Falconsai/nsfw_image_detection exported to ONNX" },
  freepik: { repo: "doppeltilde/nsfw_image_detector-ONNX", unsafe: ["medium", "high"], licence: "see Freepik/nsfw_image_detector", notes: "graded: neutral, low, medium, high" },
  "five-kinds": { repo: "onnx-community/nsfw-classifier-ONNX", unsafe: ["hentai", "porn", "sexy"], licence: "see giacomoarienti/nsfw-classifier", notes: "drawings, hentai, neutral, porn, sexy" },
};

export const DEFAULT_MODERATOR = "falconsai";

/** Zero-shot labels of the second pass: unsafe when one of `ZERO_SHOT_UNSAFE` ranks first. */
export const ZERO_SHOT_SAFE = ["everyday life", "people", "a street", "an animal", "food", "a landscape", "sport", "an object"] as const;
export const ZERO_SHOT_UNSAFE = ["nudity", "explicit sexual content", "pornography", "a naked person"] as const;

export interface ModerationResult {
  /** Probability that the image is unsafe (the unsafe labels added up). */
  unsafe: number;
  /** `unsafe >= threshold`, or the second pass said so. */
  flagged: boolean;
  /** The model's best label. */
  label: string;
  /** Every label of the model with its probability. */
  scores: Record<string, number>;
  /** Flagged by the zero-shot second pass, not by the classifier. */
  secondPass: boolean;
}

/**
 * The second pass: the best label of each image among `labels` (an image-text model ranking them
 * zero-shot).
 */
export type ZeroShotRanker = (images: readonly ImageInput[], labels: readonly string[]) => Promise<string[]>;

export interface ModeratorOptions extends Omit<LoadOptions, "model" | "resolver"> {
  Tensor: TensorFactory;
  /** A key of `MODERATORS`. */
  model?: string;
  /** `unsafe` from which an image is flagged (0.5). */
  threshold?: number;
  /** Off by default: see the module's notes. */
  secondPass?: ZeroShotRanker;
  endpoint?: string;
}

interface Preprocessor {
  width: number;
  height: number;
  rescale: number;
  normalize: boolean;
  mean: number[];
  std: number[];
}

export function readModerationPreprocessor(config: Record<string, unknown>): Preprocessor {
  const size = config["size"] as number | { height?: number; width?: number; shortest_edge?: number } | undefined;
  let width = 224;
  let height = 224;
  if (typeof size === "number") width = height = size;
  else if (size?.shortest_edge) width = height = size.shortest_edge;
  else if (size) {
    height = size.height ?? 224;
    width = size.width ?? 224;
  }
  return {
    width,
    height,
    rescale: (config["rescale_factor"] as number | undefined) ?? 1 / 255,
    normalize: config["do_normalize"] !== false,
    mean: (config["image_mean"] as number[] | undefined) ?? [0.5, 0.5, 0.5],
    std: (config["image_std"] as number[] | undefined) ?? [0.5, 0.5, 0.5],
  };
}

/** Softmax of each row, the unsafe labels added up. */
export function readModeration(logits: ArrayLike<number>, labels: readonly string[], unsafe: readonly string[], threshold: number): ModerationResult[] {
  const n = labels.length;
  const out: ModerationResult[] = [];
  for (let row = 0; row * n < logits.length; row += 1) {
    let max = -Infinity;
    for (let i = 0; i < n; i += 1) max = Math.max(max, logits[row * n + i]!);
    const exps = labels.map((_, i) => Math.exp(logits[row * n + i]! - max));
    const total = exps.reduce((a, b) => a + b, 0);
    const scores: Record<string, number> = {};
    labels.forEach((label, i) => (scores[label] = exps[i]! / total));
    const unsafeProbability = unsafe.reduce((sum, label) => sum + (scores[label] ?? 0), 0);
    const label = labels.reduce((best, l) => (scores[l]! > scores[best]! ? l : best), labels[0]!);
    out.push({ unsafe: unsafeProbability, flagged: unsafeProbability >= threshold, label, scores, secondPass: false });
  }
  return out;
}

export class ImageModerator {
  private constructor(
    readonly name: string,
    readonly entry: ModeratorEntry,
    private readonly bundle: ModelBundle,
    private readonly preprocessor: Preprocessor,
    private readonly labels: string[],
    private readonly Tensor: TensorFactory,
    readonly threshold: number,
    private readonly secondPass: ZeroShotRanker | null,
  ) {}

  static async load(options: ModeratorOptions): Promise<ImageModerator> {
    const { Tensor, model = DEFAULT_MODERATOR, threshold = 0.5, secondPass, endpoint, ...rest } = options;
    const entry = MODERATORS[model];
    if (!entry) throw new YnONNXYOLOError(`unknown moderator '${model}' (${Object.keys(MODERATORS).join(", ")})`);

    const gpu = typeof navigator !== "undefined" && "gpu" in navigator && (rest.backends === undefined || String(rest.backends).includes("webgpu"));
    const bundle = await loadModel({
      ...rest,
      extraFiles: ["preprocessor_config.json", "config.json"],
      resolver: huggingFaceResolver({ repo: entry.repo, subfolder: "onnx", manifest: false, ...(endpoint ? { endpoint } : {}) }),
      model: { name: entry.repo, computeType: (rest.computeType as string | undefined) ?? (gpu ? "float16" : "int8") },
    } as never);
    const config = JSON.parse(bundle.assetText("preprocessor_config.json") ?? "{}") as Record<string, unknown>;
    const id2label = (JSON.parse(bundle.assetText("config.json") ?? "{}") as { id2label?: Record<string, string> }).id2label ?? {};
    const labels = Object.keys(id2label)
      .map(Number)
      .sort((a, b) => a - b)
      .map((i) => id2label[String(i)]!);
    if (labels.length === 0) throw new YnONNXYOLOError(`${entry.repo} has no id2label in its config.json`);
    return new ImageModerator(model, entry, bundle, readModerationPreprocessor(config), labels, Tensor, threshold, secondPass ?? null);
  }

  get backend(): string {
    return this.bundle.backend;
  }

  /** One result per image, in order. */
  async check(images: readonly ImageInput[], batch = 16): Promise<ModerationResult[]> {
    const session = this.bundle.only;
    const { width, height } = this.preprocessor;
    const out: ModerationResult[] = [];
    for (let start = 0; start < images.length; start += batch) {
      const group = images.slice(start, start + batch);
      const pixels = new Float32Array(group.length * 3 * width * height);
      group.forEach((image, n) => pixels.set(this.#pixels(image), n * 3 * width * height));
      const result = (await session.run({ [session.inputs[0]!.name]: new this.Tensor("float32", pixels, [group.length, 3, height, width]) })) as Record<string, { data: ArrayLike<number> }>;
      out.push(...readModeration(result[session.outputs[0]!.name]!.data, this.labels, this.entry.unsafe, this.threshold));
    }
    if (this.secondPass) {
      const missed = out.map((r, i) => (r.flagged ? -1 : i)).filter((i) => i >= 0);
      if (missed.length > 0) {
        const ranked = await this.secondPass(missed.map((i) => images[i]!), [...ZERO_SHOT_SAFE, ...ZERO_SHOT_UNSAFE]);
        missed.forEach((i, k) => {
          if ((ZERO_SHOT_UNSAFE as readonly string[]).includes(ranked[k] ?? "")) {
            out[i]!.flagged = true;
            out[i]!.secondPass = true;
          }
        });
      }
    }
    return out;
  }

  /** `[3, height, width]`, rescaled and normalized. */
  #pixels(image: ImageInput): Float32Array {
    const { width, height, rescale, normalize, mean, std } = this.preprocessor;
    let rgb: Uint8Array | Uint8ClampedArray;
    let step = 3;
    if (isFrameSource(image)) {
      const canvas = new OffscreenCanvas(width, height);
      const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
      const [w, h] = frameSize(image);
      ctx.drawImage(image as CanvasImageSource, 0, 0, w, h, 0, 0, width, height);
      rgb = ctx.getImageData(0, 0, width, height).data;
      step = 4;
    } else {
      const raster = image as RasterImage;
      rgb = raster.width === width && raster.height === height && channelsOf(raster) === 3 ? raster.data : resizeLinear(raster, width, height);
    }
    const plane = width * height;
    const out = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i += 1) {
      for (let c = 0; c < 3; c += 1) {
        const v = rgb[i * step + c]! * rescale;
        out[c * plane + i] = normalize ? (v - mean[c]!) / std[c]! : v;
      }
    }
    return out;
  }

  async close(): Promise<void> {
    await this.bundle.close();
  }
}
