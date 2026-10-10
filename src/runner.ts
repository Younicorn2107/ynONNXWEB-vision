/**
 * `YOLORunner` — the entry point: a model loaded once, then images, batches, tiles (SAHI) and video
 * frames through letterbox → inference → the task's decoder → the original image's pixels →
 * tracking.
 */

import { huggingFaceResolver, loadModel, type LoadOptions, type ModelBundle, type ModelResolver, type ModelSession } from "@youni/onnxweb-core";
import { ConfigurationError, InferenceError, TaskError, type TensorFactory } from "@youni/onnxweb-core/core";

import { DEFAULT_MODEL, detectTask, publishedModel, taskFromMetadata } from "./catalogue.js";
import { COCO_CLASSES, COCO_KEYPOINT_NAMES } from "./coco.js";
import {
  contentRegion,
  decodeClassify,
  decodeDepth,
  decodeDetect,
  decodeObb,
  decodePose,
  decodeSegment,
  decodeSemantic,
  segmentBoxes,
  type DecodeOptions,
  type DecoderContext,
  type HeadLayout,
  type RawOutput,
} from "./decoders.js";
import { SAHIError } from "./errors.js";
import { fromHalf, toHalf } from "./half.js";
import { GpuMaskComposer } from "./gpuMasks.js";
import { GpuLetterbox, framePixels, frameSize, isFrameSource, type FrameSource, type ImageInput } from "./gpu.js";
import { inputShape, preprocess, STRIDE } from "./preprocess.js";
import { SAHISlicer, type SAHIOptions } from "./sahi.js";
import { KeyframeTimeline, type MotionOptions } from "./motion.js";
import { frameSignature, holdLost, likeness, rememberSeen, type PersistOptions, type SeenObject } from "./continuity.js";
import { TextEncoder, type TextEncoderOptions } from "./textEncoder.js";
import { ObjectTracker, propagate, type TrackerAlgorithm, type TrackerConfig } from "./tracker.js";
import type { ReIDEmbedder } from "./reid.js";
import { hasBoxes, type BoundingBox, type BoxResult, type DetectionResult, type FrameResult, type LetterboxMeta, type OrientedBox, type Task, type VideoResult, type YOLOResult } from "./types.js";

export interface YOLOLoadOptions extends Omit<LoadOptions, "model" | "resolver" | "computeType"> {
  Tensor: TensorFactory;
  /**
   * A published short name (`yolo26n`, `yolo26n-pose`, `yolov10s`), a Hub repository (`org/repo`,
   * its graph under `onnx/`), or the key of a model a `resolver` serves. `yolo26n` by default.
   */
  model?: string;
  /**
   * `auto` by default: `float32`. float16 halves the download but runs no faster — slower on
   * WebGPU, where a vision model is bound by kernel launches, not by memory (yolo26n: 9 ms in
   * float32, 9–10 in float16).
   */
  computeType?: string;
  /** Overrides the task told from the name and the graph's metadata. */
  task?: Task;
  /**
   * The model input side; by default the graph's, else the one it was trained at (its metadata),
   * else 224 for classification and 640 otherwise.
   */
  imgsz?: number;
  /** Where the model comes from; the Hub by default. */
  resolver?: ModelResolver | readonly ModelResolver[];
  /** A Hub mirror. */
  endpoint?: string;
  /** Open vocabulary: where the CLIP text encoder comes from. */
  textEncoder?: Omit<TextEncoderOptions, "ort" | "Tensor">;
  /**
   * On WebGPU, frames (`ImageBitmap`, `VideoFrame`, canvases) are letterboxed by a shader into a
   * GPU buffer and run through a session built for their shape — outputs on the GPU, the run
   * captured and replayed: 2 to 3 times faster per frame. On by default; off sends every frame
   * through the CPU path, byte for byte the Python letterbox.
   */
  gpuPipeline?: boolean;
}

/**
 * WebGPU, then WebAssembly, unless asked otherwise (`backends`; `auto` is the core's NPU → GPU →
 * CPU): the GPU path (letterbox shader, captured session) is WebGPU's, and WebNN compiles every new
 * input size anew.
 */
const DEFAULT_BACKENDS = ["webgpu", "wasm"] as const;

export interface RunOptions extends Omit<DecodeOptions, "classes"> {
  /**
   * Class ids to keep — or, on an open-vocabulary model (YOLOE, YOLO-World), the class prompts
   * themselves (`["person", "red car"]`).
   */
  classes?: readonly number[] | readonly string[] | null;
  /** Where the image came from, copied into the result. */
  source?: string;
  /**
   * The input side for this call (a multiple of 32; the load's by default). Smaller is faster — 480
   * runs about twice as fast as 640 — and loses small objects first. Dynamic graphs only
   * (`runner.dynamic`).
   */
  imgsz?: number;
  /**
   * Letterbox to the smallest multiples of 32 around the image instead of the square: a 16:9 frame
   * at 640 goes in as 640 × 384, the same boxes for 40 % less work. Dynamic graphs only;
   * classification keeps its centre crop.
   */
  rect?: boolean;
}

export interface SAHIRunOptions extends RunOptions, SAHIOptions {}

export interface VideoOptions {
  /** Track identities across frames; none by default. */
  tracker?: TrackerAlgorithm | null;
  trackerConfig?: TrackerConfig;
  /** BoT-SORT's appearance model, when `trackerConfig.withReid`. */
  reid?: ReIDEmbedder | null;
  /** Infer one frame in N; the others are propagated (with a tracker) or repeat the last result. */
  frameStep?: number;
  /** Kalman propagation on the frames not inferred. On by default. */
  interpolate?: boolean;
  /** The first frame's index (frame steps count from it). */
  startFrame?: number;
  run?: RunOptions;
  /**
   * Against flicker (needs a tracker): an object missed for an inference or two, in a shot that has
   * not changed, is held with its number. Off by default.
   */
  persist?: PersistOptions | null;
  /** The `source` of the `VideoResult` (`camera`, a file name). */
  source?: string;
  fps?: number;
  /** `offer`/`infer`/`resultAt`: how frames between and after keyframes are filled. */
  motion?: MotionOptions;
}

/** Python's dict repr of class names: `{0: 'person', 1: 'bicycle'}`, or a list. */
export function parseNames(raw: string | undefined): string[] | null {
  if (!raw) return null;
  const pairs = [...raw.matchAll(/(\d+)\s*:\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")/g)];
  if (pairs.length > 0) {
    return pairs.sort((a, b) => Number(a[1]) - Number(b[1])).map((m) => (m[2] ?? m[3] ?? "").replace(/\\(.)/g, "$1"));
  }
  const items = [...raw.matchAll(/'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"/g)].map((m) => m[1] ?? m[2] ?? "");
  return items.length > 0 ? items : null;
}

/** Ultralytics' `kpt_names`: `{0: ['nose', 'left_eye', …]}` — the first class's list. */
function parseKeypointNames(raw: string | undefined): string[] | null {
  if (!raw) return null;
  const list = raw.match(/\[([^\]]*)\]/);
  if (!list) return null;
  const names = [...list[1]!.matchAll(/'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"/g)].map((m) => m[1] ?? m[2] ?? "");
  return names.length > 0 ? names : null;
}

const isPlaceholder = (names: readonly string[]) => names.every((n, i) => n === String(i));

/** The nearest multiple of 32 (the stride), at least 32. */
function strideMultiple(size: number): number {
  return Math.max(STRIDE, Math.round(size / STRIDE) * STRIDE);
}

function staticDim(value: number | string | undefined): number | null {
  return typeof value === "number" && value > 0 ? value : null;
}

/**
 * The head's layout, from the graph: split `logits` + `pred_boxes` (transformers.js exports),
 * normalized coordinates (those exports, whose input is `pixel_values`), end-to-end rows
 * (Ultralytics says it in its metadata; otherwise one `[B, N, ≤8]` output).
 */
export function headLayout(session: Pick<ModelSession, "inputs" | "outputs" | "metadata">, imgsz: number): HeadLayout {
  const outputs = session.outputs.map((o) => o.name);
  const split = outputs.includes("logits") && outputs.includes("pred_boxes");
  const normalized = split || session.inputs.some((i) => i.name === "pixel_values");
  const declared = session.metadata["end2end"]?.trim().toLowerCase();
  let nmsFree: boolean;
  if (split) nmsFree = true;
  else if (declared === "true" || declared === "false") nmsFree = declared === "true";
  else if (/['"]nms['"]\s*:\s*True/.test(session.metadata["args"] ?? "")) nmsFree = true;
  else {
    const dims = session.outputs[0]?.dims ?? [];
    const last = dims[2];
    nmsFree = session.outputs.length === 1 && dims.length === 3 && typeof last === "number" && last <= 8;
  }
  return { nmsFree, normalized, split, imgsz };
}

/** The upright box around an oriented one, for tracking. */
function boxAround(o: OrientedBox): BoundingBox {
  const c = Math.abs(Math.cos(o.angle));
  const s = Math.abs(Math.sin(o.angle));
  const hw = (o.w * c + o.h * s) / 2;
  const hh = (o.w * s + o.h * c) / 2;
  return { x1: o.cx - hw, y1: o.cy - hh, x2: o.cx + hw, y2: o.cy + hh, confidence: o.confidence, classId: o.classId, className: o.className, trackId: null, isInterpolated: false };
}

interface GpuSession {
  session: ModelSession;
  /** The input, written by the letterbox shader, read by the session. */
  buffer: GPUBuffer;
  captured: boolean;
}

async function closeGpuSession(gpu: GpuSession): Promise<void> {
  await gpu.session.close();
  gpu.buffer.destroy();
}

export class YOLORunner {
  #textEncoder: TextEncoder | null = null;

  private constructor(
    readonly modelName: string,
    readonly task: Task,
    readonly imgsz: number,
    /** Whether the graph takes any input size (`imgsz` and `rect` per call). */
    readonly dynamic: boolean,
    readonly classNames: readonly string[],
    readonly keypointNames: readonly string[],
    readonly layout: HeadLayout,
    /** The graph's fixed batch size, when it has one (one image per call otherwise). */
    readonly staticBatch: number | null,
    private readonly bundle: ModelBundle,
    private readonly session: ModelSession,
    private readonly Tensor: TensorFactory,
    private readonly options: YOLOLoadOptions,
  ) {}

  static async load(options: YOLOLoadOptions): Promise<YOLORunner> {
    const { Tensor, model = DEFAULT_MODEL, task, imgsz, resolver, endpoint, textEncoder: _t, computeType: _c, gpuPipeline, backends = DEFAULT_BACKENDS, ...given } = options;
    const rest = { ...given, backends };

    const published = resolver ? null : publishedModel(model);

    const asked = options.computeType ?? "auto";
    const computeType = asked !== "auto" ? asked : !published || published.precisions.includes("float32") ? "float32" : published.precisions[0]!;
    const repo = published?.repo ?? model;
    const bundle = await loadModel({
      extraFiles: ["config.json"],
      ...rest,
      model: { name: repo, computeType },

      ...(gpuPipeline !== false ? { variants: true } : {}),
      resolver: resolver ?? huggingFaceResolver({ repo, subfolder: "onnx", manifest: false, ...(endpoint ? { endpoint } : {}) }),
    });
    const session = bundle.only;
    const meta = session.metadata;
    const resolvedTask = task ?? taskFromMetadata(meta["task"]) ?? published?.task ?? detectTask(model);
    const image = session.inputs.find((i) => (i.dims?.length ?? 0) === 4) ?? session.inputs[0]!;
    const graphSize = staticDim(image.dims?.[2]);

    const trained = Number(/\d+/.exec(meta["imgsz"] ?? "")?.[0]) || null;
    const size = graphSize ?? (imgsz ? strideMultiple(imgsz) : (trained ?? (resolvedTask === "classify" ? 224 : 640)));
    const layout = headLayout(session, size);
    const classNames = YOLORunner.#classNames(bundle, session, resolvedTask, layout);
    const keypointNames = parseKeypointNames(meta["kpt_names"]) ?? [...COCO_KEYPOINT_NAMES];
    const name = published?.name ?? (model.split(/[\\/:]/).pop() ?? model).replace(/\.onnx$/i, "");
    return new YOLORunner(name, resolvedTask, size, graphSize === null, classNames, keypointNames, layout, staticDim(image.dims?.[0]), bundle, session, Tensor, options);
  }

  /**
   * The model's class names: Ultralytics' metadata (placeholder names `0, 1…` count as missing —
   * COCO when there are 80), else a `config.json`'s `id2label` (a pose export's lists keypoints,
   * not classes), else from the head's width: COCO-80, or `class{i}`.
   */
  static #classNames(bundle: ModelBundle, session: ModelSession, task: Task, layout: HeadLayout): string[] {
    const parsed = parseNames(session.metadata["names"]);
    if (parsed) {
      if (!isPlaceholder(parsed)) return parsed;
      if (parsed.length === COCO_CLASSES.length) return [...COCO_CLASSES];
      return parsed;
    }
    if (task !== "pose") {
      try {
        const config = JSON.parse(bundle.assetText("config.json") ?? "{}") as { id2label?: Record<string, string> };
        const ids = Object.keys(config.id2label ?? {}).sort((a, b) => Number(a) - Number(b));
        if (ids.length > 0) return ids.map((id) => config.id2label![id]!);
      } catch {
        // Not fatal: carry on.
      }
    }
    if (task === "pose") return ["person"];
    let nc = 0;
    const dims = session.outputs[0]?.dims ?? [];
    if (layout.split) nc = staticDim(dims[2]) ?? 0;
    else if (dims.length === 2) nc = staticDim(dims[1]) ?? 0;
    else if (dims.length === 3 && !layout.nmsFree) nc = Math.max(0, (staticDim(dims[1]) ?? 0) - 4);
    if (nc === COCO_CLASSES.length || nc === 0) return [...COCO_CLASSES];
    return Array.from({ length: nc }, (_, i) => `class${i}`);
  }

  get backend(): string {
    return this.session.backend;
  }

  /** Whether the graph takes class prompts (a second, text-embedding input). */
  get isOpenVocab(): boolean {
    return this.session.inputs.length >= 2;
  }

  /** What the library knows of the loaded model. */
  info(): Record<string, unknown> {
    const published = publishedModel(this.modelName);
    return {
      modelName: this.modelName,
      task: this.task,
      imgsz: this.imgsz,
      dynamic: this.dynamic,
      backend: this.backend,
      classes: this.classNames.length,
      layout: this.layout,
      license: published?.license ?? this.session.metadata["license"] ?? null,
      inputs: this.session.inputs,
      outputs: this.session.outputs,
    };
  }

  /** One image: pixels, or a frame the GPU path takes as it is. */
  async run(image: ImageInput, options: RunOptions = {}): Promise<YOLOResult> {
    return (await this.runBatch([image], options))[0]!;
  }

  /** Several images, in batches the graph accepts. */
  async runBatch(images: readonly ImageInput[], options: RunOptions = {}): Promise<YOLOResult[]> {
    const { textEmbeddings, classes, prompts } = await this.#classesOf(options.classes);
    const step = this.staticBatch ?? Math.max(1, images.length);
    const results: YOLOResult[] = [];
    for (let i = 0; i < images.length; i += step) {
      results.push(...(await this.#inferDecode(images.slice(i, i + step), { ...options, classes }, textEmbeddings, prompts)));
    }
    return results;
  }

  /**
   * SAHI: the image in overlapping tiles, each inferred, merged by a cross-tile NMS (detection
   * only).
   */
  async runSahi(input: ImageInput, options: SAHIRunOptions = {}): Promise<DetectionResult> {
    if (this.task !== "detect") throw new SAHIError(`SAHI serves the detect task (this model: '${this.task}')`);
    const image = framePixels(input);
    const slicer = new SAHISlicer(options);
    const { tiles, metas } = slicer.slice(image);
    const results = (await this.runBatch(tiles, options)) as DetectionResult[];
    const merged = slicer.merge(results, metas, [image.height, image.width]);
    merged.source = options.source ?? "array";
    return merged;
  }

  /** A video or a camera, frame by frame, with tracking and interpolation. */
  video(options: VideoOptions = {}): VideoSession {
    return new VideoSession(this, options);
  }

  async #classesOf(classes: RunOptions["classes"]): Promise<{ textEmbeddings: Float32Array[] | null; classes: number[] | null; prompts: string[] | null }> {
    if (!classes || classes.length === 0) return { textEmbeddings: null, classes: null, prompts: null };
    if (classes.every((c) => typeof c === "string")) {
      if (!this.isOpenVocab) {
        throw new TaskError(
          "text classes need an open-vocabulary model (YOLOE, YOLO-World); this one has no text-embedding input. Use class ids instead.",
        );
      }
      const prompts = classes as string[];
      this.#textEncoder ??= await TextEncoder.load({ ...this.options.textEncoder, ort: this.options.ort, Tensor: this.Tensor });
      return { textEmbeddings: await this.#textEncoder.encodeBatch(prompts), classes: null, prompts };
    }
    return { textEmbeddings: null, classes: (classes as number[]).map(Number), prompts: null };
  }

  async #inferDecode(inputs: readonly ImageInput[], options: DecodeOptions & Pick<RunOptions, "source" | "imgsz" | "rect">, textEmbeddings: Float32Array[] | null, prompts: string[] | null): Promise<YOLOResult[]> {
    const size = options.imgsz ? strideMultiple(options.imgsz) : this.imgsz;
    const rect = options.rect === true && this.task !== "classify" && !this.layout.normalized;
    if (!this.dynamic && (size !== this.imgsz || rect)) {
      throw new TaskError(`${this.modelName}: the graph's input is fixed at ${this.imgsz}; another size or 'rect' needs a dynamic export`);
    }
    const imageInput = this.session.inputs.find((i) => (i.dims?.length ?? 4) === 4) ?? this.session.inputs[0]!;
    if (!textEmbeddings && imageInput.type !== "float16" && inputs.every(isFrameSource) && this.#gpuUsable()) {
      const ran = await this.#runGpu(inputs as FrameSource[], size, rect, imageInput.dims ?? [], options);
      this.lastPath = ran ? (ran.captured ? "gpu-captured" : "gpu") : "cpu";
      if (ran) return this.#label(ran.results ?? this.#decode(ran.outputs, ran.metas, options, size), options, ran.runMs, inputs.length, prompts);
    }
    this.lastPath = "cpu";
    const images = inputs.map(framePixels);
    const { tensor, dims, metas } = preprocess(images, size, { centerCrop: this.task === "classify", rect });
    const half = imageInput.type === "float16";
    const feeds: Record<string, unknown> = { [imageInput.name]: half ? new this.Tensor("float16", toHalf(tensor), dims) : new this.Tensor("float32", tensor, dims) };
    if (textEmbeddings) {
      const textInput = this.session.inputs.find((i) => i !== imageInput)!;
      const dim = textEmbeddings[0]!.length;
      const flat = new Float32Array(textEmbeddings.length * dim);
      textEmbeddings.forEach((row, i) => flat.set(row, i * dim));
      const shape = (textInput.dims?.length ?? 2) === 3 ? [1, textEmbeddings.length, dim] : [textEmbeddings.length, dim];
      feeds[textInput.name] = textInput.type === "float16" ? new this.Tensor("float16", toHalf(flat), shape) : new this.Tensor("float32", flat, shape);
    }
    let raw: Record<string, unknown>;
    try {
      raw = await this.session.run(feeds);
    } catch (error) {
      throw new InferenceError(`${this.modelName}: inference failed: ${(error as Error).message}`, { cause: error });
    }
    return this.#finish(this.#ordered(raw), metas, options, size, this.session.lastRunMs, images.length, prompts);
  }

  #finish(outputs: RawOutput[], metas: LetterboxMeta[], options: DecodeOptions & { source?: string }, size: number, runMs: number, count: number, prompts: string[] | null): YOLOResult[] {
    return this.#label(this.#decode(outputs, metas, options, size), options, runMs, count, prompts);
  }

  /** Timing, source and prompt names on decoded results. */
  #label(results: YOLOResult[], options: { source?: string }, runMs: number, count: number, prompts: string[] | null): YOLOResult[] {
    const perImage = runMs / Math.max(1, count);
    for (const result of results) {
      result.inferenceTimeMs = perImage;
      result.source = options.source ?? "array";

      if (prompts && hasBoxes(result)) {
        for (const box of result.boxes) if (box.classId >= 0 && box.classId < prompts.length) box.className = prompts[box.classId]!;
      }
    }
    return results;
  }

  /**
   * How the last inference ran: `gpu-captured` (letterboxed on the GPU, the run replayed), `gpu`
   * (on the GPU, not capturable), `cpu`.
   */
  lastPath: "gpu-captured" | "gpu" | "cpu" = "cpu";
  #letterbox: GpuLetterbox | null = null;
  #masks: GpuMaskComposer | null = null;
  /** Sessions per input shape (`NxHxW`), the latest three; null where none would build. */
  readonly #gpuSessions = new Map<string, Promise<GpuSession | null>>();
  /** One GPU inference at a time: the input buffer of a shape is reused. */
  #gpuChain: Promise<unknown> = Promise.resolve();
  #gpuFailed = false;

  #device(): GPUDevice | null {
    return (this.options.ort.env.webgpu?.device as GPUDevice | undefined) ?? null;
  }

  #gpuUsable(): boolean {
    return this.options.gpuPipeline !== false && !this.#gpuFailed && this.backend === "webgpu" && this.#device() !== null;
  }

  /**
   * Letterbox on the GPU, run the session of this shape, read the outputs back; null to take the
   * CPU path.
   */
  #runGpu(
    frames: FrameSource[],
    size: number,
    rect: boolean,
    dims: readonly (number | string)[],
    options: DecodeOptions,
  ): Promise<{ outputs: RawOutput[]; metas: LetterboxMeta[]; runMs: number; captured: boolean; results?: YOLOResult[] } | null> {
    type Ran = { read: () => Promise<RawOutput[]>; metas: LetterboxMeta[]; runMs: number; captured: boolean } | { later: () => Promise<YOLOResult[]>; metas: LetterboxMeta[]; runMs: number; captured: boolean };
    const run = async (): Promise<Ran | null> => {
      const centerCrop = this.task === "classify";
      const [height, width] = centerCrop ? [size, size] : inputShape(frames.map((f) => { const [w, h] = frameSize(f); return { width: w, height: h }; }), size, rect);
      const gpu = await this.#gpuSession(frames.length, height, width, dims);
      if (!gpu) return null;
      try {
        const metas = this.#letterbox!.write(frames, gpu.buffer, size, [height, width], centerCrop);
        const input = (this.Tensor as unknown as { fromGpuBuffer(buffer: GPUBuffer, options: { dataType: string; dims: number[] }): unknown }).fromGpuBuffer(gpu.buffer, { dataType: "float32", dims: [frames.length, 3, height, width] });
        const imageName = this.session.inputs.find((i) => (i.dims?.length ?? 4) === 4)?.name ?? this.session.inputs[0]!.name;
        const raw = await gpu.session.run({ [imageName]: input });

        if (this.task === "segment" && !this.layout.split) {
          const later = this.#segmentStage(raw, metas, size, options, !gpu.captured);
          if (later) return { later, metas, runMs: gpu.session.lastRunMs, captured: gpu.captured };
        }

        const read = this.#stage(raw, !gpu.captured);
        return { read, metas, runMs: gpu.session.lastRunMs, captured: gpu.captured };
      } catch (error) {
        this.#gpuFailed = true;
        this.options.logger?.warn(`${this.modelName}: the GPU path failed, frames go through the CPU from now on: ${(error as Error).message}`);
        return null;
      }
    };
    const next = this.#gpuChain.then(run, run);
    this.#gpuChain = next.catch(() => undefined);
    return next.then(async (ran) => {
      if (!ran) return null;
      if ("later" in ran) return { outputs: [], metas: ran.metas, runMs: ran.runMs, captured: ran.captured, results: await ran.later() };
      return { outputs: await ran.read(), metas: ran.metas, runMs: ran.runMs, captured: ran.captured };
    });
  }

  /**
   * The outputs copied into buffers of their own, the copies submitted: the GPU queue is free for
   * the next frame. The returned function maps them (in decoder order). Outputs not on the GPU are
   * read as they are.
   */
  #stage(raw: Record<string, unknown>, release: boolean): () => Promise<RawOutput[]> {
    const device = this.#device()!;
    const names = this.layout.split ? ["logits", "pred_boxes"] : this.session.outputs.map((o) => o.name);
    const encoder = device.createCommandEncoder();
    const staged = names.map((name) => {
      const tensor = raw[name] as { gpuBuffer?: GPUBuffer; getData?(release?: boolean): Promise<ArrayLike<number>>; dims?: readonly number[]; type?: string; dispose?(): void } | undefined;
      if (!tensor?.dims) throw new InferenceError(`${this.modelName}: output '${name}' is missing`);
      if (!tensor.gpuBuffer) return { tensor, copy: null };
      const count = tensor.dims.reduce((a, b) => a * b, 1);
      const bytes = Math.min(tensor.gpuBuffer.size, Math.ceil((count * (tensor.type === "float16" ? 2 : 4)) / 4) * 4);
      const copy = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      encoder.copyBufferToBuffer(tensor.gpuBuffer, 0, copy, 0, bytes);
      return { tensor, copy };
    });
    device.queue.submit([encoder.finish()]);

    if (release) for (const { tensor, copy } of staged) if (copy) tensor.dispose?.();
    return () =>
      Promise.all(
        staged.map(async ({ tensor, copy }) => {
          const dims = tensor.dims!;
          if (!copy) {
            const data = await tensor.getData!(release);
            return { data: tensor.type === "float16" ? fromHalf(data) : data, dims };
          }
          await copy.mapAsync(GPUMapMode.READ);
          const count = dims.reduce((a, b) => a * b, 1);
          const bytes = copy.getMappedRange().slice(0);
          copy.unmap();
          copy.destroy();
          const data = tensor.type === "float16" ? fromHalf(new Uint16Array(bytes, 0, count)) : new Float32Array(bytes, 0, count);
          return { data, dims };
        }),
      );
  }

  #gpuSession(batch: number, height: number, width: number, dims: readonly (number | string)[]): Promise<GpuSession | null> {
    const key = `${batch}x${height}x${width}`;
    let entry = this.#gpuSessions.get(key);
    if (entry) {
      this.#gpuSessions.delete(key);
    } else {
      entry = this.#buildGpuSession(batch, height, width, dims);
      while (this.#gpuSessions.size >= 3) {
        const [oldest, dropped] = this.#gpuSessions.entries().next().value as [string, Promise<GpuSession | null>];
        this.#gpuSessions.delete(oldest);
        void dropped.then((g) => g && closeGpuSession(g));
      }
    }
    this.#gpuSessions.set(key, entry);
    return entry;
  }

  async #buildGpuSession(batch: number, height: number, width: number, dims: readonly (number | string)[]): Promise<GpuSession | null> {
    const device = this.#device();
    if (!device) return null;
    this.#letterbox ??= new GpuLetterbox(device);

    const overrides: Record<string, number> = {};
    [batch, 3, height, width].forEach((value, i) => {
      const dim = dims[i];
      if (typeof dim === "string" && dim) overrides[dim] = value;
    });
    const errors: string[] = [];

    for (const captured of [true, false]) {
      try {
        const session = await this.bundle.variant(this.session.role, {
          ...(Object.keys(overrides).length > 0 ? { freeDimensionOverrides: overrides } : {}),
          preferredOutputLocation: "gpu-buffer",
          ...(captured ? { enableGraphCapture: true } : {}),
        });
        this.options.logger?.info(`${this.modelName}: GPU session for ${batch}×3×${height}×${width}${captured ? ", run captured" : " (not capturable)"}`);
        return { session, buffer: this.#letterbox.inputBuffer(batch, height, width), captured };
      } catch (error) {
        errors.push((error as Error).message);
      }
    }
    this.options.logger?.warn(`${this.modelName}: no GPU session for ${height}×${width}, the CPU path is used: ${errors.join(" / ")}`);
    return null;
  }

  /**
   * Segmentation with the prototypes left on the GPU. Here, in the GPU queue's turn: the head and
   * the prototypes copied aside (the next frame may run). Later, outside it: the head read back,
   * the boxes decoded, the masks composed from the copy (a bit a pixel). Null when the prototypes
   * are not a float32 GPU buffer (the caller reads everything back).
   */
  #segmentStage(raw: Record<string, unknown>, metas: LetterboxMeta[], size: number, options: DecodeOptions, release: boolean): (() => Promise<YOLOResult[]>) | null {
    const [headName, protoName] = this.session.outputs.map((o) => o.name);
    const proto = raw[protoName ?? ""] as { gpuBuffer?: GPUBuffer; dims?: readonly number[]; type?: string; dispose?(): void } | undefined;
    const head = raw[headName ?? ""] as { gpuBuffer?: GPUBuffer; dims?: readonly number[]; type?: string; dispose?(): void } | undefined;
    if (!proto?.gpuBuffer || !proto.dims || proto.type !== "float32" || !head?.gpuBuffer || !head.dims || head.type !== "float32") return null;
    const device = this.#device()!;
    const headCount = head.dims.reduce((a, b) => a * b, 1);
    const headCopy = device.createBuffer({ size: headCount * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const protoCopy = device.createBuffer({ size: proto.gpuBuffer.size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(head.gpuBuffer, 0, headCopy, 0, headCount * 4);
    encoder.copyBufferToBuffer(proto.gpuBuffer, 0, protoCopy, 0, proto.gpuBuffer.size);
    device.queue.submit([encoder.finish()]);
    if (release) {
      head.dispose?.();
      proto.dispose?.();
    }
    const headDims = head.dims;
    const [c, mh, mw] = (proto.dims.length === 4 ? proto.dims.slice(1) : proto.dims) as [number, number, number];
    this.#masks ??= new GpuMaskComposer(device);
    const composer = this.#masks;
    return async () => {
      try {
        await headCopy.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(headCopy.getMappedRange().slice(0));
        headCopy.unmap();
        const decoded = segmentBoxes([{ data, dims: headDims }], metas, this.#context(size), options);
        return await Promise.all(
          decoded.map(async ({ result, kept }, b) => {
            const meta = metas[b]!;
            const masks = await composer.compose(protoCopy, [c, mh, mw], contentRegion(meta, mh, mw), { boxes: result.boxes, coeffs: kept.extra, meta, protoBase: b * c * mh * mw }, options.maskThreshold ?? 0.5);
            return { ...result, masks };
          }),
        );
      } finally {
        headCopy.destroy();
        protoCopy.destroy();
      }
    };
  }

  /** The outputs in the order a decoder reads them (`logits` before `pred_boxes`). */
  #ordered(raw: Record<string, unknown>): RawOutput[] {
    const names = this.layout.split ? ["logits", "pred_boxes"] : this.session.outputs.map((o) => o.name);
    return names.map((name) => {
      const tensor = raw[name] as { data?: ArrayLike<number>; dims?: readonly number[]; type?: string } | undefined;
      if (!tensor?.data || !tensor.dims) throw new InferenceError(`${this.modelName}: output '${name}' is missing`);

      const data = tensor.type === "float16" ? fromHalf(tensor.data) : tensor.data;
      return { data, dims: tensor.dims };
    });
  }

  #context(imgsz: number): DecoderContext {
    const layout = imgsz === this.layout.imgsz ? this.layout : { ...this.layout, imgsz };
    return { classNames: this.classNames, modelName: this.modelName, layout, keypointNames: this.keypointNames };
  }

  #decode(outputs: RawOutput[], metas: LetterboxMeta[], options: DecodeOptions, imgsz: number): YOLOResult[] {
    const context = this.#context(imgsz);
    switch (this.task) {
      case "detect":
        return decodeDetect(outputs, metas, context, options);
      case "segment":
        return decodeSegment(outputs, metas, context, options);
      case "pose":
        return decodePose(outputs, metas, context, options);
      case "classify":
        return decodeClassify(outputs, metas, context, options);
      case "obb":
        return decodeObb(outputs, metas, context, options);
      case "semantic":
        return decodeSemantic(outputs, metas, context);
      case "depth":
        return decodeDepth(outputs, metas, context);
    }
  }

  async close(): Promise<void> {
    await this.#textEncoder?.close();
    this.#textEncoder = null;
    await this.#gpuChain;
    for (const entry of this.#gpuSessions.values()) {
      const gpu = await entry;
      if (gpu) await closeGpuSession(gpu);
    }
    this.#gpuSessions.clear();
    this.#letterbox?.destroy();
    this.#letterbox = null;
    this.#masks?.destroy();
    this.#masks = null;
    await this.bundle.close();
  }
}

/**
 * A video or camera feed, one frame at a time: infer one frame in `frameStep`, track, and fill the
 * frames between by Kalman propagation (or the last result repeated, without a tracker). `frame()`
 * per frame; `result()` the `VideoResult` so far.
 */
export class VideoSession {
  readonly tracker: ObjectTracker | null;
  readonly frameStep: number;
  readonly #interpolate: boolean;
  readonly #startFrame: number;
  readonly #frames: FrameResult[] = [];
  #processed = 0;
  #last: YOLOResult | null = null;
  #lastInferred = -1;
  #busy = false;
  /** Keyframes and the fill between them (`infer`, `offer`, `resultAt`). */
  readonly timeline = new KeyframeTimeline();
  /** Time of the last inference call, in ms (a batch counts once). */
  latencyMs = 0;
  /** The last failure of a background inference started by `offer`. */
  lastError: unknown = null;
  /**
   * The run options of the next inferences (`conf`, `classes`, `imgsz`…): replace them to change
   * them live, the tracks and keyframes kept.
   */
  run: RunOptions;
  /** Holding missed objects (`VideoOptions.persist`); replace it to change it live, null to stop. */
  persist: PersistOptions | null;
  /** The last frame's signature, and what was last seen of each object, for `persist`. */
  #signature: Uint8Array | null = null;
  readonly #seen = new Map<number, SeenObject>();
  /** The last oriented box of each numbered object, to hold it in shape. */
  readonly #seenOriented = new Map<number, OrientedBox>();

  constructor(
    private readonly runner: YOLORunner,
    private readonly options: VideoOptions,
  ) {
    this.frameStep = Math.max(1, Math.trunc(options.frameStep ?? 1));
    this.#interpolate = options.interpolate ?? true;
    this.#startFrame = options.startFrame ?? 0;
    this.run = { ...options.run };
    this.persist = options.persist ?? null;
    this.tracker = options.tracker ? new ObjectTracker(options.tracker, options.trackerConfig ?? {}, options.reid ?? null) : null;
    if (this.tracker && runner.task !== "detect" && runner.task !== "segment" && runner.task !== "pose" && runner.task !== "obb") {
      throw new ConfigurationError(`tracking needs boxes; the '${runner.task}' task has none`);
    }
  }

  /**
   * One frame: inferred, or propagated. `keep: false` leaves it out of `result()` (a live camera).
   */
  async frame(image: ImageInput, frameIdx: number, timestampS: number, keep = true): Promise<FrameResult> {
    const infer = (frameIdx - this.#startFrame) % this.frameStep === 0 || this.#last === null;
    let frame: FrameResult;
    if (infer) {
      let result = await this.runner.run(image, { ...this.run, source: this.options.source ?? "video" });
      result = await this.#track(result, image);
      this.#processed += 1;
      this.#last = result;
      this.#lastInferred = frameIdx;
      frame = { frameIdx, timestampS, result, isInterpolated: false };
    } else if (this.#interpolate && this.tracker && hasBoxes(this.#last!)) {
      frame = { frameIdx, timestampS, result: propagate(this.tracker, frameIdx, this.#lastInferred, this.#last as BoxResult), isInterpolated: true };
    } else {
      frame = { frameIdx, timestampS, result: this.#last!, isInterpolated: true };
    }
    if (keep) this.#frames.push(frame);
    return frame;
  }

  /**
   * Keyframes: these frames inferred — in one batch when the graph takes several images — and
   * tracked in frame order. Every other frame is then answered by `resultAt`.
   */
  async infer(frames: readonly { image: ImageInput; frameIdx: number }[]): Promise<void> {
    if (frames.length === 0) return;
    const ordered = [...frames].sort((a, b) => a.frameIdx - b.frameIdx);
    const started = performance.now();

    const inferred = this.runner.runBatch(ordered.map((f) => f.image), { ...this.run, source: this.options.source ?? "video" });

    inferred.catch(() => undefined);
    const turn = this.#trackTurn.then(async () => {
      const results = await inferred;
      this.latencyMs = performance.now() - started;
      for (const [i, frame] of ordered.entries()) {
        let result = results[i]!;
        result = await this.#track(result, frame.image);
        this.timeline.add(frame.frameIdx, result);
        this.#processed += 1;
      }
    });
    this.#trackTurn = turn.catch(() => undefined);
    await turn;
  }

  #trackTurn: Promise<void> = Promise.resolve();

  /**
   * The result for any frame: a keyframe's own, interpolated between two, or carried on after the
   * last (`options.motion`). Null before the first.
   */
  resultAt(frameIdx: number, motion: MotionOptions = {}): YOLOResult | null {
    return this.timeline.resultAt(frameIdx, { ...this.options.motion, ...motion });
  }

  /**
   * Live use, at the input's own rate: if the model is free, start inferring this frame (without
   * waiting for it); answer at once from what is known. The model runs as often as it can — no
   * fixed step — and every frame still gets a result.
   */
  offer(image: ImageInput, frameIdx: number, motion: MotionOptions = {}): YOLOResult | null {
    if (!this.#busy) {
      this.#busy = true;
      this.infer([{ image, frameIdx }])
        .catch((error: unknown) => {
          this.lastError = error;
        })
        .finally(() => {
          this.#busy = false;
          this.timeline.prune(frameIdx - 120);
        });
    }
    return this.resultAt(frameIdx, motion);
  }

  /**
   * Tracked, missed objects held (`persist`). Oriented boxes go through the boxes around them:
   * numbered, held and re-identified like any box, the numbers then written back on the oriented
   * boxes (a held one keeps its shape and angle, moved with its box).
   */
  async #track(result: YOLOResult, image: ImageInput): Promise<YOLOResult> {
    if (!this.tracker) return result;
    if (hasBoxes(result)) return this.#hold(await this.tracker.update(result as BoxResult, image), image);
    if (result.task !== "obb") return result;
    const view: DetectionResult = { ...result, task: "detect", boxes: result.orientedBoxes.map(boxAround) };
    const tracked = this.#hold(await this.tracker.update(view, image), image);
    const oriented: OrientedBox[] = result.orientedBoxes.map((o, i) => ({ ...o, trackId: tracked.boxes[i]?.trackId ?? null, isInterpolated: false }));
    for (const o of oriented) if (o.trackId !== null && o.trackId !== undefined) this.#seenOriented.set(o.trackId, o);

    for (const box of tracked.boxes.slice(result.orientedBoxes.length)) {
      const last = box.trackId !== null ? this.#seenOriented.get(box.trackId) : undefined;
      if (!last) continue;
      oriented.push({ ...last, cx: (box.x1 + box.x2) / 2, cy: (box.y1 + box.y2) / 2, confidence: box.confidence, isInterpolated: true });
    }
    return { ...result, orientedBoxes: oriented };
  }

  /** The tracked result, with the objects it missed held when the frames follow (`persist`). */
  #hold(result: BoxResult, image: ImageInput): BoxResult {
    rememberSeen(result, this.#seen);
    if (!this.persist || !this.tracker) {
      this.#signature = null;
      return result;
    }
    const signature = frameSignature(image);
    const alike = this.#signature ? likeness(this.#signature, signature) : 0;
    this.#signature = signature;
    const held = holdLost(result, this.tracker, this.#seen, alike, this.persist);

    const known = new Set([...this.tracker.activeTracks().map((t) => t.trackId), ...this.tracker.lostTracks().map((t) => t.trackId)]);
    for (const id of this.#seen.keys()) if (!known.has(id)) this.#seen.delete(id);
    return held;
  }

  /** Whether an inference is running (`offer` skips frames meanwhile). */
  get busy(): boolean {
    return this.#busy;
  }

  result(): VideoResult {
    return {
      source: this.options.source ?? "video",
      fps: this.options.fps ?? 0,
      totalFrames: this.#frames.length,
      processedFrames: this.#processed,
      frameStep: this.frameStep,
      frames: [...this.#frames],
    };
  }

  /** Forget the frames and the tracks (a new clip). */
  reset(): void {
    this.#frames.length = 0;
    this.#processed = 0;
    this.#last = null;
    this.#lastInferred = -1;
    this.timeline.clear();
    this.tracker?.reset();
    this.#signature = null;
    this.#seen.clear();
    this.#seenOriented.clear();
  }
}
