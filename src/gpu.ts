/** Frames to the model without leaving the GPU. */

import { PAD_COLOR, pyRound, type RasterImage } from "./preprocess.js";
import type { LetterboxMeta } from "./types.js";

/** What the GPU path takes: anything `copyExternalImageToTexture` reads. */
export type FrameSource = ImageBitmap | VideoFrame | OffscreenCanvas | HTMLCanvasElement | HTMLVideoElement | HTMLImageElement;

/** Pixels (`RasterImage`, `ImageData`) or a frame the GPU path can take as it is. */
export type ImageInput = RasterImage | FrameSource;

export function isFrameSource(image: ImageInput): image is FrameSource {
  return !("data" in image && "width" in image && "height" in image);
}

/** `[width, height]` of a frame. */
export function frameSize(source: FrameSource): [number, number] {
  if (typeof HTMLVideoElement !== "undefined" && source instanceof HTMLVideoElement) return [source.videoWidth, source.videoHeight];
  if (typeof HTMLImageElement !== "undefined" && source instanceof HTMLImageElement) return [source.naturalWidth, source.naturalHeight];
  if (typeof VideoFrame !== "undefined" && source instanceof VideoFrame) return [source.displayWidth, source.displayHeight];
  return [(source as { width: number }).width, (source as { height: number }).height];
}

/** The pixels of a frame (the CPU path, and what BoT-SORT's motion compensation reads). */
export function framePixels(source: ImageInput): RasterImage {
  if (!isFrameSource(source)) return source;
  const [width, height] = frameSize(source);
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(source as CanvasImageSource, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}

const SHADER =  `
struct Params {
  src: vec2<f32>,      // source size
  scale: vec2<f32>,    // input pixels per source pixel, per axis
  offset: vec2<f32>,   // where the source's corner lands in the input (negative: cropped)
  size: vec2<u32>,     // input width, height
  base: u32,           // first element of this image in the batch
  pad: f32,            // the letterbox gray, in [0, 1]
};
@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var linear: sampler;
@group(0) @binding(2) var<storage, read_write> input: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.size.x || id.y >= p.size.y) { return; }
  // The source point under this input pixel's centre.
  let at = (vec2<f32>(id.xy) + 0.5 - p.offset) / p.scale;
  var rgb = vec3<f32>(p.pad);
  if (all(at >= vec2<f32>(0.0)) && all(at < p.src)) {
    // Bilinear, then 8 bits, as OpenCV's INTER_LINEAR hands back.
    rgb = round(textureSampleLevel(frame, linear, at / p.src, 0.0).rgb * 255.0) / 255.0;
  }
  let plane = p.size.x * p.size.y;
  let i = p.base + id.y * p.size.x + id.x;
  input[i] = rgb.r;
  input[i + plane] = rgb.g;
  input[i + 2u * plane] = rgb.b;
}
`;

/**
 * Where each frame goes in the input: the letterbox of `preprocess`, or the classifier's centre
 * crop.
 */
export function placement(width: number, height: number, imgsz: number, shape: [number, number], centerCrop: boolean): { meta: LetterboxMeta; scale: [number, number]; offset: [number, number] } {
  if (centerCrop) {
    const scale = imgsz / Math.min(height, width);
    const rw = Math.max(imgsz, pyRound(width * scale));
    const rh = Math.max(imgsz, pyRound(height * scale));
    const top = Math.floor((rh - imgsz) / 2);
    const left = Math.floor((rw - imgsz) / 2);
    return { meta: { originalShape: [height, width], scale, padTop: -top || 0, padLeft: -left || 0 }, scale: [rw / width, rh / height], offset: [-left, -top] };
  }
  const scale = Math.min(imgsz / height, imgsz / width);
  const nw = Math.max(1, pyRound(width * scale));
  const nh = Math.max(1, pyRound(height * scale));
  const padTop = Math.floor((shape[0] - nh) / 2);
  const padLeft = Math.floor((shape[1] - nw) / 2);
  return { meta: { originalShape: [height, width], scale, padTop, padLeft }, scale: [nw / width, nh / height], offset: [padLeft, padTop] };
}

/** The letterbox as a compute shader, writing into a GPU buffer the session reads. */
export class GpuLetterbox {
  readonly #pipeline: GPUComputePipeline;
  readonly #sampler: GPUSampler;
  #texture: GPUTexture | null = null;
  readonly #uniform: GPUBuffer;

  constructor(readonly device: GPUDevice) {
    this.#pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: "main" },
    });
    this.#sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
    this.#uniform = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** A buffer for a `[batch, 3, height, width]` float32 input. */
  inputBuffer(batch: number, height: number, width: number): GPUBuffer {
    return this.device.createBuffer({ size: batch * 3 * height * width * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  }

  /** Letterbox `frames` into `target` (`[N, 3, shape]`); where each one went. */
  write(frames: readonly FrameSource[], target: GPUBuffer, imgsz: number, shape: [number, number], centerCrop = false): LetterboxMeta[] {
    const [height, width] = shape;
    return frames.map((frame, b) => {
      const [w, h] = frameSize(frame);
      const { meta, scale, offset } = placement(w, h, imgsz, shape, centerCrop);
      const texture = this.#textureFor(w, h);
      this.device.queue.copyExternalImageToTexture({ source: frame }, { texture }, [w, h]);
      const params = new ArrayBuffer(48);
      new Float32Array(params, 0, 6).set([w, h, scale[0], scale[1], offset[0], offset[1]]);
      new Uint32Array(params, 24, 3).set([width, height, b * 3 * height * width]);
      new Float32Array(params, 36, 1).set([PAD_COLOR / 255]);
      this.device.queue.writeBuffer(this.#uniform, 0, params);
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.#pipeline);
      pass.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: this.#pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: texture.createView() },
            { binding: 1, resource: this.#sampler },
            { binding: 2, resource: { buffer: target } },
            { binding: 3, resource: { buffer: this.#uniform } },
          ],
        }),
      );
      pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
      pass.end();

      this.device.queue.submit([encoder.finish()]);
      return meta;
    });
  }

  #textureFor(width: number, height: number): GPUTexture {
    if (this.#texture && this.#texture.width === width && this.#texture.height === height) return this.#texture;
    this.#texture?.destroy();
    this.#texture = this.device.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    return this.#texture;
  }

  destroy(): void {
    this.#texture?.destroy();
    this.#uniform.destroy();
  }
}
