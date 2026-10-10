/** Instance masks composed on the GPU, where the prototypes already are. */

import { pyRound } from "./preprocess.js";
import type { BoundingBox, LetterboxMeta, Mask } from "./types.js";

const SHADER =  `
struct Params {
  frame: vec2<u32>,
  grid: vec2<u32>,
  regionOrigin: vec2<u32>,
  region: vec2<u32>,
  channels: u32,
  protoBase: u32,
  words: u32,
  objects: u32,
  tapScale: vec2<f32>,
  threshold: f32,
  stride: u32,
};

@group(0) @binding(0) var<storage, read> proto: array<f32>;
@group(0) @binding(1) var<storage, read> objects: array<f32>;
@group(0) @binding(2) var<storage, read_write> bits: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p: Params;

/** OpenCV's float INTER_LINEAR tap for output index d: (i0, i1, fraction). */
fn tap(d: u32, scale: f32, src: u32) -> vec3<f32> {
  var x = (f32(d) + 0.5) * scale - 0.5;
  var s = floor(x);
  x = x - s;
  if (s < 0.0) { x = 0.0; s = 0.0; }
  if (s >= f32(src) - 1.0) { x = 0.0; s = f32(src) - 1.0; }
  return vec3<f32>(s, min(s + 1.0, f32(src) - 1.0), x);
}

/** The sigmoid of the object's coefficients against the prototypes at region cell (gx, gy). */
fn prob(k: u32, gx: u32, gy: u32) -> f32 {
  let at = min(p.grid.y - 1u, p.regionOrigin.y + gy) * p.grid.x + min(p.grid.x - 1u, p.regionOrigin.x + gx);
  let area = p.grid.x * p.grid.y;
  let base = k * p.stride + 4u;
  var logit = 0.0;
  for (var j = 0u; j < p.channels; j = j + 1u) {
    logit = logit + objects[base + j] * proto[p.protoBase + j * area + at];
  }
  return 1.0 / (1.0 + exp(-logit));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = id.z;
  if (k >= p.objects) { return; }
  let box = k * p.stride;
  let x = u32(objects[box]) + id.x;
  let y = u32(objects[box + 1u]) + id.y;
  if (x >= u32(objects[box + 2u]) || y >= u32(objects[box + 3u])) { return; }
  let tx = tap(x, p.tapScale.x, p.region.x);
  let ty = tap(y, p.tapScale.y, p.region.y);
  let x0 = u32(tx.x);
  let x1 = u32(tx.y);
  let y0 = u32(ty.x);
  let y1 = u32(ty.y);
  // Vertical first, then horizontal: the CPU path's order.
  let left = prob(k, x0, y0) * (1.0 - ty.z) + prob(k, x0, y1) * ty.z;
  let right = prob(k, x1, y0) * (1.0 - ty.z) + prob(k, x1, y1) * ty.z;
  if (left * (1.0 - tx.z) + right * tx.z >= p.threshold) {
    let i = y * p.frame.x + x;
    atomicOr(&bits[k * p.words + i / 32u], 1u << (i % 32u));
  }
}
`;

/**
 * What `composeMasks` reads of one image: its kept boxes (frame pixels, clamped) and coefficients.
 */
export interface PendingMasks {
  boxes: BoundingBox[];
  coeffs: Float32Array[];
  meta: LetterboxMeta;
  /** Where its prototypes start in the batch's prototype buffer, in floats. */
  protoBase: number;
}

export class GpuMaskComposer {
  readonly #pipeline: GPUComputePipeline;
  readonly #uniform: GPUBuffer;

  constructor(readonly device: GPUDevice) {
    this.#pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: "main" },
    });
    this.#uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /**
   * One image's masks, full frame, from the prototypes `[c, mh, mw]` in `proto` at
   * `pending.protoBase`. `region` is the content region on the prototype grid (the letterbox's
   * padding left out), as `decodeSegment` computes it.
   */
  async compose(
    proto: GPUBuffer,
    [c, mh, mw]: readonly [number, number, number],
    region: { top: number; left: number; width: number; height: number },
    pending: PendingMasks,
    threshold: number,
  ): Promise<Mask[]> {
    const n = pending.boxes.length;
    if (n === 0) return [];
    const [h0, w0] = pending.meta.originalShape;
    const words = Math.ceil((w0 * h0) / 32);
    const stride = 4 + c;
    const objects = new Float32Array(n * stride);
    const corners: [number, number, number, number][] = pending.boxes.map((box) => {
      const x1 = Math.max(0, pyRound(box.x1));
      const y1 = Math.max(0, pyRound(box.y1));
      return [x1, y1, Math.min(w0, Math.max(x1 + 1, pyRound(box.x2))), Math.min(h0, Math.max(y1 + 1, pyRound(box.y2)))];
    });
    corners.forEach((corner, k) => {
      objects.set(corner, k * stride);
      objects.set(pending.coeffs[k]!.subarray(0, c), k * stride + 4);
    });
    const params = new ArrayBuffer(64);
    new Uint32Array(params, 0, 12).set([w0, h0, mw, mh, region.left, region.top, region.width, region.height, c, pending.protoBase, words, n]);
    new Float32Array(params, 48, 3).set([region.width / w0, region.height / h0, threshold]);
    new Uint32Array(params, 60, 1).set([stride]);
    this.device.queue.writeBuffer(this.#uniform, 0, params);
    const objectBuffer = this.device.createBuffer({ size: objects.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(objectBuffer, 0, objects);
    const bitBuffer = this.device.createBuffer({ size: n * words * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const readBuffer = this.device.createBuffer({ size: n * words * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.clearBuffer(bitBuffer);
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.#pipeline);
      pass.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: this.#pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: proto } },
            { binding: 1, resource: { buffer: objectBuffer } },
            { binding: 2, resource: { buffer: bitBuffer } },
            { binding: 3, resource: { buffer: this.#uniform } },
          ],
        }),
      );

      const spanW = Math.max(...corners.map(([x1, , x2]) => x2 - x1));
      const spanH = Math.max(...corners.map(([, y1, , y2]) => y2 - y1));
      pass.dispatchWorkgroups(Math.ceil(spanW / 16), Math.ceil(spanH / 16), n);
      pass.end();
      encoder.copyBufferToBuffer(bitBuffer, 0, readBuffer, 0, n * words * 4);
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const bits = new Uint32Array(readBuffer.getMappedRange().slice(0));
      readBuffer.unmap();
      return corners.map(([x1, y1, x2, y2], k) => {
        const data = new Uint8Array(w0 * h0);
        const base = k * words;
        for (let y = y1; y < y2; y += 1) {
          for (let x = x1; x < x2; x += 1) {
            const i = y * w0 + x;
            if ((bits[base + (i >>> 5)]! >>> (i & 31)) & 1) data[i] = 1;
          }
        }
        return { data, width: w0, height: h0, box: pending.boxes[k]! };
      });
    } finally {
      objectBuffer.destroy();
      bitBuffer.destroy();
      readBuffer.destroy();
    }
  }

  destroy(): void {
    this.#uniform.destroy();
  }
}
