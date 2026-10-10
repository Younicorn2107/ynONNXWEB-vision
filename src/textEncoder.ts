/**
 * CLIP text encoder for open-vocabulary models (YOLOE, YOLO-World): free-text class prompts →
 * normalized embeddings, the second input of those graphs.
 */

import { huggingFaceResolver, loadModel, type LoadOptions, type ModelBundle, type ModelSession } from "@youni/onnxweb-core";
import { Tokenizer, type TensorFactory } from "@youni/onnxweb-core/core";

import { TextEncoderError } from "./errors.js";

/** A repository shipping a CLIP ViT-B/32 text encoder as ONNX, with its tokenizer. */
export const DEFAULT_TEXT_ENCODER = "Qdrant/clip-ViT-B-32-text";
export const CLIP_CONTEXT_LENGTH = 77;

export interface TextEncoderOptions extends Omit<LoadOptions, "model" | "resolver"> {
  Tensor: TensorFactory;
  /** A Hub repository with `model.onnx` and `tokenizer.json` at its root. */
  model?: string;
  resolver?: LoadOptions["resolver"];
  endpoint?: string;
}

export class TextEncoder {
  readonly #cache = new Map<string, Float32Array>();

  private constructor(
    readonly model: string,
    private readonly bundle: ModelBundle,
    private readonly session: ModelSession,
    private readonly tokenizer: Tokenizer,
    private readonly Tensor: TensorFactory,
  ) {}

  static async load(options: TextEncoderOptions): Promise<TextEncoder> {
    const { Tensor, model = DEFAULT_TEXT_ENCODER, endpoint, resolver, ...rest } = options;
    let bundle: ModelBundle;
    try {
      bundle = await loadModel({
        extraFiles: ["tokenizer.json"],
        ...rest,
        model: { name: model, computeType: rest.computeType as string | undefined ?? "float32" },
        resolver: resolver ?? huggingFaceResolver({ repo: model, manifest: false, ...(endpoint ? { endpoint } : {}) }),
      });
    } catch (error) {
      throw new TextEncoderError(`cannot load the CLIP text encoder ${model}: ${(error as Error).message}`, { cause: error });
    }
    const json = bundle.assetText("tokenizer.json");
    if (!json) {
      await bundle.close();
      throw new TextEncoderError(`${model} has no tokenizer.json`);
    }
    return new TextEncoder(model, bundle, bundle.only, Tokenizer.fromJson(json), Tensor);
  }

  /** One prompt → `[dim]` (512 for ViT-B/32). */
  async encode(text: string): Promise<Float32Array> {
    return (await this.encodeBatch([text]))[0]!;
  }

  /** Several prompts → one normalized row each; a prompt already seen is not encoded again. */
  async encodeBatch(texts: readonly string[]): Promise<Float32Array[]> {
    if (texts.length === 0) throw new TextEncoderError("encodeBatch() needs at least one text");
    const missing = [...new Set(texts.filter((t) => !this.#cache.has(t)))];
    if (missing.length > 0) {
      const rows = await this.#encode(missing);
      missing.forEach((t, i) => this.#cache.set(t, rows[i]!));
    }
    return texts.map((t) => this.#cache.get(t)!);
  }

  clearCache(): void {
    this.#cache.clear();
  }

  async #encode(texts: string[]): Promise<Float32Array[]> {
    const n = texts.length;
    const ids = new Array<number>(n * CLIP_CONTEXT_LENGTH).fill(0);
    texts.forEach((text, row) => {
      let tokens: number[];
      try {
        tokens = this.tokenizer.encode(text).slice(0, CLIP_CONTEXT_LENGTH);
      } catch (error) {
        throw new TextEncoderError(`tokenization failed for ${JSON.stringify(text)}`, { cause: error });
      }
      tokens.forEach((id, j) => (ids[row * CLIP_CONTEXT_LENGTH + j] = id));
    });
    const dims = [n, CLIP_CONTEXT_LENGTH];
    const typed = (type: string | undefined, values: number[]) =>
      type === "int32" ? new this.Tensor("int32", Int32Array.from(values), dims) : new this.Tensor("int64", BigInt64Array.from(values, BigInt), dims);
    const [primary, ...others] = this.session.inputs;
    const feeds: Record<string, unknown> = { [primary!.name]: typed(primary!.type, ids) };

    for (const spec of others) feeds[spec.name] = typed(spec.type, ids.map((id) => (id !== 0 ? 1 : 0)));
    let output: { data: Float32Array; dims: readonly number[] };
    try {
      output = (await this.session.run(feeds))[this.session.outputs[0]!.name] as typeof output;
    } catch (error) {
      throw new TextEncoderError(`text-encoder inference failed: ${(error as Error).message}`, { cause: error });
    }
    const dim = output.data.length / n;
    return texts.map((_, i) => {
      const row = Float32Array.from(output.data.subarray(i * dim, (i + 1) * dim));
      const norm = Math.sqrt(row.reduce((s, v) => s + v * v, 0)) || 1;
      return row.map((v) => v / norm);
    });
  }

  async close(): Promise<void> {
    await this.bundle.close();
  }
}
