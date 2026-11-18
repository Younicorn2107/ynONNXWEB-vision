/** Documents: their codes (QR, barcodes), their text in reading order, and what kind they are. */

import type { RasterImage } from "../preprocess.js";
import { channelsOf } from "../preprocess.js";
import { groupBlocks, type SceneTextReader, type TextBlock } from "./sceneText.js";

export type Quad = [number, number][];

export interface DocumentCode {
  /** `qr_code`, `ean_13`, `upc_a`… (the `BarcodeDetector` format names). */
  kind: string;
  data: string;
  /** Corners in image pixels. */
  quad: Quad;
}

export interface DocumentResult {
  text: string;
  /** Blocks of text, in reading order. */
  blocks: string[];
  codes: DocumentCode[];
  /** Whether this browser can read codes at all. */
  codesSupported: boolean;
  /** The best label, when labels were given. */
  kind: string | null;
  /** Every label with its score, best first. */
  kindScores: [string, number][];
}

/** A zero-shot text classifier: a score per label. */
export type TextRanker = (text: string, labels: readonly string[]) => Promise<Record<string, number>>;

/** Blocks sorted in reading order: by row (top over line height), then left edge. */
export function readingOrder<T extends Pick<TextBlock, "rect" | "lineHeight">>(blocks: readonly T[]): T[] {
  const row = (b: T) => Math.round(b.rect.y / Math.max(1, b.lineHeight));
  return [...blocks].sort((a, b) => row(a) - row(b) || a.rect.x - b.rect.x);
}

/** The image grey, its levels stretched to 0–255, at `scale` times the size (nearest). */
export function stretchedGrey(image: RasterImage, scale = 2): RasterImage {
  const cn = channelsOf(image);
  const { width, height, data } = image;
  const grey = new Uint8Array(width * height);
  let lo = 255;
  let hi = 0;
  for (let i = 0; i < width * height; i += 1) {
    const g = Math.round(0.299 * data[i * cn]! + 0.587 * data[i * cn + 1]! + 0.114 * data[i * cn + 2]!);
    grey[i] = g;
    if (g < lo) lo = g;
    if (g > hi) hi = g;
  }
  const span = Math.max(1, hi - lo);
  const w = width * scale;
  const h = height * scale;
  const out = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const v = Math.round(((grey[Math.floor(y / scale) * width + Math.floor(x / scale)]! - lo) * 255) / span);
      out.set([v, v, v], (y * w + x) * 3);
    }
  }
  return { data: out, width: w, height: h };
}

interface NativeBarcode {
  rawValue: string;
  format: string;
  cornerPoints: { x: number; y: number }[];
}
type NativeDetector = { detect(image: ImageData): Promise<NativeBarcode[]> };

function nativeDetector(): NativeDetector | null {
  const Detector = (globalThis as { BarcodeDetector?: new () => NativeDetector }).BarcodeDetector;
  return Detector ? new Detector() : null;
}

function toImageData(image: RasterImage): ImageData {
  const cn = channelsOf(image);
  const rgba = new Uint8ClampedArray(image.width * image.height * 4);
  for (let i = 0; i < image.width * image.height; i += 1) {
    rgba[i * 4] = image.data[i * cn]!;
    rgba[i * 4 + 1] = image.data[i * cn + 1]!;
    rgba[i * 4 + 2] = image.data[i * cn + 2]!;
    rgba[i * 4 + 3] = 255;
  }
  return new ImageData(rgba, image.width, image.height);
}

/** Every QR code and barcode of an image; null when the browser cannot read codes. */
export async function readCodes(image: RasterImage, detector: NativeDetector | null = nativeDetector()): Promise<DocumentCode[] | null> {
  if (!detector) return null;
  const found = new Map<string, DocumentCode>();
  for (const [picture, scale] of [
    [image, 1],
    [stretchedGrey(image, 2), 2],
  ] as const) {
    for (const code of await detector.detect(toImageData(picture))) {
      if (!code.rawValue) continue;
      const key = `${code.format}|${code.rawValue}`;
      if (!found.has(key)) found.set(key, { kind: code.format, data: code.rawValue, quad: code.cornerPoints.map((p) => [p.x / scale, p.y / scale]) });
    }
    if (found.size > 0) break;
  }
  return [...found.values()];
}

export interface DocumentReaderOptions {
  /** The kinds a document may be; ranked with `classify`. */
  labels?: readonly string[];
  classify?: TextRanker;
}

export class DocumentReader {
  constructor(
    private readonly ocr: SceneTextReader,
    private readonly options: DocumentReaderOptions = {},
  ) {}

  /** One document image. */
  async read(image: RasterImage): Promise<DocumentResult> {
    const lines = await this.ocr.read(image);
    const blocks = readingOrder(lines.length > 0 ? groupBlocks(lines, image) : []);
    const texts = blocks.map((b) => b.text);
    const text = texts.join("\n");
    const { labels, classify } = this.options;
    let kindScores: [string, number][] = [];
    if (labels && labels.length > 0 && classify && text.trim()) {
      const scores = await classify(text.slice(0, 2000), labels);
      kindScores = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    }
    const codes = await readCodes(image);
    return { text, blocks: texts, codes: codes ?? [], codesSupported: codes !== null, kind: kindScores[0]?.[0] ?? null, kindScores };
  }
}
