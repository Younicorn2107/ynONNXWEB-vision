/** Pixels from what a page has: an `<img>`, a canvas, a bitmap, a file, a `<video>`, the camera. */

import { SourceError } from "@youni/onnxweb-core/core";

import type { RasterImage } from "./preprocess.js";

export type ImageSource =
  | RasterImage
  | ImageData
  | ImageBitmap
  | HTMLImageElement
  | HTMLCanvasElement
  | OffscreenCanvas
  | HTMLVideoElement
  | VideoFrame
  | Blob;

function sizeOf(source: Exclude<ImageSource, RasterImage | ImageData | Blob>): [number, number] {
  if (typeof HTMLVideoElement !== "undefined" && source instanceof HTMLVideoElement) return [source.videoWidth, source.videoHeight];
  if (typeof HTMLImageElement !== "undefined" && source instanceof HTMLImageElement) return [source.naturalWidth, source.naturalHeight];
  if (typeof VideoFrame !== "undefined" && source instanceof VideoFrame) return [source.displayWidth, source.displayHeight];
  return [(source as { width: number }).width, (source as { height: number }).height];
}

/** Any image source as `ImageData` at its natural size. */
export async function readImage(source: ImageSource): Promise<RasterImage> {
  if (typeof ImageData !== "undefined" && source instanceof ImageData) return source;
  if ("data" in source && "width" in source && "height" in source && !(typeof Blob !== "undefined" && source instanceof Blob)) return source as RasterImage;
  let drawable: Exclude<ImageSource, RasterImage | ImageData | Blob>;
  let owned: ImageBitmap | null = null;
  if (typeof Blob !== "undefined" && source instanceof Blob) {
    try {
      owned = await createImageBitmap(source);
    } catch (error) {
      throw new SourceError(`cannot decode the image (${source.type || "unknown type"})`, { cause: error });
    }
    drawable = owned;
  } else {
    drawable = source as Exclude<ImageSource, RasterImage | ImageData | Blob>;
  }
  const [width, height] = sizeOf(drawable);
  if (!width || !height) throw new SourceError("the image has no pixels yet (not loaded, or a video with no frame)");
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new SourceError("no 2D canvas context");
  ctx.drawImage(drawable as CanvasImageSource, 0, 0, width, height);
  owned?.close();
  return ctx.getImageData(0, 0, width, height);
}

/** An image from a URL (same origin, or served with CORS). */
export async function loadImage(url: string, init?: RequestInit): Promise<RasterImage> {
  const response = await fetch(url, init);
  if (!response.ok) throw new SourceError(`cannot fetch ${url}: HTTP ${response.status}`);
  return readImage(await response.blob());
}

export interface Frame {
  image: RasterImage;
  frameIdx: number;
  timestampS: number;
}

/** The camera, playing in a (possibly detached) `<video>`. Stop it with `stopCamera`. */
export async function openCamera(constraints: MediaStreamConstraints["video"] = { facingMode: "environment" }, video?: HTMLVideoElement): Promise<HTMLVideoElement> {
  if (!navigator.mediaDevices?.getUserMedia) throw new SourceError("no camera access here (getUserMedia needs a secure context)");
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: constraints, audio: false });
  } catch (error) {
    throw new SourceError(`the camera could not be opened: ${(error as Error).message}`, { cause: error });
  }
  const element = video ?? document.createElement("video");
  element.muted = true;
  element.playsInline = true;
  element.srcObject = stream;
  await element.play();
  return element;
}

export function stopCamera(video: HTMLVideoElement): void {
  const stream = video.srcObject as MediaStream | null;
  stream?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

/**
 * The frames a playing `<video>` (a camera, a stream) paints, as they come. A frame painted while
 * the consumer is still busy with the previous one is skipped: the next one handed over is always
 * the latest.
 */
export async function* liveFrames(video: HTMLVideoElement, options: { signal?: AbortSignal; maxFrames?: number } = {}): AsyncGenerator<Frame> {
  let frameIdx = 0;
  const start = performance.now();
  while (!options.signal?.aborted && (options.maxFrames === undefined || frameIdx < options.maxFrames)) {
    const mediaTime = await new Promise<number | null>((resolve) => {
      if ("requestVideoFrameCallback" in video) {
        video.requestVideoFrameCallback((_now, meta) => resolve(meta.mediaTime));
      } else {
        requestAnimationFrame(() => resolve(null));
      }
      options.signal?.addEventListener("abort", () => resolve(null), { once: true });
    });
    if (options.signal?.aborted) return;
    if (video.readyState < 2) continue;
    yield { image: await readImage(video), frameIdx, timestampS: mediaTime ?? (performance.now() - start) / 1000 };
    frameIdx += 1;
  }
}

/**
 * Every frame of a video file, by seeking at `fps` (the file's own rate is not exposed to a page:
 * 30 by default). For a whole clip, not a live feed.
 */
export async function* videoFrames(
  video: HTMLVideoElement,
  options: { fps?: number; startFrame?: number; maxFrames?: number; signal?: AbortSignal } = {},
): AsyncGenerator<Frame> {
  if (video.readyState < 1) {
    await new Promise<void>((resolve, reject) => {
      video.addEventListener("loadedmetadata", () => resolve(), { once: true });
      video.addEventListener("error", () => reject(new SourceError("the video could not be loaded")), { once: true });
    });
  }
  const fps = options.fps ?? 30;
  const total = Math.floor(video.duration * fps);
  const last = options.maxFrames !== undefined ? Math.min(total, (options.startFrame ?? 0) + options.maxFrames) : total;
  video.pause();
  for (let frameIdx = options.startFrame ?? 0; frameIdx < last; frameIdx += 1) {
    if (options.signal?.aborted) return;
    const timestampS = frameIdx / fps;
    await new Promise<void>((resolve) => {
      video.addEventListener("seeked", () => resolve(), { once: true });
      video.currentTime = timestampS;
    });
    yield { image: await readImage(video), frameIdx, timestampS };
  }
}
