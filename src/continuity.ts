/**
 * Against flicker: an object the model misses for a frame or two, in a shot that has not changed,
 * is held on screen instead of blinking out.
 */

import { frameSize, isFrameSource, type ImageInput } from "./gpu.js";
import { geometryOf, movePoints, withGeometry } from "./motion.js";
import { channelsOf } from "./preprocess.js";
import type { ObjectTracker } from "./tracker.js";
import type { BoundingBox, BoxResult, KeypointSet, Mask } from "./types.js";

export interface PersistOptions {
  /**
   * How alike two frames must be (0–1) for an object missing from the second to be held: 0.95 by
   * default (consecutive frames of a shot measure 0.98–1, a cut 0.9 or less); below, the shot
   * changed and nothing is held.
   */
  continuity?: number;
  /** Only objects last seen at or above this confidence are held (0.5). */
  minConfidence?: number;
  /** For how many inferences at most an object is held (10). */
  maxFrames?: number;
  /** `predict`: moved along its track (default); `hold`: where it was last seen. */
  mode?: "predict" | "hold";
}

const SIGNATURE_W = 32;
const SIGNATURE_H = 18;

/** The frame as 32 × 18 cells of average gray, to tell whether two frames follow. */
export function frameSignature(image: ImageInput): Uint8Array {
  if (isFrameSource(image)) {
    const canvas = new OffscreenCanvas(SIGNATURE_W, SIGNATURE_H);
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

    ctx.imageSmoothingQuality = "medium";
    const [w, h] = frameSize(image);
    ctx.drawImage(image as CanvasImageSource, 0, 0, w, h, 0, 0, SIGNATURE_W, SIGNATURE_H);
    return gray(ctx.getImageData(0, 0, SIGNATURE_W, SIGNATURE_H).data, 4);
  }

  const cn = channelsOf(image);
  const out = new Uint8Array(SIGNATURE_W * SIGNATURE_H);
  const steps = 4;
  for (let cy = 0; cy < SIGNATURE_H; cy += 1) {
    for (let cx = 0; cx < SIGNATURE_W; cx += 1) {
      let sum = 0;
      for (let sy = 0; sy < steps; sy += 1) {
        for (let sx = 0; sx < steps; sx += 1) {
          const x = Math.min(image.width - 1, Math.floor(((cx + (sx + 0.5) / steps) * image.width) / SIGNATURE_W));
          const y = Math.min(image.height - 1, Math.floor(((cy + (sy + 0.5) / steps) * image.height) / SIGNATURE_H));
          const i = (y * image.width + x) * cn;
          sum += 0.299 * image.data[i]! + 0.587 * image.data[i + 1]! + 0.114 * image.data[i + 2]!;
        }
      }
      out[cy * SIGNATURE_W + cx] = Math.round(sum / (steps * steps));
    }
  }
  return out;
}

function gray(rgba: ArrayLike<number>, cn: number): Uint8Array {
  const out = new Uint8Array(rgba.length / cn);
  for (let i = 0; i < out.length; i += 1) out[i] = Math.round(0.299 * rgba[cn * i]! + 0.587 * rgba[cn * i + 1]! + 0.114 * rgba[cn * i + 2]!);
  return out;
}

/**
 * How alike two signatures are, 0–1: one minus the mean gray difference of their cells, after the
 * overall brightness is matched (an exposure change is not a cut).
 */
export function likeness(a: Uint8Array, b: Uint8Array): number {
  let meanA = 0;
  let meanB = 0;
  for (let i = 0; i < a.length; i += 1) {
    meanA += a[i]!;
    meanB += b[i]!;
  }
  const shift = (meanB - meanA) / a.length;
  let diff = 0;
  for (let i = 0; i < a.length; i += 1) diff += Math.abs(b[i]! - a[i]! - shift);
  return Math.max(0, 1 - diff / a.length / 64);
}

/** What was last seen of each numbered object: its box, and its mask or keypoints. */
export interface SeenObject {
  box: BoundingBox;
  mask?: Mask;
  keypoints?: KeypointSet;
}

/** Remember the numbered objects of `result` (observed ones only). */
export function rememberSeen(result: BoxResult, seen: Map<number, SeenObject>): void {
  result.boxes.forEach((box, i) => {
    if (box.trackId === null || box.isInterpolated) return;
    const entry: SeenObject = { box };
    if (result.task === "segment" && result.masks[i]) entry.mask = result.masks[i];
    if (result.task === "pose" && result.keypoints[i]) entry.keypoints = result.keypoints[i];
    seen.set(box.trackId, entry);
  });
}

/**
 * `result` with the tracker's lost objects put back, when the frames follow (`frameLikeness ≥
 * continuity`) — each where its track says, with its last mask or keypoints moved along.
 */
export function holdLost<R extends BoxResult>(result: R, tracker: ObjectTracker, seen: ReadonlyMap<number, SeenObject>, frameLikeness: number, options: PersistOptions = {}): R {
  const continuity = options.continuity ?? 0.95;
  if (frameLikeness < continuity) return result;
  const minConfidence = options.minConfidence ?? 0.5;
  const maxFrames = options.maxFrames ?? 10;
  const present = new Set(result.boxes.map((b) => b.trackId));
  const boxes = [...result.boxes];
  const masks = result.task === "segment" ? [...result.masks] : null;
  const keypoints = result.task === "pose" ? [...result.keypoints] : null;
  for (const lost of tracker.lostTracks()) {
    if (present.has(lost.trackId) || lost.score < minConfidence || lost.timeSinceUpdate > maxFrames) continue;
    const last = seen.get(lost.trackId);
    if (!last) continue;
    const [x1, y1, x2, y2] = options.mode === "hold" ? lost.lastSeen : lost.predicted;
    const box = withGeometry({ ...last.box, trackId: lost.trackId }, geometryOf({ ...last.box, x1, y1, x2, y2 }), lost.score, result.imageShape);
    if (!box) continue;

    if (masks) {
      if (!last.mask) continue;
      masks.push({ ...last.mask, box });
    }
    if (keypoints) {
      if (!last.keypoints) continue;
      keypoints.push({ box, points: movePoints(last.keypoints.points, geometryOf(last.keypoints.box), geometryOf(box)) });
    }
    boxes.push(box);
  }
  if (boxes.length === result.boxes.length) return result;
  if (masks) return { ...result, boxes, masks };
  if (keypoints) return { ...result, boxes, keypoints };
  return { ...result, boxes };
}
