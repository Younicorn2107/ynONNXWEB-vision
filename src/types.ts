/** Results of every YOLO task, as plain objects. */

export type Task = "detect" | "segment" | "semantic" | "depth" | "pose" | "classify" | "obb";

export const TASKS: readonly Task[] = ["detect", "segment", "semantic", "depth", "pose", "classify", "obb"];

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  classId: number;
  className: string;
  /** Set when tracking is active; null for a box no track took. */
  trackId: number | null;
  /** True when the box was propagated by the Kalman filter, not inferred. */
  isInterpolated: boolean;
}

/** A binary instance mask over the original image: 1 inside, row-major. */
export interface Mask {
  data: Uint8Array;
  width: number;
  height: number;
  box: BoundingBox;
}

export interface Keypoint {
  x: number;
  y: number;
  confidence: number;
  /** `left_knee`…; null past the names the model declares. */
  name: string | null;
}

export interface KeypointSet {
  points: Keypoint[];
  box: BoundingBox;
}

export interface ClassPrediction {
  classId: number;
  className: string;
  confidence: number;
}

export interface OrientedBox {
  cx: number;
  cy: number;
  w: number;
  h: number;
  /** Radians. */
  angle: number;
  confidence: number;
  classId: number;
  className: string;
  /** Set when tracking is active (by the box around it); null for one no track took. */
  trackId?: number | null;
  /** True when the box was held while the model missed it, not inferred. */
  isInterpolated?: boolean;
}

interface BaseResult {
  modelName: string;
  /** Where the image came from (`array`, `camera`, a file or URL name), when known. */
  source: string | null;
  inferenceTimeMs: number;
  /** `[height, width]` of the original image. */
  imageShape: [number, number];
}

export interface DetectionResult extends BaseResult {
  task: "detect";
  boxes: BoundingBox[];
}

export interface SegmentationResult extends BaseResult {
  task: "segment";
  boxes: BoundingBox[];
  masks: Mask[];
}

export interface PoseResult extends BaseResult {
  task: "pose";
  boxes: BoundingBox[];
  keypoints: KeypointSet[];
}

export interface ClassificationResult extends BaseResult {
  task: "classify";
  topK: ClassPrediction[];
}

export interface OBBResult extends BaseResult {
  task: "obb";
  orientedBoxes: OrientedBox[];
}

export interface SemanticResult extends BaseResult {
  task: "semantic";
  /** One class id per pixel of the original image, row-major. */
  mask: Int32Array;
  classNames: string[];
}

/**
 * Depth estimation: one depth per pixel of the original image (the model's unit, metres for
 * YOLO26-depth).
 */
export interface DepthResult extends BaseResult {
  task: "depth";
  /** Row-major, `imageShape[0] × imageShape[1]`. */
  depth: Float32Array;
  min: number;
  max: number;
}

export type YOLOResult =
  | DepthResult
  | DetectionResult
  | SegmentationResult
  | PoseResult
  | ClassificationResult
  | OBBResult
  | SemanticResult;

/** Results that carry boxes: what tracking, SAHI and the box exports work on. */
export type BoxResult = DetectionResult | SegmentationResult | PoseResult;

export function hasBoxes(result: YOLOResult): result is BoxResult {
  return "boxes" in result;
}

/** Where a letterboxed (or centre-cropped) image sits in the model input. */
export interface LetterboxMeta {
  /** `[height, width]` of the original image. */
  originalShape: [number, number];
  scale: number;
  padTop: number;
  padLeft: number;
}

export interface FrameResult {
  frameIdx: number;
  timestampS: number;
  result: YOLOResult;
  /** True when the frame was not inferred (Kalman propagation or the last result repeated). */
  isInterpolated: boolean;
}

export interface VideoResult {
  source: string;
  fps: number;
  totalFrames: number;
  /** Frames actually inferred. */
  processedFrames: number;
  frameStep: number;
  frames: FrameResult[];
}

export interface TileMeta {
  xOffset: number;
  yOffset: number;
  width: number;
  height: number;
}

export interface Track {
  trackId: number;
  classId: number;
  className: string;
  lastBox: BoundingBox;
  /** Frames since creation. */
  age: number;
  /** Times matched. */
  hits: number;
  /** Frames since the last match. */
  timeSinceUpdate: number;
}

const round = (value: number, digits: number) => {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
};

/**
 * One row per box, prediction or oriented box — Python's `to_records()`, with its field names, for
 * CSV and tabular exports. A semantic map has no rows.
 */
export function recordsOf(result: YOLOResult | VideoResult): Record<string, unknown>[] {
  if ("frames" in result) {
    return result.frames.flatMap((frame) =>
      recordsOf(frame.result).map((row) => ({ ...row, frame_idx: frame.frameIdx, timestamp_s: round(frame.timestampS, 3) })),
    );
  }
  switch (result.task) {
    case "detect":
    case "segment":
    case "pose":
      return result.boxes.map((b) => ({
        source: result.source,
        class_id: b.classId,
        class_name: b.className,
        confidence: round(b.confidence, 4),
        x1: round(b.x1, 2),
        y1: round(b.y1, 2),
        x2: round(b.x2, 2),
        y2: round(b.y2, 2),
        track_id: b.trackId,
        is_interpolated: b.isInterpolated,
      }));
    case "classify":
      return result.topK.map((p, i) => ({
        source: result.source,
        rank: i + 1,
        class_id: p.classId,
        class_name: p.className,
        confidence: round(p.confidence, 4),
      }));
    case "obb":
      return result.orientedBoxes.map((b) => ({
        source: result.source,
        class_id: b.classId,
        class_name: b.className,
        confidence: round(b.confidence, 4),
        cx: round(b.cx, 2),
        cy: round(b.cy, 2),
        w: round(b.w, 2),
        h: round(b.h, 2),
        angle: round(b.angle, 4),
      }));
    case "semantic":
    case "depth":
      return [];
  }
}
