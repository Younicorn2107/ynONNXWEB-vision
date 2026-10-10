/** Results for every frame from inferences on some of them. */

import type { BoundingBox, BoxResult, Keypoint, KeypointSet, YOLOResult } from "./types.js";

export interface MotionOptions {
  /** After the last keyframe: `predict` (carry on at the recent speed) or `hold` (stay put). */
  after?: "predict" | "hold";
  /**
   * Share of the speed kept from one frame to the next when predicting (0–1): lower stops sooner.
   */
  damping?: number;
  /**
   * A track slower than this (fraction of its height per frame) is taken as still: no drift from
   * jitter.
   */
  deadband?: number;
  /** Frames past its last keyframe a track is still shown. */
  maxFrames?: number;
}

const DEFAULTS: Required<MotionOptions> = { after: "predict", damping: 0.85, deadband: 0.004, maxFrames: 30 };

interface Keyframe {
  idx: number;
  result: YOLOResult;
}

/** Box as centre and size, the space motion is reasoned in. */
export interface Geometry {
  cx: number;
  cy: number;
  w: number;
  h: number;
}

export const geometryOf = (b: BoundingBox): Geometry => ({ cx: (b.x1 + b.x2) / 2, cy: (b.y1 + b.y2) / 2, w: b.x2 - b.x1, h: b.y2 - b.y1 });

export function withGeometry(b: BoundingBox, g: Geometry, confidence: number, imageShape: [number, number]): BoundingBox | null {
  const [H, W] = imageShape;
  let x1 = g.cx - g.w / 2;
  let x2 = g.cx + g.w / 2;
  let y1 = g.cy - g.h / 2;
  let y2 = g.cy + g.h / 2;
  if (W > 0 && H > 0) {
    x1 = Math.max(0, Math.min(x1, W));
    x2 = Math.max(0, Math.min(x2, W));
    y1 = Math.max(0, Math.min(y1, H));
    y2 = Math.max(0, Math.min(y2, H));
  }
  if (x2 <= x1 || y2 <= y1) return null;
  return { ...b, x1, y1, x2, y2, confidence, isInterpolated: true };
}

/**
 * The points of a keypoint set moved from one box geometry to another (translated and scaled about
 * the centre).
 */
export function movePoints(points: readonly Keypoint[], from: Geometry, to: Geometry): Keypoint[] {
  const sx = from.w > 0 ? to.w / from.w : 1;
  const sy = from.h > 0 ? to.h / from.h : 1;
  return points.map((p) => ({ ...p, x: to.cx + (p.x - from.cx) * sx, y: to.cy + (p.y - from.cy) * sy }));
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export class KeyframeTimeline {
  readonly #keys: Keyframe[] = [];

  /** Keyframes held (oldest first). */
  get keyframes(): readonly { idx: number; result: YOLOResult }[] {
    return this.#keys;
  }

  get lastIndex(): number {
    return this.#keys[this.#keys.length - 1]?.idx ?? -1;
  }

  /** A tracked result for frame `idx`. Keyframes arrive in frame order. */
  add(idx: number, result: YOLOResult): void {
    const at = this.#keys.findIndex((k) => k.idx >= idx);
    if (at < 0) this.#keys.push({ idx, result });
    else if (this.#keys[at]!.idx === idx) this.#keys[at] = { idx, result };
    else this.#keys.splice(at, 0, { idx, result });
  }

  /** Forget keyframes older than `idx` (keeping two before it, for speeds). */
  prune(idx: number): void {
    const at = this.#keys.findIndex((k) => k.idx >= idx);
    const drop = (at < 0 ? this.#keys.length : at) - 2;
    if (drop > 0) this.#keys.splice(0, drop);
  }

  clear(): void {
    this.#keys.length = 0;
  }

  /** The result for frame `idx`; null before the first keyframe. */
  resultAt(idx: number, options: MotionOptions = {}): YOLOResult | null {
    const o = { ...DEFAULTS, ...options };
    if (this.#keys.length === 0) return null;
    let after = -1;
    for (let i = 0; i < this.#keys.length; i += 1) if (this.#keys[i]!.idx <= idx) after = i;
    if (after < 0) return this.#keys[0]!.result;
    const a = this.#keys[after]!;
    if (a.idx === idx) return a.result;
    const b = this.#keys[after + 1];
    if (b) return this.#between(a, b, idx);
    return this.#beyond(after, idx, o);
  }

  /**
   * Between two keyframes: tracked boxes and keypoints interpolated, the rest from the nearer one.
   */
  #between(a: Keyframe, b: Keyframe, idx: number): YOLOResult {
    const t = (idx - a.idx) / (b.idx - a.idx);
    const near = t < 0.5 ? a.result : b.result;
    if (!("boxes" in a.result) || !("boxes" in b.result)) return near;
    const ra = a.result as BoxResult;
    const rb = b.result as BoxResult;
    const byId = (r: BoxResult) => new Map(r.boxes.map((box, i) => [box.trackId, i] as const).filter(([id]) => id !== null));
    const ia = byId(ra);
    const ib = byId(rb);
    const boxes: BoundingBox[] = [];
    const sets: KeypointSet[] = [];
    const poseA = ra.task === "pose" ? ra.keypoints : null;
    const poseB = rb.task === "pose" ? rb.keypoints : null;
    const ids = new Set([...ia.keys(), ...ib.keys()]);
    for (const id of ids) {
      const i = ia.get(id);
      const j = ib.get(id);
      if (i !== undefined && j !== undefined) {
        const ba = ra.boxes[i]!;
        const bb = rb.boxes[j]!;
        const box: BoundingBox = {
          ...ba,
          x1: lerp(ba.x1, bb.x1, t),
          y1: lerp(ba.y1, bb.y1, t),
          x2: lerp(ba.x2, bb.x2, t),
          y2: lerp(ba.y2, bb.y2, t),
          confidence: lerp(ba.confidence, bb.confidence, t),
          isInterpolated: true,
        };
        boxes.push(box);
        if (poseA && poseB) {
          const pa = poseA[i]!.points;
          const pb = poseB[j]!.points;
          sets.push({ box, points: pa.map((p, k) => ({ ...p, x: lerp(p.x, pb[k]?.x ?? p.x, t), y: lerp(p.y, pb[k]?.y ?? p.y, t), confidence: lerp(p.confidence, pb[k]?.confidence ?? p.confidence, t) })) });
        }
      } else {
        const own = i !== undefined ? { r: ra, k: i, pose: poseA, side: 0 } : { r: rb, k: j!, pose: poseB, side: 1 };
        if ((own.side === 0) !== (t < 0.5)) continue;
        const box = { ...own.r.boxes[own.k]!, isInterpolated: true };
        boxes.push(box);
        if (own.pose) sets.push({ box, points: own.pose[own.k]!.points });
      }
    }

    const rn = near as BoxResult;
    rn.boxes.forEach((box, k) => {
      if (box.trackId !== null) return;
      boxes.push(box);
      if (rn.task === "pose") sets.push(rn.keypoints[k]!);
    });
    return this.#assemble(near, boxes, sets);
  }

  /** After the last keyframe: carried on (damped speed) or held, for `maxFrames` frames. */
  #beyond(last: number, idx: number, o: Required<MotionOptions>): YOLOResult {
    const k = this.#keys[last]!;
    if (!("boxes" in k.result)) return k.result;
    const r = k.result as BoxResult;
    const d = idx - k.idx;
    if (d > o.maxFrames) return this.#assemble(r, [], []);

    const reach = o.after === "hold" ? 0 : o.damping >= 1 ? d : (o.damping * (1 - o.damping ** d)) / (1 - o.damping);
    const boxes: BoundingBox[] = [];
    const sets: KeypointSet[] = [];
    r.boxes.forEach((box, i) => {
      const now = geometryOf(box);
      let moved = now;
      if (reach > 0 && box.trackId !== null) {
        const speed = this.#speed(last, box.trackId);
        if (speed) {
          const still = Math.hypot(speed.cx, speed.cy) < o.deadband * now.h;
          moved = {
            cx: now.cx + (still ? 0 : speed.cx * reach),
            cy: now.cy + (still ? 0 : speed.cy * reach),
            w: Math.max(1, now.w + (Math.abs(speed.w) < o.deadband * now.w ? 0 : speed.w * reach)),
            h: Math.max(1, now.h + (Math.abs(speed.h) < o.deadband * now.h ? 0 : speed.h * reach)),
          };
        }
      }
      const out = withGeometry(box, moved, box.confidence, r.imageShape);
      if (!out) return;
      boxes.push(out);
      if (r.task === "pose") sets.push({ box: out, points: movePoints(r.keypoints[i]!.points, now, moved) });
    });
    return this.#assemble(r, boxes, sets);
  }

  /**
   * Per-frame change of a track's geometry over its last two keyframes; null with fewer than two.
   */
  #speed(last: number, trackId: number): Geometry | null {
    const k = this.#keys[last]!;
    const now = (k.result as BoxResult).boxes.find((b) => b.trackId === trackId);
    if (!now) return null;
    for (let i = last - 1; i >= 0; i -= 1) {
      const before = (this.#keys[i]!.result as BoxResult).boxes?.find((b) => b.trackId === trackId);
      if (!before) continue;
      const frames = k.idx - this.#keys[i]!.idx;
      if (frames <= 0) return null;
      const g1 = geometryOf(now);
      const g0 = geometryOf(before);
      return { cx: (g1.cx - g0.cx) / frames, cy: (g1.cy - g0.cy) / frames, w: (g1.w - g0.w) / frames, h: (g1.h - g0.h) / frames };
    }
    return null;
  }

  #assemble(base: YOLOResult, boxes: BoundingBox[], sets: KeypointSet[]): YOLOResult {
    if (base.task === "pose") return { ...base, boxes, keypoints: sets, inferenceTimeMs: 0 };
    if (base.task === "segment") {
      const masks = base.masks.filter((m) => boxes.some((b) => b.trackId !== null && b.trackId === m.box.trackId));
      return { ...base, boxes, masks, inferenceTimeMs: 0 };
    }
    if (base.task === "detect") return { ...base, boxes, inferenceTimeMs: 0 };
    return base;
  }
}
