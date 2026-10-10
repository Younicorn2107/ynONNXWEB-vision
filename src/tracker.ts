/** Multi-object tracking: ByteTrack and BoT-SORT. */

import { TrackerError } from "./errors.js";
import { MotionCompensator, transformPoint, transformScale } from "./gmc.js";
import { KalmanFilterXYAH, xyahToXyxy, xyxyToXyah, type Matrix, type Vector } from "./kalman.js";
import { iouMatrix } from "./nms.js";
import { framePixels, type ImageInput } from "./gpu.js";
import type { RasterImage } from "./preprocess.js";
import { appearanceCost, type ReIDEmbedder } from "./reid.js";
import type { BoundingBox, BoxResult, Track } from "./types.js";

export type TrackerAlgorithm = "bytetrack" | "botsort";

export interface TrackerConfig {
  /** Detections at or above this start or continue tracks in pass 1. */
  trackConf?: number;
  trackIou?: number;
  /** Frames a lost track is kept before it is dropped (30; 50 for BoT-SORT). */
  trackBuffer?: number;
  /** Highest association cost accepted in pass 1. */
  matchThresh?: number;
  /** The weak detections of ByteTrack's second pass start here. */
  lowConfThresh?: number;
  /** Appearance features, from `reid`: fused into association, and re-identification. */
  withReid?: boolean;
  /** Matches before a track gets its number (2): a one-frame flicker gets none. */
  minHits?: number;
  /**
   * Re-identification: the largest appearance distance (`(1 − cos) / 2`) that gives a number back.
   */
  reidThreshold?: number;
  /** Re-identification: updates a gone track's appearance is remembered (600). */
  reidMemory?: number;
}

/** A track as the tracker last knew it, for holding it on screen while it is lost. */
export interface LostTrack {
  trackId: number;
  classId: number;
  className: string;
  /** The confidence it was last seen with. */
  score: number;
  /** Updates since it was last matched. */
  timeSinceUpdate: number;
  /** Where its Kalman filter puts it now. */
  predicted: [number, number, number, number];
  /** Where it was last seen. */
  lastSeen: [number, number, number, number];
}

/** Weight of the historical appearance feature. */
const FEATURE_EMA = 0.9;

class STrack {
  age = 0;
  hits = 1;
  timeSinceUpdate = 0;
  /** The number shown, given at confirmation (`minHits`); null before. */
  trackId: number | null = null;
  lastSeen: [number, number, number, number];

  constructor(
    readonly key: number,
    public mean: Vector,
    public covariance: Matrix,
    public score: number,
    public classId: number,
    public className: string,
    public feature: Float32Array | null = null,
  ) {
    this.lastSeen = this.xyxy;
  }

  get xyxy(): [number, number, number, number] {
    return xyahToXyxy(this.mean.slice(0, 4));
  }

  updateFeature(feature: Float32Array | null): void {
    if (!feature || feature.length === 0 || feature.every((v) => v === 0)) return;
    if (!this.feature) {
      this.feature = Float32Array.from(feature);
      return;
    }
    const blended = this.feature.map((v, i) => FEATURE_EMA * v + (1 - FEATURE_EMA) * feature[i]!);
    const norm = Math.sqrt(blended.reduce((s, v) => s + v * v, 0));
    this.feature = norm > 0 ? blended.map((v) => v / norm) : blended;
  }
}

/**
 * The assignment of minimal total cost between rows and columns (each used at most once; `min(rows,
 * cols)` pairs), as SciPy's `linear_sum_assignment`.
 */
export function linearSumAssignment(cost: readonly (ArrayLike<number>)[]): [number, number][] {
  const rows = cost.length;
  const cols = rows > 0 ? cost[0]!.length : 0;
  if (rows === 0 || cols === 0) return [];
  const transposed = rows > cols;
  const n = transposed ? cols : rows;
  const m = transposed ? rows : cols;
  const at = (i: number, j: number) => (transposed ? cost[j]![i]! : cost[i]![j]!);

  const u = new Float64Array(n + 1);
  const v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1);
  const way = new Int32Array(m + 1);
  for (let i = 1; i <= n; i += 1) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(Infinity);
    const used = new Uint8Array(m + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0]!;
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= m; j += 1) {
        if (used[j]) continue;
        const cur = at(i0 - 1, j - 1) - u[i0]! - v[j]!;
        if (cur < minv[j]!) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j]! < delta) {
          delta = minv[j]!;
          j1 = j;
        }
      }
      for (let j = 0; j <= m; j += 1) {
        if (used[j]) {
          u[p[j]!] = u[p[j]!]! + delta;
          v[j] = v[j]! - delta;
        } else {
          minv[j] = minv[j]! - delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0]!;
      p[j0] = p[j1]!;
      j0 = j1;
    } while (j0 !== 0);
  }
  const pairs: [number, number][] = [];
  for (let j = 1; j <= m; j += 1) {
    if (p[j] === 0) continue;
    pairs.push(transposed ? [j - 1, p[j]! - 1] : [p[j]! - 1, j - 1]);
  }
  return pairs.sort((a, b) => a[0] - b[0]);
}

export class ObjectTracker {
  readonly algorithm: TrackerAlgorithm;
  readonly config: Required<TrackerConfig>;
  readonly kalman = new KalmanFilterXYAH();
  readonly #gmc: MotionCompensator | null;
  readonly #embedder: ReIDEmbedder | null;
  #tracks: STrack[] = [];
  #lost: STrack[] = [];
  #nextKey = 1;
  #nextId = 1;
  #frame = 0;
  /** Gone tracks' numbers and appearances, for re-identification. */
  #gallery: { trackId: number; classId: number; feature: Float32Array; frame: number }[] = [];

  /**
   * @param reid BoT-SORT's appearance model (`ReIDEmbedder.load`); used when `withReid` is on.
   * Without it, association is Kalman + IoU only.
   */
  constructor(algorithm: TrackerAlgorithm = "bytetrack", config: TrackerConfig = {}, reid: ReIDEmbedder | null = null) {
    if (algorithm !== "bytetrack" && algorithm !== "botsort") throw new TrackerError(`unknown tracker algorithm '${String(algorithm)}'`);
    this.algorithm = algorithm;
    this.config = {
      trackConf: config.trackConf ?? 0.25,
      trackIou: config.trackIou ?? 0.45,

      trackBuffer: config.trackBuffer ?? (algorithm === "botsort" ? 50 : 30),
      matchThresh: config.matchThresh ?? 0.8,
      lowConfThresh: config.lowConfThresh ?? 0.1,
      withReid: config.withReid ?? false,
      minHits: Math.max(1, config.minHits ?? 2),
      reidThreshold: config.reidThreshold ?? 0.12,
      reidMemory: config.reidMemory ?? 600,
    };
    this.#gmc = algorithm === "botsort" ? new MotionCompensator() : null;
    if (this.config.withReid && !reid) {
      throw new TrackerError("withReid needs a ReID model: pass a ReIDEmbedder, or turn withReid off");
    }
    this.#embedder = this.config.withReid ? reid : null;
  }

  /** The detections of one frame, given their track ids. */
  async update<R extends BoxResult>(detections: R, image: ImageInput | null = null): Promise<R> {
    const config = this.config;
    const boxes = detections.boxes;
    this.#frame += 1;
    for (const track of [...this.#tracks, ...this.#lost]) {
      ({ mean: track.mean, covariance: track.covariance } = this.kalman.predict(track.mean, track.covariance));
      track.age += 1;
      track.timeSinceUpdate += 1;
    }
    if (this.#gmc && image) this.#compensate(framePixels(image));

    const high = boxes.map((b, i) => (b.confidence >= config.trackConf ? i : -1)).filter((i) => i >= 0);
    const low = boxes.map((b, i) => (b.confidence >= config.lowConfThresh && b.confidence < config.trackConf ? i : -1)).filter((i) => i >= 0);
    const highFeatures = this.#embedder && image && high.length > 0 ? await this.#embedder.embed(image, high.map((i) => boxes[i]!)) : null;

    const candidates = [...this.#tracks, ...this.#lost];
    const assigned = new Map<number, STrack>();
    const matched: STrack[] = [];

    const first = this.#associate(candidates, high.map((i) => boxes[i]!), config.matchThresh, highFeatures);
    for (const [t, d] of first.matches) {
      const track = candidates[t]!;
      this.#updateTrack(track, boxes[high[d]!]!);
      if (highFeatures) track.updateFeature(highFeatures[d]!);
      assigned.set(high[d]!, track);
      matched.push(track);
    }

    const leftover = first.unmatchedTracks.map((t) => candidates[t]!);
    const second = this.#associate(leftover, low.map((i) => boxes[i]!), 0.5, null);
    for (const [t, d] of second.matches) {
      const track = leftover[t]!;
      this.#updateTrack(track, boxes[low[d]!]!);
      assigned.set(low[d]!, track);
      matched.push(track);
    }
    const unmatched = second.unmatchedTracks.map((t) => leftover[t]!);
    this.#lost = unmatched.filter((t) => t.timeSinceUpdate <= config.trackBuffer);

    for (const gone of unmatched.filter((t) => t.timeSinceUpdate > config.trackBuffer)) this.#remember(gone);
    this.#gallery = this.#gallery.filter((g) => this.#frame - g.frame <= config.reidMemory);
    this.#tracks = matched;

    for (const d of first.unmatchedDetections) {
      const track = this.#createTrack(boxes[high[d]!]!, highFeatures?.[d] ?? null);
      assigned.set(high[d]!, track);
      this.#tracks.push(track);
    }

    for (const track of this.#tracks) if (track.trackId === null && (track.hits >= config.minHits || this.#frame === 1)) this.#number(track);
    const tracked = boxes.map((b, i) => ({ ...b, trackId: assigned.get(i)?.trackId ?? null }));

    if (detections.task === "segment") return { ...detections, boxes: tracked, masks: detections.masks.map((m, i) => ({ ...m, box: tracked[i] ?? m.box })) };
    if (detections.task === "pose") return { ...detections, boxes: tracked, keypoints: detections.keypoints.map((k, i) => ({ ...k, box: tracked[i] ?? k.box })) };
    return { ...detections, boxes: tracked };
  }

  /** Drop every track and every number (a new clip). */
  reset(): void {
    this.#tracks = [];
    this.#lost = [];
    this.#gallery = [];
    this.#nextKey = 1;
    this.#nextId = 1;
    this.#frame = 0;
    this.#gmc?.reset();
  }

  /** The numbered tracks lost at the last update and not dropped yet. */
  lostTracks(): LostTrack[] {
    return this.#lost
      .filter((t) => t.trackId !== null)
      .map((t) => ({
        trackId: t.trackId!,
        classId: t.classId,
        className: t.className,
        score: t.score,
        timeSinceUpdate: t.timeSinceUpdate,
        predicted: t.xyxy,
        lastSeen: t.lastSeen,
      }));
  }

  /** The live numbered tracks, as plain data. */
  activeTracks(): Track[] {
    return this.#tracks.filter((t) => t.trackId !== null).map((t) => {
      const [x1, y1, x2, y2] = t.xyxy;
      return {
        trackId: t.trackId!,
        classId: t.classId,
        className: t.className,
        lastBox: { x1, y1, x2, y2, confidence: t.score, classId: t.classId, className: t.className, trackId: t.trackId, isInterpolated: false },
        age: t.age,
        hits: t.hits,
        timeSinceUpdate: t.timeSinceUpdate,
      };
    });
  }

  /** The live numbered tracks' Kalman states, for the interpolator (never mutated by it). */
  get liveStates(): readonly { trackId: number; classId: number; className: string; score: number; mean: Vector; covariance: Matrix }[] {
    return this.#tracks.filter((t): t is STrack & { trackId: number } => t.trackId !== null);
  }

  /** A track gone for good: its number and look, kept `reidMemory` updates. */
  #remember(track: STrack): void {
    if (track.trackId === null || !track.feature) return;
    this.#gallery.push({ trackId: track.trackId, classId: track.classId, feature: track.feature, frame: this.#frame });
  }

  /**
   * Its number for a track just confirmed: the one of a lost or gone track of the same class that
   * looks alike (the lost one is then dropped: it is this one), else a new one.
   */
  #number(track: STrack): void {
    if (this.#embedder && track.feature) {
      const inUse = new Set(this.#tracks.map((t) => t.trackId));
      let best: { trackId: number; distance: number; lost: STrack | null } | null = null;
      const consider = (trackId: number | null, classId: number, feature: Float32Array | null, lost: STrack | null) => {
        if (trackId === null || !feature || classId !== track.classId || inUse.has(trackId)) return;
        const distance = appearanceCost([feature], [track.feature!])[0]![0]!;
        if (distance <= this.config.reidThreshold && (!best || distance < best.distance)) best = { trackId, distance, lost };
      };
      for (const lost of this.#lost) consider(lost.trackId, lost.classId, lost.feature, lost);
      for (const gone of this.#gallery) consider(gone.trackId, gone.classId, gone.feature, null);
      if (best) {
        const found = best as { trackId: number; lost: STrack | null };
        track.trackId = found.trackId;
        if (found.lost) this.#lost = this.#lost.filter((t) => t !== found.lost);
        this.#gallery = this.#gallery.filter((g) => g.trackId !== found.trackId);
        return;
      }
    }
    track.trackId = this.#nextId;
    this.#nextId += 1;
  }

  #compensate(image: RasterImage): void {
    const m = this.#gmc!.estimate(image);
    const scale = transformScale(m);
    if (Math.abs(m[2]) < 1e-9 && Math.abs(m[5]) < 1e-9 && Math.abs(scale - 1) < 1e-9) return;
    for (const track of [...this.#tracks, ...this.#lost]) {
      const [x, y] = transformPoint(m, track.mean[0]!, track.mean[1]!);
      track.mean[0] = x;
      track.mean[1] = y;
      track.mean[3] = track.mean[3]! * scale;
      const [vx, vy] = [track.mean[4]!, track.mean[5]!];
      track.mean[4] = m[0] * vx + m[1] * vy;
      track.mean[5] = m[3] * vx + m[4] * vy;
    }
  }

  #associate(tracks: STrack[], detections: BoundingBox[], matchThresh: number, features: Float32Array[] | null) {
    if (tracks.length === 0 || detections.length === 0) {
      return { matches: [] as [number, number][], unmatchedTracks: tracks.map((_, i) => i), unmatchedDetections: detections.map((_, i) => i) };
    }
    const ious = iouMatrix(tracks.map((t) => t.xyxy), detections.map((d) => [d.x1, d.y1, d.x2, d.y2]));
    let cost = ious.map((row) => Array.from(row, (v) => 1 - v));
    if (features && tracks.some((t) => t.feature)) {
      const dim = features[0]!.length;
      const app = appearanceCost(tracks.map((t) => t.feature ?? new Float32Array(dim)), features);
      cost = cost.map((row, i) => (tracks[i]!.feature ? row.map((c, j) => 0.5 * c + 0.5 * app[i]![j]!) : row));
    }
    const matches: [number, number][] = [];
    const freeTracks = new Set(tracks.map((_, i) => i));
    const freeDetections = new Set(detections.map((_, i) => i));
    for (const [r, c] of linearSumAssignment(cost)) {
      if (cost[r]![c]! > matchThresh) continue;
      matches.push([r, c]);
      freeTracks.delete(r);
      freeDetections.delete(c);
    }
    return { matches, unmatchedTracks: [...freeTracks].sort((a, b) => a - b), unmatchedDetections: [...freeDetections].sort((a, b) => a - b) };
  }

  #createTrack(box: BoundingBox, feature: Float32Array | null): STrack {
    const { mean, covariance } = this.kalman.initiate(xyxyToXyah([box.x1, box.y1, box.x2, box.y2]));
    const usable = feature && feature.some((v) => v !== 0) ? Float32Array.from(feature) : null;
    const track = new STrack(this.#nextKey, mean, covariance, box.confidence, box.classId, box.className, usable);
    this.#nextKey += 1;
    return track;
  }

  #updateTrack(track: STrack, box: BoundingBox): void {
    ({ mean: track.mean, covariance: track.covariance } = this.kalman.update(track.mean, track.covariance, xyxyToXyah([box.x1, box.y1, box.x2, box.y2])));
    track.score = box.confidence;
    track.classId = box.classId;
    track.className = box.className;
    track.hits += 1;
    track.timeSinceUpdate = 0;
    track.lastSeen = [box.x1, box.y1, box.x2, box.y2];
  }
}

/**
 * Boxes for a frame that was not inferred: each live track's Kalman state advanced `frameIdx −
 * lastInferred` steps, on a copy — the tracker itself is untouched. Tracks that drift out of the
 * image are left out.
 */
export function propagate<R extends BoxResult>(tracker: ObjectTracker, frameIdx: number, lastInferred: number, last: R): R {
  const delta = Math.max(1, frameIdx - lastInferred);
  const [h, w] = last.imageShape;
  const boxes: BoundingBox[] = [];
  for (const track of tracker.liveStates) {
    let state = { mean: [...track.mean], covariance: track.covariance.map((r) => [...r]) };
    for (let i = 0; i < delta; i += 1) state = tracker.kalman.predict(state.mean, state.covariance);
    let [x1, y1, x2, y2] = xyahToXyxy(state.mean.slice(0, 4));
    if (w > 0 && h > 0) {
      x1 = Math.max(0, Math.min(x1, w));
      x2 = Math.max(0, Math.min(x2, w));
      y1 = Math.max(0, Math.min(y1, h));
      y2 = Math.max(0, Math.min(y2, h));
    }
    if (x2 <= x1 || y2 <= y1) continue;
    boxes.push({ x1, y1, x2, y2, confidence: track.score, classId: track.classId, className: track.className, trackId: track.trackId, isInterpolated: true });
  }
  return { ...last, boxes, inferenceTimeMs: 0 };
}
