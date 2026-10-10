/**
 * What each numbered object did over a video or a live feed: where it was seen (its path, drawn as
 * a trail) and when it was there (its presence, as time ranges: "person 1, 00:20–00:50 and
 * 02:14–03:50").
 */

import type { YOLOResult } from "./types.js";

export interface TrackPoint {
  frameIdx: number;
  timeS: number;
  /** The box centre, in pixels of the original image. */
  x: number;
  y: number;
}

export interface PresenceRange {
  startS: number;
  endS: number;
}

export interface TrackSummary {
  trackId: number;
  classId: number;
  className: string;
  /** Time ranges the object was present in. */
  presence: PresenceRange[];
  /** Seconds present, all ranges together. */
  totalS: number;
  /** Where the model saw it (box centres), in time order. */
  points: TrackPoint[];
}

export interface TrackHistoryOptions {
  /** A gap longer than this (seconds) ends a presence range (1). */
  gapS?: number;
  /** Points kept per object; the oldest go first (5000). */
  maxPoints?: number;
}

interface Entry {
  classId: number;
  className: string;
  presence: PresenceRange[];
  points: TrackPoint[];
}

export class TrackHistory {
  readonly #tracks = new Map<number, Entry>();
  readonly #gapS: number;
  readonly #maxPoints: number;

  constructor(options: TrackHistoryOptions = {}) {
    this.#gapS = options.gapS ?? 1;
    this.#maxPoints = options.maxPoints ?? 5000;
  }

  /** A keyframe's result, at `timeS` seconds into the video. */
  add(frameIdx: number, timeS: number, result: YOLOResult): void {
    for (const box of trackedOf(result)) {
      if (box.trackId === null) continue;
      let entry = this.#tracks.get(box.trackId);
      if (!entry) {
        entry = { classId: box.classId, className: box.className, presence: [], points: [] };
        this.#tracks.set(box.trackId, entry);
      }
      const range = entry.presence[entry.presence.length - 1];
      if (range && timeS >= range.startS && timeS - range.endS <= this.#gapS) range.endS = Math.max(range.endS, timeS);
      else if (!range || timeS > range.endS) entry.presence.push({ startS: timeS, endS: timeS });
      if (box.isInterpolated) continue;
      const last = entry.points[entry.points.length - 1];
      if (last && last.frameIdx >= frameIdx) continue;
      entry.points.push({ frameIdx, timeS, x: box.x, y: box.y });
      if (entry.points.length > this.#maxPoints) entry.points.splice(0, entry.points.length - this.#maxPoints);
    }
  }

  /** Where an object was seen, oldest first. */
  trail(trackId: number): readonly TrackPoint[] {
    return this.#tracks.get(trackId)?.points ?? [];
  }

  /** Every object seen, by number. */
  summary(): TrackSummary[] {
    return [...this.#tracks.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([trackId, e]) => ({
        trackId,
        classId: e.classId,
        className: e.className,
        presence: e.presence.map((r) => ({ ...r })),
        totalS: e.presence.reduce((s, r) => s + (r.endS - r.startS), 0),
        points: [...e.points],
      }));
  }

  clear(): void {
    this.#tracks.clear();
  }
}

/** The numbered objects of a result, as centres: boxes, or oriented boxes. */
export function trackedOf(result: YOLOResult): { trackId: number | null; classId: number; className: string; isInterpolated: boolean; x: number; y: number }[] {
  if ("boxes" in result) return result.boxes.map((b) => ({ trackId: b.trackId, classId: b.classId, className: b.className, isInterpolated: b.isInterpolated, x: (b.x1 + b.x2) / 2, y: (b.y1 + b.y2) / 2 }));
  if (result.task === "obb") return result.orientedBoxes.map((o) => ({ trackId: o.trackId ?? null, classId: o.classId, className: o.className, isInterpolated: o.isInterpolated ?? false, x: o.cx, y: o.cy }));
  return [];
}

/** `mm:ss` (or `h:mm:ss`), the way a video player shows a time. */
export function formatTimecode(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${String(m).padStart(2, "0")}:${ss}`;
}

/** A JSON-ready report: each object, its presence as timecodes, its path. */
export function trackReport(history: TrackHistory, source: string): Record<string, unknown> {
  return {
    source,
    objects: history.summary().map((t) => ({
      track_id: t.trackId,
      class_id: t.classId,
      class_name: t.className,
      presence: t.presence.map((r) => ({ start: formatTimecode(r.startS), end: formatTimecode(r.endS), start_s: round(r.startS), end_s: round(r.endS) })),
      total_s: round(t.totalS),
      path: t.points.map((p) => ({ frame: p.frameIdx, t: round(p.timeS), x: Math.round(p.x), y: Math.round(p.y) })),
    })),
  };
}

const round = (v: number) => Math.round(v * 1000) / 1000;
