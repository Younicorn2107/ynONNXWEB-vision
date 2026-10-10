import { describe, expect, it } from "vitest";

import { ObjectTracker, TrackHistory, VideoSession, formatTimecode, frameSignature, holdLost, likeness, rememberSeen, trackReport, type DetectionResult, type ReIDEmbedder, type SeenObject } from "../../src/core.js";

const frame = { data: new Uint8Array(64 * 36 * 4).fill(128), width: 64, height: 36, channels: 4 as const };

function detections(boxes: [number, number, number, number, number][]): DetectionResult {
  return {
    task: "detect",
    modelName: "test",
    source: null,
    inferenceTimeMs: 0,
    imageShape: [1000, 1000],
    boxes: boxes.map(([x1, y1, x2, y2, confidence]) => ({ x1, y1, x2, y2, confidence, classId: 0, className: "person", trackId: null, isInterpolated: false })),
  };
}

/**
 * An appearance model that tells objects apart by the 5th value given to `detections` (their
 * "look"), through x1's tenths.
 */
const embedder = (looks: Map<number, number>) =>
  ({
    embed: async (_image: unknown, boxes: readonly { x1: number }[]) =>
      boxes.map((b) => {
        const look = looks.get(Math.round(b.x1)) ?? 0;
        const v = new Float32Array(4);
        v[look] = 1;
        return v;
      }),
  }) as unknown as ReIDEmbedder;

describe("track numbers", () => {
  it("go to confirmed tracks only: a one-frame flicker gets none and uses none", async () => {
    const tracker = new ObjectTracker("bytetrack");
    const ids = async (b: [number, number, number, number, number][]) => (await tracker.update(detections(b))).boxes.map((x) => x.trackId);
    expect(await ids([[100, 100, 200, 300, 0.9]])).toEqual([1]);
    expect(await ids([[102, 100, 202, 300, 0.9], [600, 600, 650, 700, 0.9]])).toEqual([1, null]);
    expect(await ids([[104, 100, 204, 300, 0.9]])).toEqual([1]);
    expect(await ids([[106, 100, 206, 300, 0.9], [800, 100, 850, 200, 0.9]])).toEqual([1, null]);
    expect(await ids([[108, 100, 208, 300, 0.9], [802, 100, 852, 200, 0.9]])).toEqual([1, 2]);
  });

  it("are given back by re-identification to an object that looks like one gone", async () => {
    const looks = new Map([[100, 1], [700, 1], [400, 2]]);
    const tracker = new ObjectTracker("bytetrack", { withReid: true, trackBuffer: 2 }, embedder(looks));
    const ids = async (b: [number, number, number, number, number][]) => (await tracker.update(detections(b), frame)).boxes.map((x) => x.trackId);
    expect(await ids([[100, 100, 200, 300, 0.9]])).toEqual([1]);
    expect(await ids([[100, 100, 200, 300, 0.9]])).toEqual([1]);
    for (let i = 0; i < 4; i += 1) await ids([]);

    await ids([[400, 100, 500, 300, 0.9], [700, 100, 800, 300, 0.9]]);
    expect(await ids([[400, 100, 500, 300, 0.9], [700, 100, 800, 300, 0.9]])).toEqual([2, 1]);
  });
});

describe("holding missed objects", () => {
  it("puts a lost, confident object back with its number when the frames follow, not across a cut", async () => {
    const tracker = new ObjectTracker("bytetrack");
    const seen = new Map<number, SeenObject>();
    for (let i = 0; i < 3; i += 1) rememberSeen(await tracker.update(detections([[100 + i, 100, 200 + i, 300, 0.9]])), seen);
    const missed = await tracker.update(detections([]));
    const held = holdLost(missed, tracker, seen, 0.97, { continuity: 0.9 });
    expect(held.boxes).toHaveLength(1);
    expect(held.boxes[0]).toMatchObject({ trackId: 1, isInterpolated: true });
    expect(holdLost(missed, tracker, seen, 0.5, { continuity: 0.9 }).boxes).toHaveLength(0);
    expect(holdLost(missed, tracker, seen, 0.97, { minConfidence: 0.95 }).boxes).toHaveLength(0);
    expect(holdLost(missed, tracker, seen, 0.97, { mode: "hold" }).boxes[0]).toMatchObject({ x1: 102, x2: 202 });
  });

  it("tells following frames from a cut, whatever the exposure", () => {
    const a = frameSignature(frame);
    const brighter = frameSignature({ ...frame, data: new Uint8Array(frame.data.length).fill(160) });
    const other = new Uint8Array(frame.data.length);
    for (let i = 0; i < other.length; i += 4) other.fill((i / 4) % 64 < 32 ? 0 : 255, i, i + 4);
    expect(likeness(a, a)).toBe(1);
    expect(likeness(a, brighter)).toBe(1);
    expect(likeness(a, frameSignature({ ...frame, data: other }))).toBeLessThan(0.5);
  });
});

describe("track history", () => {
  it("draws observed points only, and keeps presence as time ranges split by gaps", () => {
    const history = new TrackHistory({ gapS: 1 });
    const at = (x: number, interpolated = false) => {
      const r = detections([[x, 0, x + 10, 10, 0.9]]);
      r.boxes[0]!.trackId = 1;
      r.boxes[0]!.isInterpolated = interpolated;
      return r;
    };
    history.add(0, 0, at(0));
    history.add(30, 0.5, at(5, true));
    history.add(60, 1, at(10));
    history.add(300, 5, at(50));
    expect(history.trail(1).map((p) => p.x)).toEqual([5, 15, 55]);
    const [one] = history.summary();
    expect(one!.presence).toEqual([{ startS: 0, endS: 1 }, { startS: 5, endS: 5 }]);
    expect(formatTimecode(134)).toBe("02:14");
    expect((trackReport(history, "clip.mp4")["objects"] as { presence: { start: string }[] }[])[0]!.presence[1]!.start).toBe("00:05");
  });
});

describe("oriented boxes", () => {
  it("are tracked, held in shape and angle, and kept in the history, like upright boxes", async () => {
    const frames: [number, number][][] = [[[100, 100], [500, 300]], [[104, 101], [503, 300]], [[108, 102], [506, 300]], [[112, 103]]];
    let call = 0;
    const runner = {
      task: "obb",
      runBatch: async (images: unknown[]) =>
        images.map(() => {
          const objects = frames[call++]!;
          return {
            task: "obb" as const,
            modelName: "test",
            source: null,
            inferenceTimeMs: 0,
            imageShape: [1000, 1000] as [number, number],
            orientedBoxes: objects.map(([cx, cy]) => ({ cx, cy, w: 120, h: 40, angle: 0.5, confidence: 0.9, classId: 1, className: "ship" })),
          };
        }),
    };
    const session = new VideoSession(runner as never, { tracker: "bytetrack", persist: { continuity: 0 } });
    const history = new TrackHistory();
    for (let i = 0; i < frames.length; i += 1) {
      await session.infer([{ image: frame, frameIdx: i }]);
      history.add(i, i / 10, session.timeline.resultAt(i)!);
    }
    const last = session.timeline.resultAt(3)! as { orientedBoxes: { trackId?: number | null; isInterpolated?: boolean; angle: number; cx: number }[] };
    expect(last.orientedBoxes.map((o) => o.trackId)).toEqual([1, 2]);

    expect(last.orientedBoxes[1]).toMatchObject({ isInterpolated: true, angle: 0.5 });
    expect(history.trail(1).map((p) => p.x)).toEqual([100, 104, 108, 112]);
    expect(history.trail(2)).toHaveLength(3);
  });
});
