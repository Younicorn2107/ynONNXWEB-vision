/** Non-maximum suppression, axis-aligned and rotated. */

/** Larger than any model-space coordinate. */
export const CLASS_OFFSET = 7680;

export interface NMSOptions {
  iouThreshold?: number;
  confThreshold?: number;
  /** True: across classes. */
  agnostic?: boolean;
  maxDet?: number;
}

/** IoU of two `[x1, y1, x2, y2]` boxes. */
export function iou(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const ix = Math.max(0, Math.min(a[2]!, b[2]!) - Math.max(a[0]!, b[0]!));
  const iy = Math.max(0, Math.min(a[3]!, b[3]!) - Math.max(a[1]!, b[1]!));
  const inter = ix * iy;
  const union = (a[2]! - a[0]!) * (a[3]! - a[1]!) + (b[2]! - b[0]!) * (b[3]! - b[1]!) - inter;
  return union > 0 ? inter / union : 0;
}

/** Pairwise IoU, `[a.length][b.length]`. */
export function iouMatrix(a: readonly ArrayLike<number>[], b: readonly ArrayLike<number>[]): Float64Array[] {
  return a.map((boxA) => Float64Array.from(b, (boxB) => iou(boxA, boxB)));
}

/**
 * Indices sorted by decreasing score; ties keep the later index first, as NumPy's reversed argsort
 * does.
 */
function byScore(scores: ArrayLike<number>, indices: number[]): number[] {
  return indices.sort((i, j) => scores[i]! - scores[j]! || i - j).reverse();
}

/**
 * Confidence filter then greedy NMS over `[x1, y1, x2, y2]` boxes (flat, four per box). Returns the
 * kept indices, by decreasing score.
 */
export function nms(boxes: ArrayLike<number>, scores: ArrayLike<number>, classIds: ArrayLike<number>, options: NMSOptions = {}): number[] {
  const iouThreshold = options.iouThreshold ?? 0.45;
  const confThreshold = options.confThreshold ?? 0.25;
  const maxDet = options.maxDet ?? 300;
  const candidates: number[] = [];
  for (let i = 0; i < scores.length; i += 1) if (scores[i]! >= confThreshold) candidates.push(i);
  if (candidates.length === 0) return [];
  const shifted = (i: number): [number, number, number, number] => {
    const off = options.agnostic ? 0 : Math.fround(classIds[i]! * CLASS_OFFSET);
    return [
      Math.fround(boxes[4 * i]! + off),
      Math.fround(boxes[4 * i + 1]! + off),
      Math.fround(boxes[4 * i + 2]! + off),
      Math.fround(boxes[4 * i + 3]! + off),
    ];
  };
  let order = byScore(scores, candidates);
  const keep: number[] = [];
  while (order.length > 0 && keep.length < maxDet) {
    const best = order[0]!;
    keep.push(best);
    const bestBox = shifted(best);
    order = order.slice(1).filter((i) => iou(shifted(i), bestBox) <= iouThreshold);
  }
  return keep;
}

type Point = [number, number];

/** The corners of a rotated box `[cx, cy, w, h, angle]`, counter-clockwise. */
export function corners(box: ArrayLike<number>): Point[] {
  const [cx, cy, w, h, a] = [box[0]!, box[1]!, box[2]!, box[3]!, box[4]!];
  const c = Math.cos(a);
  const s = Math.sin(a);
  const pts: Point[] = [
    [-w / 2, -h / 2],
    [w / 2, -h / 2],
    [w / 2, h / 2],
    [-w / 2, h / 2],
  ];
  return pts.map(([x, y]) => [cx + x * c - y * s, cy + x * s + y * c]);
}

function polygonArea(points: readonly Point[]): number {
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const [x1, y1] = points[i]!;
    const [x2, y2] = points[(i + 1) % points.length]!;
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
}

/** Sutherland–Hodgman: `subject` clipped by the convex `clip` polygon. */
function clipPolygon(subject: readonly Point[], clip: readonly Point[]): Point[] {
  let signed = 0;
  for (let i = 0; i < clip.length; i += 1) {
    const [x1, y1] = clip[i]!;
    const [x2, y2] = clip[(i + 1) % clip.length]!;
    signed += x1 * y2 - x2 * y1;
  }
  const edges = signed >= 0 ? clip : [...clip].reverse();
  let output: Point[] = [...subject];
  for (let i = 0; i < edges.length && output.length > 0; i += 1) {
    const [ax, ay] = edges[i]!;
    const [bx, by] = edges[(i + 1) % edges.length]!;
    const side = ([px, py]: Point) => (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    const input = output;
    output = [];
    for (let j = 0; j < input.length; j += 1) {
      const p = input[j]!;
      const q = input[(j + 1) % input.length]!;
      const sp = side(p);
      const sq = side(q);
      if (sp >= 0) output.push(p);
      if ((sp >= 0) !== (sq >= 0)) {
        const t = sp / (sp - sq);
        output.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
      }
    }
  }
  return output;
}

/** Exact IoU of two rotated boxes `[cx, cy, w, h, angle_rad]`. */
export function rotatedIou(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const inter = clipPolygon(corners(a), corners(b));
  const interArea = inter.length >= 3 ? polygonArea(inter) : 0;
  const union = a[2]! * a[3]! + b[2]! * b[3]! - interArea;
  return union > 0 ? interArea / union : 0;
}

/**
 * Greedy NMS over rotated boxes (flat, five per box: cx, cy, w, h, angle) with the exact rotated
 * IoU. Returns kept indices by decreasing score.
 */
export function rotatedNms(boxes: ArrayLike<number>, scores: ArrayLike<number>, classIds: ArrayLike<number>, options: NMSOptions = {}): number[] {
  const iouThreshold = options.iouThreshold ?? 0.45;
  const confThreshold = options.confThreshold ?? 0.25;
  const maxDet = options.maxDet ?? 300;
  const candidates: number[] = [];
  for (let i = 0; i < scores.length; i += 1) if (scores[i]! >= confThreshold) candidates.push(i);
  if (candidates.length === 0) return [];
  const box = (i: number) => [
    boxes[5 * i]! + (options.agnostic ? 0 : classIds[i]! * CLASS_OFFSET),
    boxes[5 * i + 1]!,
    boxes[5 * i + 2]!,
    boxes[5 * i + 3]!,
    boxes[5 * i + 4]!,
  ];
  let order = byScore(scores, candidates);
  const keep: number[] = [];
  while (order.length > 0 && keep.length < maxDet) {
    const best = order[0]!;
    keep.push(best);
    const bestBox = box(best);
    order = order.slice(1).filter((i) => rotatedIou(bestBox, box(i)) <= iouThreshold);
  }
  return keep;
}
