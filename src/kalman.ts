/** Kalman filter over the XYAH box state (the ByteTrack formulation). */

const STD_POSITION = 1 / 20;
const STD_VELOCITY = 1 / 160;

export type Vector = number[];
export type Matrix = number[][];

const diag = (values: readonly number[]): Matrix => values.map((v, i) => values.map((_, j) => (i === j ? v * v : 0)));
const zeros = (r: number, c: number): Matrix => Array.from({ length: r }, () => new Array<number>(c).fill(0));

function matmul(a: Matrix, b: Matrix): Matrix {
  const out = zeros(a.length, b[0]!.length);
  for (let i = 0; i < a.length; i += 1)
    for (let k = 0; k < b.length; k += 1) {
      const v = a[i]![k]!;
      if (v === 0) continue;
      for (let j = 0; j < b[0]!.length; j += 1) out[i]![j]! += v * b[k]![j]!;
    }
  return out;
}

const transpose = (a: Matrix): Matrix => a[0]!.map((_, j) => a.map((row) => row[j]!));
const add = (a: Matrix, b: Matrix): Matrix => a.map((row, i) => row.map((v, j) => v + b[i]![j]!));

/** `A⁻¹` of a small symmetric positive-definite matrix (Gauss–Jordan with pivoting). */
function inverse(m: Matrix): Matrix {
  const n = m.length;
  const a = m.map((row, i) => [...row, ...row.map((_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let r = col + 1; r < n; r += 1) if (Math.abs(a[r]![col]!) > Math.abs(a[pivot]![col]!)) pivot = r;
    [a[col], a[pivot]] = [a[pivot]!, a[col]!];
    const p = a[col]![col]!;
    for (let j = 0; j < 2 * n; j += 1) a[col]![j]! /= p;
    for (let r = 0; r < n; r += 1) {
      if (r === col) continue;
      const f = a[r]![col]!;
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j += 1) a[r]![j]! -= f * a[col]![j]!;
    }
  }
  return a.map((row) => row.slice(n));
}

export class KalmanFilterXYAH {
  readonly #motion: Matrix;
  readonly #update: Matrix;

  constructor() {
    this.#motion = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => (i === j || j === i + 4 ? 1 : 0)));
    this.#update = Array.from({ length: 4 }, (_, i) => Array.from({ length: 8 }, (_, j) => (i === j ? 1 : 0)));
  }

  /** A track's state from its first `[x, y, a, h]` measurement. */
  initiate(measurement: readonly number[]): { mean: Vector; covariance: Matrix } {
    const h = measurement[3]!;
    return {
      mean: [...measurement, 0, 0, 0, 0],
      covariance: diag([
        2 * STD_POSITION * h, 2 * STD_POSITION * h, 1e-2, 2 * STD_POSITION * h,
        10 * STD_VELOCITY * h, 10 * STD_VELOCITY * h, 1e-5, 10 * STD_VELOCITY * h,
      ]),
    };
  }

  /** One frame ahead: constant velocity plus process noise. */
  predict(mean: Vector, covariance: Matrix): { mean: Vector; covariance: Matrix } {
    const h = mean[3]!;
    const noise = diag([
      STD_POSITION * h, STD_POSITION * h, 1e-2, STD_POSITION * h,
      STD_VELOCITY * h, STD_VELOCITY * h, 1e-5, STD_VELOCITY * h,
    ]);
    const next = this.#motion.map((row) => row.reduce((s, v, j) => s + v * mean[j]!, 0));
    return { mean: next, covariance: add(matmul(matmul(this.#motion, covariance), transpose(this.#motion)), noise) };
  }

  /** The state corrected by an observed `[x, y, a, h]`. */
  update(mean: Vector, covariance: Matrix, measurement: readonly number[]): { mean: Vector; covariance: Matrix } {
    const h = mean[3]!;
    const projectedMean = mean.slice(0, 4);
    const projectedCov = add(matmul(matmul(this.#update, covariance), transpose(this.#update)), diag([STD_POSITION * h, STD_POSITION * h, 1e-1, STD_POSITION * h]));
    const gain = matmul(matmul(covariance, transpose(this.#update)), inverse(projectedCov));
    const innovation = measurement.map((v, i) => v - projectedMean[i]!);
    const nextMean = mean.map((v, i) => v + gain[i]!.reduce((s, g, j) => s + g * innovation[j]!, 0));
    const correction = matmul(matmul(gain, projectedCov), transpose(gain));
    const nextCov = covariance.map((row, i) => row.map((v, j) => v - correction[i]![j]!));
    return { mean: nextMean, covariance: nextCov };
  }
}

/** `[x1, y1, x2, y2]` → `[cx, cy, a, h]`. */
export function xyxyToXyah(box: readonly number[]): number[] {
  const w = Math.max(1e-6, box[2]! - box[0]!);
  const h = Math.max(1e-6, box[3]! - box[1]!);
  return [box[0]! + w / 2, box[1]! + h / 2, w / h, h];
}

/** `[cx, cy, a, h]` → `[x1, y1, x2, y2]`. */
export function xyahToXyxy(xyah: readonly number[]): [number, number, number, number] {
  const [cx, cy, a, h] = [xyah[0]!, xyah[1]!, xyah[2]!, xyah[3]!];
  const w = a * h;
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
}
