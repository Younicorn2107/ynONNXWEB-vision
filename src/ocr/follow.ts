/** Text read on one frame, carried to the frames that follow. */

import { IDENTITY, MotionCompensator, transformPoint, transformScale, type Affine, type MotionCompensatorOptions } from "../gmc.js";
import type { RasterImage } from "../preprocess.js";
import type { TextBlock, TextLine } from "./sceneText.js";

/** `a` after `b`: a point goes through `b`, then `a`. */
function compose(a: Readonly<Affine>, b: Readonly<Affine>): Affine {
  return [
    a[0] * b[0] + a[1] * b[3],
    a[0] * b[1] + a[1] * b[4],
    a[0] * b[2] + a[1] * b[5] + a[2],
    a[3] * b[0] + a[4] * b[3],
    a[3] * b[1] + a[4] * b[4],
    a[3] * b[2] + a[4] * b[5] + a[5],
  ];
}

function invert(m: Readonly<Affine>): Affine {
  const det = m[0] * m[4] - m[1] * m[3];
  if (det === 0) return [...IDENTITY];
  const a = m[4] / det;
  const b = -m[1] / det;
  const c = -m[3] / det;
  const d = m[0] / det;
  return [a, b, -(a * m[2] + b * m[5]), c, d, -(c * m[2] + d * m[5])];
}

export interface TextFollowerOptions extends MotionCompensatorOptions {
  /** Frames remembered, to place blocks read this many frames ago (300). */
  memory?: number;
}

export class TextFollower {
  readonly #motion: MotionCompensator;
  readonly #memory: number;
  /** Per frame index: the motion from the first frame tracked to that one. */
  readonly #since = new Map<number, Affine>();
  #last: Affine = [...IDENTITY];
  #lastIndex = -1;

  constructor(options: TextFollowerOptions = {}) {
    const { memory = 300, ...motion } = options;

    this.#motion = new MotionCompensator({ downscale: 1, maxFeatures: 40, window: 11, levels: 2, ...motion });
    this.#memory = memory;
  }

  /**
   * A frame shown, in order. `frame` may be a reduced copy: `scale` is the blocks' pixels per pixel
   * of it (a 320-wide copy of a 1280-wide video: 4), which keeps this to a few milliseconds a
   * frame.
   */
  track(frame: RasterImage, index: number, scale = 1): void {
    const m = this.#motion.estimate(frame);
    const step: Affine = [m[0], m[1], m[2] * scale, m[3], m[4], m[5] * scale];
    this.#last = this.#lastIndex < 0 ? [...IDENTITY] : compose(step, this.#last);
    this.#lastIndex = index;
    this.#since.set(index, this.#last);
    for (const key of this.#since.keys()) {
      if (key > index - this.#memory) break;
      this.#since.delete(key);
    }
  }

  /** The motion from frame `index` to the last frame tracked (identity when `index` is unknown). */
  motionSince(index: number): Affine {
    const then = this.#since.get(index);
    return then && index !== this.#lastIndex ? compose(this.#last, invert(then)) : [...IDENTITY];
  }

  /** Blocks read on frame `index`, where the camera has taken them since. */
  place(blocks: readonly TextBlock[], index: number): TextBlock[] {
    const m = this.motionSince(index);
    if (m.every((v, i) => v === IDENTITY[i])) return [...blocks];
    const scale = transformScale(m);
    const turn = Math.atan2(m[3], m[0]);
    const move = (line: TextLine): TextLine => ({
      ...line,
      quad: line.quad.map(([x, y]) => transformPoint(m, x, y)) as TextLine["quad"],
      angle: line.angle + turn,
    });
    return blocks.map((block) => {
      const lines = block.lines.map(move);
      const xs = lines.flatMap((l) => l.quad.map((p) => p[0]));
      const ys = lines.flatMap((l) => l.quad.map((p) => p[1]));
      const x = Math.min(...xs);
      const y = Math.min(...ys);
      return {
        ...block,
        lines,
        rect: { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y },
        angle: block.angle + turn,
        lineHeight: block.lineHeight * scale,
      };
    });
  }

  /** A new clip or scene: nothing carried over. */
  reset(): void {
    this.#motion.reset();
    this.#since.clear();
    this.#last = [...IDENTITY];
    this.#lastIndex = -1;
  }
}
