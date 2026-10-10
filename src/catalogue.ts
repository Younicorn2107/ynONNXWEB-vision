/** Which YOLO models a page can load, and what task each serves. */

import type { Task } from "./types.js";

export interface PublishedYOLO {
  /** The short name callers use (`yolo26n`, `yolo26n-pose`). */
  name: string;
  repo: string;
  task: Task;
  license: "AGPL-3.0";
  category: "copyleft";
  /**
   * Precisions offered, the default first: float16 (half the download, the same boxes); float32.
   * The int8 graphs (dynamic quantization of a convolutional net) lose too much and run slowly on
   * WebGPU: not offered.
   */
  precisions: readonly string[];
}

const SIZES_26 = ["n", "s", "m", "l", "x"] as const;

export const PUBLISHED_MODELS: readonly PublishedYOLO[] = [
  ...SIZES_26.map((size) => ({
    name: `yolo26${size}`,
    repo: `onnx-community/yolo26${size}-ONNX`,
    task: "detect" as const,
    license: "AGPL-3.0" as const,
    category: "copyleft" as const,
    precisions: ["float16", "float32"],
  })),
  ...SIZES_26.map((size) => ({
    name: `yolo26${size}-pose`,
    repo: `onnx-community/yolo26${size}-pose-ONNX`,
    task: "pose" as const,
    license: "AGPL-3.0" as const,
    category: "copyleft" as const,
    precisions: ["float16", "float32"],
  })),
];

/** The default model: the smallest YOLO26 detector. */
export const DEFAULT_MODEL = "yolo26n";

/** A published model by its short name, or by its repository. */
export function publishedModel(name: string): PublishedYOLO | null {
  const key = name.trim().toLowerCase();
  return PUBLISHED_MODELS.find((m) => m.name === key || m.repo.toLowerCase() === key) ?? null;
}

const TASK_SUFFIXES: readonly (readonly [string, Task])[] = [
  ["-seg", "segment"],
  ["-sem", "semantic"],
  ["-depth", "depth"],
  ["-pose", "pose"],
  ["-cls", "classify"],
  ["-obb", "obb"],
];

/**
 * The task of a model reference, from its name (`yolo26n-seg` → `segment`). The file's stem first;
 * when it carries no suffix (`org/yolov8n-pose:onnx/model.onnx`), the whole reference.
 */
export function detectTask(model: string): Task {
  const stem = (model.split(/[\\/:]/).pop() ?? model).replace(/\.[^.]+$/, "").toLowerCase();
  for (const [suffix, task] of TASK_SUFFIXES) if (stem.includes(suffix)) return task;
  const full = model.toLowerCase();
  for (const [suffix, task] of TASK_SUFFIXES) if (full.includes(suffix)) return task;
  return "detect";
}

/** Ultralytics' metadata `task` names, as this library's. */
export function taskFromMetadata(value: string | undefined): Task | null {
  switch (value?.trim().toLowerCase()) {
    case "detect":
      return "detect";
    case "segment":
      return "segment";
    case "pose":
      return "pose";
    case "classify":
      return "classify";
    case "obb":
      return "obb";
    case "depth":
      return "depth";
    case "semantic":
    case "semseg":
      return "semantic";
    default:
      return null;
  }
}
