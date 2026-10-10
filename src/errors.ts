/** Errors only the YOLO library raises, over the core's hierarchy. */

import { YnONNXError, registerErrors } from "@youni/onnxweb-core/core";

export class YnONNXYOLOError extends YnONNXError {
  static override readonly code: string = "YnONNXYOLOError";
}

/** Sliced inference asked with impossible tiles, or on a task it does not serve. */
export class SAHIError extends YnONNXYOLOError {
  static override readonly code: string = "SAHIError";
}

/** A tracker asked for an unknown algorithm, or without what it needs. */
export class TrackerError extends YnONNXYOLOError {
  static override readonly code: string = "TrackerError";
}

/** The CLIP text encoder of open-vocabulary models failed. */
export class TextEncoderError extends YnONNXYOLOError {
  static override readonly code: string = "TextEncoderError";
}

registerErrors(YnONNXYOLOError, SAHIError, TrackerError, TextEncoderError);
