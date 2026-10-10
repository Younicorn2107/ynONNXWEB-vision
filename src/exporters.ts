/**
 * Results written out: JSON and CSV (the core's renderers, over this library's records), YOLO label
 * files and PASCAL VOC XML. An annotated image is `annotate()` on a canvas, then the canvas's own
 * encoder.
 */

import { ExportError, toCsv, toJson } from "@youni/onnxweb-core/core";

import { recordsOf, type VideoResult, type YOLOResult } from "./types.js";

/**
 * YOLO label format, one line per box: `class_id cx cy w h`, normalized to `[0, 1]` — what
 * Ultralytics retrains on.
 */
export function toYoloTxt(result: YOLOResult): string {
  const [h, w] = result.imageShape;
  if (h <= 0 || w <= 0) throw new ExportError("YOLO labels need the image size (imageShape) to normalize the boxes");
  const boxes = "boxes" in result ? result.boxes : [];
  const lines = boxes.map((b) => {
    const cx = (b.x1 + b.x2) / 2 / w;
    const cy = (b.y1 + b.y2) / 2 / h;
    return `${b.classId} ${cx.toFixed(6)} ${cy.toFixed(6)} ${((b.x2 - b.x1) / w).toFixed(6)} ${((b.y2 - b.y1) / h).toFixed(6)}`;
  });
  return lines.length > 0 ? `${lines.join("\n")}\n` : "";
}

const escapeXml = (text: string) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Python's `round` to an integer: halves to the even neighbour. */
function roundEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff !== 0.5) return Math.round(value);
  return floor % 2 === 0 ? floor : floor + 1;
}

/** PASCAL VOC XML (LabelImg, CVAT), as ElementTree writes it. */
export function toVocXml(result: YOLOResult): string {
  const [h, w] = result.imageShape;
  const source = result.source ?? "unknown";
  const parts = source.split(/[\\/]/);
  const filename = parts[parts.length - 1] ?? source;
  const folder = parts.length > 1 ? parts[parts.length - 2]! : "";
  const leaf = (tag: string, text: string, depth: number) => `${"  ".repeat(depth)}<${tag}>${escapeXml(text)}</${tag}>`;
  const lines = [
    "<?xml version='1.0' encoding='utf-8'?>",
    "<annotation>",
    folder ? leaf("folder", folder, 1) : "  <folder />",
    leaf("filename", filename, 1),
    leaf("path", source, 1),
    "  <size>",
    leaf("width", String(w), 2),
    leaf("height", String(h), 2),
    leaf("depth", "3", 2),
    "  </size>",
    leaf("segmented", "0", 1),
  ];
  for (const b of "boxes" in result ? result.boxes : []) {
    lines.push(
      "  <object>",
      leaf("name", b.className, 2),
      leaf("pose", "Unspecified", 2),
      leaf("truncated", "0", 2),
      leaf("difficult", "0", 2),
      "    <bndbox>",
      leaf("xmin", String(roundEven(b.x1)), 3),
      leaf("ymin", String(roundEven(b.y1)), 3),
      leaf("xmax", String(roundEven(b.x2)), 3),
      leaf("ymax", String(roundEven(b.y2)), 3),
      "    </bndbox>",
      "  </object>",
    );
  }
  lines.push("</annotation>");
  return lines.join("\n");
}

/** JSON with Python's field names; masks become their shape, not megabytes of zeros. */
export function toYoloJson(result: YOLOResult | VideoResult | readonly YOLOResult[]): string {
  return toJson(result, { policy: "shape" });
}

/** One row per box (or prediction), Python's columns. */
export function toYoloCsv(result: YOLOResult | VideoResult): string {
  return toCsv(recordsOf(result));
}

export type ExportFormat = "json" | "csv" | "txt" | "xml";

/** A result in one of the formats, as a string. */
export function exportResult(result: YOLOResult | VideoResult, format: ExportFormat | string): string {
  switch (format.toLowerCase().replace(/^\./, "")) {
    case "json":
      return toYoloJson(result);
    case "csv":
      return toYoloCsv(result);
    case "txt":
      if ("frames" in result) throw new ExportError("YOLO labels are per image: export each frame's result");
      return toYoloTxt(result);
    case "xml":
      if ("frames" in result) throw new ExportError("VOC XML is per image: export each frame's result");
      return toVocXml(result);
    default:
      throw new ExportError(`unknown export format '${format}' (known: csv, json, txt, xml; images: annotate() on a canvas)`);
  }
}

const MIME: Readonly<Record<string, string>> = { json: "application/json", csv: "text/csv", txt: "text/plain", xml: "application/xml" };

/** The same, as a Blob ready for a download link. */
export function exportResultBlob(result: YOLOResult | VideoResult, format: ExportFormat | string): Blob {
  const name = format.toLowerCase().replace(/^\./, "");
  return new Blob([exportResult(result, name)], { type: `${MIME[name] ?? "text/plain"};charset=utf-8` });
}
