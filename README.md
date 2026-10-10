# onnxweb-vision

Computer vision in the browser: YOLO detection, segmentation, pose, tracking, and reading text in images.

- **YOLO** (Ultralytics YOLO26 and others): detection, instance and semantic segmentation, depth, classification, pose, oriented boxes — preprocessing on the GPU, any input size.
- **Video**: tracking (ByteTrack/BoT-SORT with camera motion compensation and re-identification), sliced inference for small objects.
- **Text in images**: PP-OCR detection and recognition, lines followed from frame to frame, translated and painted back in place.
- **Drawing**: boxes, masks, keypoints, blurring.

```ts
import * as ort from "onnxruntime-web";
import { YOLORunner, loadImage, annotate } from "@youni/onnxweb-vision";

const runner = await YOLORunner.load({ ort, Tensor: ort.Tensor, model: "yolo26n", backends: "webgpu" });
const image = await loadImage("/photo.png");
const result = await runner.run(image, { conf: 0.25 });
```

Weights: Ultralytics YOLO is AGPL-3.0 (a web service using it must publish its modified sources); PP-OCR Apache-2.0.

## Install

Builds against `@youni/onnxweb-core`:

```sh
npm install
npm run build   # dist/
npm test        # unit tests (Node, onnxruntime-web on WebAssembly)
```

## Licence

Apache-2.0 for the code. Model weights keep their own licence (listed above where it matters).
