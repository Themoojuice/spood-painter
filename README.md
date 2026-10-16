# spood-painter

macro pics to artwork sticks. This is **spoodpaint** in your phone browser.

Turns a macro photo of a jumping spider into a gouache-and-ink painting, entirely on your phone.
No server, no API fees: the cutout networks and the painting code run in the browser.

**Use it:** open the site in Chrome on Android, then menu ⋮ → **Add to Home screen** (or **Install app**).
After that, in your gallery: **Share → spoodpaint**. Tap the spider's big front eyes, hit **Paint**.

First launch downloads about 240 MB once (two cutout networks, about 185 MB, plus the Python runtime).
Afterwards it works offline.

## How it works

| Step | Runs in | What |
|---|---|---|
| Cutout | onnxruntime-web (WebGPU, falls back to multi-threaded WebAssembly) | ISNet (rough outline) + BiRefNet lite (edges), BiRefNet run twice: whole photo, then a close-up crop; fused exactly as in `cutout.py` |
| Painting | Pyodide (Python + numpy/scipy/scikit-image/OpenCV compiled to WebAssembly) | the unchanged `spoodpaint.py`, `eyes.py`, `limbs.py` from v0.4 |
| Share target, offline, threads | `sw.js` service worker | receives shared photos, caches the app, adds the cross-origin-isolation headers GitHub Pages can't set |

### Model conversion notes
- **Deformable convolution**: BiRefNet's decoder uses torchvision's `deform_conv2d`. It is exported as the
  standard ONNX `DeformConv` op (opset 19), which onnxruntime-web runs exactly (max error 1e-7).
  An earlier version rebuilt it from `GridSample` + 1x1 convs; that was exact too, but onnxruntime-web keeps
  every intermediate tensor of a run alive, and its 49-step loops needed >4 GB.
- **Low-memory decoder tail** (`tools/export/lowmem.py`): the last stage is linear, so the 1x1 output conv is
  moved before the upsample and the image branch's two 3x3 convs are collapsed into one 5x5 conv. Nothing
  big is computed at full resolution any more. Max difference vs the original: 2e-5.
- Input is a fixed 1120x1120 square (training also resized to a fixed size). 896 was tested and rejected:
  it dropped half of a camouflaged spider's abdomen on a test photo. 1120 matched the laptop's masks at
  98% overlap on both test photos.
- Weights are stored as float16 with a Cast back to float32 at load: half the download, float32 maths.

### Memory
WebAssembly memory never shrinks and is capped at 4 GB per worker. The cutout networks run in their own worker
(`networker.js`, ~2.8 GB peak at 1120 px on the CPU path), which is terminated after every cutout so the
painting (Python, its own worker) gets the memory back. Photos larger than 4000 px are scaled down first
(50 MP -> 12 MP).

## Files
- `index.html`, `app.js`: UI (photo intake, pinch-zoom eye taps, results)
- `worker.js`: loads Pyodide and runs the pipeline
- `networker.js`: onnxruntime-web + the two cutout networks (created per cutout, then terminated)
- `sw.js`: service worker (share target, offline cache, isolation headers)
- `py/`: the Python pipeline (`web_pipeline.py` is the browser glue; the rest is v0.4 plus a `VIBRANCE` setting)
- `models/`: the two ONNX models
- `tools/`: how the models were made (`export/`) and how this site is assembled (`build.py`)
- `pyodide/`, `ort/`: self-hosted runtimes (same origin keeps isolation simple and makes offline work)

Licences: BiRefNet (MIT), ISNet / rembg weights (Apache-2.0), Pyodide (MPL-2.0), onnxruntime-web (MIT).
