# compare: score a render against a photo and a target painting

Aligns a painting/render to its source photo (SIFT similarity, or silhouette moments + ECC when features fail,
then smooth Farneback flow), cuts the spider out with ISNet, and scores lightness detail bands
(0.75-12 px difference-of-Gaussians): "same marks" = band correlation, "same places" = correlation of
local detail energy. See the "Spoodpaint target-style specification" doc for targets.

- `pair.py`: find which photo each painting came from (feature inliers).
- `analyse_lib.py`: mask, alignment and band scoring (`compare(painting, photo)`).
- `control.py`: scores Gemini paintings and a spoodpaint render of the same photo side by side.

Photo/painting pairs are not in this repository; point `U` and `PAIRS` at your own files.
Needs: opencv-python, numpy, onnxruntime, and models/isnet.onnx (rembg isnet-general-use).
