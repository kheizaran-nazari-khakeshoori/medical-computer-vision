# Grad-CAM Explainability

## Why
Black-box predictions lack clinical trust. Grad-CAM provides heatmap localization.

## How it Works
- Hook activations + gradients on `model.layer4[-1]` (`src/gradcam.py:19`)
- `generate()` (`src/gradcam.py:29`): forward → score.backward() → `weights = grads.mean((1,2))` → `cam = ReLU( Σ weights * activations )` → resize to 224×224
- Overlay: `cv2.COLORMAP_JET` + alpha blend (`src/visualization.py:8`)

```python
from src.inference import predict_with_heatmap
from PIL import Image
res = predict_with_heatmap(Image.open("data/normal/sample.jpg"))
heatmap = res["heatmap"]  # 224x224 float [0,1]
overlay = overlay_heatmap(image, heatmap)
```

## Usage in App
`app/main.py:60` — `predict_with_heatmap(image, model=cached_model)` + `overlay_heatmap` shown if `settings["show_heatmap"]`.

## Limitations
- Coarse (7×7 feature map) → not segmentation; see `src/experimental/segmentation_model.py` for future UNet.
- Requires correct target layer; for EfficientNet use `model.features[-1]`.

