# Grad-CAM Usage
```python
from src.gradcam import GradCAM
from src.model import get_model
model = get_model()
cam = GradCAM(model, model.layer4[-1])
heatmap = cam.generate(tensor)
```
Overlay with `src.visualization.overlay_heatmap`.
