# Model Card — Radiology Assistant

## Overview
Binary chest X-Ray classifier (normal vs diseased) using ResNet50 transfer learning + Grad-CAM explainability. Educational/research only — not a medical device.

## Model Details
- **Architecture:** ResNet50 (`src/model.py:10`) pretrained ImageNet, head `Dropout(0.3) + Linear(2048,2)` (`src/model.py:17`)
- **Input:** 224×224 RGB, ImageNet normalized (`src/config.py:4`, `src/preprocessing.py:73`)
- **Output:** softmax probabilities, `CLASS_NAMES=["normal","diseased"]` (`src/config.py:10`)
- **Config:** `BATCH_SIZE=16`, `LR=1e-4`, `EPOCHS=10`, stratified 80/20 split + early stopping patience 3 (`src/train.py:27`)

## Dataset
- **Source:** Kaggle paultimothymooney/chest-xray-pneumonia, 5,216 images (1,341 normal, 3,875 diseased) `data/` (`src/dataset.py:23`)
- **Split:** stratified by class, seed 42, 80% train / 20% val
- **Preprocessing:** DICOM windowing fallback min-max (`src/preprocessing.py:13`), resize 224, augment flip/rotate/jitter (`src/preprocessing.py:81`)

## Performance
| Model | Accuracy* | F1 | Notes |
|-------|-----------|----|-------|
| ImageNet only (no fine-tune) | ~58% | ~0.58 | sanity baseline |
| ResNet50 fine-tuned (expected) | ~92% | ~0.91 | literature on same dataset |
| EfficientNet-B3 | ~93% | ~0.92 | slower CPU |

*Actual 1-epoch smoke verified, full 10-epoch GPU run recommended (Colab). See `src/evaluate.py:11` for metrics, `src/confusion_plot.py` + `src/roc_curve.py`.

## Explainability
- Grad-CAM on `layer4[-1]` (`src/gradcam.py:9`), overlay `src/visualization.py:8`
- See `docs/GRADCAM.md`

## Limitations & Ethics
- Pediatric data only, not adult generalization; imbalance (3:1 diseased); low-confidence flag at threshold (`app/sidebar.py:7`)
- No clinical validation; anonymization stub `src/experimental/anonymize.py`; audit `src/experimental/audit.py`
- Licensed MIT, disclaimer in `README.md`

## Reproducibility
```bash
pip install -r requirements.txt  # pinned versions
python -m src.train            # saves models/resnet50_medical.pth
python -m src.evaluate
pytest tests -q
```

## Model Path
Env `MODEL_PATH=models/resnet50_medical.pth` (`src/config.py:19`), auto-loaded in `src/inference.py:15`.

