# Data

Place datasets as:

```
data/
  normal/*.jpg|png|dcm
  diseased/*.jpg|png|dcm
  # or for pneumonia/tumor:
  # data/pneumonia/*.jpg
  # data/covid/*.jpg
```

Use helpers:
- `python -m src.split --source data/raw --output data/processed`
- `python src/validation_report.py --path data`
- `src/validation.py: validate_image()` checks format/size/corruption before upload
- DICOM: put `.dcm` files directly under `data/<class>/` - handled by `src/preprocessing.py:load_image` and `src/dicom_loader.py`
