# API Documentation

## Preprocessing
- `src.preprocessing.load_image(path)` -> PIL Image (supports JPG/PNG/DICOM)
- `src.preprocessing.preprocess_image(image)` -> torch.Tensor

## Model
- `src.model.get_model(num_classes)` -> ResNet50

## Inference
- `src.inference.predict(image)` -> {label, confidence, probabilities}
- `src.inference.predict_with_heatmap(image)` -> includes heatmap

## Grad-CAM
- `src.gradcam.GradCAM(model, target_layer).generate(tensor)` -> heatmap

## Dataset
- `src.dataset.MedicalImageDataset(root_dir)` expects `data/<class>/*.jpg`

## Reports
- `src.report.generate_report(patient_id, prediction, output_path)`

## Run App
```bash
streamlit run app/main.py
```
