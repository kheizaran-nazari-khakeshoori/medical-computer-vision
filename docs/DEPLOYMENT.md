# Deployment Guide
## Docker
```bash
docker build -t radiology .
docker run -p 8501:8501 radiology
```
## Streamlit Cloud
Set `app/main.py` as entrypoint.
## Environment
- `MODEL_PATH=models/resnet50_medical.pth`
