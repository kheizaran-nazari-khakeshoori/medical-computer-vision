# Deployment Guide
## Docker (recommended)
```bash
docker compose up --build  # http://localhost:8501
docker build -t radiology . && docker run -p 8501:8501 -p 8000:8000 radiology
```
Healthcheck: `GET /health` and `GET /metrics` (`src/api.py:12`).

## Streamlit Cloud
Set `app/main.py` as entrypoint. Add `MODEL_PATH` secret.

## Environment
Copy `.env.example` to `.env`:
```
MODEL_PATH=models/resnet50_medical.pth
DATA_DIR=data
```
See `src/config.py:15`.

## API
```bash
uvicorn src.api:app --port 8000
curl -F "file=@data/normal/sample.jpg" http://localhost:8000/predict
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

## ONNX & Benchmark
```bash
python -m src.export_onnx   # -> models/model.onnx
python -m src.benchmark     # latency report
python -m src.model_compress # quantization demo
```

## CI
`pyproject.toml:1` + `.github/workflows/ci.yml:1`, local `make lint && make test`.

