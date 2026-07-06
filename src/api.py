"""API endpoint for prediction service."""

import io
import time

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from PIL import Image

from src.inference import predict

app = FastAPI(title="Radiology Assistant API", version="1.0.0")
_request_count = 0
_start = time.time()


@app.get("/health")
async def health():
    return {"status": "ok", "model": "resnet50", "uptime_s": int(time.time() - _start)}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    # simple prometheus-style metrics without extra dep
    uptime = int(time.time() - _start)
    return f"# HELP api_requests_total total requests\n# TYPE api_requests_total counter\napi_requests_total {_request_count}\n# HELP api_uptime_seconds uptime\napi_uptime_seconds {uptime}\n"


@app.post("/predict")
async def predict_endpoint(file: UploadFile):
    global _request_count
    _request_count += 1
    # basic validation: limit 10MB
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty file")
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="file too large (max 10MB)")
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid image: {e}")
        # quality check soft warning header could be added
        result = predict(image)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")
