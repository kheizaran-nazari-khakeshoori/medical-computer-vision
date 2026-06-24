"""API endpoint for prediction service."""
import io

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

from src.inference import predict

app = FastAPI(title="Radiology Assistant API", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "resnet50"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile):
    if not file.content_type or not file.content_type.startswith(("image/", "application/")):
        # allow but warn
        pass
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty file")
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid image: {e}")
        result = predict(image)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")
