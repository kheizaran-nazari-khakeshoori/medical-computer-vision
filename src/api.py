"""API endpoint for prediction service."""
from fastapi import FastAPI, UploadFile
from PIL import Image
import io
from src.inference import predict
app = FastAPI(title="Radiology Assistant API")
@app.post("/predict")
async def predict_endpoint(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result = predict(image)
    return result
