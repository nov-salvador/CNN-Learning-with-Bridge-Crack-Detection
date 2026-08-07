from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from PIL import Image
from io import BytesIO

import torch
from src.inference.model_loader import load_model
from src.inference.predictor import predict_image
from configs.config import *
from app.schemas import PredictionResponse, HealthResponse
from app.routes.predict import router as predict_router
from app.routes.health import router as health_router

app = FastAPI(
  title="Bridge Crack Detection API",
  description="CNN model for detecting bridge cracks",
  version="1.0.0"
)

app.state.model = load_model()
app.include_router(predict_router)
app.include_router(health_router)

@app.get("/")
def home():
  return {
    "message": "Bridge Crack Detection API is running!"
  }
