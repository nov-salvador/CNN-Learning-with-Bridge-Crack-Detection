from fastapi import Request, File, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse

from PIL import Image
from io import BytesIO

import torch
from src.inference.model_loader import load_model
from src.inference.predictor import predict_image
from configs.config import *
from app.schemas import PredictionResponse, HealthResponse

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)):
  if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
    raise HTTPException(status_code=400, detail="Only JPEG, JPG and PNG images are supported.")
  MAX_DIMENSION = 4000
  try:
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    image.load()
  except Exception:
    raise HTTPException(status_code=400, detail="Invalid image file.")
  
  if max(image.size) > MAX_DIMENSION:
    raise HTTPException(
      status_code=400,
      detail="Image dimensions are too large."
    )

  model = request.app.state.model
  
  result = predict_image(
    model=model,
    image=image,
    class_names=CLASS_NAMES
  )

  del image
  del contents

  return{
      "prediction": result['prediction'],
      "confidence": round(result['confidence'] * 100, 2)
  }
