from fastapi import APIRouter

from configs.config import DEVICE
from app.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health():
  return {
    "status": "healthy",
    "model": "loaded",
    "device": f"{DEVICE}"
  }