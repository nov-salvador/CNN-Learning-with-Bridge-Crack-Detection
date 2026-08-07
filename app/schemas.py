from pydantic import BaseModel

class PredictionResponse(BaseModel):
  prediction: str
  confidence: float 

class HealthResponse(BaseModel):
  status: str
  model: str
  device: str