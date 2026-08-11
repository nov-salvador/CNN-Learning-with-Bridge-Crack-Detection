import torch

from PIL import Image

from configs.config import *

from src.models.model import get_model

from src.inference.predictor import predict_image
from src.inference.model_loader import load_model

def main():
  model = load_model(isTraining=False)

  image_path = SAMPLE_DIR_5
  image = Image.open(image_path)

  if not image_path.exists():
    raise FileNotFoundError(f"{image_path} not found")

  print(f"Predicting using {MODEL_NAME} on image {image_path}") 
  
  result= predict_image(
    model=model,
    image=image,
    class_names=CLASS_NAMES
  )

  print(f"Prediction : {result['prediction']}")
  print(f"Confidence : {result['confidence'] * 100:.2f}%")

if __name__ == "__main__":
  main()