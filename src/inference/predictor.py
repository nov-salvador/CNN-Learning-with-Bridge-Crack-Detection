import torch
from PIL import Image

from configs.config import *

from src.models.model import get_model
from src.dataset.dataset import val_transform

def predict_image(model, image, class_names):
  image = image.convert("RGB")
  image = val_transform(image)
  image = image.unsqueeze(0)
  image = image.to(DEVICE)

  model.eval()
  with torch.no_grad():
  
    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(
      probabilities,
      dim=1
    )
    return {
      "prediction" : class_names[predicted.item()], 
      "confidence": confidence.item(),
      "probabilities": probabilities.squeeze().cpu()
    }