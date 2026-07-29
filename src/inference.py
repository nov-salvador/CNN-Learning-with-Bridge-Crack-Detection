import torch
from PIL import Image

from configs.config import *

from src.models.model import get_model
from src.dataset.dataset import val_transform

def predict_image(model, image_path, class_names):
  image =  Image.open(image_path).convert("RGB")
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
    return (class_names[predicted.item()], confidence.item())

def main():
  CLASS_NAMES = {
    0: "crack",
    1: "no_crack"
  }
  model = get_model().to(DEVICE)

  checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=DEVICE)

  model.load_state_dict(checkpoint["model_state_dict"])

  image_path = SAMPLE_DIR_1


  prediction, confidence = predict_image(
    model=model,
    image_path=image_path,
    class_names=CLASS_NAMES
  )

  print(f"Prediction : {prediction}")
  print(f"Confidence : {confidence * 100:.2f}%")

if __name__ == "__main__":
  main()