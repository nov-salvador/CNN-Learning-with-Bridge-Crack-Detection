import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from pathlib import Path


from src.dataset.dataset import get_test_loader
from configs.config import *
from src.models.model import get_model
from src.train.engine import test_one_epoch


def main():
  test_loader = get_test_loader()

  model = get_model().to(DEVICE)

  best_model_state = torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=DEVICE)

  model.load_state_dict(best_model_state)

  class_weights =  CRITERION_WEIGHTS.to(DEVICE)
  criterion = nn.CrossEntropyLoss(weight=class_weights)

  metrics = test_one_epoch(
    model=model,
    test_loader=test_loader,
    criterion=criterion,
    device=DEVICE
  )
  print("=" * 40)
  print("Test Results")
  print("=" * 40)
  
  print(f"Accuracy : {metrics['accuracy']:.4f}")
  print(f"Precision: {metrics['precision']:.4f}")
  print(f"Recall   : {metrics['recall']:.4f}")
  print(f"F1 Score : {metrics['f1']:.4f}")
  print(f"Loss     : {metrics['loss']:.4f}")
  print("\nConfusion Matrix")
  print(metrics["cm"])

  print(metrics['classification_report'])

if __name__ == "__main__":
  main()