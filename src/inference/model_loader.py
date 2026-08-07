from src.models.model import get_model
import torch
from configs.config import *


def load_model():
  model = get_model().to(DEVICE)
  checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pth", map_location= DEVICE)
  model.load_state_dict(checkpoint)
  model.eval()

  return model