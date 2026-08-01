from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

from configs.config import NUM_CLASSES

def get_model():
  model = resnet18(weights=ResNet18_Weights.DEFAULT)
  for params in model.parameters():
    params.requires_grad = False

  for params in model.layer3.parameters():
    params.requires_grad = True

  for params in model.layer4.parameters():
    params.requires_grad = True
    
  num_in_feature = model.fc.in_features

  model.fc = nn.Linear(num_in_feature, NUM_CLASSES)
    
  for params in model.fc.parameters():
    params.requires_grad = True
  
  return model
