from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from configs.config import *

train_transform = transforms.Compose([
  transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(15),
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  )
])

val_transform = transforms.Compose([
  transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  )
])

def get_train_loader():
  train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)

  train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
  )
  return train_loader
  

def get_val_loader():
  val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transform)

  val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
  )
  return val_loader

def get_test_loader():
  test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=val_transform)

  test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
  )
  return test_loader