import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from dataset import BridgeCrackDataset
from models.model import SimpleCNN

#Configuration
TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
best_val_loss = float('inf')
patience = 3
counter = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Dataset and Dataloader
train_dataset = BridgeCrackDataset(root_dir=TRAIN_DIR)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = BridgeCrackDataset(root_dir=VAL_DIR)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#Model
# model = SimpleCNN().to(device=device) # used for scratch
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 2)

#Freeze the backbone
for param in model.parameters():
  param.requires_grad = False

#Unfreeze the FC 
for param in model.fc.parameters():
  param.requires_grad = True

#Loss
criterion = nn.CrossEntropyLoss()

#Optimizer and LR scheduler
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) #used in scratch model
optimizer = torch.optim.Adam(model.fc.parameters(), lr= LEARNING_RATE) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)

#Training
for epoch in range(EPOCHS):
  model.train()

  train_loss = 0
  train_correct = 0
  train_total = 0

  for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    train_correct += (predicted == labels).sum().item()
    train_total += labels.size(0)  

  avg_train_loss = train_loss / len(train_loader)
  train_accuracy = 100 * train_correct/train_total

  model.eval()
  val_loss = 0
  val_correct = 0
  val_total = 0

  with torch.no_grad():
    for images, labels in val_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)

      val_loss += loss.item()
      _, predicted = outputs.max(1)
      val_correct += (predicted == labels).sum().item()
      val_total += labels.size(0)

  avg_val_loss = val_loss / len(val_loader)
  val_accuracy = 100 * val_correct/val_total

  if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    counter = 0

    torch.save(model.state_dict(), "checkpoints/best_model.pth")

  else:
    counter += 1

  scheduler.step()
  checkpoints = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_val_loss": best_val_loss
  }
  torch.save(checkpoints, "checkpoints/last_checkpoint.pth")
  print(
    f"Epoch [{epoch+1}/{EPOCHS}] "
    f"Learning Rate [{optimizer.param_groups[0]['lr']}]"
    f"Train Loss: {avg_train_loss:.4f} "
    f"Train Acc: {train_accuracy:.2f}% "
    f"Val Loss: {avg_val_loss:.4f} "
    f"Val Acc: {val_accuracy:.2f}%"
  )

  if counter >= patience:
    print("Early Stopping")
    break 