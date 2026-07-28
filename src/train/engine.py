import torch

def train_one_epoch(model, train_loader, criterion, optimizer, device):
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

  return avg_train_loss, train_accuracy

def val_one_epoch(model, val_loader, criterion, device):
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
  return avg_val_loss, val_accuracy
