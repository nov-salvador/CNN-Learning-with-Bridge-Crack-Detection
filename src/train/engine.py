import torch
from sklearn.metrics import (
  accuracy_score, 
  precision_score, 
  recall_score, 
  f1_score,
  confusion_matrix,
  classification_report
)

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

def test_one_epoch(model, test_loader, criterion, device):
  model.eval()

  all_labels = []
  all_predictions = []
  test_loss = 0

  with torch.no_grad():
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      loss = criterion(outputs, labels)

      test_loss += loss.item()
      _, predicted = outputs.max(1)
      avg_test_loss = test_loss / len(test_loader)
      all_labels.extend(labels.cpu().numpy())
      all_predictions.extend(predicted.cpu().numpy())

  metrics = {
    "loss": avg_test_loss,
    "accuracy": accuracy_score(all_labels, all_predictions),
    "precision": precision_score(all_labels, all_predictions),
    "recall": recall_score(all_labels, all_predictions),
    "f1": f1_score(all_labels, all_predictions),
    "cm": confusion_matrix(all_labels, all_predictions)
  }

  return metrics
