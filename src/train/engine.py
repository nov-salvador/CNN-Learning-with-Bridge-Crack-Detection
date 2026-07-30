import torch
import tqdm

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

  train_loader_tqdm = tqdm(train_loader, desc="Train")

  for batch_idx, (images, labels) in enumerate(train_loader_tqdm, start=1):
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

    train_loss_tqdm = train_loss / batch_idx
    train_accuracy_tqdm = 100 * train_correct/train_total

    train_loader_tqdm.set_postfix(
        loss=f"{train_loss_tqdm:.4f}",
        acc=f"{train_accuracy_tqdm:.2f}%"
    )

  avg_train_loss = train_loss / len(train_loader)
  train_accuracy = 100 * train_correct/train_total

  return avg_train_loss, train_accuracy

def val_one_epoch(model, val_loader, criterion, device):
  model.eval()
  val_loss = 0
  val_correct = 0
  val_total = 0

  val_loader_tqdm = tqdm(val_loader, desc="Train")

  with torch.no_grad():
    for batch_idx , (images, labels) in enumerate(val_loader, start=1):
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      loss = criterion(outputs, labels)

      val_loss += loss.item()
      _, predicted = outputs.max(1)
      val_correct += (predicted == labels).sum().item()
      val_total += labels.size(0)

      val_loss_tqdm = val_loss / batch_idx
      val_accuracy_tqdm = 100 * val_correct/val_total
  
      val_loader_tqdm.set_postfix(
          loss=f"{val_loss_tqdm:.4f}",
          acc=f"{val_accuracy_tqdm:.2f}%"
      )

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
