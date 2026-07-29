import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import *
from src.models.custom_model import SimpleCNN
from src.models.model import get_model
from src.dataset.dataset import get_train_loader, get_val_loader

from src.train.engine import train_one_epoch, val_one_epoch

from src.utils.checkpoint import save_best_model, save_checkpoint, load_checkpoint
from src.utils.logger import TrainingLogger

def main():
  train_loader = get_train_loader()
  val_loader = get_val_loader()

  model = get_model().to(DEVICE)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=LEARNING_RATE
  )
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode='min',
    factor=0.1,
    patience=3
  )

  logger = TrainingLogger()

  counter = 0
  patience = 3

  if RESUME_TRAINING:
    start_epoch, best_val_loss = load_checkpoint(
      model=model,
      optimizer=optimizer,
      scheduler=scheduler,
      checkpoint_dir= CHECKPOINT_DIR / "last_checkpoint.pth"
    )
  else:
    start_epoch = 0 
    best_val_loss = float('inf')

  for epoch in range(start_epoch, EPOCHS):

    #Training Starts Here
    avg_train_loss, train_accuracy = train_one_epoch(
      model=model,
      train_loader=train_loader,
      criterion=criterion,
      optimizer=optimizer,
      device=DEVICE
    )

    #Validation Starts Here
    avg_val_loss, val_accuracy = val_one_epoch(
      model=model,
      val_loader=val_loader,
      criterion=criterion,
      device=DEVICE
    )

    scheduler.step(avg_val_loss)

    #Saving the best model
    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      counter = 0
      save_best_model(model=model, checkpoint_dir=CHECKPOINT_DIR)
    else:
      counter += 1

   #Saving a checkpoint
    save_checkpoint(
      epoch=epoch,
      model=model,
      optimizer=optimizer,
      scheduler=scheduler,
      best_val_loss=best_val_loss,
      checkpoint_dir=CHECKPOINT_DIR
    )

    #Logging the metrics
    logger.log(
      epoch=epoch + 1,
      train_loss=avg_train_loss,
      val_loss=avg_val_loss,
      train_accuracy=train_accuracy,
      val_accuracy=val_accuracy,
      learning_rate=optimizer.param_groups[0]['lr']
    )

    print(
      f"Epoch [{epoch+1}/{EPOCHS}] "
      f"Learning Rate [{optimizer.param_groups[0]['lr']:.6f}] "
      f"Train Loss: {avg_train_loss:.4f} "
      f"Train Acc: {train_accuracy:.2f}% "
      f"Val Loss: {avg_val_loss:.4f} "
      f"Val Acc: {val_accuracy:.2f}%"
    )

    if counter >= patience:
      print("Early Stopping")
      break
  logger.close()

if __name__ == "__main__":
  main()