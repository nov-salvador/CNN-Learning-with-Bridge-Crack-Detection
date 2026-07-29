import csv

from torch.utils.tensorboard import SummaryWriter

from configs.config import *

class TrainingLogger():
  def __init__(self):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_LOG_FILE.exists():
      with open(CSV_LOG_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
          "epoch",
          "train_loss",
          "val_loss",
          "train_accuracy",
          "val_accuracy",
          "learning_rate"
        ])

    self.writer = SummaryWriter(log_dir=TENSORBOARD_DIR)

  def log(
    self, 
    epoch,
    train_loss,
    val_loss,
    train_accuracy,
    val_accuracy,
    learning_rate  
  ):
    with open(CSV_LOG_FILE, "a", newline="") as file:
      writer = csv.writer(file)
      writer.writerow([
        epoch,
        train_loss,
        val_loss,
        train_accuracy,
        val_accuracy,
        learning_rate  
      ])

    self.writer.add_scalar(
      "Loss/Train",
      train_loss,
      epoch
    )

    self.writer.add_scalar(
      "Loss/Validation",
      val_loss,
      epoch
    )

    self.writer.add_scalar(
      "Accuracy/Train",
      train_accuracy,
      epoch
    )

    self.writer.add_scalar(
      "Accuracy/Validation",
      val_accuracy,
      epoch
    )

    self.writer.add_scalar(
      "Learning Rate",
      learning_rate,
      epoch
    )

  def close(self):
    self.writer.close()

if __name__ == "__main__":
  logger = TrainingLogger()
  logger.log(
    epoch=1,
    train_loss=0.50,
    val_loss=0.45,
    train_accuracy=0.81,
    val_accuracy=0.85,
    learning_rate=0.002
  )
  print("CSV updated!")