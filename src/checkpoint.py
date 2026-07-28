import torch

def save_best_model(model, checkpoint_dir):
    torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
    print("Best model saved!")

def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, checkpoint_dir):
  checkpoints = {
          "epoch": epoch + 1,
          "model_state_dict": model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
          "scheduler_state_dict": scheduler.state_dict(),
          "best_val_loss": best_val_loss
  }
  torch.save(checkpoints, checkpoint_dir / "last_checkpoint.pth")
