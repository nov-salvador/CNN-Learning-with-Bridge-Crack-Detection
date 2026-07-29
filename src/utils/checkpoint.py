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

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir, device):
    start_epoch = 0
    best_val_loss = float('inf')
    if checkpoint_dir.exists():
        checkpoint = torch.load(checkpoint_dir, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']

        print(f"Resuming from epoch {start_epoch}")

    return start_epoch, best_val_loss