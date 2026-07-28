import torch
from pathlib import Path

#Paths
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "processed/train"
VAL_DIR = DATA_DIR / "processed/val"
TEST_DIR = DATA_DIR / "processed/test"

#Image Size
IMAGE_SIZE = 224

#Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

#Model
NUM_CLASSES = 2

#Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Outputs
OUTPUT_DIR = Path("outputs")
PREDICTION_DIR = OUTPUT_DIR / "predictions"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

#Checkpoint
CHECKPOINT_DIR = Path("checkpoints")