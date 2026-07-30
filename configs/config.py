import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

#Paths of data
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "processed/train"
VAL_DIR = DATA_DIR / "processed/val"
TEST_DIR = DATA_DIR / "processed/test"
SAMPLE_DIR_1 = DATA_DIR / "sample/sample_1.jpg"
SAMPLE_DIR_2 = DATA_DIR / "sample/sample_2.jpg"
SAMPLE_DIR_3 = DATA_DIR / "sample/sample_3.jpg"
SAMPLE_DIR_4 = DATA_DIR / "sample/sample_4.jpg"
SAMPLE_DIR_5 = DATA_DIR / "sample/sample_5.jpg"

#Dataset
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SOURCE_DIR = DATA_DIR / "raw"
IMAGE_OUTPUT_DIR = DATA_DIR / "processed"

#Image Size
IMAGE_SIZE = 224

#Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 2
CRITERION_WEIGHTS = torch.tensor([1.87, 0.68])
RESUME_TRAINING = False

#Model
NUM_CLASSES = 2

#Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Outputs
OUTPUT_DIR = BASE_DIR / "outputs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

#Checkpoint
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

#Logs
LOG_DIR = BASE_DIR / "logs"
CSV_LOG_FILE = LOG_DIR / "training_log.csv"
TENSORBOARD_DIR = LOG_DIR / "tensorboard"