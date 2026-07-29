from pathlib import Path
import random
import shutil

#Configuration
SOURCE_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

#Spiliting
def split_class(source_dir, class_name):
  for extension in ("*.jpg", "*.jpeg", "*.png"):
    image_paths = list(source_dir.glob(extension))

  random.shuffle(image_paths)

  train_end = int(len(image_paths) * TRAIN_RATIO)
  val_end = train_end + int(len(image_paths) * VAL_RATIO)

  train_images = image_paths[:train_end]
  val_images = image_paths[train_end:val_end]
  test_images = image_paths[val_end:]

  for split_name, image_list in [
    ("train", train_images),
    ("val", val_images),
    ("test", test_images),
  ]:
    destination = OUTPUT_DIR / split_name / class_name
    destination.mkdir(parents=True, exist_ok=True)

    for image_path in image_list:
      shutil.copy2(image_path, destination)

#main
if OUTPUT_DIR.exists():
  shutil.rmtree(OUTPUT_DIR)
  
split_class(SOURCE_DIR / "crack")
split_class(SOURCE_DIR / "no_crack")

print("Dataset split completed")