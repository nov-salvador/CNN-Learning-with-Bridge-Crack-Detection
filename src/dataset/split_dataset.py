from pathlib import Path
import random
import shutil

from configs.config import *

random.seed(42)

#Spiliting
def split_class(source_dir, class_name):
  image_paths = []
  for extension in ("*.jpg", "*.jpeg", "*.png"):
    image_paths.extend(source_dir.glob(extension))

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
    destination = IMAGE_OUTPUT_DIR / split_name / class_name
    destination.mkdir(parents=True, exist_ok=True)
    
    for image_path in image_list:
      shutil.copy2(image_path, destination)

#main
if IMAGE_OUTPUT_DIR.exists():
  shutil.rmtree(IMAGE_OUTPUT_DIR)
  
split_class(SOURCE_DIR / "crack", "crack")
split_class(SOURCE_DIR / "no_crack", "no_crack")

print("Dataset split completed")