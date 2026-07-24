from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class BridgeCrackDataset(Dataset):

  def __init__(self, root_dir, transform=None):
    super().__init__()

    self.root_dir = Path(root_dir)
    self.transform = transform

    self.image_paths, self.labels = self._load_dataset()

  def _load_dataset(self):
    image_paths = []
    labels = []
    classes = {
        "no_crack": 0,
        "crack": 1,
    }
    for class_name, label in classes.items():
      class_dir = self.root_dir / class_name
      for image_path in class_dir.glob("*"):
        image_paths.append(image_path)
        labels.append(label)

    return image_paths, labels
  
  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, index):
    image = Image.open(self.image_paths[index]).convert("RGB")
    label = self.labels[index]
    if self.transform:
      image = self.transform(image)
    return image, label