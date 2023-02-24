# Standard library
import os

# Third-party libraries and modules
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    """
    CustomDataset
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask > 0.0] = 1.0  # better approach
        # mask[mask == 255.0] = 1.0  # bad practice

        if self.transform is not None:
            image = self.transform[0](image)
            augmented = self.transform[1](image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask, name

