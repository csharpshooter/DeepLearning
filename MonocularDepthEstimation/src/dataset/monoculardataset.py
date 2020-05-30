import numpy as np
import torch.utils.data
from PIL import Image


class MonocularDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transforms=None):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.images = images
        self.labels = labels

    def __getitem__(self, idx):
        images = []
        labels = []
        # load images and masks

        bg_fg = Image.open(self.images[idx])  # .convert("RGB")
        bg = Image.open(self.labels[idx]["bg_path"])  # .convert("RGB")
        mask = Image.open(self.labels[idx]["masks"]).convert("RGB")
        dm = Image.open(self.labels[idx]["depth_mask"])

        labels.append(self.images[idx])
        labels.append(self.labels[idx]["bg_path"])
        labels.append(self.labels[idx]["masks"])
        labels.append(self.labels[idx]["depth_mask"])

        if self.transforms is not None:
            bg_fg = self.transforms(bg_fg)

        if self.transforms is not None:
            bg = self.transforms(bg)

        if self.transforms is not None:
            mask = self.transforms(mask)

        if self.transforms is not None:
            dm = self.transforms(dm)

        images.append(np.array(bg_fg, np.float32))
        images.append(np.array(bg, np.float32))
        images.append(np.array(mask, np.float32))
        images.append(np.array(dm, np.float32))

        return images, labels

    def __len__(self):
        return len(self.images)
