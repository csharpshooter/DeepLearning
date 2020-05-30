import io
import os
# from imageio import imread
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class TinyImagenetDataset(Dataset):
    """Tiny Imagenet dataset."""

    def __init__(self, image_data, image_labels, transform=None):
        self.transform = transform
        self.image_data = image_data
        self.image_labels = image_labels

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_data[idx]))
        label = self.image_labels[idx]
        # label = label.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'label': label}

        if self.is_grayscale_image(image):
            image = np.stack((image,) * 3, axis=-1)
            # image = np.squeeze(np.stack((image,) * 3, -1))

        if self.transform:
            image = self.transform(Image.fromarray(image))

        return image, label

    def is_grayscale_image(self, image):
        return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)
