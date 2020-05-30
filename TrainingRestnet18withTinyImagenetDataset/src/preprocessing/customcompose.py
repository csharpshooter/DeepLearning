from albumentations import Compose, ElasticTransform, Flip, CoarseDropout, RandomCrop, pytorch, Normalize, Resize, \
    HorizontalFlip, Rotate, PadIfNeeded, CenterCrop, Cutout
import numpy as np


class CustomCompose:
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self, img):
        img = np.array(img)
        img = self.transforms(image=img)["image"]
        return img
