from PIL import Image
import numpy as np
from torchvision.datasets import CIFAR10


class CustomCifar10(CIFAR10):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(img)

        if self.transform:
            # Convert PIL image to numpy array
            image = np.array(image)
            # Apply transformations
            augmented = self.transform(image=image)
            # Convert numpy array to PIL Image
            image = augmented['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
