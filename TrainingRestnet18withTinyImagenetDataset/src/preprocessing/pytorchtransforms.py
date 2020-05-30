from torchvision import transforms


class PytorchTransforms(object):

    def gettraintransforms(self, mean, std):
        # Train Phase transformations
        return transforms.Compose([
            transforms.Pad(padding=1, padding_mode="edge"),
            transforms.RandomHorizontalFlip(p=1),  # randomly flip and rotate
            transforms.RandomRotation(20),
            transforms.RandomCrop(size=(64, 64), padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(scale=(0.10, 0.10), ratio=(1, 1), p=1),
        ])

    def gettesttransforms(self, mean, std):
        # Test Phase transformations
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
