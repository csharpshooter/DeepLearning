from src import preprocessing
from src.preprocessing import customcompose



class PreprocHelper:

    def getalbumentationstraintesttransforms(mean, std):
        preproc = preprocessing.AlbumentaionsTransforms()
        train_transforms = preproc.gettraintransforms(mean, std)
        test_transforms = preproc.gettesttransforms(mean, std)
        compose_train = customcompose.CustomCompose(train_transforms)
        compose_test = customcompose.CustomCompose(test_transforms)
        return compose_train, compose_test

    def getpytorchtransforms(mean, std, size=224):
        preproc = preprocessing.PytorchTransforms()
        train_transforms = preproc.gettraintransforms(mean, std, size)
        test_transforms = preproc.gettesttransforms(mean, std)
        return train_transforms, test_transforms

    def get_transform(train):
        from src.train.torchvision import transforms as T
        transforms = [T.ToTensor()]
        # converts the image, a PIL image, into a PyTorch Tensor
        if train:
            # during training, randomly flip the training images
            # and ground-truth for data augmentation
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)
