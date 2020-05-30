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

    def getpytorchtransforms(mean, std):
        preproc = preprocessing.PytorchTransforms()
        train_transforms = preproc.gettraintransforms(mean, std)
        test_transforms = preproc.gettesttransforms(mean, std)
        return train_transforms, test_transforms
