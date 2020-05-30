import cv2
from albumentations import Compose, Flip, pytorch, Normalize, OneOf, MotionBlur, MedianBlur, Blur, \
    ShiftScaleRotate, OpticalDistortion, GridDistortion, HueSaturationValue, CoarseDropout, GaussNoise, RandomCrop, \
    PadIfNeeded


class AlbumentaionsTransforms(object):

    def gettraintransforms(self, mean, std, p=1):
        # Train Phase transformations

        albumentations_transform = Compose([
            # RandomRotate90(),
            PadIfNeeded(72, 72, border_mode=cv2.BORDER_REFLECT, always_apply=True),
            RandomCrop(64, 64, True),
            Flip(),
            GaussNoise(p=0.8, mean=mean),
            OneOf([
                MotionBlur(p=0.4),
                MedianBlur(blur_limit=3, p=0.2),
                Blur(blur_limit=3, p=0.2),
            ], p=0.4),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.6),
            OneOf([
                OpticalDistortion(p=0.8),
                GridDistortion(p=0.4),
            ], p=0.6),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.6),
            CoarseDropout(always_apply=True, max_holes=1, min_holes=1, max_height=16, max_width=16,
                          fill_value=(255 * .6), min_height=16, min_width=16),
            Normalize(mean=mean, std=std, always_apply=True),
            pytorch.ToTensorV2(always_apply=True),

        ], p=p)

        return albumentations_transform;

    def gettesttransforms(self, mean, std):
        # Test Phase transformations
        return Compose([
            Normalize(mean=mean, std=std, always_apply=True),
            pytorch.ToTensorV2(always_apply=True),
        ])
