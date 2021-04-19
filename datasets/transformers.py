from torchvision import transforms

import torch
import torchvision.transforms.functional as TF


class RandomHorizontalFlip:

    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, target = sample

        if torch.rand(1) < self.p:
            image = transforms.RandomHorizontalFlip(1)(image)
            target[[3, 5]] = target[[5, 3]]

        return image, target


class RandomVerticalFlip:

    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, target = sample

        if torch.rand(1) < self.p:
            image = transforms.RandomVerticalFlip(1)(image)
            target[[2, 4]] = target[[4, 2]]

        return image, target


class RandomRightRotation:

    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, target = sample

        if torch.rand(1) < self.p:
            image = TF.rotate(image, -90)
            target[[3, 4, 5, 2]] = target[[2, 3, 4, 5]]

        return image, target
