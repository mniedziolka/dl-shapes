import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class ShapesDataset(Dataset):

    def __init__(
            self,
            csv_file,
            root_dir,
            transform_images=None,
            transform_all=None
    ):
        self.shapes_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform_images = transform_images
        self.transform_all = transform_all

    def __len__(self):
        return len(self.shapes_frame)

    def __getitem__(self, idx):
        raise NotImplementedError('You chose abstract dataset!')


class ShapesClassificationDataset(ShapesDataset):

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        shapes = self.shapes_frame.iloc[idx, 1:]
        shapes = np.array([shapes])
        shapes = shapes.astype('float').flatten()
        shapes[shapes > 0] = 1

        img_name = os.path.join(self.root_dir,
                                self.shapes_frame.iloc[idx, 0])

        image = io.imread(img_name)

        image = transforms.ToPILImage()(image)

        if self.transform_all:
            image, shapes = self.transform_all((image, shapes))

        if self.transform_images:
            image = self.transform_images(image)

        return image, shapes


class ShapesCounterDataset(ShapesDataset):

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        shapes = self.shapes_frame.iloc[idx, 1:]
        shapes = np.array([shapes])
        shapes = shapes.astype('float').flatten()

        img_name = os.path.join(self.root_dir,
                                self.shapes_frame.iloc[idx, 0])

        image = io.imread(img_name)

        image = transforms.ToPILImage()(image)

        if self.transform_all:
            image, shapes = self.transform_all((image, shapes))

        if self.transform_images:
            image = self.transform_images(image)

        return image, shapes
