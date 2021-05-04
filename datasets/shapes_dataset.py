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


class ShapesCounterDataset135(ShapesDataset):

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

        # 0 -> squares        [0 - 8, 9 - 17, 18 - 26, 27 - 35, 36 - 44]
        # 1 -> circles        [45 - 53, 54 - 62, 63 - 71, 72 - 80]
        # 2 -> triangle_up    [81 - 89, 90 - 98, 99 - 107]
        # 3 -> triangle_right [108 - 116, 117 - 125]
        # 4 -> triangle_down  [126 - 134]
        # 5 -> triangle_left

        indexes = np.argwhere(shapes > 0)

        assert len(indexes) == 2
        first, second = indexes
        assert first < second
        how_many = shapes[first]

        labels = np.zeros(135)

        ind = 0
        if first == 1:
            ind += 45
        elif first == 2:
            ind += 81
        elif first == 3:
            ind += 108
        elif first == 4:
            ind += 126

        # print(int(second))
        # print(int(first))
        # print(ind)
        ind += (int(second) - int(first) - 1) * 9
        ind += int(how_many) - 1

        # print(ind)
        # print(type(ind))
        # exit()

        # index = int(index)
        # print(ind)
        # labels[int(ind)] = 1
        #
        # print(labels)
        return image, ind
