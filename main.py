import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import neptune
from torch.autograd import Variable

from datasets.shapes_dataset import ShapesClassificationDataset
from datasets.transformers import RandomVerticalFlip, RandomHorizontalFlip, RandomRightRotation

from models.fc import FCNet
from models.cnn import ConvNet

from training import train_and_evaluate_model, setup_neptune

# dataset = ShapesClassificationDataset(
#     "data/labels.csv",
#     "data/images",
#     transform_all=transforms.Compose([
#         RandomHorizontalFlip(1),
#         RandomVerticalFlip(1),
#         RandomRightRotation(1),
#         # transforms.ToTensor(),
#         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ]),
# )

# df = pd.read_csv("data/labels.csv")
# df[:9000].to_csv("data/train.csv", index=False)
# df[9000:10000].to_csv("data/val.csv", index=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

setup_neptune()

transform_images = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
    transforms.Normalize(0.5, 0.5)
])

transform_all = transforms.Compose([
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        RandomRightRotation(0.5),
])

train_set = ShapesClassificationDataset(
    "data/train.csv",
    "data/images",
    transform_all=transform_all,
    transform_images=transform_images
)

validation_set = ShapesClassificationDataset(
    "data/val.csv",
    "data/images",
    transform_all=None,
    transform_images=transform_images
)

batch_size = 1000

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

classes = ('squares', 'circles', 'triangle_up', 'triangle_right',
           'triangle_down', 'triangle_left')

model = ConvNet().to(device)

criterion = nn.BCEWithLogitsLoss(reduction='sum')
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters())

hist = train_and_evaluate_model(model, criterion, optimizer,
                                train_loader, train_set,
                                validation_loader, validation_set,
                                device, num_epochs=100)

print('Finished Training')



