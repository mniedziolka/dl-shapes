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

transform_images = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
])

train_set = ShapesClassificationDataset(
    "data/train.csv",
    "data/images",
    transform_all=None,
    transform_images=transform_images
)

validation_set = ShapesClassificationDataset(
    "data/val.csv",
    "data/images",
    transform_all=None,
    transform_images=transform_images
)

batch_size = 100

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

classes = ('squares', 'circles', 'triangle_up', 'triangle_right',
           'triangle_down', 'triangle_left')

net = FCNet().to(device)

criterion = nn.BCEWithLogitsLoss(reduction='sum')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):  # loop over the dataset multiple times
    print("EPOKA")
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs.float(), labels.float())
        # loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

running_loss_test = 0.0
running_corrects_test = 0

for inputs, labels in validation_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs)

    loss = criterion(outputs.float(), labels.float())

    _, preds = torch.topk(outputs, 2, 1)
    print(preds)

    running_loss_test += loss.detach() * inputs.size(0)
    running_corrects_test += torch.sum(preds == labels.data)

acc_test = running_corrects_test.float() / len(validation_set)
print(acc_test)
print('Finished Training')



