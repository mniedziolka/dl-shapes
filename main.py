import argparse
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

from datasets.shapes_dataset import ShapesClassificationDataset, ShapesCounterDataset
from datasets.transformers import RandomVerticalFlip, RandomHorizontalFlip, RandomRightRotation

from models.shapes_classifier import ShapesClassifier
from models.shapes_counter import ShapesCounter

from training import train_and_evaluate_model, setup_neptune

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_classifier(transform_images, transform_all):
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

    batch_size = 1000

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                                    shuffle=False, num_workers=2)

    classes = ('squares', 'circles', 'triangle_up', 'triangle_right',
               'triangle_down', 'triangle_left')

    model = ShapesClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters())

    hist = train_and_evaluate_model(model, criterion, optimizer,
                                    train_loader, train_set,
                                    validation_loader, validation_set,
                                    device, num_epochs=100)

    return hist


def train_counter(transform_images, transform_all):
    train_set = ShapesCounterDataset(
        "data/train.csv",
        "data/images",
        transform_all=None,
        transform_images=transform_images
    )

    validation_set = ShapesCounterDataset(
        "data/val.csv",
        "data/images",
        transform_all=None,
        transform_images=transform_images
    )

    batch_size = 10

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                                    shuffle=False, num_workers=2)

    model = ShapesCounter().to(device)

    def counter_loss(outputs, labels):
        soft_outputs = F.softmax(outputs.view(outputs.shape[0], 6, 10), dim=2)

        j = torch.cat([torch.arange(0, 10, device=device).unsqueeze(0)] * 6, 0)
        j = torch.cat([j.unsqueeze(0)] * soft_outputs.shape[0], 0)

        stretched_labels = labels.view(-1, 1).repeat(1, 10).view(soft_outputs.shape[0], 6, 10)

        loss = (soft_outputs * (stretched_labels - j)**2).sum()

        return loss

    criterion = counter_loss
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters())

    hist = train_and_evaluate_model(model, criterion, optimizer,
                                    train_loader, train_set,
                                    validation_loader, validation_set,
                                    device, save_every_nth_all=1000, num_epochs=100)

    return hist


def main(args):
    if args.neptune:
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

    if args.model == 'classifier':
        hist = train_classifier(transform_images, transform_all)

    elif args.model == 'counter':
        hist = train_counter(transform_images, transform_all)

    else:
        raise ValueError('Unknown model')

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work with shapes networks.')
    parser.add_argument('-n', '--neptune', action='store_true',
                        help='use neptune.ai for logging')
    parser.add_argument('-m', '--model', action='store', type=str, required=True,
                        help='which model should be trained')

    args = parser.parse_args()
    main(args)
