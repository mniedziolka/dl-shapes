import argparse
import json
import neptune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from datasets.shapes_dataset import ShapesClassificationDataset, ShapesCounterDataset
from datasets.transformers import RandomVerticalFlip, RandomHorizontalFlip, RandomRightRotation

from models.shapes_classifier import ShapesClassifier
from models.shapes_counter import ShapesCounter

from training import train_and_evaluate_model, setup_neptune, upload_file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 500
WORKERS = 2


def train_classifier(transform_images, transform_all):
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

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=WORKERS)

    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE,
                                                    shuffle=False, num_workers=WORKERS)

    classes = ('squares', 'circles', 'triangle_up', 'triangle_right',
               'triangle_down', 'triangle_left')

    model = ShapesClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    hist = train_and_evaluate_model(model, criterion, optimizer,
                                    train_loader, train_set,
                                    validation_loader, validation_set,
                                    device, num_epochs=100)

    torch.save(model, 'classifier.pt')

    return hist


def train_counter(transform_images, transform_all):
    train_set = ShapesCounterDataset(
        "data/train.csv",
        "data/images",
        transform_all=transform_all,
        transform_images=transform_images
    )

    validation_set = ShapesCounterDataset(
        "data/val.csv",
        "data/images",
        transform_all=None,
        transform_images=transform_images
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=WORKERS)

    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE,
                                                    shuffle=False, num_workers=WORKERS)

    model = ShapesCounter().to(device)

    def counter_loss(outputs, labels):
        # print(outputs.shape)
        # print(labels.shape)
        # print()

        soft_outputs = F.softmax(outputs, dim=2)

        j = torch.cat([torch.arange(0, 10, device=device, dtype=torch.float, requires_grad=True).unsqueeze(0)] * 6, 0)
        j = torch.cat([j.unsqueeze(0)] * soft_outputs.shape[0], 0)
        # print(soft_outputs)

        stretched_labels = labels.view(-1, 1).repeat(1, 10).view(soft_outputs.shape[0], 6, 10)
        # print(j)
        # print(stretched_labels)
        # print(soft_outputs)
        # print(labels)

        _, chosen_classes = torch.max(soft_outputs, 2)
        non_zero = torch.sum(chosen_classes > 0, dim=1)
        reg = torch.sum(non_zero - torch.ones_like(non_zero) * 2) ** 2

        loss = torch.sum(
            torch.sum(soft_outputs * (j - stretched_labels)**2, dim=1)
        )

        # print(loss)
        # print(reg)
        # print()
        # print(loss)
        # exit()

        # print(soft_outputs * (stretched_labels - j)**2)
        # print(labels)
        # print(F.softmax(outputs.view(outputs.shape[0], 6, 10), dim=2))
        # print(loss)
        # exit()

        return loss # + reg

    criterion = counter_loss
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    hist = train_and_evaluate_model(model, criterion, optimizer,
                                    train_loader, train_set,
                                    validation_loader, validation_set,
                                    device, num_epochs=100)

    return hist


def main(args):
    global BATCH_SIZE, WORKERS

    if args.neptune:
        setup_neptune(args.model)

    if args.entropy:
        BATCH_SIZE = 1000
        WORKERS = 8

    print('-' * 10)
    print(f'Settings: batch = {BATCH_SIZE}, workers = {WORKERS}')
    print('-' * 10)

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

    hist = None

    if args.model == 'classifier':
        hist = train_classifier(transform_images, transform_all)

    elif args.model == 'counter':
        hist = train_counter(transform_images, transform_all)

    else:
        raise ValueError('Unknown model')

    if args.file:
        with open(args.file, 'w') as f:
            json.dump(hist, f)

        if args.neptune:
            upload_file(args.file)

    print('-' * 10)
    print('Finished Training')
    print('-' * 10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work with shapes networks.')
    parser.add_argument('-n', '--neptune', action='store_true',
                        help='use neptune.ai for logging')
    parser.add_argument('-m', '--model', action='store', type=str, required=True,
                        help='which model should be trained')
    parser.add_argument('-e', '--entropy', action='store_true',
                        help='development or training environment')
    parser.add_argument('-f', '--file', action='store', type=str,
                        help='file for storing output for plotting')

    args = parser.parse_args()
    main(args)
