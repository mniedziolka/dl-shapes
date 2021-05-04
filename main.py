import argparse
import json
import neptune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from datasets.shapes_dataset import ShapesClassificationDataset, ShapesCounterDataset, ShapesCounterDataset135
from datasets.transformers import RandomVerticalFlip, RandomHorizontalFlip, RandomRightRotation

from metrics import counter_loss, counter135_loss
from models.shapes_classifier import ShapesClassifier
from models.shapes_counter import ShapesCounter, ShapesCounter135
from training import train_and_evaluate_model, setup_neptune, upload_file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 100
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

    model = ShapesClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    hist = train_and_evaluate_model(model, criterion, optimizer,
                                    train_loader, train_set,
                                    validation_loader, validation_set,
                                    device, num_epochs=150)

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

    criterion = counter_loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    hist = train_and_evaluate_model(model, criterion, optimizer,
                                    train_loader, train_set,
                                    validation_loader, validation_set,
                                    device, num_epochs=300)

    return hist


def train_counter135(transform_images, transform_all):
    train_set = ShapesCounterDataset135(
        "data/train.csv",
        "data/images",
        transform_all=transform_all,
        transform_images=transform_images
    )

    validation_set = ShapesCounterDataset135(
        "data/val.csv",
        "data/images",
        transform_all=None,
        transform_images=transform_images
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=WORKERS)

    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE,
                                                    shuffle=False, num_workers=WORKERS)

    model = ShapesCounter135().to(device)

    # criterion = counter135_loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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
        WORKERS = 4

    print('-' * 10)
    print(f'Settings: batch = {BATCH_SIZE}, workers = {WORKERS}')
    print('-' * 10)

    transform_images = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
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
    elif args.model == 'counter135':
        hist = train_counter135(transform_images, transform_all)
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
