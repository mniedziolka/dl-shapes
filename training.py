from numba import jit

import neptune.new as neptune
import numpy as np
import time
import torch

tracked_values = ['train/loss',
                  'train/acc',
                  'validation/loss',
                  'validation/acc',
                  'train/batch_loss']

NEPTUNE = False
npt_run = None
hist_run = {key: [] for key in tracked_values}


def setup_neptune():
    global NEPTUNE, npt_run

    NEPTUNE = True
    npt_run = neptune.init(project='uw-niedziol/dl-shapes',
                           source_files=['*.py'])


def log_values(key, value):
    if NEPTUNE:
        npt_run[key].log(value)

    hist_run[key].append(value)


def calculate_accuracy(chosen_classes, labels):
    return torch.sum(
        torch.all(
            chosen_classes == torch.nonzero(labels, as_tuple=True)[1].view(-1, 2),
            dim=1)
    )


def train_and_evaluate_model(
        model,
        criterion,
        optimizer,
        train_loader,
        train_set,
        val_loader,
        val_set,
        device,
        num_epochs=10,
        save_every_nth_all=1,
        save_every_nth_batch_loss=50
):
    try:
        for epoch in range(num_epochs):
            epoch_start = time.time()
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            # training phase
            model.train()

            running_loss_train = 0.0
            running_corrects_train = 0.0

            i = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss_train += loss.detach() * inputs.size(0)

                _, chosen_classes = torch.topk(outputs, 2, 1)
                if epoch % save_every_nth_all == 0:
                    running_corrects_train += calculate_accuracy(chosen_classes, labels)

                if i % save_every_nth_batch_loss == 0:
                    log_values('train/batch_loss', loss.item())

                i += 1

            epoch_loss_train = running_loss_train / len(train_set)
            log_values('train/loss', epoch_loss_train)
            print(f'[TRAIN] loss: {epoch_loss_train}')

            if epoch % save_every_nth_all == 0:
                epoch_acc_train = running_corrects_train / len(train_set)
                log_values('train/acc', epoch_acc_train)
                print(f'[TRAIN] accuracy: {epoch_acc_train}')

            # evaluating phase
            model.eval()

            running_loss_val = 0.0
            running_corrects_val = 0.0

            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss_val += loss.detach() * inputs.size(0)

                _, chosen_classes = torch.topk(outputs, 2, 1)
                if epoch % save_every_nth_all == 0:
                    running_corrects_val += calculate_accuracy(chosen_classes, labels)

            epoch_loss_test = running_loss_val / len(val_set)
            log_values('validation/loss', epoch_loss_test)
            print(f'[TEST] loss: {epoch_loss_test}')

            if epoch % save_every_nth_all == 0:
                epoch_acc_test = running_corrects_val / len(val_set)
                log_values('validation/acc', epoch_acc_test)
                print(f'[TEST] accuracy: {epoch_acc_test}')

            epoch_end = time.time()
            print("Epoch elapsed time = %s" % (epoch_end - epoch_start))

    except KeyboardInterrupt:
        print('Interrupt')
        pass

    return hist_run
