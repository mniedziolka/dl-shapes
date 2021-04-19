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
    acc = 0.0

    for i in range(labels.shape[0]):
        labels_row = labels[i]
        chosen_row = chosen_classes[i]

        # acc += np.all(labels_row[chosen_row])
        acc += torch.all(labels_row[chosen_row])

    return acc


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
        save_every_nth_batch_loss=50
):
    try:
        for epoch in range(num_epochs):

            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            # training phase
            model.train()

            running_loss_train = 0.0
            running_corrects_train = 0.0

            i = 0
            start = time.time()
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
                if epoch % 1 == 0:

                    running_corrects_train += calculate_accuracy(chosen_classes, labels)
                    # running_corrects_train += calculate_accuracy(chosen_classes.detach().cpu().numpy(),
                    #                                              labels.detach().cpu().numpy())

                if i % save_every_nth_batch_loss == 0:
                    log_values('train/batch_loss', loss.item())
                i += 1
            end = time.time()
            print("Elapsed time = %s" % (end - start))

            epoch_loss_train = running_loss_train / len(train_set)
            if epoch % 1 == 0:
                epoch_acc_train = running_corrects_train / len(train_set)

            if epoch % 1 == 0:
                print('train loss: {:.4f}, train acc: {:.4f}'. \
                      format(epoch_loss_train.item(),
                             epoch_acc_train))

            log_values('train/loss', epoch_loss_train.item())
            if epoch % 1 == 0:
                log_values('train/acc', epoch_acc_train)

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
                if epoch % 1 == 0:
                    running_corrects_val += calculate_accuracy(chosen_classes, labels)

                    # running_corrects_val += calculate_accuracy(chosen_classes.detach().cpu().numpy(),
                    #                                            labels.detach().cpu().numpy())

            epoch_loss_test = running_loss_val / len(val_set)
            if epoch % 1 == 0:
                epoch_acc_test = running_corrects_val / len(val_set)

            if epoch % 1 == 0:
                print('validation loss: {:.4f}, test acc: {:.4f}\n'. \
                      format(epoch_loss_test.item(),
                             epoch_acc_test))

            log_values('validation/loss', epoch_loss_test.item())
            if epoch % 1 == 0:
                log_values('validation/acc', epoch_acc_test)

    except KeyboardInterrupt:
        print('Interrupt')
        pass

    return hist_run
