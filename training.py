import torch

import neptune.new as neptune

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


def calculate_accuracy(outputs, labels):
    acc = 0
    _, preds = torch.topk(outputs, 2, 1)

    for i in range(labels.shape[0]):
        labels_row = labels[i]
        preds_row = preds[i]

        acc += torch.all(labels_row[preds_row])

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
            running_corrects_train = 0

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
                if epoch % 10 == 0:
                    running_corrects_train += calculate_accuracy(outputs, labels)

                if i % save_every_nth_batch_loss == 0:
                    log_values('train/batch_loss', loss.item())
                i += 1

            epoch_loss_train = running_loss_train / len(train_set)
            if epoch % 10 == 0:
                epoch_acc_train = running_corrects_train.float() / len(train_set)

            if epoch % 10 == 0:
                print('train loss: {:.4f}, train acc: {:.4f}'. \
                      format(epoch_loss_train.item(),
                             epoch_acc_train.item()))

            log_values('train/loss', epoch_loss_train.item())
            if epoch % 10 == 0:
                log_values('train/acc', epoch_acc_train.item())

            # evaluating phase
            model.eval()

            running_loss_val = 0.0
            running_corrects_val = 0

            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss_val += loss.detach() * inputs.size(0)
                if epoch % 10 == 0:
                    running_corrects_val += calculate_accuracy(outputs, labels)

            epoch_loss_test = running_loss_val / len(val_set)
            if epoch % 10 == 0:
                epoch_acc_test = running_corrects_val.float() / len(val_set)

            if epoch % 10 == 0:
                print('validation loss: {:.4f}, test acc: {:.4f}\n'. \
                      format(epoch_loss_test.item(),
                             epoch_acc_test.item()))

            log_values('validation/loss', epoch_loss_test.item())
            if epoch % 10 == 0:
                log_values('validation/acc', epoch_acc_test.item())

    except KeyboardInterrupt:
        print('Interrupt')
        pass

    return hist_run
