import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def counter_loss(outputs, labels):
    soft_outputs = F.softmax(outputs, dim=2)

    j = torch.cat([torch.arange(0, 10, device=device, dtype=torch.float, requires_grad=True).unsqueeze(0)] * 6, 0)
    j = torch.cat([j.unsqueeze(0)] * soft_outputs.shape[0], 0)

    stretched_labels = labels.view(-1, 1).repeat(1, 10).view(soft_outputs.shape[0], 6, 10)

    loss = torch.sum(
        torch.sum(soft_outputs * (j - stretched_labels) ** 2, dim=1)
    )

    return loss


def counter135_loss(outputs, labels):
    labels_max = torch.argmax(labels, dim=1)

    return nn.CrossEntropyLoss()(outputs, labels_max)


def calculate_classification(outputs, labels):
    _, chosen_classes = torch.topk(outputs, 2, 1, sorted=False)

    chosen_classes, _ = torch.sort(chosen_classes, dim=1)

    accuracy = torch.sum(
        torch.all(
            chosen_classes == torch.nonzero(labels, as_tuple=True)[1].view(-1, 2),
            dim=1
        )
    )

    return accuracy


def calculate_counter(outputs, labels):
    soft_outputs = F.softmax(outputs, dim=2)

    _, chosen_classes = torch.max(soft_outputs, 2)

    accuracy = torch.sum(torch.all(chosen_classes == labels, dim=1))

    return accuracy


def calculate_counter135(outputs, labels):
    _, preds = torch.max(outputs, 1)

    return torch.sum(preds == labels)
