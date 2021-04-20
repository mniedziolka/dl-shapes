import torch.nn as nn
import torch.nn.functional as F


class ShapesCounter(nn.Module):
    def __init__(self):
        super(ShapesCounter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=12,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=12,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=80,
                               kernel_size=3,
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=80,
                               out_channels=120,
                               kernel_size=3,
                               padding=1)

        self.conv5 = nn.Conv2d(in_channels=120,
                               out_channels=50,
                               kernel_size=3,
                               padding=1)

        self.conv5_bn = nn.BatchNorm2d(50)

        self.pool_small = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear(2450, 1000)
        self.bn1 = nn.BatchNorm1d(num_features=1000)
        self.fc2 = nn.Linear(1000, 60)

    def forward(self, x):
        # print(x.shape)
        # With pooling.
        x = F.relu(self.conv1(x))
        x = self.pool_small(x)
        x = F.relu(self.conv2(x))
        x = self.pool_small(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5_bn(self.conv5(x)))

        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape)

        # x = F.relu(F.dropout(self.fc1(x), p=0.6))
        # x = F.relu(F.dropout(self.fc2(x), p=0.2))

        x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc2(x)

        x = x.view(x.shape[0], 6, 10)

        return x
