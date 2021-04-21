import torch.nn as nn
import torch.nn.functional as F


class ShapesCounter(nn.Module):
    def __init__(self):
        super(ShapesCounter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=12,
                               kernel_size=2,
                               padding=1)
        self.conv1_bn = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(in_channels=12,
                               out_channels=20,
                               kernel_size=3,
                               padding=1)
        self.conv2_bn = nn.BatchNorm2d(20)

        self.conv3 = nn.Conv2d(in_channels=20,
                               out_channels=60,
                               kernel_size=3,
                               padding=1)
        self.conv3_bn = nn.BatchNorm2d(60)

        self.conv4 = nn.Conv2d(in_channels=60,
                               out_channels=120,
                               kernel_size=3,
                               padding=1)
        self.conv4_bn = nn.BatchNorm2d(120)

        self.conv5 = nn.Conv2d(in_channels=120,
                               out_channels=250,
                               kernel_size=3,
                               padding=1)
        self.conv5_bn = nn.BatchNorm2d(250)

        self.conv6 = nn.Conv2d(in_channels=250,
                               out_channels=120,
                               kernel_size=3,
                               padding=1)
        self.conv6_bn = nn.BatchNorm2d(120)

        self.conv7 = nn.Conv2d(in_channels=120,
                               out_channels=40,
                               kernel_size=5,
                               padding=1)
        self.conv7_bn = nn.BatchNorm2d(40)

        self.pool_small = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear(1800, 1000)
        self.bn1 = nn.BatchNorm1d(num_features=1000)
        self.fc2 = nn.Linear(1000, 400)
        self.bn2 = nn.BatchNorm1d(num_features=400)
        self.fc3 = nn.Linear(400, 60)

    def forward(self, x):
        # print(x.shape)
        # With pooling.
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool_small(x)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool_small(x)

        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.relu(self.conv7_bn(self.conv7(x)))

        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape)

        # x = F.relu(F.dropout(self.bn1(self.fc1(x)), p=0.6))
        # x = F.relu(F.dropout(self.fc2(x), p=0.2))

        # x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        # x = F.relu(self.fc3(x))
        x = self.fc3(x)

        x = x.view(x.shape[0], 6, 10)

        return x
