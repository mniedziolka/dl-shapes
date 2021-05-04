import torch.nn as nn
import torch.nn.functional as F


class ShapesCounter(nn.Module):
    def __init__(self):
        super(ShapesCounter, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=4,
                      kernel_size=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=(4, 4),
                      padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(5, 5),
                      padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(6, 6),
                      stride=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(7, 7),
                      padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(8, 8)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 60),
            nn.BatchNorm1d(60),
            # nn.ReLU()
        )
        #
        # self.fc3 = nn.Sequential(
        #     nn.Linear(200, 60)
        # )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape)

        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(x)

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        x = x.view(x.shape[0], 6, 10)

        return x


class ShapesCounter135(nn.Module):

    def __init__(self):
        super(ShapesCounter135, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=4,
                      kernel_size=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=(4, 4),
                      padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(5, 5),
                      padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(6, 6),
                      stride=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(7, 7),
                      padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(8, 8)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512, 135),
            nn.BatchNorm1d(135),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)

        return x
