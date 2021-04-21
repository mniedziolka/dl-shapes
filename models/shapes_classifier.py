import torch.nn as nn
import torch.nn.functional as F


class ShapesClassifier(nn.Module):
    def __init__(self):
        super(ShapesClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((2, 2)),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(5, 5),
                      stride=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(7, 7),
                      stride=(5, 5),
                      padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128, 100),
            nn.Dropout(0.2),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(100, 50),
            nn.Dropout(0.1),
            nn.BatchNorm1d(50),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(100, 6)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)

        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # x = F.relu(F.dropout(self.fc1(x), p=0.6))
        # x = F.relu(F.dropout(self.fc2(x), p=0.2))

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        return x
