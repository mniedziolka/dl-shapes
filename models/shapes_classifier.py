import torch.nn as nn
import torch.nn.functional as F


class ShapesClassifier(nn.Module):
    def __init__(self):
        super(ShapesClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=(2, 2),
                      padding=(1, 1)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(4, 4),
                      padding=(1, 1)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=400,
                      kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=400,
                      out_channels=100,
                      kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=100,
                      out_channels=40,
                      kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(360, 200),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, 80),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(80, 6)
        )

    def forward(self, x):
        # print(x.shape)
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
        x = self.fc3(x)

        # x = F.relu(F.dropout(self.fc1(x), p=0.6))
        # x = F.relu(F.dropout(self.fc2(x), p=0.2))

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        return x
