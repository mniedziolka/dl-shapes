import torch.nn as nn
import torch.nn.functional as F


class ShapesClassifier(nn.Module):
    def __init__(self):
        super(ShapesClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=128,
                               out_channels=80,
                               kernel_size=3,
                               padding=1)
        self.conv5 = nn.Conv2d(in_channels=80,
                               out_channels=40,
                               kernel_size=3,
                               padding=1)

        self.pool_small = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear(360, 200)
        self.fc2 = nn.Linear(200, 80)
        self.fc3 = nn.Linear(80, 6)

    def forward(self, x):
        # print(x.shape)
        # With pooling.
        x = F.relu(self.conv1(x))
        x = self.pool_small(x)
        x = F.relu(self.conv2(x))
        x = self.pool_small(x)
        x = F.relu(self.conv3(x))
        x = self.pool_small(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # print(x.shape)
        x = x.view(x.shape[0], -1)

        x = F.relu(F.dropout(self.fc1(x), p=0.6))
        x = F.relu(F.dropout(self.fc2(x), p=0.2))

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
