import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PilkasModelis(nn.Module):
    def __init__(self, input_shape=(1, 500, 500)):
        super(PilkasModelis, self).__init__()
        self.conv1 = nn.Conv2d(1, 350, 5, 2)
        self.conv2 = nn.Conv2d(350, 250, 3, 2)
        self.conv3 = nn.Conv2d(250, 256, 5, 2)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(9216, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 102)

        return None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
