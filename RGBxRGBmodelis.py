import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RGBxRGBmodelis(nn.Module):
    def __init__(self, RGBmodel1, RGBmodel2):
        super(RGBxRGBmodelis, self).__init__()

        self.Model1 = RGBmodel1
        self.Model2 = RGBmodel2
        self.Layer1 = nn.Linear(102*2, 1024)
        self.Layer2 = nn.Linear(1024, 256)
        self.Layer3 = nn.Linear(256, 512)
        self.Layer4 = nn.Linear(512, 128)
        self.Layer5 = nn.Linear(128, 102)
        return None

    def forward(self, _x):

        x1 = self.Model1(_x)
        x2 = self.Model2(_x)
        x = th.cat((x1, x2), dim=1)
        x = F.relu(self.Layer1(x))
        x = F.relu(self.Layer2(x))
        x = F.relu(self.Layer3(x))
        x = F.relu(self.Layer4(x))

        x = self.Layer5(x)
        return x
