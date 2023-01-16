import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RGBxPilkasModelis(nn.Module):
    def __init__(self, grayModel, rgbModel):
        super(RGBxPilkasModelis, self).__init__()

        self.grayModel = grayModel
        self.rgbModel = rgbModel
        self.Layer1 = nn.Linear(102*2, 1024)
        self.Layer2 = nn.Linear(1024, 256)
        self.Layer3 = nn.Linear(256, 512)
        self.Layer4 = nn.Linear(512, 128)
        self.Layer5 = nn.Linear(128, 256)
        self.Layer6 = nn.Linear(256, 102)

    def forward(self, _x):

        _grayImage, _rgbImage = _x
        x1 = self.grayModel(_grayImage)
        x2 = self.rgbModel(_rgbImage)
        x = th.cat((x1, x2), dim=1)
        x = F.relu(self.Layer1(x))
        x = F.relu(self.Layer2(x))
        x = F.relu(self.Layer3(x))
        x = F.relu(self.Layer4(x))
        x = F.relu(self.Layer5(x))

        x = self.Layer6(x)
        return x
