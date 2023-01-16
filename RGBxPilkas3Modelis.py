import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RGBxPilkas3Modelis(nn.Module):
    def __init__(self, rgbModel, edgeModel, skeletonModel, LBPmodel):
        super(RGBxPilkas3Modelis, self).__init__()

        self.rgbModel = rgbModel
        self.edgeModel = edgeModel
        self.skeletonModel = skeletonModel
        self.LBPmodel = LBPmodel
        self.Layer1 = nn.Linear(102*4, 1024)
        self.Layer2 = nn.Linear(1024, 512)
        self.Layer3 = nn.Linear(512, 102)

    def forward(self, _x):

        _rgb, _edge, _skeleton, _lbp = _x
        x1 = self.rgbModel(_rgb)
        x2 = self.edgeModel(_edge)
        x3 = self.skeletonModel(_skeleton)
        x4 = self.LBPmodel(_lbp)
        x = th.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.Layer1(x))
        x = F.relu(self.Layer2(x))

        x = self.Layer3(x)
        return x
