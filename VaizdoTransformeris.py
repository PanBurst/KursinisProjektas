import torch as th
import torch.nn as nn


class Transfomer():
    def __init__(self):
        model = th.hub.load('facebookresearch/deit:main',
                            'deit_tiny_patch16_224', pretrained=True)
        for param in model.parameters():  # freeze model
            param.requires_grad = False
        n_inputs = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 102)
        )
        return model
