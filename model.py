import torch.nn as nn


class DepthNetModel(nn.Module):
    def __init__(self):
        super(DepthNetModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(200, 100, 7, stride=1),
            nn.ReLU(),
            nn.Conv2d(100, 100, 5, stride=1),
            nn.ReLU(),
            nn.Conv2d(100, 50, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(50, 1, 1, stride=1),
        )

    def forward(self, features):
        out = self.layer(features)
        return out


class ColorNetModel(nn.Module):
    def __init__(self):
        super(ColorNetModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(15, 100, 7, stride=1),
            nn.ReLU(),
            nn.Conv2d(100, 100, 5, stride=1),
            nn.ReLU(),
            nn.Conv2d(100, 50, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(50, 3, 1, stride=1),
        )

    def forward(self, features):
        out = self.layer(features)
        return out

     
