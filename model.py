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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return nn.functional.sigmoid(self.net(x).view(batch_size))
