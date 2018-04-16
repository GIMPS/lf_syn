import torch
import torch.nn as nn

class depthNet(nn.Module):
    def __init__(self):
        super(depthNet, self).__init__()
        self.conv1= nn.Conv2d(200,100,stride=1)

        self.conv2=nn.Conv2d(100,100,stride=1)
        self.conv3=nn.Conv2d(100,50,stride=1)
        self.conv4 = nn.Conv2d(50, 1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, features):
        out=self.conv1(features)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.relu(out)
        out=self.conv3(out)
        out=self.relu(out)
        out=self.conv4(out)
        return out



class colorNet(nn.Module):
    def __init__(self):
        super(colorNet, self).__init__()
        self.conv1= nn.Conv2d(15,100,stride=1)
        self.conv2=nn.Conv2d(100,100,stride=1)
        self.conv3=nn.Conv2d(100,50,stride=1)
        self.conv4 = nn.Conv2d(50, 3, stride=1)
        self.relu = nn.ReLU()

    def forward(self, features):
        out=self.conv1(features)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.relu(out)
        out=self.conv3(out)
        out=self.relu(out)
        out=self.conv4(out)
        return out