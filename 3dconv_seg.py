import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import cv2
import os


class Conv3DSegNet(nn.Module):

    def __init__(self):
        super(Conv3DSegNet, self).__init__()
        self.conv1 = nn.Conv3d(13, 64, 3)
        self.conv2 = nn.Conv3d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.deconv1 = nn.ConvTranspose2d(256, 256, 3)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 3)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 3)
        self.deconv4 = nn.ConvTranspose2d(64, 1, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        print (x.size())
        x = F.relu(self.conv2(x))
        print (x.size())
        x = x.view(-1, 128, 28, 28)
        x = F.relu(self.conv3(x))
        print (x.size())
        x = F.relu(self.conv4(x))
        x = self.deconv1(x)
        print (x.size())
        x = self.deconv2(x)
        print (x.size())
        x = self.deconv3(x)
        print (x.size())
        x = self.deconv4(x)
        print (x.size())
        return x

model = Conv3DSegNet()
input = torch.randn(8,13,5,32,32)
output = model(input)
