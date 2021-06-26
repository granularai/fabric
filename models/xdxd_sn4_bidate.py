import os
import torch
from torch import nn
from torchvision.models import vgg16


class XDXD_SpaceNet4_UNetVGG16(nn.Module):
    def __init__(self, n_channels=4, n_classes=1,
                 num_filters=32, pretrained=False):
        super().__init__()
        self.encoder = vgg16(pretrained=pretrained).features
        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            self.encoder[0], self.relu, self.encoder[2], self.relu)
        self.conv2 = nn.Sequential(
            self.encoder[5], self.relu, self.encoder[7], self.relu)
        self.conv3 = nn.Sequential(
            self.encoder[10], self.relu, self.encoder[12], self.relu,
            self.encoder[14], self.relu)
        self.conv4 = nn.Sequential(
            self.encoder[17], self.relu, self.encoder[19], self.relu,
            self.encoder[21], self.relu)
        self.conv5 = nn.Sequential(
            self.encoder[24], self.relu, self.encoder[26], self.relu,
            self.encoder[28], self.relu)

        self.center = XDXD_SN4_DecoderBlock(512, num_filters * 8 * 2,
                                            num_filters * 8)
        self.dec5 = XDXD_SN4_DecoderBlock(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = XDXD_SN4_DecoderBlock(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = XDXD_SN4_DecoderBlock(
            256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = XDXD_SN4_DecoderBlock(
            128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = XDXD_SN4_ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, n_classes, kernel_size=1)

    def forward(self, x):
        x_d1 = x[:, 0, :, :, :]
        x_d2 = x[:, 1, :, :, :]

        conv1_d1 = self.conv1(x_d1)
        conv2_d1 = self.conv2(self.pool(conv1_d1))
        conv3_d1 = self.conv3(self.pool(conv2_d1))
        conv4_d1 = self.conv4(self.pool(conv3_d1))
        conv5_d1 = self.conv5(self.pool(conv4_d1))
        center_d1 = self.center(self.pool(conv5_d1))

        conv1_d2 = self.conv1(x_d2)
        conv2_d2 = self.conv2(self.pool(conv1_d2))
        conv3_d2 = self.conv3(self.pool(conv2_d2))
        conv4_d2 = self.conv4(self.pool(conv3_d2))
        conv5_d2 = self.conv5(self.pool(conv4_d2))
        center_d2 = self.center(self.pool(conv5_d2))

        dec5 = self.dec5(torch.cat([center_d1 * center_d2, conv5_d1 * conv5_d2], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4_d1 * conv4_d2], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3_d1 * conv3_d2], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2_d1 * conv2_d2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1_d1 * conv1_d2], 1))
        x_out = self.final(dec1)
        return torch.sigmoid(x_out).squeeze(1)


class XDXD_SN4_ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class XDXD_SN4_DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(XDXD_SN4_DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            XDXD_SN4_ConvRelu(in_channels, middle_channels),
            XDXD_SN4_ConvRelu(middle_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)