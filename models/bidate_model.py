import torch
import torch.nn as nn

from .unet_parts import down, outconv, up, inconv


class BiDateNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(BiDateNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x_d1 = x[:, 0, :]
        x_d2 = x[:, 1, :]
        x1_d1 = self.inc(x_d1)
        x2_d1 = self.down1(x1_d1)
        x3_d1 = self.down2(x2_d1)
        x4_d1 = self.down3(x3_d1)
        x5_d1 = self.down4(x4_d1)

        x1_d2 = self.inc(x_d2)
        x2_d2 = self.down1(x1_d2)
        x3_d2 = self.down2(x2_d2)
        x4_d2 = self.down3(x3_d2)
        x5_d2 = self.down4(x4_d2)

        x = self.up1(torch.relu(x5_d2 * x5_d1), torch.relu(x4_d2 * x4_d1))
        x = self.up2(x, torch.relu(x3_d2 * x3_d1))
        x = self.up3(x, torch.relu(x2_d2 * x2_d1))
        x = self.up4(x, torch.relu(x1_d2 * x1_d1))
        x = self.outc(x)
        x = x.squeeze()
        x = torch.sigmoid(x)
        return x
