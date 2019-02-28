import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

DROPOUT = 0.5

class UNetBlock(nn.Module):
    def __init__(self, filters_in, filters_out):
        super(UNetBlock, self).__init__()
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.conv1 = nn.Conv2d(filters_in, filters_out, (3, 3), padding=1)
        self.norm1 = nn.BatchNorm2d(filters_out)
        self.conv2 = nn.Conv2d(filters_out, filters_out, (3, 3), padding=1)
        self.norm2 = nn.BatchNorm2d(filters_out)

        self.activation = nn.ReLU()

    def forward(self, x):
        conved1 = self.conv1(x)
        conved1 = self.activation(conved1)
        conved1 = self.norm1(conved1)
        conved2 = self.conv2(conved1)
        conved2 = self.activation(conved2)
        conved2 = self.norm2(conved2)
        return conved2

class UNetDownBlock(UNetBlock):
    def __init__(self, filters_in, filters_out, pool=True):
        super(UNetDownBlock, self).__init__(filters_in, filters_out)
        if pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = lambda x: x

    def forward(self, x):
        return self.pool(super().forward(x))

class UNetUpBlock(UNetBlock):
    def __init__(self, filters_in, filters_out):
        super().__init__(filters_in, filters_out)
        self.upconv = nn.Conv2d(filters_in, filters_in // 2, (3, 3), padding=1)
        self.upnorm = nn.BatchNorm2d(filters_in // 2)

    def forward(self, x, x_combd):
        x = F.upsample(x, size=x_combd.size()[-2:], mode='bilinear', align_corners=True)
        x = self.upnorm(self.activation(self.upconv(x)))
        x = torch.cat((x, x_combd), 1)
        return super().forward(x)

class UNet(nn.Module):
    def __init__(self, layers, init_filters, num_channels, fusion_method=None):
        super(UNet, self).__init__()
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.init_filters = init_filters
        self.fusion_method = fusion_method

        filter_size = init_filters
        for _ in range(layers - 1):
            self.down_layers.append(
                UNetDownBlock(filter_size, filter_size*2)
            )
            filter_size *= 2
        self.down_layers.append(UNetDownBlock(filter_size, filter_size * 2, pool=False))
        for i in range(layers):
            self.up_layers.append(
                UNetUpBlock(filter_size * 2, filter_size)
            )
            filter_size //= 2

        self.data_norm = nn.BatchNorm2d(num_channels)
        self.init_layer = nn.Conv2d(num_channels, init_filters, (7, 7), padding=3)
        self.activation = nn.ReLU()
        self.init_norm = nn.BatchNorm2d(init_filters)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x1, x2):
        x1 = self.data_norm(x1)
        x1 = self.init_norm(self.activation(self.init_layer(x1)))

        x2 = self.data_norm(x2)
        x2 = self.init_norm(self.activation(self.init_layer(x2)))

        saved_x = [[x1, x2]]
        for layer in self.down_layers:
            saved_x.append([x1,x2])
            x1 = self.dropout(layer(x1))
            x2 = self.dropout(layer(x2))

        is_first = True
        for layer, saved_x in zip(self.up_layers, reversed(saved_x)):
            if not is_first:
                is_first = False
                x1 = self.dropout(x1)

            if self.fusion_method == 'cat':
                x_comb = torch.cat(saved_x, 1)
            if self.fusion_method == 'add':
                x_combd = self.activation(saved_x[0] + saved_x[1])
            if self.fusion_method == 'sub':
                x_combd = self.activation(saved_x[0] - saved_x[1])
            if self.fusion_method == 'mul':
                x_combd = self.activation(saved_x[0] * saved_x[1])
            if self.fusion_method == 'div':
                x_combd = self.activation(saved_x[0] / saved_x[1])

            x1 = layer(x1, x_combd)
        return x1

class UNetCD(UNet):
    def __init__(self, out_dim, *args, **kwargs):
        super(UNetCD, self).__init__(*args, **kwargs)
        self.output_layer = nn.Conv2d(self.init_filters, out_dim, (3, 3), padding=1)

    def forward(self, x1, x2):
        x = super().forward(x1, x2)
        # Note that we don't perform the sigmoid here.
        return self.output_layer(x)

class UNetS2LatePooling(nn.Module):
    def __init__(self, layers, init_filters, fusion_method=None):
        super(UNetS2LatePooling, self).__init__()
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.init_filters = init_filters
        self.fusion_method = fusion_method

        filter_size = init_filters
        for i in range(layers - 1):
            if i == 0:
                self.down_layers.append(
                    UNetDownBlock(filter_size, filter_size)
                )
            elif i == 2:
                self.down_layers.append(
                    UNetDownBlock(filter_size, filter_size*2-32)
                )
            else:
                self.down_layers.append(
                    UNetDownBlock(filter_size, filter_size*2)
                )
            filter_size *= 2
        self.down_layers.append(UNetDownBlock(filter_size, filter_size * 2, pool=False))
        for i in range(layers):
            self.up_layers.append(
                UNetUpBlock(filter_size * 2, filter_size)
            )
            filter_size //= 2

        self.data_norm1 = nn.BatchNorm2d(4)
        self.init_layer1 = nn.Conv2d(4, init_filters, (7, 7), padding=3)
        self.data_norm2 = nn.BatchNorm2d(6)
        self.init_layer2 = nn.Conv2d(6, init_filters, (7, 7), padding=3)
        self.data_norm3 = nn.BatchNorm2d(3)
        self.init_layer3 = nn.Conv2d(3, init_filters, (7, 7), padding=2)
        self.activation = nn.ReLU()
        self.init_norm1 = nn.BatchNorm2d(init_filters)
        self.init_norm2 = nn.BatchNorm2d(init_filters)
        self.init_norm3 = nn.BatchNorm2d(init_filters)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x1_1, x1_2, x1_3, x2_1, x2_2, x2_3):
        x1_1 = self.data_norm1(x1_1)
        x1_1 = self.init_norm1(self.activation(self.init_layer1(x1_1)))

        x2_1 = self.data_norm1(x2_1)
        x2_1 = self.init_norm1(self.activation(self.init_layer1(x2_1)))
        
        x1_2 = self.data_norm2(x1_2)
        x1_2 = self.init_norm2(self.activation(self.init_layer2(x1_2)))

        x2_2 = self.data_norm2(x2_2)
        x2_2 = self.init_norm2(self.activation(self.init_layer2(x2_2)))
        
        x1_3 = self.data_norm3(x1_3)
        x1_3 = self.init_norm3(self.activation(self.init_layer3(x1_3)))

        x2_3 = self.data_norm3(x2_3)
        x2_3 = self.init_norm3(self.activation(self.init_layer3(x2_3)))
        
        saved_x = [[x1_1, x2_1]]
        i = 0
        for layer in self.down_layers:
            if i == 1:
                x1_1 = torch.cat((x1_1,x1_2), dim=1)
                x2_1 = torch.cat((x2_1,x2_2), dim=1)
            if i == 3:
                x1_1 = torch.cat((x1_1,x1_3), dim=1)
                x2_1 = torch.cat((x2_1,x2_3), dim=1)
            
            saved_x.append([x1_1,x2_1])
            x1_1 = self.dropout(layer(x1_1))
            x2_1 = self.dropout(layer(x2_1))
            
            i += 1
        
        is_first = True
        for layer, saved_x in zip(self.up_layers, reversed(saved_x)):
            if not is_first:
                is_first = False
                x1_1 = self.dropout(x1_1)
                
            if self.fusion_method == 'cat':
                x_combd = torch.cat(saved_x, 1)
            if self.fusion_method == 'add':
                x_combd = self.activation(saved_x[0] + saved_x[1])
            if self.fusion_method == 'sub':
                x_combd = self.activation(saved_x[0] - saved_x[1])
            if self.fusion_method == 'mul':
                x_combd = self.activation(saved_x[0] * saved_x[1])
            if self.fusion_method == 'div':
                x_combd = self.activation(saved_x[0] / saved_x[1])
            
            x1_1 = layer(x1_1, x_combd)
        return x1_1

class UNetClassifyS2(UNetS2LatePooling):
    def __init__(self, out_dim, *args, **kwargs):
        super(UNetClassifyS2, self).__init__(*args, **kwargs)
        self.output_layer = nn.Conv2d(self.init_filters, out_dim, (3, 3), padding=1)

    def forward(self, x1_1, x1_2, x1_3, x2_1, x2_2, x2_3):
        x = super().forward(x1_1, x1_2, x1_3, x2_1, x2_2, x2_3)
        # Note that we don't perform the sigmoid here.
        return self.output_layer(x)