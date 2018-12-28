import glob
import cv2
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import random
import pandas as pd
import math
from tqdm import tqdm_notebook as tqdm
from utils.dataloaders import OneraPreloader, onera_siamese_loader_late_pooling, full_onera_loader

def get_iou(mask1, mask2):
    if np.sum(mask1 & mask2) == 0:
        return 0
    return np.sum(mask1 & mask2) / np.sum(mask1 | mask2)

DROPOUT = 0.0

class UNetBlock(nn.Module):
    def __init__(self, filters_in, filters_out):
        super().__init__()
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
        super().__init__(filters_in, filters_out)
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

    def forward(self, x, cross_x1, cross_x2):
        x = F.upsample(x, size=cross_x1.size()[-2:], mode='bilinear')
        x = self.upnorm(self.activation(self.upconv(x)))
        x = torch.cat((x, self.activation(cross_x2*cross_x1)), 1)
        return super().forward(x)

class UNet(nn.Module):
    def __init__(self, layers, init_filters):
        super().__init__()
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.init_filters = init_filters

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
        
        saved_x_copy = saved_x
        is_first = True
        for layer, saved_x in zip(self.up_layers, reversed(saved_x)):
            if not is_first:
                is_first = False
                x1_1 = self.dropout(x1)
            x1_1 = layer(x1_1, saved_x[0], saved_x[1])
        return x1_1

class UNetClassify(UNet):
    def __init__(self, *args, **kwargs):
        init_val = kwargs.pop('init_val', 0.5)
        super().__init__(*args, **kwargs)
        self.output_layer = nn.Conv2d(self.init_filters, 1, (3, 3), padding=1)

    def forward(self, x1_1, x1_2, x1_3, x2_1, x2_2, x2_3):
        x = super().forward(x1_1, x1_2, x1_3, x2_1, x2_2, x2_3)
        # Note that we don't perform the sigmoid here.
        return self.output_layer(x)

# From: https://github.com/pytorch/pytorch/issues/1249
def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def weighted_dice_coef(y_pred, y_true):
    mean = 0.022826975609355593
    w_1 = 1/mean**2
    w_0 = 1/(1-mean)**2
    y_true_f_1 = y_true.view(-1)
    y_pred_f_1 = y_pred.view(-1)
    y_true_f_0 = 1-y_true.view(-1)
    y_pred_f_0 = 1-y_pred.view(-1)

    intersection_0 = (y_true_f_0 * y_pred_f_0).sum()
    intersection_1 = (y_true_f_1 * y_pred_f_1).sum()

    return 1 - 2 * (w_0 * intersection_0 + w_1 * intersection_1) / ((w_0 * (y_true_f_0.sum() + y_pred_f_0.sum())) + (w_1 * (y_true_f_1.sum() + y_pred_f_1.sum())))

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()

def get_loss(loss):
    if loss == 'dice':
        print('dice')
        return dice_loss
    elif loss == 'wdice':
        print('wdice')
        return weighted_dice_coef
    elif loss == 'focal':
        print('focal')
        return w(FocalLoss(0.023))
    else:
        print('bce')
        return w(nn.BCEWithLogitsLoss())

USE_CUDA = torch.cuda.is_available()
DEVICE = 2
def w(v):
    if USE_CUDA:
        return v.cuda(DEVICE)
    return v

epochs = 100
batch_size = 128
input_size = 32
layers = 4
lr = 0.01
init_filters = 32
loss_func = 'focal'
init_val = 0.022826975609355593
bands = ['B01','B02', 'B03', 'B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
data_dir = '../datasets/onera/'
weights_dir = '../weights/onera/'
train_csv = '../datasets/onera/train.csv'
test_csv = '../datasets/onera/test.csv'

net = w(UNetClassify(layers=layers, init_filters=init_filters, init_val=init_val))
criterion = get_loss(loss_func)
optimizer = optim.Adam(net.parameters(), lr=lr)

full_load = full_onera_loader(data_dir, bands)
train_dataset = OneraPreloader(data_dir , train_csv, input_size, full_load, onera_siamese_loader_late_pooling)
train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = OneraPreloader(data_dir , test_csv, input_size, full_load, onera_siamese_loader_late_pooling)
test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

best_iou = -1.0
best_net_dict = None
best_epoch = -1
best_loss = 1000.0

for epoch in tqdm(range(epochs)):
    net.train()
    train_losses = []
    for batch_img1_cat1, batch_img1_cat2, batch_img1_cat3, batch_img2_cat1, batch_img2_cat2, batch_img2_cat3, labels in train:
        batch_img1_cat1 = w(autograd.Variable(batch_img1_cat1))
        batch_img1_cat2 = w(autograd.Variable(batch_img1_cat2))
        batch_img1_cat3 = w(autograd.Variable(batch_img1_cat3))
        
        batch_img2_cat1 = w(autograd.Variable(batch_img2_cat1))
        batch_img2_cat2 = w(autograd.Variable(batch_img2_cat2))
        batch_img2_cat3 = w(autograd.Variable(batch_img2_cat3))
        
        labels = w(autograd.Variable(labels))

        optimizer.zero_grad()
        output = net(batch_img1_cat1, batch_img1_cat2, batch_img1_cat3, batch_img2_cat1, batch_img2_cat2, batch_img2_cat3)
        loss = criterion(output, labels.view(-1,1,input_size,input_size).float())
        loss.backward()
        train_losses.append(loss.item())

        optimizer.step()
    print('train loss', np.mean(train_losses))
    
    optimizer.zero_grad()
    net.eval()
    losses = []
    iou = []
    to_show = random.randint(0, len(test) - 1)
    for batch_img1_cat1, batch_img1_cat2, batch_img1_cat3, batch_img2_cat1, batch_img2_cat2, batch_img2_cat3, labels_true in test:
        labels = w(autograd.Variable(labels_true))
        
        batch_img1_cat1 = w(autograd.Variable(batch_img1_cat1))
        batch_img1_cat2 = w(autograd.Variable(batch_img1_cat2))
        batch_img1_cat3 = w(autograd.Variable(batch_img1_cat3))
        
        batch_img2_cat1 = w(autograd.Variable(batch_img2_cat1))
        batch_img2_cat2 = w(autograd.Variable(batch_img2_cat2))
        batch_img2_cat3 = w(autograd.Variable(batch_img2_cat3))
        
        output = net(batch_img1_cat1, batch_img1_cat2, batch_img1_cat3, batch_img2_cat1, batch_img2_cat2, batch_img2_cat3)
        
        loss = criterion(output, labels.view(-1,1,input_size,input_size).float())
        losses += [loss.item()] * batch_size
        
        result = (F.sigmoid(output).data.cpu().numpy() > 0.5)
        
        for label, res in zip(labels_true, result):
            label = label.cpu().numpy()[:, :] > 0.5
#             print (label.min(), label.max(), res.min(), res.max())
            iou.append(get_iou(label, res))

    cur_iou = np.mean(iou)
    if cur_iou > best_iou or (cur_iou == best_iou and np.mean(losses) < best_loss):
        best_iou = cur_iou
        best_epoch = epoch
        best_loss = np.mean(losses)
        torch.save(net.state_dict(), '../weights/onera/unet_siamese_late_pooling_prod_relu_inp32_13band_2dates_' + loss_func + '_hm_cnc_all_14_cities.pt')
    print(np.mean(losses), np.mean(iou), best_loss, best_iou)

    