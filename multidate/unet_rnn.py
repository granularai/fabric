import sys, glob, cv2, random, math

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import datasets, models, transforms

sys.path.append('../utils')
sys.path.append('../models')
from dataloaders import *
from unet_blocks import *
from metrics_and_losses import *

USE_CUDA = torch.cuda.is_available()
DEVICE = 1
def w(v):
    if USE_CUDA:
        return v.cuda(DEVICE)
    return v

DROPOUT = 0.5

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
    def __init__(self, input_size, filters_in, filters_out, pooling=True):
        super().__init__(filters_in, filters_out)
        self.filters_out = filters_out
        self.input_size = input_size
        self.pooling = pooling
        
        if pooling:
            self.pool = nn.MaxPool2d(2)
            self.recurrent_weights = nn.Parameter(torch.Tensor(filters_out, input_size//2, input_size//2))
            self.recurrent_activation = nn.ReLU()
        else:
            self.pool = lambda x: x
            self.recurrent_weights = nn.Parameter(torch.Tensor(filters_out, input_size, input_size))
            self.recurrent_activation = nn.ReLU()

    def forward(self, xinp):
        if self.pooling:
            h = w(torch.randn(xinp.size()[1], self.filters_out, self.input_size//2, self.input_size//2))
            xout = w(Variable(torch.zeros(xinp.size()[0], xinp.size()[1], self.filters_out, self.input_size//2, self.input_size//2)))
        else:
            h = w(torch.randn(xinp.size()[1], self.filters_out, self.input_size, self.input_size))
            xout = w(Variable(torch.zeros(xinp.size()[0], xinp.size()[1], self.filters_out, self.input_size, self.input_size)))
            
        for i in range(xinp.size()[0]):
            xs = self.pool(super().forward(xinp[i]))
            xs = self.recurrent_activation(self.recurrent_weights * h + xs)
            xout[i] = xs
            
        return xs, xout

class UNetUpBlock(UNetBlock):
    def __init__(self, filters_in, filters_out):
        super().__init__(filters_in, filters_out)
        self.upconv = nn.Conv2d(filters_in, filters_in // 2, (3, 3), padding=1)
        self.upnorm = nn.BatchNorm2d(filters_in // 2)

    def forward(self, x, cross_x):
        x = F.upsample(x, size=cross_x.size()[-2:], mode='bilinear')
        x = self.upnorm(self.activation(self.upconv(x)))
        x = torch.cat((x, cross_x), 1)
        return super().forward(x)

class UNet(nn.Module):
    def __init__(self, input_size, layers, init_filters):
        super().__init__()
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.input_size = input_size
        self.init_filters = init_filters

        
        self.data_norm = nn.BatchNorm2d(13)
        self.init_layer = nn.Conv2d(13, init_filters, (7, 7), padding=(3,3))
        self.activation = nn.ReLU()
        self.init_norm = nn.BatchNorm2d(init_filters)
        self.dropout = nn.Dropout(DROPOUT)
        
        self.recurrent_weights = nn.Parameter(torch.Tensor(init_filters, input_size, input_size))
        self.recurrent_activation = nn.Tanh()
        
        filter_size = init_filters
        for i in range(layers - 1):
            self.down_layers.append(
                UNetDownBlock(input_size, filter_size, filter_size * 2)
            )
            filter_size *= 2
            input_size = input_size // 2
        self.down_layers.append(UNetDownBlock(input_size, filter_size, filter_size * 2, pooling=False))
        for i in range(layers):
            self.up_layers.append(
                UNetUpBlock(filter_size * 2, filter_size)
            )
            filter_size //= 2

    def forward(self, xinp):
        h = w(torch.randn(xinp.size()[1], self.init_filters, self.input_size, self.input_size))
        xout = w(Variable(torch.zeros(xinp.size()[0], xinp.size()[1], self.init_filters, self.input_size, self.input_size)))
        
        for i in range(xinp.size()[0]):
            xs = self.data_norm(xinp[i])
            xs = self.init_norm(self.activation(self.init_layer(xs)))
            xs = self.recurrent_activation(self.recurrent_weights * h + xs)
            xout[i] = xs
            
        saved_x = [xs]
        for layer in self.down_layers:
            saved_x.append(xs)
            xs, xout = layer(xout)
            xs = self.dropout(xs)
        is_first = True
        for layer, saved_x in zip(self.up_layers, reversed(saved_x)):
            if not is_first:
                is_first = False
                xs = self.dropout(xs)
            xs = layer(xs, saved_x)
        return xs

class UNetClassify(UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_layer = nn.Conv2d(self.init_filters, 1, (3, 3), padding=1)

    def forward(self, x):
        x = super().forward(x)
        # Note that we don't perform the sigmoid here.
        return self.output_layer(x)


weight_factor = 0.023
epochs = 100
batch_size = 64
input_size = 64
layers = 6
lr = 0.01
init_filters = 32
loss_func = 'focal'
bands = ['B01','B02', 'B03', 'B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
data_dir = '../../datasets/onera/'
weight_path = '../../weights/onera/unet_recurrent_relu_inp64_13band_5dates_' + loss_func + '_hm_cnc_all_14_cities.pt'
train_csv = '../../datasets/onera/train_all_64x64.csv'
test_csv = '../../datasets/onera/test_64x64.csv'

net = w(UNetClassify(input_size=input_size, layers=layers, init_filters=init_filters))
# weights = torch.load(weight_path)
# net.load_state_dict(weights)

criterion = get_loss(loss_func, weight_factor)
optimizer = optim.Adam(net.parameters(), lr=lr)

full_load = full_onera_multidate_loader(data_dir, bands)
train_dataset = OneraPreloader(data_dir , train_csv, input_size, full_load, onera_loader)
train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = OneraPreloader(data_dir , test_csv, input_size, full_load, onera_loader)
test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

best_iou = -1.0
best_net_dict = None
best_epoch = -1
best_loss = 1000.0

for epoch in tqdm(range(epochs)):
    net.train()
    train_losses = []
    for batch_img, labels in train:
        batch_img = batch_img.permute(2,0,1,3,4)
        batch_img = w(autograd.Variable(batch_img))
        
        labels = w(autograd.Variable(labels))

        optimizer.zero_grad()
        output = net(batch_img)
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
    for batch_img, labels_true in test:
        batch_img = batch_img.permute(2,0,1,3,4)
        labels = w(autograd.Variable(labels_true))
        batch_img = w(autograd.Variable(batch_img))
       
        output = net(batch_img)
        
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
        torch.save(net.state_dict(), weight_path)
    print(np.mean(losses), np.mean(iou), best_loss, best_iou)
