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
DEVICE = 2
def w(v):
    if USE_CUDA:
        return v.cuda(DEVICE)
    return v

weight_factor = 0.023
epochs = 100
batch_size = 128
input_size = 32
layers = 4
lr = 0.01
init_filters = 32
loss_func = 'focal'
bands = ['B01','B02', 'B03', 'B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
data_dir = '../../datasets/onera/'
weight_path = '../../weights/onera/unet_siamese_late_pooling_mul_relu_inp32_13band_2dates_' + loss_func + '_hm_cnc_all_14_cities.pt'
train_csv = '../../datasets/onera/train.csv'
test_csv = '../../datasets/onera/test.csv'

net = w(UNetClassifyS2(layers=layers, init_filters=init_filters, fusion_method='mul'))

criterion = get_loss(loss_func, weight_factor)
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
        torch.save(net.state_dict(), weight_path)
    print(np.mean(losses), np.mean(iou), best_loss, best_iou)

    