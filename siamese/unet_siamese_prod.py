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
DEVICE = 0
def w(v):
    if USE_CUDA:
        return v.cuda(DEVICE)
    return v

input_size = 64
stride = 16
num_workers = 4
gpu_id = 0
augmentation = True
train_city_split = [0,1]

epochs = 200
batch_size = 64
layers = 6
lr = 0.01
init_filters = 32

loss_func = 'focal'
weight_factor = 2

bands = ['B01','B02', 'B03', 'B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
num_channels = len(bands) #how to pass bands in param
fusion_method = 'mul' #cat, add, mul, sub, div

data_dir = '../../datasets/onera/'
weight_path = '../../weights/onera/unet_siamese_prod_relu_inp64_4band_2dates_focal_6layers_1e-3lr_hm_cnc_all_14_cities.pt'
train_csv = '../../datasets/onera/train_64x64.csv'
test_csv = '../../datasets/onera/test_64x64.csv'
log_path = '../../logs/onera/'

###############################
#1. CHANGE PRECISION, Recall, F1Score, IOU in metrics_and_losses
#2. Data augmentation in dataloader
#3. sampling code in dataloader
#4. log creator code
#5. args handling
###############################

exit()
full_load = full_onera_loader(data_dir, bands)
train_dataset = OneraPreloader(data_dir , train_csv, input_size, full_load, onera_siamese_loader)
train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

test_dataset = OneraPreloader(data_dir , test_csv, input_size, full_load, onera_siamese_loader)
test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

net = w(UNetClassify(layers=layers, init_filters=init_filters, num_channels=num_channels, fusion_method=fusion_method))

criterion = get_loss(loss_func, weight_factor)
optimizer = optim.Adam(net.parameters(), lr=lr)

best_iou = -1.0
best_net_dict = None
best_epoch = -1
best_loss = 1000.0

for epoch in tqdm(range(epochs)):
    net.train()
    train_losses = []
    for batch_img1, batch_img2, labels in train:
        batch_img1 = w(autograd.Variable(batch_img1))
        batch_img2 = w(autograd.Variable(batch_img2))
        labels = w(autograd.Variable(labels))

        optimizer.zero_grad()
        output = net(batch_img1, batch_img2)
        loss = criterion(output, labels.view(-1,1,input_size,input_size).float())
        loss.backward()
        train_losses.append(loss.item())

        optimizer.step()
        torch.cuda.empty_cache()
    print('train loss', np.mean(train_losses))

    optimizer.zero_grad()
    net.eval()
    losses = []
    iou = []
    to_show = random.randint(0, len(test) - 1)
    for batch_img1, batch_img2, labels_true in test:
        labels = w(autograd.Variable(labels_true))
        batch_img1 = w(autograd.Variable(batch_img1))
        batch_img2 = w(autograd.Variable(batch_img2))
        output = net(batch_img1, batch_img2)
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
