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
from utils.dataloaders import OneraPreloader, onera_siamese_loader, full_onera_loader

DROPOUT = 0.5

USE_CUDA = torch.cuda.is_available()

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v

epochs = 100
batch_size = 128
input_size = 32
layers = 5
lr = 0.01
init_filters = 64
loss_func = 'focal'
init_val = 0.022826975609355593
bands = ['B01','B02', 'B03', 'B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
data_dir = '../datasets/onera/'
weights_dir = '../weights/onera/'
train_csv = '../datasets/onera/train.csv'
test_csv = '../datasets/onera/test.csv'

net = w(UNetClassify(layers=layers, init_filters=init_filters, init_val=init_val))
weights = torch.load('../weights/onera/unet_siamese_prod_relu_inp32_13band_2dates_focal_hm_cnc_all_14_cities.pt')
net.load_state_dict(weights)

criterion = get_loss(loss_func)
optimizer = optim.Adam(net.parameters(), lr=lr)

full_load = full_onera_loader(data_dir, bands)
train_dataset = OneraPreloader(data_dir , train_csv, input_size, full_load, onera_siamese_loader)
train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = OneraPreloader(data_dir , test_csv, input_size, full_load, onera_siamese_loader)
test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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
        import copy
        best_net_dict = copy.deepcopy(net.state_dict())
        best_loss = np.mean(losses)
        torch.save(best_net_dict, '../weights/onera/unet_siamese_prod_relu_inp32_13band_2dates_' + loss_func + '_hm_cnc_all_14_cities.pt')
    print(np.mean(losses), np.mean(iou), best_loss, best_iou)
