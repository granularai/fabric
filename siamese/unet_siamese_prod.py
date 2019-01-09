import sys, glob, cv2, random, math, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import classification _report

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

parser = argparse.ArgumentParser(description='Siamese Change Detection network training')

parser.add_argument('--gpu_id', type=int, default=0, required=False, help='gpu to use for training')
parser.add_argument('--patch_size', type=int, default=32, required=False, help='input patch size')
parser.add_argument('--stride', type=int, default=16, required=False, help='stride at which to sample patches')
parser.add_argument('--num_workers', type=int, default=4, required=False, help='number of cpus to use for data preprocessing')
parser.add_argument('--augmentation', default=True, required=False, help='train with or without augmentation')

parser.add_argument('--epochs', type=int, default=50, required=False, help='number of eochs to train')
parser.add_argument('--layers', type=int, default=5, required=False, help='number of layers in unet')
parser.add_argument('--batch_size', type=int, default=128, required=False, help='batch size for training')
parser.add_argument('--init_filters', type=int, default=32, required=False, help='initial filter size of unet')
parser.add_argument('--bands', type=int, default=13, required=False, help='number of bands to use as input 4:[B02, B03, B04, B08] or 13:All bands')
parser.add_argument('--loss_func', default='focal', required=False, help='Loss function to use for training, bce, dice, focal')
parser.add_argument('--weight_factor', default=1.5, required=False, help='if focal loss is used pass gamma')
parser.add_argument('--fusion', default='mul', required=False, help='fusion of two dates, cat, add, mul, sub, div')
parser.add_argument('--lr', default=0.01, required=False, help='Learning rate')

parser.add_argument('--val_cities', default='0,1', required=False, help='''cities to use for validation,
                            0:abudhabi, 1:aguasclaras, 2:beihai, 3:beirut, 4:bercy, 5:bordeaux, 6:cupertino, 7:hongkong, 8:mumbai,
                            9:nantes, 10:paris, 11:pisa, 12:rennes, 14:saclay_e''')
parser.add_argument('--data_dir', required=True, help='data directory for training')
parser.add_argument('--weight_dir', required=True, help='directory to save weights')
parser.add_argument('--log_dir', required=True, help='directory to save training log')

opt = parser.parse_args()

print (opt)
USE_CUDA = torch.cuda.is_available()
def w(v):
    if USE_CUDA:
        return v.cuda(opt.gpu_id)
    return v

###############################
#1. CHANGE PRECISION, Recall, F1Score, IOU in metrics_and_losses
#2. Data augmentation in dataloader
#3. sampling code in dataloader
#4. log creator code
#5. args handling
###############################

if opt.bands==13:
    bands = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
elif opt.bands==4:
    bands = ['B02','B03','B04','B08']
else:
    print ('Only 4 and 13 bands are allowed for now.')
    exit()

train_metadata, val_metadata = get_train_val_metadata(opt.data_dir, opt.val_cities, opt.patch_size, opt.stride)
full_load = full_onera_loader(opt.data_dir, bands)
train_dataset = OneraPreloader(opt.data_dir , train_metadata, opt.patch_size, full_load, onera_siamese_loader)
train = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True)
val_dataset = OneraPreloader(opt.data_dir , val_metadata, opt.patch_size, full_load, onera_siamese_loader)
val = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True)

net = w(UNetClassify(layers=opt.layers, init_filters=opt.init_filters, num_channels=opt.bands, fusion_method=opt.fusion_method))

criterion = get_loss(opt.loss_func, opt.weight_factor)
optimizer = optim.Adam(net.parameters(), lr=opt.lr)

best_iou = -1.0
best_net_dict = None
best_epoch = -1
best_loss = 1000.0

for epoch in tqdm(range(opt.epochs)):
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
    gts = []
    preds = []

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
            iou.append(get_iou(label, res))
            gts += list(label.flatten())
            preds += list(res.flatten())

    cur_iou = np.mean(iou)
    stats = classification_report(gts, preds, labels=[0,1], target_names=['nochange','change'])

    if cur_iou > best_iou or (cur_iou == best_iou and np.mean(losses) < best_loss):
        best_iou = cur_iou
        best_epoch = epoch
        best_loss = np.mean(losses)
        torch.save(net.state_dict(), weight_path)

    print('val loss', np.mean(losses), 'epoch iou', np.mean(iou), 'best loss', best_loss, 'best iou', best_iou)
    print (stats)
