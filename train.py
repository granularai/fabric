import sys, glob, cv2, random, math, argparse
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.metrics import precision_recall_fscore_support as prfs

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import datasets, models, transforms

sys.path.append('.')
from utils.dataloaders import *
from models.bidate_model import *
from utils.metrics import *

from moonshot import alert

parser = argparse.ArgumentParser(description='Training change detection network')

parser.add_argument('--patch_size', type=int, default=120, required=False, help='input patch size')
parser.add_argument('--stride', type=int, default=10, required=False, help='stride at which to sample patches')
parser.add_argument('--aug', default=True, required=False, help='Do augmentation or not')

parser.add_argument('--gpu_ids', default='0,1,2,3', required=False, help='gpus ids for parallel training')
parser.add_argument('--num_workers', type=int, default=90, required=False, help='Number of cpu workers')

parser.add_argument('--epochs', type=int, default=10, required=False, help='number of eochs to train')
parser.add_argument('--batch_size', type=int, default=256, required=False, help='batch size for training')
parser.add_argument('--loss', type=str, default='bce', required=False, help='bce,focal')
parser.add_argument('--gamma', type=float, default=2, required=False, help='if focal loss is used pass gamma')
parser.add_argument('--lr', type=float, default=0.01, required=False, help='Learning rate')

parser.add_argument('--val_cities', default='0,1', required=False, help='''cities to use for validation,
                            0:abudhabi, 1:aguasclaras, 2:beihai, 3:beirut, 4:bercy, 5:bordeaux, 6:cupertino, 7:hongkong, 8:mumbai,
                            9:nantes, 10:paris, 11:pisa, 12:rennes, 14:saclay_e''')

parser.add_argument('--data_dir', default='../datasets/onera/', required=False, help='data directory for training')
parser.add_argument('--weight_dir', default='../weights/', required=False, help='directory to save weights')
parser.add_argument('--log_dir', default='../logs/', required=False, help='directory to save training log')

opt = parser.parse_args()

if opt.loss == 'bce':
    path = 'cd_patchSize_' + str(opt.patch_size) + '_stride_' + str(opt.stride) + \
            '_batchSize_' + str(opt.batch_size) + '_loss_' + opt.loss  + \
            '_lr_' + str(opt.lr) + '_epochs_' + str(opt.epochs) +\
            '_valCities_' + opt.val_cities 

if opt.loss == 'focal':
    path = 'cd_patchSize_' + str(opt.patch_size) + '_stride_' + str(opt.stride) + \
            '_batchSize_' + str(opt.batch_size) + '_loss_' + opt.loss + '_gamma_' + str(opt.gamma) + \
            '_lr_' + str(opt.lr) + '_epochs_' + str(opt.epochs) +\
            '_valCities_' + opt.val_cities 

weight_path = opt.weight_dir + path + '.pt'
log_path = opt.log_dir + path + '.log'

fout = open(log_path, 'w')
fout.write(str(opt))


train_samples, test_samples = get_train_val_metadata(opt.data_dir, opt.val_cities, opt.patch_size, opt.stride)
print ('train samples : ', len(train_samples))
print ('test samples : ', len(test_samples))
fout.write('\n')
fout.write('train samples:' + str(len(train_samples)) + ' test samples:' + str(len(test_samples)))
fout.write('\n')

full_load = full_onera_loader(opt.data_dir)

train_dataset = OneraPreloader(opt.data_dir, train_samples, full_load, opt.patch_size, opt.aug)
test_dataset = OneraPreloader(opt.data_dir, test_samples, full_load, opt.patch_size, opt.aug)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

model = BiDateNet(13, 1).cuda()
model = nn.DataParallel(model, device_ids=[int(x) for x in opt.gpu_ids.split(',')])

if opt.loss == 'bce':
    criterion = nn.BCEWithLogitsLoss()
if opt.loss == 'focal':
    criterion = FocalLoss(opt.gamma)

optimizer = optim.SGD(model.parameters(), lr=opt.lr)


best_f1s = -1
for epoch in range(opt.epochs):
    train_losses = []
    train_corrects = []
    train_precisions = []
    train_recalls = []
    train_f1scores = []
    
    model.train()

    t = trange(len(train_loader))
    for batch_img1, batch_img2, labels in train_loader:
        batch_img1 = autograd.Variable(batch_img1).cuda()
        batch_img2 = autograd.Variable(batch_img2).cuda()
        
        labels = autograd.Variable(labels).float().cuda()
        labels = labels.view(-1, 1, opt.patch_size, opt.patch_size)

        optimizer.zero_grad()
        preds = model(batch_img1, batch_img2)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        preds = torch.sigmoid(preds) > 0.5
        corrects = 100 * (preds == labels.byte()).sum() / (labels.size()[0] * opt.patch_size * opt.patch_size)
        
        train_report = prfs(labels.data.cpu().numpy().flatten(), preds.data.cpu().numpy().flatten(), average='binary')
        
        train_losses.append(loss.item())
        train_corrects.append(corrects.item())
        train_precisions.append(train_report[0])
        train_recalls.append(train_report[1])
        train_f1scores.append(train_report[2])
        
        t.set_postfix(loss=loss.data.tolist(), accuracy=corrects.data.tolist())
        t.update()
        
    train_loss = np.mean(train_losses)
    train_acc = np.mean(train_corrects)
    train_prec = np.mean(train_precisions)
    train_rec = np.mean(train_recalls)
    train_f1s = np.mean(train_f1scores)
    print ('train loss : ', train_loss, ' train accuracy : ', train_acc, ' avg. precision : ', train_prec, 'avg. recall : ', train_rec, ' avg. f1 score : ', train_f1s)


    model.eval()
    
    test_losses = []
    test_corrects = []
    test_precisions = []
    test_recalls = []
    test_f1scores = []
    
    t = trange(len(test_loader))
    for batch_img1, batch_img2, labels in test_loader:
        batch_img1 = autograd.Variable(batch_img1).cuda()
        batch_img2 = autograd.Variable(batch_img2).cuda()
        
        labels = autograd.Variable(labels).float().cuda()
        labels = labels.view(-1, 1, opt.patch_size, opt.patch_size)

        preds = model(batch_img1, batch_img2)
        loss = criterion(preds, labels)

        preds = torch.sigmoid(preds) > 0.5
        corrects = 100 * (preds == labels.byte()).sum() / (labels.size()[0] * opt.patch_size * opt.patch_size)
    
        test_report = prfs(labels.data.cpu().numpy().flatten(), preds.data.cpu().numpy().flatten(), average='binary')
        
        test_losses.append(loss.item())
        test_corrects.append(corrects.item())
        test_precisions.append(test_report[0])
        test_recalls.append(test_report[1])
        test_f1scores.append(test_report[2])
        
        t.set_postfix(loss=loss.data.tolist(), accuracy=corrects.data.tolist())
        t.update()

    test_loss = np.mean(test_losses)
    test_acc = np.mean(test_corrects)
    test_prec = np.mean(test_precisions)
    test_rec = np.mean(test_recalls)
    test_f1s = np.mean(test_f1scores)
    print ('test loss : ', test_loss, ' test accuracy : ', test_acc, ' avg. precision : ', test_prec, 'avg. recall : ', test_rec, ' avg. f1 score : ', test_f1s)

    fout.write('train loss : ' + str(train_loss) + ' test loss : ' + str(test_loss) + '\n')
    
    if test_f1s < best_f1s:
        torch.save(model, weight_path)
        best_f1s = test_loss

    alert.slack_alert({
    'author_name':'Sagar',
    'text':'UNet-BiDate: Change Detection',
    'fields':[
        {
            "title": "Progress",
            "value": str(epoch) + '/' + str(opt.epochs) + " epochs",
            "short": False
         },
        {
            "title": "Train Accuracy",
            "value": str(train_acc),
            "short": False
        },
        {
            "title": "Train Loss",
            "value": str(train_loss),
            "short": False
        },
        {
            "title": "Train Average Precision",
            "value": str(train_prec),
            "short": False
        },
        {
            "title": "Train Average Recall",
            "value": str(train_rec),
            "short": False
        },
        {
            "title": "Train Average F1 Score",
            "value": str(train_f1s),
            "short": False
        },
        {
            "title": "Test Accuracy",
            "value": str(test_acc),
            "short": False
        },
        {
            "title": "Test Loss",
            "value": str(test_loss),
            "short": False
        },
        {
            "title": "Test Average Precision",
            "value": str(test_prec),
            "short": False
        },
        {
            "title": "Test Average Recall",
            "value": str(test_rec),
            "short": False
        },
        {
            "title": "Test Average F1 Score",
            "value": str(test_f1s),
            "short": False
        }
    ]})
    
fout.close()