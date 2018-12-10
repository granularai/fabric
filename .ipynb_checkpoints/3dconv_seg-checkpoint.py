from __future__ import print_function, division
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function, Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from utils.dataloaders import OneraPreloader
from sklearn.metrics import f1_score, precision_score

use_gpu = torch.cuda.is_available()
DEVICE = 0

#Data statistics
input_size = 32
num_classes = 2

#Training parameters
lr = 0.01
momentum = 0.9
step_size = 20
gamma = 1
num_epochs = 200
batch_size = 128

#data_stat
all_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
bands_mean = {}
bmean = [1570.68786621,  1365.48986816,  1284.81140137,  1298.93725586,
        1430.97460938,  1860.87902832,  2081.93554688,  1994.85632324,
        2214.60205078,   641.43334961,    14.36389923,  1957.03930664,
        1419.28393555]
for i in range(len(all_bands)):
    bands_mean[all_bands[i]] = bmean[i]
    
bands_min = {}
bmin = [1.01300000e+03,   1.00000000e+00,   1.00000000e+00,
         1.00000000e+00,   1.99000000e+02,   1.22000000e+02,
         1.00000000e+00,   1.48000000e+02,   8.20000000e+01,
         4.10000000e+01,   2.00000000e+00,   6.10000000e+01,
         2.10000000e+01]
for i in range(len(all_bands)):
    bands_min[all_bands[i]] = bmin[i]
    
bands_max = {}
bmax = [5089.,  28000.,  28000.,  28000.,  26901.,  28000.,  25591.,
        28000.,  26305.,   5412.,    158.,  25104.,  19940.]
for i in range(len(all_bands)):
    bands_max[all_bands[i]] = bmax[i]
    
bands_std = {}
bstd = [269.31314087,   414.13943481,   536.9800415 ,   764.6965332 ,
         711.48406982,   745.4911499 ,   831.77252197,   865.61376953,
         902.31213379,   318.97601318,     8.65729427,  1007.27667236,
         860.51831055]
for i in range(len(all_bands)):
    bands_std[all_bands[i]] = bstd[i]

bands = ['B02', 'B03', 'B04', 'B08']
data_dir = '../datasets/onera/'
weights_dir = '../weights/onera/'
train_csv = '../datasets/onera/train.csv'
test_csv = '../datasets/onera/test.csv'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class BinaryCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        """
        Binary cross entropy loss 2D
        Args:
            weight:
            size_average:
        """
        super(BinaryCrossEntropyLoss2d, self).__init__()
#         weight = torch.from_numpy(np.array([1/18393175.0,1/2447273.0])).float().cuda()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)  # Flatten
        targets_flat = targets.view(-1)  # Flatten
        return self.bce_loss(probs_flat, targets_flat)

    
class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, logits, targets):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (targets.size() == logits.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), logits.size()))

        max_val = (-logits).clamp(min=0)
        loss = logits - logits * targets + max_val + ((-max_val).exp() + (-logits - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-logits * (targets * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()
    
    
class Conv3DSegNet(nn.Module):
    def __init__(self):
        super(Conv3DSegNet, self).__init__()
        
        self.conv11 = nn.Conv3d(4, 16, kernel_size=(1,3,3), padding=(0,1,1))
        self.conv12 = nn.Conv3d(16, 16, kernel_size=(1,3,3), padding=(0,1,1))
        
        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.conv43d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv42d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv41d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        self.conv33d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv32d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv31d = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        self.conv22d = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv21d = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        self.conv12d = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv11d = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
         # Stage 1
        x11 = F.relu(self.conv11(x))
        x12 = F.relu(self.conv12(x11))
        x1p, id1 = F.max_pool3d(x12,kernel_size=2, stride=2,return_indices=True)

#         print (x1p.size())
        x1p = x1p.view(-1, x1p.size()[1], x1p.size()[3], x1p.size()[4])
        id1 = id1.view(-1, x1p.size()[1], id1.size()[3], id1.size()[4])
        
        # Stage 2
        x21 = F.relu(self.conv21(x1p))
        x22 = F.relu(self.conv22(x21))
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x31 = F.relu(self.conv31(x2p))
        x32 = F.relu(self.conv32(x31))
        x33 = F.relu(self.conv33(x32))
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)


        # Stage 4
        x41 = F.relu(self.conv41(x3p))
        x42 = F.relu(self.conv42(x41))
        x43 = F.relu(self.conv43(x42))
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)
        
        # Stage 4d
        x4d = F.max_unpool2d(x4p, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.conv43d(x4d))
        x42d = F.relu(self.conv42d(x43d))
        x41d = F.relu(self.conv41d(x42d))
        
        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.conv33d(x3d))
        x32d = F.relu(self.conv32d(x33d))
        x31d = F.relu(self.conv31d(x32d))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.conv22d(x2d))
        x21d = F.relu(self.conv21d(x22d))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.conv12d(x1d))
        x11d = self.conv11d(x12d)
        
        out = torch.squeeze(x11d, dim=1)
        return out

def train_model(model, criterion, optimizer, scheduler, num_epochs=2000):
    """
        Training model with given criterion, optimizer for num_epochs.
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_losses = []
    train_accs = []
    
    test_losses = []
    test_accs = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            losses = AverageMeter()
            dice_coeffs = AverageMeter()

            for data in dataloaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda(DEVICE))
                    labels = Variable(labels.type(torch.FloatTensor).cuda(DEVICE))
                else:
                    inputs, labels = Variable(inputs), Variable(labels.type(torch.FloatTensor))

                optimizer.zero_grad()
                logits = model(inputs)

                probs = F.sigmoid(logits)
#                 pred = (probs > 0.5).float()
#                 _, pred = torch.max(logits.data, 1)

                # backward + optimize
                loss =  dice_coeff(probs, labels) + criterion[0](probs, labels) + criterion[2](probs, labels) 
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # print statistics
#                 acc = torch.sum(pred == labels) / (input_size * input_size)
#                 acc = dice_coeff(pred, labels)
                losses.update(loss.item(), labels.size()[0])
#                 dice_coeffs.update(acc.item(), labels.size()[0])
             
            epoch_loss = losses.avg
            epoch_acc = 0#dice_coeffs.avg
            
            if phase=='train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
                print ('##############################################################')
                print ("{} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))
            else:
                test_losses.append(epoch_loss)
                test_accs.append(epoch_acc)
                print ("{} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))
                print ('##############################################################')


            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model, weights_dir + '3dconv_seg.pt')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    model = torch.load(weights_dir + '3dconv_seg.pt')

    return model


#Dataload and generator initialization
image_datasets = {'train': OneraPreloader(data_dir , train_csv, input_size, bands, bands_mean, bands_std, bands_max),
                    'test': OneraPreloader(data_dir , test_csv, input_size, bands, bands_mean, bands_std, bands_max)}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
file_name = __file__.split('/')[-1].split('.')[0]

#Create model and initialize/freeze weights
model_conv = Conv3DSegNet()

if use_gpu:
    model_conv = model_conv.cuda(DEVICE)

# inp = torch.randn(8, 4, 2, 128, 128).cuda()
# out = model_conv(inp)
# print (out.size())
#Initialize optimizer and loss function

class_weights = torch.from_numpy(np.array([1/10699569.0,1/811215.0])).float().cuda()
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion1 = BinaryCrossEntropyLoss2d()
criterion2 = DiceCoeff()
criterion3 = FocalLoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

#Train model
model_conv = train_model(model_conv, [criterion1, criterion2, criterion3], optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs)
