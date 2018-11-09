from __future__ import print_function, division
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
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
lr = 0.0001
momentum = 0.9
step_size = 50
gamma = 1
num_epochs = 50
batch_size = 32

data_dir = '../datasets/onera/'
weights_dir = '../weights/onera/'
train_csv = '../datasets/onera/train.csv'
test_csv = '../datasets/onera/test.csv'

class Conv3DSegNet(nn.Module):
    def __init__(self):
        super(Conv3DSegNet, self).__init__()
        
        self.conv11 = nn.Conv3d(13, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm3d(64)
        self.conv12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm3d(64)
        
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)
        
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256)
        
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128)
        
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64)
        
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64)
        self.conv11d = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        
    def forward(self, x):
         # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool3d(x12,kernel_size=2, stride=2,return_indices=True)

        x1p = x1p.view(-1, 64, 16, 16)
        id1 = id1.view(-1, 64, 16, 16)
        
        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)


        # Stage 3d
        x3d = F.max_unpool2d(x3p, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        
        return x11d

def train_model(model, criterion, optimizer, scheduler, num_epochs=2000):
    """
        Training model with given criterion, optimizer for num_epochs.
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_loss = []
    train_acc = []
    train_f1_micro = []
    train_f1_macro = []
    train_change_precision = []
    
    test_acc = []
    test_loss = []
    test_f1_micro = []
    test_f1_macro = []
    test_change_precision = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0
            running_preds = []
            running_trues = []

            for data in dataloaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda(DEVICE))
                    labels = Variable(labels.type(torch.LongTensor).cuda(DEVICE))
                else:
                    inputs, labels = Variable(inputs), Variable(labels.type(torch.LongTensor))

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy().flatten()
                labels = labels.cpu().numpy().flatten()
                running_corrects += np.sum(preds == labels) / (32*32)
                running_preds += list(preds)
                running_trues += list(labels)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_f1_macro = f1_score(running_trues, running_preds, average='macro')
            epoch_f1_micro = f1_score(running_trues, running_preds, average='micro')
            epoch_change_precision = precision_score(running_trues, running_preds, average='binary', pos_label=1)
            
            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                train_f1_micro.append(epoch_f1_micro)
                train_f1_macro.append(epoch_f1_macro)
                train_change_precision.append(epoch_change_precision)
                print ('##############################################################')
                print ("{} loss = {}, acc = {}, macro = {}, micro = {}, change_precision = {},".format(phase, epoch_loss, epoch_acc, epoch_f1_macro, epoch_f1_micro, epoch_change_precision))
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)
                test_f1_micro.append(epoch_f1_micro)
                test_f1_macro.append(epoch_f1_macro)
                test_change_precision.append(epoch_change_precision)
                print ("{} loss = {}, acc = {}, macro = {}, micro = {}, change_precision = {},".format(phase, epoch_loss, epoch_acc, epoch_f1_macro, epoch_f1_micro, epoch_change_precision))
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
image_datasets = {'train': OneraPreloader(data_dir , train_csv, input_size),
                    'test': OneraPreloader(data_dir , test_csv, input_size)}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
file_name = __file__.split('/')[-1].split('.')[0]

#Create model and initialize/freeze weights
model_conv = Conv3DSegNet()

if use_gpu:
    model_conv = model_conv.cuda(DEVICE)

#Initialize optimizer and loss function

class_weights = torch.from_numpy(np.array([1/(1-0.0220811521931),1/0.0220811521931])).float().cuda()
criterion = nn.CrossEntropyLoss(class_weights)

optimizer_conv = optim.SGD(model_conv.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

#Train model
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs)
