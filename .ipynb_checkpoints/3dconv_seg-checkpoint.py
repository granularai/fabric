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
        self.conv1 = nn.Conv3d(13, 64, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 3)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3)
        self.deconv3 = nn.ConvTranspose2d(64, 1, 2)

    def forward(self, x):
        #print (x.size())
        x = F.relu(self.conv1(x))
        #print (x.size())
        x = x.view(-1, 64, 31, 31)
        x = F.relu(self.conv2(x))
        #print (x.size())
        x = F.relu(self.conv3(x))
        #print (x.size())
        x = self.deconv1(x)
        #print (x.size())
        x = self.deconv2(x)
        #print (x.size())
        x = self.deconv3(x)
        #print (x.size())
        x = x.view(-1, 32, 32)
        return x

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
                    labels = Variable(labels.type(torch.FloatTensor).cuda(DEVICE))
                else:
                    inputs, labels = Variable(inputs), Variable(labels.type(torch.FloatTensor))

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                
                preds = outputs > 0.5
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
# weight1 = np.ones((32,32)) * 1/(1-0.0220811521931)
# weight2 = np.ones((32,32)) * 1/0.0220811521931
# class_weights = np.transpose(np.stack([weight1, weight2]), (1,2,0))
# class_weights = torch.from_numpy(class_weights).cuda(DEVICE)
# print (class_weights.size())
criterion = nn.BCEWithLogitsLoss()

optimizer_conv = optim.SGD(model_conv.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

#Train model
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs)