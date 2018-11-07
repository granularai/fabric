from __future__ import print_function, division
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from config import GTEA as DATA
from utils.folder import OneraPreloader


use_gpu = torch.cuda.is_available()
DEVICE = 1

#Data statistics
num_classes = 2
class_map = DATA.rgb['class_map']

#Training parameters
lr = 0.01
momentum = 0.9
step_size = 10
gamma = 1
num_epochs = 10
batch_size = 32
data_dir = '../datasets/onera/images/'
label_dir = '../datasets/onera/training_labels/'
weights_dir = '../weights/onera/'
train_csv = '../datasets/onera/train.csv'
test_csv = '../datasets/onera/test.csv'

class Conv3DSegNet(nn.Module):
    def __init__(self):
        super(Conv3DSegNet, self).__init__()
        self.conv1 = nn.Conv3d(13, 64, 3)
        self.conv2 = nn.Conv3d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.deconv1 = nn.ConvTranspose2d(256, 256, 3)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 3)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 3)
        self.deconv4 = nn.ConvTranspose2d(64, 2, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print (x.size())
        x = F.relu(self.conv2(x))
        #print (x.size())
        x = x.view(-1, 128, 28, 28)
        x = F.relu(self.conv3(x))
        #print (x.size())
        x = F.relu(self.conv4(x))
        x = self.deconv1(x)
        #print (x.size())
        x = self.deconv2(x)
        #print (x.size())
        x = self.deconv3(x)
        #print (x.size())
        x = self.deconv4(x)
        #print (x.size())
        return x.view(-1, 32, 32)

def train_model(model, criterion, optimizer, scheduler, num_epochs=2000):
    """
        Training model with given criterion, optimizer for num_epochs.
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_loss = []
    train_acc = []
    test_acc = []
    test_loss = []

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

            for data in dataloaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda(DEVICE))
                    labels = Variable(labels.cuda(DEVICE))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                print ('##############################################################')
                print ("{} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)
                print (" {} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))
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
image_datasets = {'train': OneraPreloader(data_dir , train_csv, class_map, data_transforms['train']),
                    'test': OneraPreloader(data_dir , test_csv, class_map, data_transforms['test'])}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
file_name = __file__.split('/')[-1].split('.')[0]

#Create model and initialize/freeze weights
model_conv = torchvision.models.resnet50(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False
model_conv = ResNet50Bottom(model_conv)

if use_gpu:
    model_conv = model_conv.cuda(DEVICE)

#Initialize optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

#Train model
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs)