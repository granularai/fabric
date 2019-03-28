from comet_ml import Experiment as CometExperiment
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
from utils.parser import get_parser_with_args
from utils.helpers import get_loaders, define_output_paths, download_dataset, get_criterion, load_model, initialize_metrics, get_mean_metrics, set_train_metrics, set_val_metrics

from polyaxon_client.tracking import Experiment, get_log_level, get_data_paths, get_outputs_path
from polystores.stores.manager import StoreManager

import logging


###
### Initialize experiments for polyaxon and comet.ml
###

comet = CometExperiment('QQFXdJ5M7GZRGri7CWxwGxPDN', project_name="cd_lulc")
experiment = Experiment()
logging.basicConfig(level=logging.INFO)




###
### Initialize Parser and define arguments
###

parser = get_parser_with_args()
opt = parser.parse_args()

###
### Set up environment: define paths, download data, and set device
###

weight_path, log_path = define_output_paths(opt)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))
download_dataset(opt.dataset, comet)
train_loader, val_loader = get_loaders(opt)



###
### Load Model then define other aspects of the model
###

logging.info('LOADING Model')
model = load_model(opt, device)

criterion = get_criterion(opt)
criterion_lulc = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=opt.lr)




###
### Set starting values
###

best_metrics = {'cd_val_f1scores':-1}



###
### Begin Training
###
logging.info('STARTING training')
for epoch in range(opt.epochs):

    train_metrics, val_metrics = initialize_metrics()
    with comet.train():
        model.train()
        logging.info('SET model mode to train!')
        batch_iter = 0
        for batch_img1, batch_img2, labels, masks in train_loader:
            logging.info("batch: "+str(batch_iter)+" - "+str(batch_iter+opt.batch_size))
            batch_iter = batch_iter+opt.batch_size
            batch_img1 = autograd.Variable(batch_img1).to(device)
            batch_img2 = autograd.Variable(batch_img2).to(device)

            labels = autograd.Variable(labels).long().to(device)
            masks = autograd.Variable(masks).long().to(device)

            optimizer.zero_grad()
            cd_preds, lulc_preds = model(batch_img1, batch_img2)
            cd_loss = criterion(cd_preds, labels)
            lulc_loss = criterion_lulc(lulc_preds, masks)

            loss = cd_loss + lulc_loss
            loss.backward()
            optimizer.step()

            _, cd_preds = torch.max(cd_preds, 1)
            _, lulc_preds = torch.max(lulc_preds, 1)

            cd_corrects = 100 * (cd_preds.byte() == labels.squeeze().byte()).sum() / (labels.size()[0] * opt.patch_size * opt.patch_size)
            lulc_corrects = 100 * (lulc_preds.byte() == masks.squeeze().byte()).sum() / (masks.size()[0] * opt.patch_size * opt.patch_size)
            cd_train_report = prfs(labels.data.cpu().numpy().flatten(), cd_preds.data.cpu().numpy().flatten(), average='binary', pos_label=1)
            lulc_train_report = prfs(masks.data.cpu().numpy().flatten(), lulc_preds.data.cpu().numpy().flatten(), average='weighted')

            train_metrics = set_train_metrics(train_metrics, cd_loss, cd_corrects, cd_train_report, lulc_loss, lulc_corrects, lulc_train_report)
            mean_train_metrics = get_mean_metrics(train_metrics)
            comet.log_metrics(mean_train_metrics)

            del batch_img1, batch_img2, labels, masks


        print(mean_train_metrics)

    with comet.validate():
        model.eval()

        for batch_img1, batch_img2, labels, masks in val_loader:
            batch_img1 = autograd.Variable(batch_img1).to(device)
            batch_img2 = autograd.Variable(batch_img2).to(device)

            labels = autograd.Variable(labels).long().to(device)
            masks = autograd.Variable(masks).long().to(device)

            cd_preds, lulc_preds = model(batch_img1, batch_img2)
            cd_loss = criterion(cd_preds, labels)
            lulc_loss = criterion_lulc(lulc_preds, masks)

            _, cd_preds = torch.max(cd_preds, 1)
            _, lulc_preds = torch.max(lulc_preds, 1)

            cd_corrects = 100 * (cd_preds.byte() == labels.squeeze().byte()).sum() / (labels.size()[0] * opt.patch_size * opt.patch_size)
            lulc_corrects = 100 * (lulc_preds.byte() == masks.squeeze().byte()).sum() / (masks.size()[0] * opt.patch_size * opt.patch_size)
            cd_val_report = prfs(labels.data.cpu().numpy().flatten(), cd_preds.data.cpu().numpy().flatten(), average='binary', pos_label=1)
            lulc_val_report = prfs(masks.data.cpu().numpy().flatten(), lulc_preds.data.cpu().numpy().flatten(), average='weighted')


            val_metrics = set_val_metrics(val_metrics, cd_loss, cd_corrects, cd_val_report, lulc_loss, lulc_corrects, lulc_val_report)
            mean_val_metrics = get_mean_metrics(val_metrics)
            comet.log_metrics(mean_val_metrics)

            del batch_img1, batch_img2, labels, masks

        print (mean_val_metrics)

    if mean_val_metrics['cd_val_f1scores'] > best_metrics['cd_val_f1scores']:
        torch.save(model, '/tmp/checkpoint_epoch_'+str(epoch)+'.pt')
        experiment.outputs_store.upload_file('/tmp/checkpoint_epoch_'+str(epoch)+'.pt')
        best_metrics = mean_val_metrics

    epoch_metrics = {'epoch':epoch,
                        **mean_train_metrics, **mean_val_metrics}

    experiment.log_metrics(**epoch_metrics)
    comet.log_epoch_end(epoch)
