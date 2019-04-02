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
from utils.helpers import get_loaders, define_output_paths, download_dataset, get_criterion, load_model, initialize_metrics, get_mean_metrics, set_metrics, log_images

from polyaxon_client.tracking import Experiment, get_log_level, get_data_paths, get_outputs_path
from polystores.stores.manager import StoreManager

import logging


###
### Initialize experiments for polyaxon and comet.ml
###

comet = CometExperiment('QQFXdJ5M7GZRGri7CWxwGxPDN', project_name="cd_lulc_hptuning_adam", auto_param_logging=False, parse_args=False)
comet.log_other('status', 'started')
experiment = Experiment()
logging.basicConfig(level=logging.INFO)




###
### Initialize Parser and define arguments
###

parser = get_parser_with_args()
opt = parser.parse_args()
comet.log_parameters(vars(opt))

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
# optimizer = optim.SGD(model.parameters(), lr=opt.lr)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-2)





###
### Set starting values
###

best_metrics = {'cd_f1scores':-1, 'cd_recalls':-1, 'cd_precisions':-1}



###
### Begin Training
###
logging.info('STARTING training')
for epoch in range(opt.epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()
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

            loss = cd_loss + lulc_loss*0
            loss.backward()
            optimizer.step()

            _, cd_preds = torch.max(cd_preds, 1)
            _, lulc_preds = torch.max(lulc_preds, 1)

            cd_corrects = 100 * (cd_preds.byte() == labels.squeeze().byte()).sum() / (labels.size()[0] * opt.patch_size * opt.patch_size)
            lulc_corrects = 100 * (lulc_preds.byte() == masks.squeeze().byte()).sum() / (masks.size()[0] * opt.patch_size * opt.patch_size)
            cd_train_report = prfs(labels.data.cpu().numpy().flatten(), cd_preds.data.cpu().numpy().flatten(), average='binary', pos_label=1)
            lulc_train_report = prfs(masks.data.cpu().numpy().flatten(), lulc_preds.data.cpu().numpy().flatten(), average='weighted')

            train_metrics = set_metrics(train_metrics, cd_loss, cd_corrects, cd_train_report, lulc_loss, lulc_corrects, lulc_train_report)
            mean_train_metrics = get_mean_metrics(train_metrics)
            comet.log_metrics(mean_train_metrics)

            del batch_img1, batch_img2, labels, masks


        print("EPOCH TRAIN METRICS", mean_train_metrics)

    with comet.validate():
        model.eval()

        first_batch = True
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

            if first_batch:
                log_images(comet, epoch, batch_img1, batch_img2, labels, masks, cd_preds, lulc_preds)
                first_batch=False

            cd_corrects = 100 * (cd_preds.byte() == labels.squeeze().byte()).sum() / (labels.size()[0] * opt.patch_size * opt.patch_size)
            lulc_corrects = 100 * (lulc_preds.byte() == masks.squeeze().byte()).sum() / (masks.size()[0] * opt.patch_size * opt.patch_size)
            cd_val_report = prfs(labels.data.cpu().numpy().flatten(), cd_preds.data.cpu().numpy().flatten(), average='binary', pos_label=1)
            lulc_val_report = prfs(masks.data.cpu().numpy().flatten(), lulc_preds.data.cpu().numpy().flatten(), average='weighted')


            val_metrics = set_metrics(val_metrics, cd_loss, cd_corrects, cd_val_report, lulc_loss, lulc_corrects, lulc_val_report)
            mean_val_metrics = get_mean_metrics(val_metrics)
            comet.log_metrics(mean_val_metrics)
            del batch_img1, batch_img2, labels, masks

        print ("EPOCH VALIDATION METRICS", mean_val_metrics)


        #
        #
        # code for outputting full city results

        d1_bands = glob.glob(data_dir + 'Safes/' + safe1 + '/GRANULE/**/IMG_DATA/*_B*.jp2')
        d2_bands = glob.glob(data_dir + 'Safes/' + safe2 + '/GRANULE/**/IMG_DATA/*_B*.jp2')

        template_img = rasterio.open(d1_bands[2])
        profile = template_img.profile

        d1_bands.sort()
        d2_bands.sort()

        d1d2 = read_bands(d1_bands + d2_bands)
        print ('Bands read')

        d1, d2 = stack_bands(d1d2, height=template_img.height, width=template_img.width)

        d1 = d1.transpose(1,2,0)
        d2 = d2.transpose(1,2,0)

        patches1, hs, ws, lc, lr, h, w = get_patches(d1, patch_dim=opt.patch_size)
        patches1 = patches1.transpose(0,3,1,2)

        print ('Patches1 Created')

        patches2, hs, ws, lc, lr, h, w = get_patches(d2, patch_dim=opt.patch_size)
        patches2 = patches2.transpose(0,3,1,2)

        print ('Patches2 Created')

        out = []
        for i in range(0,patches1.shape[0],batch_size):
            batch1 = torch.from_numpy(patches1[i:i+batch_size,:,:,:]).to(device)
            batch2 = torch.from_numpy(patches2[i:i+batch_size,:,:,:]).to(device)

            preds = model(batch1, batch2)
            del batch1
            del batch2

            preds = F.sigmoid(preds) > 0.5
            preds = preds.data.cpu().numpy()
            out.append(preds)

        profile['dtype'] = 'uint8'
        profile['driver'] = 'GTiff'
        fout = rasterio.open(results_dir + tid + '_' + date1 + '_' + date2 + '.tif', 'w', **profile)
        fout.write(np.asarray([mask]).astype(np.uint8))
        fout.close()

    if (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores']) or (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls']) or (mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions']):
        torch.save(model, '/tmp/checkpoint_epoch_'+str(epoch)+'.pt')
        experiment.outputs_store.upload_file('/tmp/checkpoint_epoch_'+str(epoch)+'.pt')
        best_metrics = mean_val_metrics

    log_train_metrics = {"train_"+k:v for k,v in mean_train_metrics.items()}
    log_val_metrics = {"validate_"+k:v for k,v in mean_val_metrics.items()}
    epoch_metrics = {'epoch':epoch,
                        **log_train_metrics, **log_val_metrics}

    experiment.log_metrics(**epoch_metrics)
    comet.log_other('status', 'running') # this is placed after the first epoch because this likely means the training process is sound
    comet.log_epoch_end(epoch)
comet.log_other('status', 'complete')
