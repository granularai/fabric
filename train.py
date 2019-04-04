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
from utils.helpers import get_loaders, define_output_paths, download_dataset, get_criterion, load_model, initialize_metrics, get_mean_metrics, set_metrics, log_images, log_figure, _scale

from polyaxon_client.tracking import Experiment, get_log_level, get_data_paths, get_outputs_path
from polystores.stores.manager import StoreManager

import logging


###
### Initialize experiments for polyaxon and comet.ml
###

comet = CometExperiment('QQFXdJ5M7GZRGri7CWxwGxPDN', project_name="val_image_2", auto_param_logging=False, parse_args=False)
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
optimizer = optim.SGD(model.parameters(), lr=opt.lr)
# optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-2)





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
        for batch_img1, batch_img2, labels in train_loader:
            # logging.info("batch: "+str(batch_iter)+" - "+str(batch_iter+opt.batch_size))
            batch_iter = batch_iter+opt.batch_size
            batch_img1 = autograd.Variable(batch_img1).to(device)
            batch_img2 = autograd.Variable(batch_img2).to(device)

            labels = autograd.Variable(labels).long().to(device)


            optimizer.zero_grad()
            cd_preds = model(batch_img1, batch_img2)
            cd_loss = criterion(cd_preds, labels)

            loss = cd_loss
            loss.backward()
            optimizer.step()

            _, cd_preds = torch.max(cd_preds, 1)

            cd_corrects = 100 * (cd_preds.byte() == labels.squeeze().byte()).sum() / (labels.size()[0] * opt.patch_size * opt.patch_size)
            cd_train_report = prfs(labels.data.cpu().numpy().flatten(), cd_preds.data.cpu().numpy().flatten(), average='binary', pos_label=1)


            train_metrics = set_metrics(train_metrics, cd_loss, cd_corrects, cd_train_report)
            mean_train_metrics = get_mean_metrics(train_metrics)
            comet.log_metrics(mean_train_metrics)

            del batch_img1, batch_img2, labels
            # break # temporary break to ignore training

        print("EPOCH TRAIN METRICS", mean_train_metrics)

    with comet.validate():
        model.eval()

        first_batch = True
        for batch_img1, batch_img2, labels in val_loader:
            batch_img1 = autograd.Variable(batch_img1).to(device)
            batch_img2 = autograd.Variable(batch_img2).to(device)

            labels = autograd.Variable(labels).long().to(device)

            cd_preds = model(batch_img1, batch_img2)

            cd_loss = criterion(cd_preds, labels)

            _, cd_preds = torch.max(cd_preds, 1)

            if first_batch:
                log_images(comet, epoch, batch_img1, batch_img2, labels, cd_preds)
                first_batch=False

            cd_corrects = 100 * (cd_preds.byte() == labels.squeeze().byte()).sum() / (labels.size()[0] * opt.patch_size * opt.patch_size)

            cd_val_report = prfs(labels.data.cpu().numpy().flatten(), cd_preds.data.cpu().numpy().flatten(), average='binary', pos_label=1)


            val_metrics = set_metrics(val_metrics, cd_loss, cd_corrects, cd_val_report)
            mean_val_metrics = get_mean_metrics(val_metrics)
            comet.log_metrics(mean_val_metrics)
            del batch_img1, batch_img2, labels

        print ("EPOCH VALIDATION METRICS", mean_val_metrics)



        ###
        ### Output full test image
        ###

        # load day 1 and 2 bands
        d1_bands = glob.glob(opt.data_dir + 'images/' + opt.validation_city + '/imgs_1/*')
        d2_bands = glob.glob(opt.data_dir + 'images/' + opt.validation_city + '/imgs_2/*')

        # sort bands to ensure that B01 -> B12 order
        d1_bands.sort()
        d2_bands.sort()

        # load band 2 from d1 bands to get template image dimensions, profile
        template_img = rasterio.open(d1_bands[2])
        profile = template_img.profile

        # read all the bands from d1 and d2 by simply rio opening the files
        d1d2 = read_bands(d1_bands + d2_bands)
        print ('Bands read')

        # using city_loader, lets get a stack of all bands of dimension (2,13,H,W)
        print("template values h,w", template_img.height, template_img.width)
        imgs_stacked = city_loader([opt.data_dir + 'images/' + opt.validation_city, template_img.width,template_img.height])

        d1 = imgs_stacked[0]
        d2 = imgs_stacked[1]

        # flip images
        d1 = d1.transpose(1,2,0)
        d2 = d2.transpose(1,2,0)

        patches1, hs, ws, lc, lr, h, w = get_patches(d1, patch_dim=opt.patch_size)

        patches1 = patches1.transpose(0,3,1,2)

        print ('Patches1 Created')

        patches2, hs, ws, lc, lr, h, w = get_patches(d2, patch_dim=opt.patch_size)
        patches2 = patches2.transpose(0,3,1,2)

        print ('Patches2 Created')

        out = []
        for i in range(0,patches1.shape[0],opt.batch_size):
            batch1 = torch.from_numpy(patches1[i:i+opt.batch_size,:,:,:]).to(device)
            batch2 = torch.from_numpy(patches2[i:i+opt.batch_size,:,:,:]).to(device)

            preds = model(batch1, batch2)

            del batch1
            del batch2

            # preds = F.sigmoid(preds) > 0.5
            _, cd_preds = torch.max(preds, 1)
            print(cd_preds.shape)
            cd_preds = cd_preds.data.cpu().numpy()
            out.append(cd_preds)

        mask = get_bands(out[0], hs, ws, lc, lr, h, w, patch_size=opt.patch_size)

        torch_mask = torch.from_numpy(mask).float().to(device)

        print("MASK DIMS", mask.shape)

        file_path = opt.validation_city+'_epoch_'+str(epoch)
        cv2.imwrite(file_path+'.png', _scale(mask))
        comet.log_image(file_path+'.png')

        preview1 = stretch_8bit(cv2.imread(opt.data_dir + 'images/' + opt.validation_city + '/pair/img1.png', 1))
        preview2 = stretch_8bit(cv2.imread(opt.data_dir + 'images/' + opt.validation_city + '/pair/img2.png', 1))
        groundtruth = torch.from_numpy(cv2.imread(opt.data_dir + 'labels/' + opt.validation_city + '/cm/cm.png', 0))
        log_figure(comet, img1=preview1, img2=preview2, groundtruth=groundtruth, prediction=torch_mask, fig_name=file_path)

    if (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores']) or (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls']) or (mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions']):
        torch.save(model, '/tmp/checkpoint_epoch_'+str(epoch)+'.pt')
        experiment.outputs_store.upload_file('/tmp/checkpoint_epoch_'+str(epoch)+'.pt')
        comet.log_asset('/tmp/checkpoint_epoch_'+str(epoch)+'.pt')
        best_metrics = mean_val_metrics

    log_train_metrics = {"train_"+k:v for k,v in mean_train_metrics.items()}
    log_val_metrics = {"validate_"+k:v for k,v in mean_val_metrics.items()}
    epoch_metrics = {'epoch':epoch,
                        **log_train_metrics, **log_val_metrics}

    experiment.log_metrics(**epoch_metrics)
    comet.log_other('status', 'running') # this is placed after the first epoch because this likely means the training process is sound
    comet.log_epoch_end(epoch)
comet.log_other('status', 'complete')
