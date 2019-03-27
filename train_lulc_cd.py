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

from polyaxon_client.tracking import Experiment, get_log_level, get_data_paths, get_outputs_path
from polystores.stores.manager import StoreManager



# from moonshot import alert


import logging

comet = CometExperiment('QQFXdJ5M7GZRGri7CWxwGxPDN', project_name="cd_lulc")

logging.basicConfig(level=logging.INFO)


def get_weight_filename(weight_file):
    return '{}/{}'.format(get_outputs_path(), 'checkpoint.pth.tar')



parser = argparse.ArgumentParser(description='Training change detection network')

parser.add_argument('--patch_size', type=int, default=120, required=False, help='input patch size')
parser.add_argument('--stride', type=int, default=10, required=False, help='stride at which to sample patches')
parser.add_argument('--aug', default=True, required=False, help='Do augmentation or not')
parser.add_argument('--mask', default=True, required=False, help='Load LULC mask and train with it')

parser.add_argument('--gpu_ids', default='0,1,2,3', required=False, help='gpus ids for parallel training')
parser.add_argument('--num_workers', type=int, default=90, required=False, help='Number of cpu workers')

parser.add_argument('--epochs', type=int, default=10, required=False, help='number of eochs to train')
parser.add_argument('--batch_size', type=int, default=256, required=False, help='batch size for training')
parser.add_argument('--lr', type=float, default=0.01, required=False, help='Learning rate')

parser.add_argument('--loss', type=str, default='bce', required=False, help='bce,focal,dice,jaccard,tversky')
parser.add_argument('--gamma', type=float, default=2, required=False, help='if focal loss is used pass gamma')
parser.add_argument('--alpha', type=float, default=0.5, required=False, help='if tversky loss is used pass alpha')
parser.add_argument('--beta', type=float, default=0.5, required=False, help='if tversky loss is used pass beta')

parser.add_argument('--val_cities', default='0,1', required=False, help='''cities to use for validation,
                            0:abudhabi, 1:aguasclaras, 2:beihai, 3:beirut, 4:bercy, 5:bordeaux, 6:cupertino, 7:hongkong, 8:mumbai,
                            9:nantes, 10:paris, 11:pisa, 12:rennes, 14:saclay_e''')

parser.add_argument('--data_dir', default='../datasets/onera/', required=False, help='data directory for training')
parser.add_argument('--weight_dir', default='../weights/', required=False, help='directory to save weights')
parser.add_argument('--weight_file', default='', required=False, help='if defined and available, will preload weights from this file')
parser.add_argument('--log_dir', default='../logs/', required=False, help='directory to save training log')


opt = parser.parse_args()

if opt.mask:
    model_name = 'lulc_cd'
else:
    model_name = 'cd'

if opt.loss == 'bce' or opt.loss == 'dice' or opt.loss == 'jaccard':
    path = model_name + '_patchSize_' + str(opt.patch_size) + '_stride_' + str(opt.stride) + \
            '_batchSize_' + str(opt.batch_size) + '_loss_' + opt.loss  + \
            '_lr_' + str(opt.lr) + '_epochs_' + str(opt.epochs) +\
            '_valCities_' + opt.val_cities

if opt.loss == 'focal':
    path = model_name + '_patchSize_' + str(opt.patch_size) + '_stride_' + str(opt.stride) + \
            '_batchSize_' + str(opt.batch_size) + '_loss_' + opt.loss + '_gamma_' + str(opt.gamma) + \
            '_lr_' + str(opt.lr) + '_epochs_' + str(opt.epochs) +\
            '_valCities_' + opt.val_cities

if opt.loss == 'tversky':
    path = model_name + '_patchSize_' + str(opt.patch_size) + '_stride_' + str(opt.stride) + \
            '_batchSize_' + str(opt.batch_size) + '_loss_' + opt.loss + '_alpha_' + str(opt.alpha) + '_beta_' + str(opt.beta) + \
            '_lr_' + str(opt.lr) + '_epochs_' + str(opt.epochs) +\
            '_valCities_' + opt.val_cities

weight_file = opt.weight_file

weight_path = opt.weight_dir + path + '.pt'
log_path = opt.log_dir + path + '.log'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))
logging.info('STARTING data download')
data_paths = list(get_data_paths().values())[0]
data_store = StoreManager(path=data_paths)
data_store.download_dir('onera')
experiment = Experiment()

train_samples, test_samples = get_train_val_metadata(opt.data_dir, opt.val_cities, opt.patch_size, opt.stride)
print ('train samples : ', len(train_samples))
print ('test samples : ', len(test_samples))

logging.info('STARTING Dataset Creation')

full_load = full_onera_loader(opt.data_dir, load_mask=opt.mask)

train_dataset = OneraPreloader(opt.data_dir, train_samples, full_load, opt.patch_size, opt.aug, opt.mask)
test_dataset = OneraPreloader(opt.data_dir, test_samples, full_load, opt.patch_size, opt.aug, opt.mask)

logging.info('STARTING Dataloading')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

logging.info('LOADING Model')

if opt.mask:
    model = BiDateLULCNet(13, 2, 5).to(device)
    model = nn.DataParallel(model, device_ids=[int(x) for x in opt.gpu_ids.split(',')])
else:
    model = BiDateNet(13, 2).to(device)
    model = nn.DataParallel(model, device_ids=[int(x) for x in opt.gpu_ids.split(',')])
#
# if os.path.exists(opt.weight_dir) and weight_file in os.listdir(opt.weight_dir):
#     model = torch.load(opt.weight_dir + 'cd_patchSize_90_stride_10_batchSize_512_loss_tversky_alpha_0.08_beta_0.92_lr_0.01_epochs_10_valCities_0,1.pt')

if opt.loss == 'bce':
    criterion = nn.BCEWithLogitsLoss()
if opt.loss == 'focal':
    criterion = FocalLoss(opt.gamma)
if opt.loss == 'dice':
    criterion = dice_loss
if opt.loss == 'jaccard':
    criterion = jaccard_loss
if opt.loss == 'tversky':
    criterion = TverskyLoss(alpha=opt.alpha, beta=opt.beta)

if opt.mask:
    criterion_lulc = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=opt.lr)


cd_best_f1s = -1
best_metric = {}

logging.info('STARTING training')

with comet.train():
    for epoch in range(opt.epochs):

        cd_train_losses = []
        cd_train_corrects = []
        cd_train_precisions = []
        cd_train_recalls = []
        cd_train_f1scores = []

        lulc_train_losses = []
        lulc_train_corrects = []
        lulc_train_precisions = []
        lulc_train_recalls = []
        lulc_train_f1scores = []


        model.train()
        logging.info('SET model mode to train!')

        t = trange(len(train_loader))
        logging.info(t)
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

            cd_train_losses.append(cd_loss.item())
            cd_train_corrects.append(cd_corrects.item())
            cd_train_precisions.append(cd_train_report[0])
            cd_train_recalls.append(cd_train_report[1])
            cd_train_f1scores.append(cd_train_report[2])

            lulc_train_losses.append(lulc_loss.item())
            lulc_train_corrects.append(lulc_corrects.item())
            lulc_train_precisions.append(lulc_train_report[0])
            lulc_train_recalls.append(lulc_train_report[1])
            lulc_train_f1scores.append(lulc_train_report[2])

            t.set_postfix(cd_loss=cd_loss.data.tolist(), lulc_loss=lulc_loss.data.tolist(), cd_accuracy=cd_corrects.data.tolist(), lulc_accuracy=lulc_corrects.data.tolist())
            t.update()

            del batch_img1
            del batch_img2
            del labels
            del masks

        cd_train_loss = np.mean(cd_train_losses)
        cd_train_acc = np.mean(cd_train_corrects)
        cd_train_prec = np.mean(cd_train_precisions)
        cd_train_rec = np.mean(cd_train_recalls)
        cd_train_f1s = np.mean(cd_train_f1scores)

        lulc_train_loss = np.mean(lulc_train_losses)
        lulc_train_acc = np.mean(lulc_train_corrects)
        lulc_train_prec = np.mean(lulc_train_precisions)
        lulc_train_rec = np.mean(lulc_train_recalls)
        lulc_train_f1s = np.mean(lulc_train_f1scores)

        print('cd train loss : ', cd_train_loss, ' cd train accuracy : ', cd_train_acc, ' cd avg. precision : ', cd_train_prec, ' cd avg. recall : ', cd_train_rec, ' cd avg. f1 score : ', cd_train_f1s)
        print('lulc train loss : ', lulc_train_loss, ' lulc train accuracy : ', lulc_train_acc, ' lulc avg. precision : ', lulc_train_prec, ' lulc avg. recall : ', lulc_train_rec, ' lulc avg. f1 score : ', lulc_train_f1s)
        # fout.write('train loss : ' + str(train_loss) + ' train accuracy : ' + str(train_acc) + ' avg. precision : ' + str(train_prec) + ' avg. recall : ' + str(train_rec) + ' avg. f1 score : ' + str(train_f1s) + '\n')

        model.eval()

        cd_test_losses = []
        cd_test_corrects = []
        cd_test_precisions = []
        cd_test_recalls = []
        cd_test_f1scores = []

        lulc_test_losses = []
        lulc_test_corrects = []
        lulc_test_precisions = []
        lulc_test_recalls = []
        lulc_test_f1scores = []

        t = trange(len(test_loader))
        for batch_img1, batch_img2, labels, masks in test_loader:
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

            cd_test_report = prfs(labels.data.cpu().numpy().flatten(), cd_preds.data.cpu().numpy().flatten(), average='binary', pos_label=1)
            lulc_test_report = prfs(masks.data.cpu().numpy().flatten(), lulc_preds.data.cpu().numpy().flatten(), average='weighted')

            cd_test_losses.append(cd_loss.item())
            cd_test_corrects.append(cd_corrects.item())
            cd_test_precisions.append(cd_test_report[0])
            cd_test_recalls.append(cd_test_report[1])
            cd_test_f1scores.append(cd_test_report[2])

            lulc_test_losses.append(lulc_loss.item())
            lulc_test_corrects.append(lulc_corrects.item())
            lulc_test_precisions.append(lulc_test_report[0])
            lulc_test_recalls.append(lulc_test_report[1])
            lulc_test_f1scores.append(lulc_test_report[2])

            t.set_postfix(cd_loss=cd_loss.data.tolist(), lulc_loss=lulc_loss.data.tolist(), cd_accuracy=cd_corrects.data.tolist(), lulc_accuracy=lulc_corrects.data.tolist())
            t.update()

            del batch_img1
            del batch_img2
            del labels
            del masks

        cd_test_loss = np.mean(cd_test_losses)
        cd_test_acc = np.mean(cd_test_corrects)
        cd_test_prec = np.mean(cd_test_precisions)
        cd_test_rec = np.mean(cd_test_recalls)
        cd_test_f1s = np.mean(cd_test_f1scores)

        lulc_test_loss = np.mean(lulc_test_losses)
        lulc_test_acc = np.mean(lulc_test_corrects)
        lulc_test_prec = np.mean(lulc_test_precisions)
        lulc_test_rec = np.mean(lulc_test_recalls)
        lulc_test_f1s = np.mean(lulc_test_f1scores)

        print ('cd test loss : ', cd_test_loss, ' cd test accuracy : ', cd_test_acc, ' cd avg. precision : ', cd_test_prec, ' cd avg. recall : ', cd_test_rec, ' cd avg. f1 score : ', cd_test_f1s)
        print ('lulc test loss : ', lulc_test_loss, ' lulc test accuracy : ', lulc_test_acc, ' lulc avg. precision : ', lulc_test_prec, ' lulc avg. recall : ', lulc_test_rec, ' lulc avg. f1 score : ', lulc_test_f1s)

        if cd_test_f1s > cd_best_f1s:
            torch.save(model, '/tmp/checkpoint_epoch_'+str(epoch)+'.pt')
            experiment.outputs_store.upload_file('/tmp/checkpoint_epoch_'+str(epoch)+'.pt')
            cd_best_f1s = cd_test_f1s
            best_metric['cd train loss'] = str(cd_train_loss)
            best_metric['cd test loss'] = str(cd_test_loss)
            best_metric['cd train accuracy'] = str(cd_train_acc)
            best_metric['cd test accuracy'] = str(cd_test_acc)
            best_metric['cd train avg. precision'] = str(cd_train_prec)
            best_metric['cd test avg. precision'] = str(cd_test_prec)
            best_metric['cd train avg. recall'] = str(cd_train_rec)
            best_metric['cd test avg. recall'] = str(cd_test_rec)
            best_metric['cd train avg. f1 score'] = str(cd_train_f1s)
            best_metric['cd test avg. f1 score'] = str(cd_test_f1s)

            best_metric['lulc train loss'] = str(lulc_train_loss)
            best_metric['lulc test loss'] = str(lulc_test_loss)
            best_metric['lulc train accuracy'] = str(lulc_train_acc)
            best_metric['lulc test accuracy'] = str(lulc_test_acc)
            best_metric['lulc train avg. precision'] = str(lulc_train_prec)
            best_metric['lulc test avg. precision'] = str(lulc_test_prec)
            best_metric['lulc train avg. recall'] = str(lulc_train_rec)
            best_metric['lulc test avg. recall'] = str(lulc_test_rec)
            best_metric['lulc train avg. f1 score'] = str(lulc_train_f1s)
            best_metric['lulc test avg. f1 score'] = str(lulc_test_f1s)

        experiment.log_metrics(epoch=epoch,
                                cd_train_f1_score=cd_train_f1s,
                                cd_train_recall=cd_train_rec,
                                cd_train_prec=cd_train_prec,
                                cd_train_loss=cd_train_loss,
                                cd_train_accuracy=cd_train_acc,
                                cd_test_f1_score=cd_test_f1s,
                                cd_test_recall=cd_test_rec,
                                cd_test_prec=cd_test_prec,
                                cd_test_loss=cd_test_loss,
                                cd_test_accuracy=cd_test_acc,
                                lulc_train_f1_score=lulc_train_f1s,
                                lulc_train_recall=lulc_train_rec,
                                lulc_train_prec=lulc_train_prec,
                                lulc_train_loss=lulc_train_loss,
                                lulc_train_accuracy=lulc_train_acc,
                                lulc_test_f1_score=lulc_test_f1s,
                                lulc_test_recall=lulc_test_rec,
                                lulc_test_prec=lulc_test_prec,
                                lulc_test_loss=lulc_test_loss,
                                lulc_test_accuracy=lulc_test_acc)
        comet.log_metrics({'epoch':epoch,
                            'cd_train_f1_score':cd_train_f1s,
                            'cd_train_recall':cd_train_rec,
                            'cd_train_prec':cd_train_prec,
                            'cd_train_loss':cd_train_loss,
                            'cd_train_accuracy':cd_train_acc,
                            'cd_test_f1_score':cd_test_f1s,
                            'cd_test_recall':cd_test_rec,
                            'cd_test_prec':cd_test_prec,
                            'cd_test_loss':cd_test_loss,
                            'cd_test_accuracy':cd_test_acc,
                            'lulc_train_f1_score':lulc_train_f1s,
                            'lulc_train_recall':lulc_train_rec,
                            'lulc_train_prec':lulc_train_prec,
                            'lulc_train_loss':lulc_train_loss,
                            'lulc_train_accuracy':lulc_train_acc,
                            'lulc_test_f1_score':lulc_test_f1s,
                            'lulc_test_recall':lulc_test_rec,
                            'lulc_test_prec':lulc_test_prec,
                            'lulc_test_loss':lulc_test_loss,
                            'lulc_test_accuracy':lulc_test_acc})
