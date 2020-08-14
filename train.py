import os
import logging
import json
import tarfile
import shutil

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from polyaxon.tracking import Run

from basecamp.metrics.metrics import Metrics
from basecamp.loss import get_loss
from basecamp.runner.runner import Runner

from models.bidate_model import BiDateNet
from utils.dataloader import get_dataloaders
from basecamp.grain.grain import Grain


def local_testing():
    if 'POLYAXON_NO_OP' in os.environ:
        if os.environ['POLYAXON_NO_OP'] == 'true':
            return True
    else:
        False


experiment = None
if not local_testing():
    experiment = Run()

### diagnose issue

import glob

for filename in glob.iglob('/data' + '**/**', recursive=True):
    print(filename)


for filename in glob.iglob('/artifacts' + '**/**', recursive=True):
    print(filename)

##

grain_exp = Grain(polyaxon_exp=experiment)
args = grain_exp.parse_args_from_json('metadata.json')

logging.basicConfig(level=logging.INFO)

# """
# Set up environment: define paths, download data, and set device
# """
#
# if not local_testing():
#     if not os.path.exists(args.polyaxon_data_path):
#         os.makedirs(args.polyaxon_data_path)
#     tf = tarfile.open(args.nfs_data_path)
#     tf.extractall(args.polyaxon_data_path)
#     args.dataset_dir = os.path.join(args.polyaxon_data_path,
#                                     'change_detection/')
#
# train_loader, val_loader = get_dataloaders(args)
#
# """
# Load Model then define other aspects of the model
# """
# logging.info('LOADING Model')
# model = BiDateNet(n_channels=len(args.band_ids), n_classes=1).cuda(args.gpu)
#
# criterion = get_loss(args)
# optimizer = optim.SGD(model.parameters(), lr=args.lr)
# # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
#
#
# runner = Runner(model=model, optimizer=optimizer,
#                 criterion=criterion, train_loader=train_loader,
#                 val_loader=val_loader, args=args,
#                 polyaxon_exp=experiment)
#
# best_dc = -1
#
# logging.info('STARTING training')
# for epoch in range(args.epochs):
#     """
#     Begin Training
#     """
#     logging.info('SET model mode to train!')
#     runner.set_epoch_metrics()
#     train_metrics = runner.train_model()
#     eval_metrics = runner.eval_model()
#     print(train_metrics)
#     print(eval_metrics)
#
#     """
#     Store the weights of good epochs based on validation results
#     """
#     if eval_metrics['val_dc'] > best_dc:
#         if not local_testing():
#             base = os.environ.get('POLYAXON_RUN_OUTPUTS_PATH')
#             save_path = os.path.join(base,
#                                      'checkpoint_epoch_' + str(epoch) + '.pt')
#             torch.save(model, save_path)
#             experiment.log_artifact(save_path)
#         else:
#             if not os.path.exists(args.weight_dir):
#                 os.makedirs(args.weight_dir)
#
#             torch.save(model, os.path.join(args.weight_dir,
#                        'checkpoint_epoch_'+str(epoch)+'.pt'))
#
#         best_dc = eval_metrics['val_dc']
