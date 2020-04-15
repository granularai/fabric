import os
import logging
import json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from polyaxon_client.tracking import Experiment

from basecamp.metrics.metrics import Metrics
from basecamp.metrics.loss import DiceLoss
from basecamp.runner.runner import Runner

from models.bidate_model import BiDateNet
# from utils.dataloader import get_dataloaders
from utils.parser import get_args

def local_testing():
    if 'POLYAXON_NO_OP' in os.environ:
        if os.environ['POLYAXON_NO_OP'] == 'true':
            return True
    else:
        False

experiment = None
if local_testing():
    experiment = Experiment()

args = get_args()
print (args)

logging.basicConfig(level=logging.INFO)

"""
Set up environment: define paths, download data, and set device
"""


train_loader, val_loader = get_dataloaders(args)

"""
Load Model then define other aspects of the model
"""
logging.info('LOADING Model')
model = Dummy(n_channels=len(args.band_ids), n_classes=args.num_classes).cuda(args.gpu)

criterion = DiceLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)

if not local_testing():
    """
    Initialize experiments for polyaxon
    """
    train_metrics = Metrics(polyaxon_exp=experiment, metrics_strings=['dc'])
    val_metrics = Metrics(polyaxon_exp=experiment, metrics_strings=['dc'])
    runner = Runner(device=args.gpu, model=model,
                optimizer=optimizer, criterion=criterion,
                train_loader=train_loader, val_loader=val_loader,
                train_metrics=train_metrics, val_metrics=val_metrics,
                polyaxon_exp=experiment)
else:
    train_metrics = Metrics(metrics_strings=['dc'])
    val_metrics = Metrics(metrics_strings=['dc'])
    runner = Runner(device=args.gpu, model=model,
                optimizer=optimizer, criterion=criterion,
                train_loader=train_loader, val_loader=val_loader,
                train_metrics=train_metrics, val_metrics=val_metrics,
                polyaxon_exp=None)

logging.info('STARTING training')
for epoch in range(args.epochs):
    """
    Begin Training
    """
    logging.info('SET model mode to train!')
    runner.set_epoch_metrics()
    train_metrics = runner.train_model()
    eval_metrics = runner.eval_model()

    """
    Store the weights of good epochs based on validation results
    """
    torch.save(model, '/tmp/checkpoint_epoch_'+str(epoch)+'.pt')
    upload_file_path = '/tmp/checkpoint_epoch_'+str(epoch)+'.pt'

    if not local_testing():
        experiment.outputs_store.upload_file(upload_file_path)
