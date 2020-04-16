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
    experiment = Experiment()

grain_exp = Grain(polyaxon_exp=experiment)
args = grain_exp.parse_args_from_json('metadata.json')

logging.basicConfig(level=logging.INFO)

"""
Set up environment: define paths, download data, and set device
"""

train_loader, val_loader = get_dataloaders(args)

"""
Load Model then define other aspects of the model
"""
logging.info('LOADING Model')
model = BiDateNet(n_channels=len(args.band_ids), n_classes=1).cuda(args.gpu)

criterion = DiceLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)


runner = Runner(model=model,
            optimizer=optimizer, criterion=criterion,
            train_loader=train_loader, val_loader=val_loader,
            args=args, polyaxon_exp=experiment)


logging.info('STARTING training')
for epoch in range(args.epochs):
    """
    Begin Training
    """
    logging.info('SET model mode to train!')
    runner.set_epoch_metrics()
    train_metrics = runner.train_model()
    eval_metrics = runner.eval_model()
    print (train_metrics)
    print (eval_metrics)
    """
    Store the weights of good epochs based on validation results
    """
    torch.save(model, '/tmp/checkpoint_epoch_'+str(epoch)+'.pt')
    upload_file_path = '/tmp/checkpoint_epoch_'+str(epoch)+'.pt'

    if not local_testing():
        experiment.outputs_store.upload_file(upload_file_path)
