import os
import logging
import tarfile
from shutil import copytree, ignore_patterns

import torch
import torch.optim as optim

from polyaxon.tracking import Run

from phobos.loss import get_loss
from phobos.runner import Runner
from phobos.grain import Grain

from models.bidate_model import BiDateNet
from utils.dataloader import get_dataloaders

from inspect import getmodule, signature


def local_testing():
    if 'POLYAXON_NO_OP' in os.environ:
        if os.environ['POLYAXON_NO_OP'] == 'true':
            return True
    else:
        False


experiment = None
if not local_testing():
    experiment = Run()

grain_exp = Grain(polyaxon_exp=experiment)
args = grain_exp.parse_args_from_json('metadata.json')

logging.basicConfig(level=logging.INFO)
"""
Set up environment: define paths, download data, and set device
"""


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


if not local_testing():
    if not os.path.exists(args.local_artifacts_path):
        os.makedirs(args.local_artifacts_path)
    tf = tarfile.open(args.nfs_data_path)
    tf.extractall(args.local_artifacts_path)
    args.dataset_dir = os.path.join(args.local_artifacts_path, 'onera/')

    # log code to artifact/code folder
    print('local files')
    list_files('.')
    code_path = os.path.join(experiment.get_artifacts_path(), 'code')
    print('prev code path files')
    list_files(code_path)
    copytree('.', code_path, ignore=ignore_patterns('.*'))
    print('new code path files')
    list_files(code_path)

    # set artifact/weight folder
    args.weight_dir = os.path.join(experiment.get_artifacts_path(), 'weights')

if not os.path.exists(args.weight_dir):
    os.makedirs(args.weight_dir)

train_loader, val_loader = get_dataloaders(args)
"""
Load Model then define other aspects of the model
"""
logging.info('LOADING Model')
model = grain_exp.load_model(BiDateNet,
                             n_channels=len(args.band_ids),
                             n_classes=1)
print("MODEL LOADED")
"""
DEBUG: Load Model then define other aspects of the model
"""

if args.gpu > -1:
    model = model.to(args.gpu)

criterion = get_loss(args)
optimizer = optim.SGD(model.parameters(), lr=args.lr)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)

runner = Runner(model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                args=args,
                polyaxon_exp=experiment)

best_dc = -1

logging.info('STARTING training')
for epoch in range(args.epochs):
    """
    Begin Training
    """
    logging.info('SET model mode to train!')
    runner.set_epoch_metrics()
    train_metrics = runner.train_model()
    eval_metrics = runner.eval_model()
    print(train_metrics)
    print(eval_metrics)
    """
    Store the weights of good epochs based on validation results
    """
    if eval_metrics['val_dc'] > best_dc:
        cpt_path = os.path.join(args.weight_dir,
                                'checkpoint_epoch_' + str(epoch) + '.pt')
        torch.save(model.state_dict(), cpt_path)
        best_dc = eval_metrics['val_dc']
