from utils import getLoadersMap
from models import UNetMultiDate

import tarfile
import os, logging

import torch
import torch.nn as nn

from phobos.grain import Grain
from phobos.runner import Runner

from polyaxon.tracking import Run


################### Polyaxon / Local ###################
"""
Initialization to use datalab or local system for training.
"""

experiment = None
if not Runner.local_testing():
    experiment = Run()


################### Polyaxon / Local ###################

################### Arguments ###################

"""Initialize all arguments passed via metadata.json
"""
args = Grain(yaml='metadata.yaml',polyaxon_exp=experiment)

################### Arguments ###################

############## Input & Output from Grain ###############

inputs, outputs = args.get_inputs_outputs()

logging.basicConfig(level=logging.WARNING)

########################################################


################### Setup Data and Weight ###################

if not Runner.local_testing():
    """
    When using datalab for training, we need to see how data is stored in datastore 
    and copy, untar, or pass url properly depending on how we use the datastore. 
    This will require a bit of effort in understanding the structure of the dataset, 
    like are train, val tarred together or are they different. Are we using webdataset
    shards, etc. We will eventually move to a unified framwork under webdataset-aistore
    for all dataset coming from Europa and all third party open-source datasets.
    """
    if not os.path.exists(args.local_artifacts_path):
        os.makedirs(args.local_artifacts_path)

    tf = tarfile.open(args.nfs_data_path)
    tf.extractall(args.local_artifacts_path)
    #tf = tarfile.open(os.path.join(args.nfs_data_path, 'test.tar.gz'))
    #tf.extractall(args.local_artifacts_path)
    args.dataset_path = os.path.join(args.local_artifacts_path,args.dataset_name.lower())

    # log code to artifact/code folder
    # code_path = os.path.join(experiment.get_artifacts_path(), 'code')
    # copytree('.', code_path, ignore=ignore_patterns('.*'))

    # set artifact/weight folder
    args.weight_dir = os.path.join(experiment.get_artifacts_path(), 'weights')

if not os.path.exists(args.weight_dir):
    os.makedirs(args.weight_dir)


################### Setup Data and Weight Directories ###################

loaders = getLoadersMap(args,inputs)

print(f'train loader length : {len(loaders["train"])}')
print(f'val loader length : {len(loaders["val"])}\n')
'''
AIS webdataset eg,
def preproc(data):
    inp1 = data['x.pth']
    inp1 = torch.unsqueeze(inp1,0)

    out1 = data['y.cls']

    x = {'inp1': inp1}
    y = {'out1': out1}
    
    return x,y

urlmap = { 
    'train': 'http://aistore.granular.ai/v1/objects/test_ais/train/train-{0..4}.tar?provider=gcp',
    'val': 'http://aistore.granular.ai/v1/objects/test_ais/val/val-{0..4}.tar?provider=gcp',
}
transmap = {'train': preproc, 'val': preproc }

loaders = getWebDataLoaders(
    posixes=urlmap,
    transforms=transmap,
    shuffle=True,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    distributed=args.distributed,
)
'''

################### Intialize Model ###################
"""
Load Model then define other aspects of the model
"""

shape = inputs.heads['rgbn'].shape.H
device = torch.device("cuda",0)
n_channels = inputs.heads['rgbn'].shape.C

n_classes  = outputs.heads['cd'].num_classes
if n_classes == 2:
    n_classes = 1

if args.model == 'unetmultidate':
    model = args.load_model(UNetMultiDate,
                                n_channels=n_channels,
                                n_classes=n_classes,
                                patch_size=shape,
                                device=device
                                )

if args.distributed:
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
elif args.num_gpus > 1:
    model = nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

if args.pretrained_checkpoint:
    """
    If you have any pretrained weights that you want to load for the model, this 
    is the place to do it.
    """
    pretrained = torch.load(args.pretrained_checkpoint)
    model.load_state_dict(pretrained)

if args.resume_checkpoint:
    """If we want to resume training from some checkpoints.
    """
    weight = torch.load(args.resume_checkpoint)
    model.load_state_dict(weight)

################### Intialize Model ###################

################### Intialize Runner ###################

runner = Runner(
    model=model,
    device=args.device,
    train_loader=loaders['train'],
    val_loader=loaders['val'], 
    inputs=inputs, 
    outputs=outputs, 
    optimizer=args.optimizer, 
    optimizer_args=args.optimizer_args,
    scheduler=args.scheduler,
    scheduler_args=args.scheduler_args,
    mode=args.mode,
    distributed=args.distributed,
    verbose=args.verbose,
    max_iters=args.max_iters,
    frequency=args.frequency, 
    tensorboard_logging=True, 
    polyaxon_exp=experiment
)

################### Intialize Runner ###################

################### Train ###################
"""Dice coeffiecient is used to select best model weights.
Use metric as you think is best for your problem.
"""

best_val = -1e5
best_metrics = None

logging.info('STARTING training')

for step, outputs in runner.trainer():
    if runner.master():
        print(f'step: {step}')
        outputs.print()

        val_f1 = outputs.heads['cd'].means['val_metrics']['f1']
        if val_f1 > best_val:
            best_val = val_f1
            cpt_path = os.path.join(args.weight_dir,
                                    'checkpoint_epoch_'+ str(step) + '.pt')
            state_dict = model.module.state_dict() if runner.distributed \
                else model.state_dict()
            torch.save(state_dict, cpt_path)

################### Train ###################