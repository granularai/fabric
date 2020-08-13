import logging
from polyaxon_client.tracking import get_data_paths
from polystores.stores.manager import StoreManager
import time


import torch
import torch.utils.data
import torch.nn as nn
import numpy as np

from utils.dataloaders import (get_train_val_metadata,
                               full_onera_loader,
                               OneraPreloader)
from models.bidate_model import BiDateNet
from utils.metrics import TverskyLoss, jaccard_loss, FocalLoss, dice_loss

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


def initialize_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values

    Returns
    -------
    dict
        a dictionary of metrics

    """
    metrics = {
        'cd_losses': [],
        'cd_corrects': [],
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
    }

    return metrics


def get_mean_metrics(metric_dict):
    """takes a dictionary of lists for metrics and returns dict of mean values

    Parameters
    ----------
    metric_dict : dict
        A dictionary of metrics

    Returns
    -------
    dict
        dict of floats that reflect mean metric value

    """
    return {k: np.mean(v) for k, v in metric_dict.items()}


def set_metrics(metric_dict, cd_loss, cd_corrects, cd_report):
    """Updates metric dict with batch metrics

    Parameters
    ----------
    metric_dict : dict
        dict of metrics
    cd_loss : dict(?)
        loss value
    cd_corrects : dict(?)
        number of correct results (to generate accuracy
    cd_report : list
        precision, recall, f1 values

    Returns
    -------
    dict
        dict of  updated metrics


    """
    metric_dict['cd_losses'].append(cd_loss.item())
    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])

    return metric_dict


def log_patches(comet, epoch, batch_img1, batch_img2, labels, cd_preds):
    """Logs specified patches with real image patch and groundtruth label to comet

    Parameters
    ----------
    comet : comet_ml.Experiment
        instance of comet.ml logger
    epoch : int
        current epoch
    batch_img1 : np.array
        date 1 image stack correlated to batch of predictions
    batch_img2 : np.array
        date 2 image stack correlated to batch of predictions
    labels : torch.tensor
        groundtruth array correlated to batch of predictions
    cd_preds : torch.tensor
        batch of predictions


    """
    batch_size = batch_img1.shape[0]
    samples = list(range(0, batch_size, 10))
    for sample in samples:
        sample_img1 = _denorm_image(batch_img1, sample)
        sample_img2 = _denorm_image(batch_img2, sample)

        # log cd
        cd_figname = 'epoch_'+str(epoch)+'_cd_sample_'+str(sample)
        log_figure(comet,
                   sample_img1,
                   sample_img2,
                   labels[sample].cpu().numpy(),
                   cd_preds[sample].cpu().numpy(),
                   fig_name=cd_figname)


def _denorm_image(img_tsr, sample):
    """takes a tensor and returns a normalized array

    Parameters
    ----------
    img_tsr : torch.tensor
        cuda tensor of image
    sample : int
        sample of interest from image_tensor

    Returns
    -------
    np.array
        scaled, flipped array

    """
    # select the sample of interest from bands 2-4, flip for rgb, move depth
    trans_torch = torch.flip(img_tsr[sample][1:4, :, :], [0]).permute(1, 2, 0)

    # Convert to a numpy array
    np_arr = trans_torch.cpu().numpy()
    return scale(np_arr).astype(int)


def scale(x, out_range=(0, 255)):
    """scales an array to specified range (default: 0-255)

    Parameters
    ----------
    x : np.array
        Array to be scaled
    out_range : tuple(int,int)
        output array scale

    Returns
    -------
    np.array
        array with same dimensions but scaled values

    """
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return (y *
            (out_range[1] - out_range[0]) +
            (out_range[1] + out_range[0]) /
            2)


def log_figure(comet, img1, img2, groundtruth, prediction, fig_name=''):
    """logs a set of arrays to a figure and uploads to comet

    Parameters
    ----------
    comet : comet_ml.Experiment
        comet.ml instance
    img1 : np.array
        3 band image1 array
    img2 : np.array
        3 band image1 array
    groundtruth : np.array
        groundtruth array of depth 1
    prediction : np.array
        prediction array of depth 1
    fig_name : string
        log name of figure

    """
    fig, axarr = plt.subplots(2, 2)
    axarr[0, 0].set_title("Date 1")
    axarr[0, 0].imshow(img1)
    axarr[0, 1].set_title("Date 2")
    axarr[0, 1].imshow(img2)
    axarr[1, 0].set_title("Groundtruth")
    axarr[1, 0].imshow(groundtruth)
    axarr[1, 1].set_title("Prediction")
    axarr[1, 1].imshow(prediction)
    plt.setp(axarr, xticks=[], yticks=[])

    comet.log_figure(figure=fig, figure_name=fig_name)

    plt.close(fig=fig)


def get_loaders(opt):
    """Given user arguments, loads dataset metadata, loads full onera dataset,
       defines a preloader and returns train and val dataloaders

    Parameters
    ----------
    opt : dict
        Dictionary of options/flags

    Returns
    -------
    (DataLoader, DataLoader)
        returns train and val dataloaders

    """
    train_samples, val_samples = get_train_val_metadata(opt.dataset_dir,
                                                        opt.validation_cities,
                                                        opt.patch_size,
                                                        opt.stride)
    print('train samples : ', len(train_samples))
    print('val samples : ', len(val_samples))

    logging.info('STARTING Dataset Creation')

    full_load = full_onera_loader(opt.dataset_dir, opt)

    train_dataset = OneraPreloader(opt.dataset_dir,
                                   train_samples,
                                   full_load,
                                   opt.patch_size,
                                   opt.augmentation)
    val_dataset = OneraPreloader(opt.dataset_dir,
                                 val_samples,
                                 full_load,
                                 opt.patch_size,
                                 False)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return train_loader, val_loader


def get_criterion(opt):
    """get the user selected loss function

    Parameters
    ----------
    opt : dict
        Dictionary of options/flags

    Returns
    -------
    method
        loss function

    """

    if opt.loss_function == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    if opt.loss_function == 'focal':
        criterion = FocalLoss(opt.focal_gamma)
    if opt.loss_function == 'dice':
        criterion = dice_loss
    if opt.loss_function == 'jaccard':
        criterion = jaccard_loss
    if opt.loss_function == 'tversky':
        criterion = TverskyLoss(alpha=opt.tversky_alpha, beta=opt.tversky_beta)

    return criterion


def load_model(opt, device):
    """Loads the model specific to user flags

    Parameters
    ----------
    opt : dict
        User specified flags/options
    device : string
        device on which to train model

    Returns
    -------
    torch.nn.DataParallel
        DataParallel model

    """
    device_ids = list(range(opt.num_gpus))
    model = BiDateNet(13, 2).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)

    return model

# def get_weight_filename(weight_file):
#     return '{}/{}'.format(get_outputs_path(), 'checkpoint.pth.tar')
