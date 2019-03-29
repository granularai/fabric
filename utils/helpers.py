import logging
from polyaxon_client.tracking import get_data_paths, get_outputs_path
from polystores.stores.manager import StoreManager
import time
import tarfile


import torch
import torch.utils.data
import torch.nn as nn

import sys

sys.path.append('..')
from utils.dataloaders import *
from models.bidate_model import *
from utils.metrics import *

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)




def initialize_metrics():
    metrics={
        'cd_losses': [],
        'cd_corrects': [],
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
        'lulc_losses': [],
        'lulc_corrects': [],
        'lulc_precisions': [],
        'lulc_recalls': [],
        'lulc_f1scores': []
    }

    return metrics

def get_mean_metrics(metric_dict):
    return {k:np.mean(v) for k,v in metric_dict.items()}

def set_metrics(metric_dict, cd_loss, cd_corrects, cd_report, lulc_loss, lulc_corrects, lulc_report):
    metric_dict['cd_losses'].append(cd_loss.item())
    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])
    metric_dict['lulc_losses'].append(lulc_loss.item())
    metric_dict['lulc_corrects'].append(lulc_corrects.item())
    metric_dict['lulc_precisions'].append(lulc_report[0])
    metric_dict['lulc_recalls'].append(lulc_report[1])
    metric_dict['lulc_f1scores'].append(lulc_report[2])

    return metric_dict

def log_images(comet, epoch, batch_img1, batch_img2, labels, masks, cd_preds, lulc_preds):
    # batch_img1
    # img.save(filename)
    # comet.log_image(filename)
    print('batch_img1 shape', batch_img1.shape)
    print('batch_img2 shape', batch_img2.shape)
    print('labels shape', labels.shape)
    print('masks shape', masks.shape)
    print('cd_preds shape', cd_preds.shape)
    print('lulc_preds shape', lulc_preds.shape)

    batch_size = batch_img1.shape[0]
    samples = list(range(0,batch_size,10))
    for sample in samples:
        sample_img1 = _denorm_image(batch_img1, sample)
        sample_img2 = _denorm_image(batch_img2, sample)

        #log cd
        cd_figname='epoch_'+str(epoch)+'_change_detection_sample_'+str(sample)
        _log_figure(comet, sample_img1, sample_img2, labels[sample], cd_preds[sample], fig_name=cd_figname)

        #log lulc
        lulc_figname='epoch_'+str(epoch)+'_landuse_landcover_sample_'+str(sample)
        _log_figure(comet, sample_img1, sample_img2, masks[sample], lulc_preds[sample], fig_name=lulc_figname)

def _denorm_image(image_tensor, sample):
    np_arr = torch.flip(image_tensor[sample][1:4,:,:],[0]).permute(1,2,0).cpu().numpy()
    return _scale(np_arr).astype(int)

def _scale(x, out_range=(0, 255)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def _log_figure(comet, batch1_img, batch2_img, groundtruth, prediction, fig_name=''):
    fig, axarr = plt.subplots(1,4)
    axarr[0].set_title("Date 1")
    axarr[0].imshow(batch1_img)
    axarr[1].set_title("Date 2")
    axarr[1].imshow(batch2_img)
    axarr[2].set_title("Groundtruth")
    axarr[2].imshow(groundtruth.cpu().numpy())
    axarr[3].set_title("Prediction")
    axarr[3].imshow(prediction.cpu().numpy())
    plt.setp(axarr, xticks=[], yticks=[])

    comet.log_figure(figure=fig, figure_name=fig_name)



def get_loaders(opt):
    """Given user arguments, loads dataset metadata, loads full onera dataset, defines a preloader and returns train and val dataloaders

    Parameters
    ----------
    opt : dict
        Dictionary of options/flags

    Returns
    -------
    (DataLoader, DataLoader)
        returns train and val dataloaders

    """
    train_samples, val_samples = get_train_val_metadata(opt.data_dir, opt.val_cities, opt.patch_size, opt.stride)
    print('train samples : ', len(train_samples))
    print('val samples : ', len(val_samples))

    logging.info('STARTING Dataset Creation')

    full_load = full_onera_loader(opt.data_dir, load_mask=opt.mask)

    train_dataset = OneraPreloader(opt.data_dir, train_samples, full_load, opt.patch_size, opt.aug, opt.mask)
    val_dataset = OneraPreloader(opt.data_dir, val_samples, full_load, opt.patch_size, False, opt.mask)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_loader, val_loader



def download_dataset(target_dataset, comet):
    """download and extract the dataset from GCS

    Parameters
    ----------
    target_dataset : string
        `target_dataset` is the file name at the base of attached cloud storage eg (GCS: /data)


    """
    data_paths = list(get_data_paths().values())[0]
    data_store = StoreManager(path=data_paths)

    logging.info('STARTING tar download')
    comet.log_dataset_info(name=target_dataset, version=None, path=data_paths)
    start = time.time()
    data_store.download_file(target_dataset)
    end = time.time()
    logging.info('DOWNLOAD time taken: '+ str(end - start))
    comet.log_dataset_hash(target_dataset)
    if target_dataset.endswith('.tar.gz'):
        logging.info('STARTING untarring')
        tf = tarfile.open(target_dataset)
        tf.extractall()
        logging.info('COMPLETING untarring')




def define_output_paths(opt):
    """Uses user defined options (or defaults) to define an appropriate output path

    Parameters
    ----------
    opt : dict
        Dictionary of options/flags

    Returns
    -------
    string
        output path

    """
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

    weight_path = opt.weight_dir + path + '.pt'
    log_path = opt.log_dir + path + '.log'
    return weight_path, log_path


def get_criterion(opt):
    """Short summary.

    Parameters
    ----------
    opt : type
        Description of parameter `opt`.

    Returns
    -------
    type
        Description of returned object.

    """

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
    if opt.mask:
        model = BiDateLULCNet(13, 2, 5).to(device)
        model = nn.DataParallel(model, device_ids=[int(x) for x in opt.gpu_ids.split(',')])
    else:
        model = BiDateNet(13, 2).to(device)
        model = nn.DataParallel(model, device_ids=[int(x) for x in opt.gpu_ids.split(',')])

    return model

# def get_weight_filename(weight_file):
#     return '{}/{}'.format(get_outputs_path(), 'checkpoint.pth.tar')
