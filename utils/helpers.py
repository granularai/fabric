import logging
from polyaxon_client.tracking import Experiment, get_log_level, get_data_paths, get_outputs_path
from polystores.stores.manager import StoreManager
import time
import tarfile


logging.basicConfig(level=logging.INFO)






def download_dataset(target_dataset):
    """download and extract the dataset from GCS

    Parameters
    ----------
    target_dataset : string
        `target_dataset` is the file name at the base of attached cloud storage eg (GCS: /data)


    """
    data_paths = list(get_data_paths().values())[0]
    data_store = StoreManager(path=data_paths)

    logging.info('STARTING tar download')
    start = time.time()
    data_store.download_file(target_dataset)
    end = time.time()
    logging.info('DOWNLOAD time taken: '+ str(end - start))
    if target_dataset.endswith('.tar.gz'):
        logging.info('STARTING untarring')
        tf = tarfile.open(target_dataset)
        tf.extractall()
        logging.info('COMPLETING untarring')




def define_output_path(opt):
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
