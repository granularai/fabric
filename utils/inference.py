import sys
import os, csv, random, math, json
import glob

import rasterio
import cv2

from sklearn.feature_extraction import image

from multiprocessing import Pool

from PIL import Image
import numpy as np
import pandas as pd

import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize

import torch
import torch.utils.data as data
from torch.autograd import Variable

from torchvision import transforms
from torchvision.transforms import functional
from functools import partial

import matplotlib.pyplot as plt

sys.path.append('..')
from utils.dataloaders import city_loader, read_bands, stretch_8bit
from utils.helpers import log_figure, scale


def generate_patches(opt):
    """Generates patches for 2 dates based on user defined options for inference.
        NOTE: expects 2 images of same shape, depth=13 under imgs_1, imgs_2 in city name folder

    Parameters
    ----------
    opt : dict
        Dictionary of options/flags

    Returns
    -------
    tuple(patches1, patches2, hs, ws, lc, lr, h, w)
        2 image patch stacks and metadata for reconstruction

    """
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

    # TEMPORARY FIX: switching width and height seems to fix image generation...
    imgs_stacked = city_loader([opt.data_dir + 'images/' + opt.validation_city, template_img.width,template_img.height])

    d1 = imgs_stacked[0]
    d2 = imgs_stacked[1]

    # move image depth
    d1 = d1.transpose(1,2,0)
    d2 = d2.transpose(1,2,0)

    patches1, hs, ws, lc, lr, h, w = _get_patches(d1, patch_dim=opt.patch_size)

    patches1 = patches1.transpose(0,3,1,2)

    print ('Patches1 Created')

    patches2, hs, ws, lc, lr, h, w = _get_patches(d2, patch_dim=opt.patch_size)
    patches2 = patches2.transpose(0,3,1,2)

    print ('Patches2 Created')
    return patches1, patches2, hs, ws, lc, lr, h, w


def log_full_image(out, hs, ws, lc, lr, h, w, opt, epoch, device, comet):
    """Given a list of patch arrays, constructs an inference output and logs to comet

    Parameters
    ----------
    out : list
        list of np.arrays
    hs : int
        height count
    ws : int
        row count
    lc : int
        last column
    lr : int
        last row
    h : int
        height of original image
    w : int
        width of original image
    opt : dict
        Dictionary of options/flags
    epoch : int
        current epoch
    comet : comet_ml.Experiment
        instance of comet.ml logger

    """
    out = np.vstack(out)

    mask = _get_bands(out, hs, ws, lc, lr, h, w, patch_size=opt.patch_size)

    torch_mask = torch.from_numpy(mask).float().to(device)

    file_path = opt.validation_city+'_epoch_'+str(epoch)
    cv2.imwrite(file_path+'.png', scale(mask))
    comet.log_image(file_path+'.png')

    preview1 = stretch_8bit(cv2.imread(opt.data_dir + 'images/' + opt.validation_city + '/pair/img1.png', 1))
    preview2 = stretch_8bit(cv2.imread(opt.data_dir + 'images/' + opt.validation_city + '/pair/img2.png', 1))
    groundtruth = torch.from_numpy(cv2.imread(opt.data_dir + 'labels/' + opt.validation_city + '/cm/cm.png', 0))
    log_figure(comet, img1=preview1, img2=preview2, groundtruth=groundtruth, prediction=torch_mask, fig_name=file_path)



def _get_patches(bands, patch_dim=64):
    """given a 13 band image, get a stack of patches with metadata for reconstruction
    (last row/column position, corner position, original image size)

    Parameters
    ----------
    bands : np.array
        image array with depth 13
    patch_dim : int
        height and width of patches for model

    Returns
    -------
    tuple
        patches with metadata for reconstruction

    """
    patches = image.extract_patches(bands, (patch_dim, patch_dim, 13), patch_dim)
    hs, ws = patches.shape[0], patches.shape[1]
    patches = patches.reshape(-1, patch_dim, patch_dim, 13)

    last_row = bands[bands.shape[0]-patch_dim:,:,:]
    last_column = bands[:,bands.shape[1]-patch_dim:,:]
    corner = np.asarray([bands[bands.shape[0]-patch_dim:,bands.shape[1]-patch_dim:,:]])

    last_column = image.extract_patches(last_column, (patch_dim,patch_dim,13), patch_dim).reshape(-1, patch_dim, patch_dim, 13)
    last_row = image.extract_patches(last_row, (patch_dim,patch_dim,13), patch_dim).reshape(-1, patch_dim, patch_dim, 13)

    lc = last_column.shape[0]
    lr = last_row.shape[0]

    patches = np.vstack((patches, last_column, last_row, corner))
    return patches, hs, ws, lc, lr, bands.shape[0], bands.shape[1]

def _get_bands(patches, hs, ws, lc, lr, h, w, patch_size=64):
    """This method allows us to reconstruct an image from a stack of patches and return a composite numpy array

    Parameters
    ----------
    patches : np.array
        a stack of patches including last row, last column and corner patch
    hs : int
        height count
    ws : int
        row count
    lc : int
        last column
    lr : int
        last row
    h : int
        height of original image
    w : int
        width of original image
    patch_size : int
        user input for patch size

    Returns
    -------
    np.array
        2d image array (h,w)

    """
    corner = patches[-1]
    last_row = patches[-lr-1:-1]
    last_column = patches[-lc-lr-1:-lr-1]
    patches = patches[:-lc-lr-1]

    img = np.zeros((h,w))
    k = 0
    for i in range(hs):
        for j in range(ws):
            img[i*patch_size:i*patch_size+patch_size,j*patch_size:j*patch_size+patch_size] = patches[k]
            k += 1

    for i in range(lc):
        img[i*patch_size:i*patch_size+patch_size,w-patch_size:] = last_column[i]

    for i in range(lr):
        img[h-patch_size:,i*patch_size:i*patch_size+patch_size] = last_row[i]

    img[h-patch_size:,w-patch_size:] = corner

    return img
