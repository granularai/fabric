import os, math, cv2, sys, glob, random, argparse
from multiprocessing import Pool
from itertools import product
import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append('../utils')
sys.path.append('../models')
from dataloaders import *
from unet_blocks import *
from metrics_and_losses import *




def read_band(band):
    return rasterio.open(band).read()[0]

def read_bands(band_paths):
    pool = Pool(26)
    bands = pool.map(read_band, band_paths)
    pool.close()
    return bands

def _match_band(two_date):
    return match_band(two_date[1], two_date[0])

def match_bands(date1, date2):
    pool = Pool(13)
    date2 = pool.map(_match_band, [[date1[i], date2[i]] for i in range(len(date1))])
    pool.close()
    return date2

def _resize(band):
    return cv2.resize(band, (10980, 10980))

def stack_bands(bands):
    pool = Pool(26)
    bands = pool.map(_resize, bands)
    pool.close()
    pool = Pool(26)
    bands = pool.map(stretch_8bit, bands)
    pool.close()

    return np.stack(bands[:13]).astype(np.float32), np.stack(bands[13:]).astype(np.float32)

parser = argparse.ArgumentParser(description='Inference on Yiwu tiles')
parser.add_argument('--gpu_id', type=int, default=0, required=False)
parser.add_argument('--tile_id', required=True)

opt = parser.parse_args()

dates = """T50RQS 20151126T024032 20170228T023631 20171225T024121 20180728T023549
T51RTM 20151126T024032 20170228T023631 20171225T024121 20181001T023551
T51RTN 20151126T024032 20170228T023631 20171225T024121 20180728T023549
T50RQT 20151126T024032 20170228T023631 20171225T024121 20181001T023551"""
samples = {}
for line in dates.split('\n'):
    row = line.split()
    samples[row[0]] = row[1:]


input_size = 64
weight_file = '../../weights/onera/unet_siamese_prod_relu_inp64_13band_2dates_focal_hm_cnc_all_14_cities.pt'
results_dir = '../../../Yiwu/cd_out/'

USE_CUDA = torch.cuda.is_available()
def w(v):
    if USE_CUDA:
        return v.cuda(opt.gpu_id)
    return v

# model = w(UNetClassify(layers=6, init_filters=32, num_channels=13, fusion_method='mul', out_dim=1))
# weights = torch.load(weight_file, map_location='cuda:' + str(opt.gpu_id))
# model.load_state_dict(weights)

dates = samples[opt.tile_id]
# dates = ['20151126T024032','20151126T024032']
# dates = ['20151126T024032',' 20170228T023631']
# dates = ['20170228T023631','20151126T024032']
dates.sort()

a = glob.glob('../../../Yiwu/SAFES/*20151126T024032*T51RTM*/GRANULE/**/IMG_DATA/*_B*.jp2')
b = glob.glob('../../../Yiwu/SAFES/*20170228T023631*T51RTM*/GRANULE/**/IMG_DATA/*_B*.jp2')
c = glob.glob('../../../Yiwu/SAFES/*20171225T024121*T51RTM*/GRANULE/**/IMG_DATA/*_B*.jp2')


        #get band paths for each date
        d1_bands = a
        d2_bands = b
        
        print (d1_bands[0], "%%%%%%%%%%%%%%%%", d2_bands[0])
        print ('##########################')
        print ('##########################')
        print ('##########################')

        d1_bands.sort()
        d2_bands.sort()

        #load bands for two dates and do preprocessing
        d1d2 = read_bands(d1_bands + d2_bands)
        d1d2[13:] = match_bands(d1d2[:13], d1d2[13:])
        d1, d2 = stack_bands(d1d2)

        out = np.zeros((d1.shape[1], d1.shape[2]))

        batches1 = []
        batches2 = []
        ijs = []
        for i in range(0,d1.shape[1],64):
            for j in range(0,d1.shape[2],64):
                if i+input_size <= d1.shape[1] and j+input_size <= d1.shape[2]:
                    batches1.append(d1[:,i:i+input_size,j:j+input_size])
                    batches2.append(d2[:,i:i+input_size,j:j+input_size])
                    ijs.append([i,j])
                elif i+input_size>d1.shape[1] and j+input_size<=d1.shape[2]:
                    batches1.append(d1[:,d1.shape[1]-input_size:d1.shape[1],j:j+input_size])
                    batches2.append(d2[:,d2.shape[1]-input_size:d2.shape[1],j:j+input_size])
                    ijs.append([d1.shape[1]-input_size,j])
                elif i+input_size<=d1.shape[1] and j+input_size>d1.shape[2]:
                    batches1.append(d1[:,i:i+input_size,d1.shape[2]-input_size:d1.shape[2]])
                    batches2.append(d2[:,i:i+input_size,d2.shape[2]-input_size:d2.shape[2]])
                    ijs.append([i,d1.shape[2]-input_size])
                else:
                    batches1.append(d1[:,d1.shape[1]-input_size:d1.shape[1],
                                         d1.shape[2]-input_size:d1.shape[2]])
                    batches2.append(d2[:,d2.shape[1]-input_size:d2.shape[1],
                                         d2.shape[2]-input_size:d2.shape[2]])
                    ijs.append([d1.shape[1]-input_size,d1.shape[2]-input_size])

                if len(batches1) == 110:
                    inp1 = w(torch.from_numpy(np.asarray(batches1) / 255.))
                    inp2 = w(torch.from_numpy(np.asarray(batches2) / 255.))
                    logits = model(inp1, inp2)
                    pred = F.sigmoid(logits) > 0.5
                    pred = pred.data.cpu().numpy()

                    batches1 = []
                    batches2 = []

                    del inp1
                    del inp2

                    for c in range(len(ijs)):
                        out[ijs[c][0]:ijs[c][0]+input_size,ijs[c][1]:ijs[c][1]+input_size] = pred[c]

                    ijs = []

        profile = rasterio.open(d1_bands[1]).profile
        profile['dtype'] = 'uint8'
        profile['driver'] = 'GTiff'
        fout = rasterio.open(results_dir + opt.tile_id + '_' + dates[i] + '_' + dates[j] + '.tif', 'w', **profile)
        fout.write(np.asarray([out]).astype(np.uint8))
        fout.close()
