import os, math, cv2, sys, glob, random, argparse, csv
from multiprocessing import Pool
from itertools import product
import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append('.')
from utils.dataloaders import *
from models.unet_blocks import *
from models.metrics_and_losses import *

def inference(tid, d1, d2, profile, date1, date2, model):
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


    profile['dtype'] = 'uint8'
    profile['driver'] = 'GTiff'
    fout = rasterio.open(results_dir + tid + '_' + date1 + '_' + date2 + '.tif', 'w', **profile)
    fout.write(np.asarray([out]).astype(np.uint8) * 255)
    fout.close()
    print (results_dir + tid + '_' + date1 + '_' + date2 + '.tif')


parser = argparse.ArgumentParser(description='Inference on Yiwu tiles')
parser.add_argument('--gpu_id', type=int, default=0, required=False)
# parser.add_argument('--tile_id', required=True)

opt = parser.parse_args()

# dates = """T50RQS 20151126T024032 20170228T023631 20171225T024121 20180728T023549
# T51RTM 20151126T024032 20170228T023631 20171225T024121 20181001T023551
# T51RTN 20151126T024032 20170228T023631 20171225T024121 20180728T023549
# T50RQT 20151126T024032 20170228T023631 20171225T024121 20181001T023551"""
# samples = {}
# for line in dates.split('\n'):
#     row = line.split()
#     samples[row[0]] = row[1:]


input_size = 64
weight_file = '../weights/onera/unet_siamese_prod_relu_inp64_13band_2dates_focal_hm_cnc_all_14_cities.pt'

data_dir = '/media/Drive1/CDTiles/'
results_dir = data_dir + 'cd_out/'

fin = open(data_dir + '100_cities_distinct_pairs.csv', 'r')
r = csv.reader(fin)
pairs = []
for row in r:
    pairs.append(row)
    
if opt.gpu_id == 0:
    pairs_gpu = pairs[:30]
if opt.gpu_id == 1:
    pairs_gpu = pairs[30:60]
if opt.gpu_id == 2:
    pairs_gpu = pairs[60:90]
if opt.gpu_id == 3:
    pairs_gpu = pairs[90:120]
    
USE_CUDA = torch.cuda.is_available()
def w(v):
    if USE_CUDA:
        return v.cuda(opt.gpu_id)
    return v

model = w(UNetClassify(layers=6, init_filters=32, num_channels=13, fusion_method='mul', out_dim=1))
weights = torch.load(weight_file, map_location='cuda:' + str(opt.gpu_id))
model.load_state_dict(weights)
model = model.eval()

# dates = samples[opt.tile_id]
# dates = ['20151126T024032','20151126T024032']
# dates = ['20151126T024032',' 20170228T023631']
# dates = ['20170228T023631','20151126T024032']
# dates.sort()

for pair in pairs_gpu:
    
    date1 = pair[0]
    date2 = pair[2]
    
    d1_bands = glob.glob(data_dir + 'SAFES/' + pair[1] + '/GRANULE/**/IMG_DATA/*_B*.jp2')
    d2_bands = glob.glob(data_dir + 'SAFES/' + pair[3] + '/GRANULE/**/IMG_DATA/*_B*.jp2')

#     print (d1_bands, d2_bands)
    profile = rasterio.open(d1_bands[2]).profile


    d1_bands.sort()
    d2_bands.sort()

    #load bands for two dates and do preprocessing
    d1d2 = read_bands(d1_bands + d2_bands)
    d1d2[13:] = match_bands(d1d2[:13], d1d2[13:])
    d1, d2 = stack_bands(d1d2)

    d1_d2 = inference(pair[-1], d1, d2, profile, date1, date2, model)
