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




input_size = 64
batch_size = 1200

weight_file = '../weights/onera/unet_siamese_prod_relu_inp64_13band_2dates_focal_hm_cnc_all_14_cities.pt'

data_dir = '/media/Drive2/Demo/'
results_dir = data_dir + '/ChangeDetection/'

fin = open(data_dir + 'demo_map.csv', 'r')
r = csv.reader(fin)

pairs = []
for row in r:
    pairs.append(row)

fin.close()    
    
model = UNetClassify(layers=6, init_filters=32, num_channels=13, fusion_method='mul', out_dim=1).cuda()
weights = torch.load(weight_file, map_location=lambda storage, loc: storage.cuda(0))
model.load_state_dict(weights)
model = nn.DataParallel(model, device_ids=[0,1,2,3])
model.eval()

for pair in pairs:
    tid = pair[0]
    date1 = pair[1]
    date2 = pair[2]
    safe1 = pair[3]
    safe2 = pair[4]
    
    if not os.path.exists(results_dir + tid + '_' + date1 + '_' + date2 + '.tif'):
        d1_bands = glob.glob(data_dir + 'Safes/' + safe1 + '/GRANULE/**/IMG_DATA/*_B*.jp2')
        d2_bands = glob.glob(data_dir + 'Safes/' + safe2 + '/GRANULE/**/IMG_DATA/*_B*.jp2')

        profile = rasterio.open(d1_bands[2]).profile


        d1_bands.sort()
        d2_bands.sort()

        d1d2 = read_bands(d1_bands + d2_bands)
        print ('Bands read')
        d1d2[13:] = match_bands(d1d2[:13], d1d2[13:])
        print ('Bands matched')
        d1, d2 = stack_bands(d1d2)

        print ('Bands Loaded')

        d1 = d1.transpose(1,2,0)
        d2 = d2.transpose(1,2,0)

        patches1, hs, ws, lc, lr, h, w = get_patches(d1)
        patches1 = patches1.transpose(0,3,1,2)

        print ('Patches1 Created')

        patches2, hs, ws, lc, lr, h, w = get_patches(d2)
        patches2 = patches2.transpose(0,3,1,2)

        print ('Patches2 Created')

        out = []
        for i in range(0,patches1.shape[0],batch_size):
            batch1 = torch.from_numpy(patches1[i:i+batch_size,:,:,:]).cuda()
            batch2 = torch.from_numpy(patches2[i:i+batch_size,:,:,:]).cuda()

            preds = model(batch1, batch2)
            del batch1
            del batch2

            preds = F.sigmoid(preds) > 0.5
            preds = preds.data.cpu().numpy()
            out.append(preds)



        out = np.vstack(out)
        mask = get_bands(out, hs, ws, lc, lr, h, w)

#         print (mask.shape)

        profile['dtype'] = 'uint8'
        profile['driver'] = 'GTiff'
        fout = rasterio.open(results_dir + tid + '_' + date1 + '_' + date2 + '.tif', 'w', **profile)
        fout.write(np.asarray([mask]).astype(np.uint8))
        fout.close()
        
#         break


    
    
    