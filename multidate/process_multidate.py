import os
import numpy as np
import rasterio
import glob
from scipy import ndimage
from pyproj import Proj, transform
import cv2
from multiprocessing import Pool

import sys
sys.path.append('../utils')
from dataloaders import stretch_8bit

def get_pix_from_s2(inProj, outProj, a, s2a, x, y):
    x1,y1 = a * (x,y)
    x2, y2 = transform(inProj,outProj,x1,y1)
    return  ~s2a * (x2,y2)


data_path = '../../datasets/onera/images/'
safe_path = '/media/Drive1/onera_safes/*'

cities = os.listdir(data_path)

bands = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
# bands = ['B02','B03','B04']

for city in cities:
    print (city)
    if 'txt' not in city:
        if not os.path.exists(data_path + city + '/cropped_safes/'):
            os.mkdir(data_path + city + '/cropped_safes/')

        file_name = glob.glob(data_path + city + '/imgs_1/*.tif')[0]
        if 'S2A' in file_name:
            grid_name = file_name.split('_')[-2]
        else:
            grid_name = file_name.split('_')[0]

        city_safes = glob.glob(safe_path + grid_name + '*.SAFE')

        source_bands = {}
        for band in bands:
            source_bands[band] = rasterio.open(file_name[:-7] + band + '.tif')

        for city_safe in city_safes:
            if not os.path.exists(data_path + city + '/cropped_safes/' + city_safe.split('/')[-1]):
                os.mkdir(data_path + city + '/cropped_safes/' + city_safe.split('/')[-1])

            def process_safe(band):
                base_name = glob.glob(city_safe + "/GRANULE/**/IMG_DATA/**.jp2")[0][:-7]
                s2b_r = rasterio.open(base_name + band + '.jp2')
                s2b = s2b_r.read()[0]

                affine = source_bands[band].transform
                s2affine = s2b_r.transform

                inProj = Proj(**source_bands[band].crs)
                outProj = Proj(**s2b_r.crs, preserve_units=True)

                band_out = np.zeros((source_bands[band].shape[0],source_bands[band].shape[1]))

                for i in range(source_bands[band].shape[0]):
                    for j in range(source_bands[band].shape[1]):
                        x,y = get_pix_from_s2(inProj, outProj, affine, s2affine, source_bands[band].shape[1]-j,source_bands[band].shape[0]-i)
            #                         print (i,j,x,y)
                        band_out[i,j] = s2b[int(round(y)),int(round(x))]


                band_out = ndimage.rotate(band_out, 180)

                profile = source_bands[band].profile
                dst = rasterio.open(data_path + city + '/cropped_safes/' + city_safe.split('/')[-1] + '/' + band + '.tif', 'w', **profile)
                dst.write(band_out.astype(np.uint16), 1)
                dst.close()

            pool = Pool(13)
            pool.map(process_safe, bands)







            break
