import os
import numpy as np
import rasterio
import glob
from scipy import ndimage

inProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:32639', preserve_units=True)

def get_pix_from_s2(a, s2a, x, y):
    x1,y1 = a * (x,y)
    x2, y2 = transform(inProj,outProj,x1,y1)
    return  ~s2a * (x2,y2)

data_path = '../../datasets/onera/images/'
safe_path = '/media/Drive1/onera_safes/*'

cities = os.listdir(data_path)

bands = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']

for city in cities:
    if 'txt' not in city:
        if not os.path.exists(data_path + city + '/cropped_safes/'):
            os.mkdir(data_path + city + '/cropped_safes/')

        file_name = os.listdir(data_path + city + '/imgs_1/')[0]
        if 'S2A' in file_name:
            grid_name = file_name.split('_')[-2]
        else:
            grid_name = file_name.split('_')[0]

        city_safes = glob.glob(safe_path + grid_name + '*.SAFE')

        source_bands = {}
        for band in bands:
            source_bands[band] = rasterio.open(file_name[:-7] + band + '.jp2')

        for city_safe in city_safes:
            os.mkdir(data_path + city + '/cropped_safes/' + city_safe)

            base_name = glob.glob(city_safe + "/GRANULE/**/IMG_DATA/**.jp2")[0][:-7]

            for band in bands:
                s2b_r = rasterio.open(base_name + band + '.jp2')
                s2b = s2b_r.read()

                a = source_bands[band].affine
                s2a = s2b_r.affine


                band_out = np.zeros((source_bands[band].shape[0],source_bands[band].shape[1]))

                for i in range(source_bands[band].shape[0]):
                    for j in range(source_bands[band].shape[1]):
                        x,y = get_pix_from_s2(a, s2a, source_bands[band].shape[1]-j,source_bands[band].shape[0]-i)
                        band_out[i,j,:] = s2b[int(round(y)),int(round(x)),:]


                band_out = ndimage.rotate(band_out, 180)

                profile = source_bands[band].profile
                dst = rasterio.open(data_path + city + '/cropped_safes/' + city_safe + '/' + band + '.jp2', 'w', **profile)
                dst.write(band_out, 1)
                dst.close()
