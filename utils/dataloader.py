import os
import random
import glob

import rasterio
import cv2

from multiprocessing import Pool

import numpy as np

import torch.utils.data as data


def read_band(band):
    r = rasterio.open(band)
    data = r.read()[0]
    r.close()
    return data


def read_bands(band_paths):
    pool = Pool(26)
    bands = pool.map(read_band, band_paths)
    pool.close()
    return bands


def _resize(band, height, width):
    band = cv2.resize(band, (height, width))
    band = stretch_8bit(band)
    return band


def stretch_8bit(band, lower_percent=2, higher_percent=98):
    a = 0
    b = 255
    real_values = band.flatten()
    real_values = real_values[real_values > 0]
    c = np.percentile(real_values, lower_percent)
    d = np.percentile(real_values, higher_percent)
    t = a + (band - c) * ((b - a) / (d - c))
    t[t < a] = a
    t[t > b] = b
    return t.astype(np.uint8)


def get_train_val_metadata(data_dir, val_cities, patch_size, stride):
    cities = [i for i in os.listdir(data_dir + 'labels/') if not
              i.startswith('.') and os.path.isdir(data_dir+'labels/'+i)]
    cities.sort()
    train_cities = list(set(cities).difference(set(val_cities)))

    train_metadata = []
    print('cities:', cities)
    print('train_cities:', train_cities)
    for city in train_cities:
        city_label = cv2.imread(data_dir + 'labels/' + city + '/cm/cm.png', 0) / 255

        for i in range(0, city_label.shape[0], stride):
            for j in range(0, city_label.shape[1], stride):
                if ((i + patch_size) <= city_label.shape[0] and
                        (j + patch_size) <= city_label.shape[1]):
                    train_metadata.append([city, i, j])

    val_metadata = []
    for city in val_cities:
        city_label = cv2.imread(data_dir + 'labels/' + city + '/cm/cm.png', 0) / 255
        for i in range(0, city_label.shape[0], stride):
            for j in range(0, city_label.shape[1], stride):
                if ((i + patch_size) <= city_label.shape[0] and
                        (j + patch_size) <= city_label.shape[1]):
                    val_metadata.append([city, i, j])

    return train_metadata, val_metadata


def label_loader(label_path):
    label = cv2.imread(label_path + '/cm/' + 'cm.png', 0) / 255
    return label


def city_loader(city_meta):
    city = city_meta[0]
    h = city_meta[1]
    w = city_meta[2]
    opt = city_meta[3]

    band_path = glob.glob(city + '/imgs_1/*')[0][:-7]
    bands_date1 = []
    for i in range(len(opt.band_ids)):
        band = rasterio.open(band_path + opt.band_ids[i] +
                             '.tif').read()[0].astype(np.float32)
        band = (band - opt.band_means[opt.band_ids[i]]) / opt.band_stds[opt.band_ids[i]]
        band = cv2.resize(band, (h, w))
        bands_date1.append(band)

    band_path = glob.glob(city + '/imgs_2/*')[0][:-7]
    bands_date2 = []
    for i in range(len(opt.band_ids)):
        band = rasterio.open(band_path + opt.band_ids[i] +
                             '.tif').read()[0].astype(np.float32)
        band = (band - opt.band_means[opt.band_ids[i]]) / opt.band_stds[opt.band_ids[i]]
        band = cv2.resize(band, (h, w))
        bands_date2.append(band)

    band_stacked = np.stack((bands_date1, bands_date2))

    return band_stacked


def full_onera_loader(data_dir, opt):
    cities = [i for i in os.listdir(data_dir + 'labels/') if not
              i.startswith('.') and os.path.isdir(data_dir+'labels/'+i)]

    label_paths = []
    for city in cities:
        label_paths.append(data_dir + 'labels/' + city)

    pool = Pool(len(label_paths))
    city_labels = pool.map(label_loader, label_paths)

    city_paths_meta = []
    i = 0
    for city in cities:
        city_paths_meta.append([data_dir + 'images/' + city,
                                city_labels[i].shape[1],
                                city_labels[i].shape[0], opt])
        i += 1

    city_loads = pool.map(city_loader, city_paths_meta)

    pool.close()

    dataset = {}
    for cp in range(len(label_paths)):
        city = label_paths[cp].split('/')[-1]

        dataset[city] = {'images': city_loads[cp],
                         'labels': city_labels[cp].astype(np.uint8)}

    return dataset


def onera_siamese_loader(dataset, city, x, y, size, aug):
    out_img = np.copy(dataset[city]['images'][:, :, x:x+size, y:y+size])
    out_lbl = np.copy(dataset[city]['labels'][x:x+size, y:y+size])

    if aug:
        rot_deg = random.randint(0, 3)
        out_img = np.rot90(out_img, rot_deg, [2, 3]).copy()
        out_lbl = np.rot90(out_lbl, rot_deg, [0, 1]).copy()

        if random.random() > 0.5:
            out_img = np.flip(out_img, axis=2).copy()
            out_lbl = np.flip(out_lbl, axis=0).copy()

        if random.random() > 0.5:
            out_img = np.flip(out_img, axis=3).copy()
            out_lbl = np.flip(out_lbl, axis=1).copy()

    return out_img[0], out_img[1], out_lbl


class OneraPreloader(data.Dataset):

    def __init__(self, root, metadata, full_load, input_size, aug=False):
        random.shuffle(metadata)

        self.full_load = full_load
        self.root = root
        self.imgs = metadata
        self.loader = onera_siamese_loader
        self.aug = aug
        self.input_size = input_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index
                   of the target class.
        """
        city, x, y = self.imgs[index]

        return self.loader(self.full_load,
                           city,
                           x,
                           y,
                           self.input_size,
                           self.aug)

    def __len__(self):
        return len(self.imgs)
