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


def get_train_val_metadata(args):
    cities = [i for i in os.listdir(args.dataset_dir + 'labels/') if not
              i.startswith('.') and os.path.isdir(args.dataset_dir + 'labels/'+i)]
    cities.sort()
    training_cities = list(set(cities).difference(set(args.validation_cities)))

    train_metadata = []
    print ('cities :', cities)
    print ('train_cities :', training_cities)
    print ('val cities :', args.validation_cities)

    for city in training_cities:
        city_label = cv2.imread(args.dataset_dir + 'labels/' + city + '/cm/cm.png', 0) / 255

        for i in range(0, city_label.shape[0], args.stride):
            for j in range(0, city_label.shape[1], args.stride):
                if ((i + args.input_shape[2]) <= city_label.shape[0] and
                        (j + args.input_shape[2]) <= city_label.shape[1]) and \
                        np.sum(city_label[i:i+args.input_shape[2], j:j+args.input_shape[2]]) > args.train_thres:
                    train_metadata.append([city, i, j])

    val_metadata = []
    for city in args.validation_cities:
        city_label = cv2.imread(args.dataset_dir + 'labels/' + city + '/cm/cm.png', 0) / 255
        for i in range(0, city_label.shape[0], args.input_shape[2]):
            for j in range(0, city_label.shape[1], args.input_shape[2]):
                if ((i + args.input_shape[2]) <= city_label.shape[0] and
                        (j + args.input_shape[2]) <= city_label.shape[1]):
                    val_metadata.append([city, i, j])

    return train_metadata, val_metadata


def label_loader(label_path):
    label = cv2.imread(label_path + '/cm/' + 'cm.png', 0) / 255
    return label


def city_loader(city_meta):
    city = city_meta[0]
    h = city_meta[1]
    w = city_meta[2]
    args = city_meta[3]

    band_path = glob.glob(city + '/imgs_1/*')[0][:-7]
    bands_date1 = []
    for i in range(len(args.band_ids)):
        band = rasterio.open(band_path + args.band_ids[i] +
                             '.tif').read()[0].astype(np.float32)
        band = (band - args.band_means[args.band_ids[i]]) / args.band_stds[args.band_ids[i]]
        band = cv2.resize(band, (h, w))
        bands_date1.append(band)

    band_path = glob.glob(city + '/imgs_2/*')[0][:-7]
    bands_date2 = []
    for i in range(len(args.band_ids)):
        band = rasterio.open(band_path + args.band_ids[i] +
                             '.tif').read()[0].astype(np.float32)
        band = (band - args.band_means[args.band_ids[i]]) / args.band_stds[args.band_ids[i]]
        band = cv2.resize(band, (h, w))
        bands_date2.append(band)

    band_stacked = np.stack((bands_date1, bands_date2))

    return band_stacked


def full_onera_loader(args):
    cities = [i for i in os.listdir(args.dataset_dir + 'labels/') if not
              i.startswith('.') and os.path.isdir(args.dataset_dir + 'labels/'+i)]

    label_paths = []
    for city in cities:
        label_paths.append(args.dataset_dir + 'labels/' + city)

    pool = Pool(min(len(label_paths), args.num_workers))
    city_labels = pool.map(label_loader, label_paths)

    city_paths_meta = []
    i = 0
    for city in cities:
        city_paths_meta.append([args.dataset_dir + 'images/' + city,
                                city_labels[i].shape[1],
                                city_labels[i].shape[0], args])
        i += 1

    city_loads = pool.map(city_loader, city_paths_meta)

    pool.close()

    dataset = {}
    for cp in range(len(label_paths)):
        city = label_paths[cp].split('/')[-1]

        dataset[city] = {'images': city_loads[cp],
                         'labels': city_labels[cp].astype(np.uint8)}

    return dataset


def onera_siamese_loader(dataset, city, x, y, aug, args):
    out_img = np.copy(dataset[city]['images'][:, :, x : x + args.input_shape[2], y : y + args.input_shape[2]])
    out_lbl = np.copy(dataset[city]['labels'][x : x + args.input_shape[2], y : y + args.input_shape[2]])

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

    return out_img, out_lbl


class OneraPreloader(data.Dataset):

    def __init__(self, metadata, full_load, aug=False, args=None):
        random.shuffle(metadata)

        self.full_load = full_load
        self.samples = metadata
        self.loader = onera_siamese_loader
        self.aug = aug
        self.args = args

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index
                   of the target class.
        """
        city, x, y = self.samples[index]

        return self.loader(self.full_load, city, x, y, self.aug, self.args)

    def __len__(self):
        return len(self.samples)

def get_dataloaders(args):
    """Given user arguments, loads dataset metadata, loads full onera dataset,
       defines a preloader and returns train and val dataloaders
    Parameters
    ----------
    args : dict
        Dictionary of argsions/flags
    Returns
    -------
    (DataLoader, DataLoader)
        returns train and val dataloaders
    """
    train_samples, val_samples = get_train_val_metadata(args)
    print('train samples : ', len(train_samples))
    print('val samples : ', len(val_samples))


    full_load = full_onera_loader(args)

    train_dataset = OneraPreloader(train_samples,
                                   full_load, True, args)
    val_dataset = OneraPreloader(val_samples,
                                 full_load, False, args)


    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)
    val_loader = data.DataLoader(val_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)
    return train_loader, val_loader
