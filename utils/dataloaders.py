import sys
import os, csv, random
import glob

import rasterio 
import cv2

from PIL import Image
import numpy as np

import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize

import torch
import torch.utils.data as data
from torch.autograd import Variable

from torchvision.transforms import functional


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(images_list):

    classes = {}
    class_id = 0
    for image in images_list:
        if image[1] not in classes:
            classes[image[1]] = class_id
            class_id += 1

    return classes.keys(), classes

def make_dataset(dir, images_list, class_to_idx):
    images = []

    for image in images_list:
        images.append((dir + image[0], int(image[1])))

    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def npy_seq_loader(seq):
    out = []
    for s in seq:
        out.append(np.load(s))
    out = np.asarray(out)

    return out

def full_onera_loader(path):
    cities = os.listdir(path + 'train_labels/')

    dataset = {}
    for city in cities:
        if '.txt' not in city:
            base_path1 = glob.glob(path + 'images/' + city + '/imgs_1/*.tif')[0][:-7]
            base_path2 = glob.glob(path + 'images/' + city + '/imgs_2/*.tif')[0][:-7]
            label_r = rasterio.open(path + 'train_labels/' + city + '/cm/' + city + '-cm.tif')

            bands1_stack = []
            bands2_stack = []
            for band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']:
                band_r = rasterio.open(base_path1 + band + '.tif')
                band_d = band_r.read()[0]
                band_d = cv2.resize(band_d, (label_r.shape[1], label_r.shape[0]))
                bands1_stack.append(band_d)

                band_r = rasterio.open(base_path2 + band + '.tif')
                band_d = band_r.read()[0]
                band_d = cv2.resize(band_d, (label_r.shape[1], label_r.shape[0]))
                bands2_stack.append(band_d)

            label = label_r.read()[0]
            label -= 1
            
            two_dates = np.asarray([bands1_stack, bands2_stack]).astype(np.float32)
            two_dates = np.transpose(two_dates, (1,0,2,3))
            dataset[city] = {'images':two_dates , 'labels': label}

    return dataset


def onera_loader(dataset, city, x, y, size):
    return dataset[city]['images'][:, : ,x:x+size, y:y+size], dataset[city]['labels'][x:x+size, y:y+size]

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class ImagePreloader(data.Dataset):

    def __init__(self, root, csv_file, class_map, transform=None, target_transform=None,
                 loader=default_loader):

        r = csv.reader(open(csv_file, 'r'), delimiter=',')

        images_list = []

        for row in r:
            images_list.append([row[0],row[1]])


        random.shuffle(images_list)
        classes, class_to_idx = class_map.keys(), class_map
        imgs = make_dataset(root, images_list, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class OneraPreloader(data.Dataset):

    def __init__(self, root, csv_file, input_size):

        r = csv.reader(open(csv_file, 'r'), delimiter=',')

        images_list = []

        for row in r:
            images_list.append([row[0], int(row[1]), int(row[2])])


        random.shuffle(images_list)

        self.full_load = full_onera_loader(root)
        self.input_size = input_size
        self.root = root
        self.imgs = images_list
        self.loader = onera_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        city, x, y = self.imgs[index]

        img, target = self.loader(self.full_load, city, x, y, self.input_size)
        return img, target

    def __len__(self):
        return len(self.imgs)
