import sys
import os, csv, random
import glob

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

def make_sequence_dataset(dir, sequences_list):
    sequences = []

    for sequence in sequences_list:
        images = []
        for image in sequence[0]:
            images.append(dir + image)

        sequences.append([images, int(sequence[1])])

    return sequences

def make_bimode_sequence_dataset(rgb_dir, flow_dir, sequences_list):
    sequences = []

    for sequence in sequences_list:
        rgb_images = []
        flow_images = []

        for image in sequence[0]:
            rgb_images.append(rgb_dir + image)
            flow_images.append(flow_dir + image)

        sequences.append([rgb_images, flow_images, int(sequence[1])])

    return sequences

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

def onera_loader(path, city, x, y):
    base_path1 = glob.glob(path + 'images/' + city + '/imgs_1/*.tif')[0][:-7]
    base_path2 = glob.glob(path + 'images/' + city + '/imgs_2/*.tif')[0][:-7]
    label_r = rasterio.open(path + 'train_labels/' + city + '/cm/' + city + '-cm.tif')

    bands1_stack = []
    bands2_stack = []
    for band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']:
        band_r = rasterio.open(base_path1 + band + '.tif')
        band = band_r.read()[0]
        band = cv2.resize(band, label_r.shape)
        bands1_stack.append(band)

        band_r = rasterio.open(base_path2 + band + '.tif')
        band = band_r.read()[0]
        band = cv2.resize(band, label_r.shape)
        bands2_stack.append(band)

    

def rgb_sequence_loader(paths, mean, std, inp_size, rand_crop_size, resize_size):
    irand = random.randint(0, inp_size[0] - rand_crop_size[0])
    jrand = random.randint(0, inp_size[1] - rand_crop_size[1])
    flip = random.random()
    batch = []
    for path in paths:
        img = Image.open(path)
        img = img.convert('RGB')
        img = functional.center_crop(img, (inp_size[0], inp_size[1]))
        img = functional.crop(img, irand, jrand, rand_crop_size[0], rand_crop_size[1])
        img = functional.resize(img, resize_size)
        if flip < 0.5:
            img = functional.hflip(img)
        tensor = functional.to_tensor(img)
        tensor = functional.normalize(tensor, mean, std)
        batch.append(tensor)

    batch = torch.stack(batch)

    return batch

def flow_sequence_loader(paths, mean, std, inp_size, rand_crop_size, resize_size):
    irand = random.randint(0, inp_size[0] - rand_crop_size[0])
    jrand = random.randint(0, inp_size[1] - rand_crop_size[1])
    flip = random.random()
    batch = []
    for path in paths:
        img = Image.open(path)
        img = img.convert('RGB')
        img = functional.resize(img, resize_size)
        img = functional.crop(img, irand, jrand, rand_crop_size[0], rand_crop_size[1])
        if flip < 0.5:
            img = functional.hflip(img)
        tensor = functional.to_tensor(img)
        tensor = functional.normalize(tensor, mean, std)
        batch.append(tensor)

    batch = torch.stack(batch)

    return batch

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

    def __init__(self, root, csv_file):

        r = csv.reader(open(csv_file, 'r'), delimiter=',')

        images_list = []

        for row in r:
            images_list.append([row[0], row[1], row[2], row[3]])


        random.shuffle(images_list)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.loader = onera_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, x, y, target = self.imgs[index]

        img, target = self.loader(path, x, y)

        return img, target

    def __len__(self):
        return len(self.imgs)