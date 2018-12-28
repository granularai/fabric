import glob
import cv2
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import random
import pandas as pd
import math
from tqdm import tqdm_notebook as tqdm
from utils.dataloaders import BuildingsPreloader as Preloader
from utils.dataloaders import full_buildings_loader

def uncombine(mask):
    max_val = np.max(mask) + 1
    results = []
    for i in range(1, max_val):
        results.append(mask == i)
    return results

def iou(mask1, mask2):
    return np.sum(mask1 & mask2) / np.sum(mask1 | mask2)

def evaluate_split(labels, y_pred):
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    #print("Number of true objects:", true_objects)
    #print("Number of predicted objects:", pred_objects)

    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    #print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if tp + fp + fn == 0:
            p = 1.0
        else:
            p = tp / (tp + fp + fn)
        #print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    #print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def evaluate_combined(combined_mask_true, combined_mask_pred):
    return evaluate_split(combined_mask_true, combined_mask_pred)

def evaluate_naive_tuple(tup):
    return evaluate_naive(*tup)

def classify_naive(image, factor, kernel_sz):
    if np.median(image) < 127:
        thresholded = (image > np.mean(image) + np.std(image) * factor).astype(np.uint8) * 255
    else:
        thresholded = (image < np.mean(image) - np.std(image) * factor).astype(np.uint8) * 255

    kernel = np.ones((kernel_sz, kernel_sz))
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    _, connected = cv2.connectedComponents(thresholded)

    return connected

def evaluate_naive(folder, factor, kernel_sz):
    image = glob.glob(folder + '/images/*')[0]
    image = imread(image)
    masks = glob.glob(folder + '/masks/*')
    total_mask = None
    for i, m in enumerate(masks):
        m = (imread(m) // 255).astype(np.int32)
        if total_mask is None:
            total_mask = m
        else:
            total_mask += m * (i+1)

    connected = classify_naive(image, factor, kernel_sz)

    return evaluate_combined(total_mask, connected)

def rle(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def rle_combined(combined):
    all_rle = []
    if np.max(combined) == 0:
        combined[0, 0] = 1
    max_val = np.max(combined) + 1
    for i in range(1, max_val):
        all_rle.append(rle(combined == i))
    return all_rle

# TODO: test rle by encoding and decoding and figuring out if it matches

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(lab_img, cut_off = 0.5):
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)

def open_res_csv(key=''):
    cur = 0
    while True:
        path = '_%s_submit_%03d.csv' % (key, cur)
        if not os.path.exists(path):
            return open(path, 'w')
        cur += 1

def find_clusters(img):
    return cv2.connectedComponents((img > 0.5).astype(np.uint8))[1]


DROPOUT = 0.5

def compute_size(inp_size, kernel_size, stride, padding):
    return (inp_size - kernel_size + 2 * padding) / stride + 1

class UNetBlock(nn.Module):
    def __init__(self, filters_in, filters_out):
        super().__init__()
        self.filters_in = filters_in
        self.filters_out = filters_out
        
        self.conv1 = nn.Conv2d(filters_in, filters_out, (3, 3), padding=1)
        self.norm1 = nn.BatchNorm2d(filters_out)
        self.conv2 = nn.Conv2d(filters_out, filters_out, (3, 3), padding=1)
        self.norm2 = nn.BatchNorm2d(filters_out)

        self.activation = nn.ReLU()

    def forward(self, x):
        conved1 = self.conv1(x)
        conved1 = self.activation(conved1)
        conved1 = self.norm1(conved1)
        conved2 = self.conv2(conved1)
        conved2 = self.activation(conved2)
        conved2 = self.norm2(conved2)
        return conved2

class UNetDownBlock(UNetBlock):
    def __init__(self, input_size, filters_in, filters_out, pooling=True):
        super().__init__(filters_in, filters_out)
        self.filters_out = filters_out
        self.input_size = input_size
        self.pooling = pooling
        
        if pooling:
            self.pool = nn.MaxPool2d(2)
            self.recurrent_weights = nn.Parameter(torch.Tensor(filters_out, input_size//2, input_size//2))
            self.recurrent_activation = nn.Tanh()
        else:
            self.pool = lambda x: x
            self.recurrent_weights = nn.Parameter(torch.Tensor(filters_out, input_size, input_size))
            self.recurrent_activation = nn.Tanh()

    def forward(self, xinp):
        if self.pooling:
            h = torch.randn(xinp.size()[1], self.filters_out, self.input_size//2, self.input_size//2).cuda()
            xout = Variable(torch.zeros(xinp.size()[0], xinp.size()[1], self.filters_out, self.input_size//2, self.input_size//2)).cuda()
        else:
            h = torch.randn(xinp.size()[1], self.filters_out, self.input_size, self.input_size).cuda()
            xout = Variable(torch.zeros(xinp.size()[0], xinp.size()[1], self.filters_out, self.input_size, self.input_size)).cuda()
            
        for i in range(xinp.size()[0]):
            xs = self.pool(super().forward(xinp[i]))
            xs = self.recurrent_activation(self.recurrent_weights * h + xs)
            xout[i] = xs
            
        return xs, xout

class UNetUpBlock(UNetBlock):
    def __init__(self, filters_in, filters_out):
        super().__init__(filters_in, filters_out)
        self.upconv = nn.Conv2d(filters_in, filters_in // 2, (3, 3), padding=1)
        self.upnorm = nn.BatchNorm2d(filters_in // 2)

    def forward(self, x, cross_x):
        x = F.upsample(x, size=cross_x.size()[-2:], mode='bilinear')
        x = self.upnorm(self.activation(self.upconv(x)))
        x = torch.cat((x, cross_x), 1)
        return super().forward(x)

class UNet(nn.Module):
    def __init__(self, input_size, layers, init_filters):
        super().__init__()
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.input_size = input_size
        self.init_filters = init_filters

        
        self.data_norm = nn.BatchNorm2d(4)
        self.init_layer = nn.Conv2d(4, init_filters, (7, 7), padding=(3,3))
        self.activation = nn.ReLU()
        self.init_norm = nn.BatchNorm2d(init_filters)
        self.dropout = nn.Dropout(DROPOUT)
        
        self.recurrent_weights = nn.Parameter(torch.Tensor(init_filters, input_size, input_size))
        self.recurrent_activation = nn.Tanh()
        
        filter_size = init_filters
        for i in range(layers - 1):
            self.down_layers.append(
                UNetDownBlock(input_size, filter_size, filter_size * 2)
            )
            filter_size *= 2
            input_size = input_size // 2
        self.down_layers.append(UNetDownBlock(input_size, filter_size, filter_size * 2, pooling=False))
        for i in range(layers):
            self.up_layers.append(
                UNetUpBlock(filter_size * 2, filter_size)
            )
            filter_size //= 2

    def forward(self, xinp):
        h = torch.randn(xinp.size()[1], self.init_filters, self.input_size, self.input_size).cuda()
        xout = Variable(torch.zeros(xinp.size()[0], xinp.size()[1], self.init_filters, self.input_size, self.input_size)).cuda()
        
        for i in range(xinp.size()[0]):
            xs = self.data_norm(xinp[i])
            xs = self.init_norm(self.activation(self.init_layer(xs)))
            xs = self.recurrent_activation(self.recurrent_weights * h + xs)
            xout[i] = xs
            
        saved_x = [xs]
        for layer in self.down_layers:
            saved_x.append(xs)
            xs, xout = layer(xout)
            xs = self.dropout(xs)
        is_first = True
        for layer, saved_x in zip(self.up_layers, reversed(saved_x)):
            if not is_first:
                is_first = False
                xs = self.dropout(xs)
            xs = layer(xs, saved_x)
        return xs

class UNetClassify(UNet):
    def __init__(self, *args, **kwargs):
        init_val = kwargs.pop('init_val', 0.5)
        super().__init__(*args, **kwargs)
        self.output_layer = nn.Conv2d(self.init_filters, 1, (3, 3), padding=1)

        for name, param in self.named_parameters():
            typ = name.split('.')[-1]
            if typ == 'bias':
                if 'output_layer' in name:
                    # Init so that the average will end up being init_val
                    param.data.fill_(-math.log((1-init_val)/init_val))
                else:
                    param.data.zero_()

    def forward(self, x):
        x = super().forward(x)
        # Note that we don't perform the sigmoid here.
        return self.output_layer(x)

# From: https://github.com/pytorch/pytorch/issues/1249
def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()

def get_loss(loss):
    if loss[0] == 'dice':
        print('dice')
        return dice_loss
    elif loss[0] == 'focal':
        print('focal')
        return w(FocalLoss(loss[1]))
    else:
        print('bce')
        return w(nn.BCEWithLogitsLoss())

USE_CUDA = torch.cuda.is_available()

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v


def fit(epochs, verbose=False, layers=4, lr=0.001, init_filters=32, loss='dice', init_val=0.5):
    net = w(UNetClassify(input_size=input_size, layers=layers, init_filters=init_filters, init_val=init_val))
    criterion = get_loss(loss)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    full_load = full_buildings_loader(data_dir)
    train_dataset = Preloader(data_dir , train_csv, input_size, full_load)
    train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    test_dataset = Preloader(data_dir , test_csv, input_size, full_load)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    best_iou = -1.0
    best_net_dict = None
    best_epoch = -1
    best_loss = 1000.0

    for epoch in tqdm(range(epochs)):
        net.train()
        train_losses = []
        for batch, labels in train:
            batch = batch.permute(2,0,1,3,4)
            batch = w(autograd.Variable(batch))
            labels = w(autograd.Variable(labels))

            optimizer.zero_grad()
            output = net(batch)
            loss = criterion(output, labels.view(-1,1,input_size,input_size).float())
            loss.backward()
            train_losses.append(loss.item())

            optimizer.step()
        print('train loss', np.mean(train_losses))

        net.eval()
        losses = []
        iou = []
        to_show = random.randint(0, len(test) - 1)
        for batch, labels_true in test:
            labels = w(autograd.Variable(labels_true))
            batch = batch.permute(2,0,1,3,4)
            batch = w(autograd.Variable(batch))
            output = net(batch)
            loss = criterion(output, labels.view(-1,1,input_size,input_size).float())
            losses += [loss.item()] * batch.size()[0]
            result = (F.sigmoid(output).data.cpu().numpy() > 0.5).astype(np.uint8)
            for label, res in zip(labels_true, result):
                label = label.cpu().numpy()[:, :]
#                 plt.imshow(label, cmap='tab20c')
#                 plt.show()
#                 plt.imshow(find_clusters(res), cmap='tab20c')
#                 plt.show()
                iou.append(evaluate_combined(label, res[0]))

        cur_iou = np.mean(iou)
        if cur_iou > best_iou or (cur_iou == best_iou and np.mean(losses) < best_loss):
            best_iou = cur_iou
            best_epoch = epoch
            import copy
            best_net_dict = copy.deepcopy(net.state_dict())
            best_loss = np.mean(losses)
        print(np.mean(losses), np.mean(iou), best_loss, best_iou)
    return best_iou, best_loss, best_epoch, best_net_dict



batch_size = 10
input_size = 64
data_dir = '../datasets/buildings_change/'
weights_dir = '../weights/buildings_change/'
train_csv = '../datasets/buildings_change/train.csv'
test_csv = '../datasets/buildings_change/test.csv'

best_iou, best_loss, best_epoch, best_net_dict = fit(1)
torch.save(best_net_dict, weights_dir + 'unet3d_4band_5dates_dice.pt')