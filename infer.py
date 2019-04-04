# from comet_ml import Experiment as CometExperiment
# import sys, glob, cv2, random, math, argparse
# import numpy as np
# import pandas as pd
# from tqdm import trange
# from sklearn.metrics import precision_recall_fscore_support as prfs
#
# import torch
# import torch.utils.data
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.autograd as autograd
# from torch.autograd import Variable
# from torchvision import datasets, models, transforms
#
# sys.path.append('.')
# from utils.dataloaders import *
# from models.bidate_model import *
# from utils.metrics import *
# from utils.parser import get_parser_with_args
#
# import logging
#
# weight_file='../../../checkpoint_epoch_2.pt'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# model = BiDateLULCNet(13, 2, 5).to(device)
# model = nn.DataParallel(model, device_ids=['cpu'])
#
# # weights = torch.load(weight_file, map_location=lambda storage, loc: storage.to(device))
# weights = torch.load(weight_file, map_location='cpu')
# # print(weights.module)
# model.load_state_dict(weights.state_dict())
#
# model.eval()
#
# #
# # for batch_img1, batch_img2 in val_loader:
# #     batch_img1 = autograd.Variable(batch_img1).to(device)
# #     batch_img2 = autograd.Variable(batch_img2).to(device)
# #
# #     labels = autograd.Variable(labels).long().to(device)
# #     masks = autograd.Variable(masks).long().to(device)
# #
# #     cd_preds, lulc_preds = model(batch_img1, batch_img2)
# #
# #     cd_loss = criterion(cd_preds, labels)
# #     lulc_loss = criterion_lulc(lulc_preds, masks)
# #
# #     _, cd_preds = torch.max(cd_preds, 1)
# #     _, lulc_preds = torch.max(lulc_preds, 1)
# #
# #     if first_batch:
# #         log_images(comet, epoch, batch_img1, batch_img2, labels, masks, cd_preds, lulc_preds)
# #         first_batch=False
# #
# #     cd_corrects = 100 * (cd_preds.byte() == labels.squeeze().byte()).sum() / (labels.size()[0] * opt.patch_size * opt.patch_size)
# #     lulc_corrects = 100 * (lulc_preds.byte() == masks.squeeze().byte()).sum() / (masks.size()[0] * opt.patch_size * opt.patch_size)
# #     cd_val_report = prfs(labels.data.cpu().numpy().flatten(), cd_preds.data.cpu().numpy().flatten(), average='binary', pos_label=1)
# #     lulc_val_report = prfs(masks.data.cpu().numpy().flatten(), lulc_preds.data.cpu().numpy().flatten(), average='weighted')
#
#
# def image_loader(base, img_folder, h=10980, w=10980):
#     arr = np.zeros([13,h,w])
#     print('loading image '+img_folder+'...')
#     band_path = glob.glob(base + img_folder + '*')[0][:-7]
#     band = rasterio.open(band_path + band_ids[1] + '.tif').read()[0].astype(np.float32)
#     for i in range(len(band_ids)):
#         print('__band '+str(i+1))
#         band = rasterio.open(band_path + band_ids[i] + '.tif').read()[0].astype(np.float32)
#         # band = (band - band_means[band_ids[i]]) / band_stds[band_ids[i]]
#         band = cv2.resize(band, (h,w))
#         arr[i]=band
#     # return band_stacked
#     return torch.from_numpy(arr)
#
# img1 = image_loader('../../../chongqing', '/imgs_1/')
# # img2 = image_loader('../../../chongqing', '/img_2/').transpose(1,2,0)
#
#
# # batch1 = get_patches(img1)
# # batch2 = get_patches(img2)
# print(img1.shape)
#
# cv2.imwrite('test.jpg',torch.flip(img1[1:4,:,:],[0]).permute(1,2,0).numpy())
# # img1 = img1.unfold()
#
# # cd_preds, lulc_preds = model(batch1, batch2)
