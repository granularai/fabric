import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

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

def get_loss(loss, weight_factor, gpu_id):
    if loss == 'dice':
        print('dice')
        return dice_loss
    elif loss == 'focal':
        print('focal')
        return FocalLoss(weight_factor)
    elif loss == 'ce':
        print('weighted cross entropy')
        return nn.CrossEntropyLoss(torch.from_numpy(np.asarray([weight_factor, 1-weight_factor])).cuda(gpu_id).float())
    else:
        print('bce')
        return nn.BCEWithLogitsLoss()

def iou(mask1, mask2):
    return np.sum(mask1 & mask2) / np.sum(mask1 | mask2)


def get_iou(mask1, mask2):
    if np.sum(mask1 & mask2) == 0:
        return 0
    return np.sum(mask1 & mask2) / np.sum(mask1 | mask2)