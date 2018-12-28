import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import cv2
import os


class RNNAENet(nn.Module):

    def __init__(self, hidden_dim):
        super(RNNAENet, self).__init__()

        self.rnn_encoder = nn.RNN(13, hidden_dim, batch_first=True)
        self.rnn_decoder = nn.RNN(hidden_dim, 13, batch_first=True)

    def forward(self, seq, h0, h1):
        encoder_out, he = self.rnn_encoder(seq, h0)
        decoder_out, hd = self.rnn_decoder(encoder_out, h1)
        return decoder_out, hd

model = RNNAENet(32)
input = torch.randn(10,14,13)
h0 = torch.randn(1, 10, 32)
h1 = torch.randn(1, 10, 13)
output, hd = model(input, h0, h1)
print (output.size(), hd.size())
