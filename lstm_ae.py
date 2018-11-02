import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn import preprocessing
import numpy as np

import cv2
import os 

class LSTMAENet(nn.Module):

    def __init__(self, hidden_dim, layer1_dim):
        super(LSTMAENet, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(13, hidden_dim, batch_first=True) 
        self.linear1 = nn.Linear(hidden_dim, layer1_dim)
        self.linear2 = nn.Linear(layer1_dim, 13)

    def forward(self, seq):
        lstm_out = self.lstm(seq)[0]
        tmp1 = self.linear1(lstm_out)
        _out = self.linear2(tmp1)
        base_out = _out
        return base_out
    

#print (lstm_ae_net)

cities = os.listdir('../datasets/onera/hist_matched_npys/')
cities = ['beihai.npy']
for city in cities:
    lstm_ae_net = LSTMAENet(32, 64)
    
    data = np.load('../datasets/onera/hist_matched_npys/' + city)
    data = data.astype(np.float16)
    
    h_w = [data.shape[2], data.shape[3]]
    data = np.transpose(data, (2, 3, 0, 1))
    data = data.reshape(-1, data.shape[2], data.shape[3])
    
    print (data.shape)
#     restack = []
    
#     for i in range(0, data.shape[1], data.shape[1]//10):
#         restack.append(data[:,i,:])
        
#     data = np.stack(restack, axis=1)
    restack = np.copy(data)
    print (data.shape)
    
    lstm_ae_net.cuda()
#     lstm_ae_net.zero_grad()

#     #loss_function = nn.L1Loss()#loss_fu 
#     loss_function = nn.MSELoss()
#     lr = 0.0001
#     optimizer = optim.SGD(lstm_ae_net.parameters(), lr=lr)

#     for epoch in range(10):
#         data = np.random.permutation(data)
#         epoch_loss = []
#         for i in range(0, data.shape[0], 256):
#             batch = data[i:i+256, :, :]
#             batch = Variable(torch.from_numpy(batch).type(torch.FloatTensor).cuda())
#             lstm_ae_net.zero_grad()
#             out = lstm_ae_net(batch)
#             loss = loss_function(out, batch)
#             loss.backward()
#             optimizer.step()

#             #print (loss.data[0])
#             epoch_loss.append(loss.data[0])

#         print (city[:-4] + " epoch " + str(epoch) +  ": " + str(np.mean(epoch_loss)))

#     torch.save(lstm_ae_net, '../weights/arch32x64_hm_' + city[:-3] + 'pt')


    lstm_ae_net = torch.load('../weights/arch32x64_hm_' + city[:-3] + 'pt')

    output = []
    tot_out = []

    for i in range(0, data.shape[0], 10000):
        batch = restack[i:i+10000, :, :]
        batch_t = Variable(torch.from_numpy(batch).type(torch.FloatTensor).cuda())
        out = lstm_ae_net(batch_t)
        out = out.data.cpu().numpy()
        tot_out += list(out)
        error = ((batch - out) ** 2).mean(axis=-1).mean(axis=-1)
        output += list(error)
    
    tot_out = np.asarray(tot_out)
    tot_out = tot_out.reshape(h_w[0],h_w[1],tot_out.shape[1],tot_out.shape[2])
    print (restack.shape, tot_out.shape)
    np.save('../datasets/onera/results/arch32x64_hm_' + city[:-3] + '.npy', tot_out)
    output = np.asarray(output)
    slope = (255 - 0 ) / (output.max() - output.min())
    output = 0 + slope * (output - output.min())
    output = output.reshape(h_w[0], h_w[1])
    cv2.imwrite('../datasets/onera/results/arch32x64_hm_' + city[:-3] + 'png', output)