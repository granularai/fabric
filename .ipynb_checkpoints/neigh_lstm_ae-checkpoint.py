import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import cv2
import os 

class LSTMAENet(nn.Module):

    def __init__(self, hidden_dim, layer1_dim):
        super(LSTMAENet, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(117, hidden_dim, batch_first=True) 
        self.linear1 = nn.Linear(hidden_dim, layer1_dim)
        self.linear2 = nn.Linear(layer1_dim, 117)

    def forward(self, seq):
        lstm_out = self.lstm(seq)[0]
        tmp = self.linear1(lstm_out)
        tmp = self.linear2(tmp)
        base_out = tmp
        return base_out
    

#print (lstm_ae_net)

cities = os.listdir('../datasets/onera/hist_matched_npys/')

for city in cities:
    lstm_ae_net = LSTMAENet(128, 256)
    
    data = np.load('../datasets/onera/hist_matched_npys/' + city)
    data = data.astype(np.float16)
    h_w = [data.shape[2], data.shape[3]]
    data = np.transpose(data, (2, 3, 0, 1))
    neigh_data = []

    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            neigh = data[i:i+3, j:j+3, :, :]
            neigh = np.transpose(neigh, (2, 0, 1, 3))
            neigh = neigh.reshape(neigh.shape[0], -1)
            if neigh.shape[1] == 117:
                neigh_data.append(neigh)

    neigh_data = np.asarray(neigh_data)
    print (neigh_data.shape, data.shape)
    
#     restack = []
    
#     for i in range(0, data.shape[1], data.shape[1]//10):
#         restack.append(data[:,i,:])
        
#     data = np.stack(restack, axis=1)
    restack = np.copy(neigh_data)
    
    lstm_ae_net.cuda()
    lstm_ae_net.zero_grad()

    #loss_function = nn.L1Loss()#loss_fu 
    loss_function = nn.MSELoss()
    lr = 0.0001
    optimizer = optim.SGD(lstm_ae_net.parameters(), lr=lr)

    for epoch in range(5):
        data = np.random.permutation(neigh_data)
        epoch_loss = []
        for i in range(0, neigh_data.shape[0], 256):
            batch = neigh_data[i:i+256, :, :]
            batch = Variable(torch.from_numpy(batch).type(torch.FloatTensor).cuda())
            lstm_ae_net.zero_grad()
            out = lstm_ae_net(batch)
            loss = loss_function(out, batch)
            loss.backward()
            optimizer.step()

            #print (loss.data[0])
            epoch_loss.append(loss.data[0])

        print (city[:-4] + " epoch " + str(epoch) +  ": " + str(np.mean(epoch_loss)))

    torch.save(lstm_ae_net, '../weights/neigh3x3_hm_' + city[:-3] + 'pt')


    lstm_ae_net = torch.load('../weights/neigh3x3_hm_' + city[:-3] + 'pt')

    output = []

    for i in range(0, restack.shape[0], 10000):
        batch = restack[i:i+10000, :, :]
        batch_t = Variable(torch.from_numpy(batch).type(torch.FloatTensor).cuda())
        out = lstm_ae_net(batch_t)
        out = out.data.cpu().numpy()
        error = ((batch - out) ** 2).mean(axis=-1).mean(axis=-1)
        output += list(error)

    output = np.asarray(output)
    print (output.shape)
    slope = (255 - 0 ) / (output.max() - output.min())
    output = 0 + slope * (output - output.min())
    output = output.reshape(h_w[0]-2, h_w[1]-2)
    cv2.imwrite('../datasets/onera/results/neigh3x3_hm_' + city[:-3] + 'png', output)
    del output
    del data
    del restack