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

class LSTMReconClassNet(nn.Module):

    def __init__(self, hidden_dim, layer1_dim):
        super(LSTMBinClassNet, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(13, hidden_dim, batch_first=True) 
        self.linear1 = nn.Linear(hidden_dim, layer1_dim)
        self.linear2 = nn.Linear(layer1_dim, 13)
        self.linear3 = nn.Linear(layer)
    def forward(self, seq):
        lstm_out = self.lstm(seq)[0]
        outputs = self.linear1(lstm_out)
        
        outputs_mean = Variable(torch.zeros(outputs.size()[0], 2)).cuda()
        for i in range(outputs.size()[0]):
            outputs_mean[i] = outputs[i].mean(dim=0)
            
        return outputs_mean
    

#print (lstm_ae_net)

cities = os.listdir('../datasets/onera/train_labels/')
for city in cities:
    
    lstm_bin_class_net = LSTMBinClassNet(32)
    
    data = np.load('../datasets/onera/hist_matched_npys/' + city + '.npy')
    labels = cv2.imread('../datasets/onera/train_labels/' + city + '/cm/new_' + city + '.png', 0)
    labels[labels == 255] = 1
    
    data = data.astype(np.float16)
    
    h_w = [data.shape[2], data.shape[3]]
    data = np.transpose(data, (2, 3, 0, 1))
    data = data.reshape(-1, data.shape[2], data.shape[3])
    
    labels = labels.reshape(-1)
    
    print (data.shape, labels.shape)
    
    train_ids = range(data.shape[0])
    
#     restack = []
    
#     for i in range(0, data.shape[1], data.shape[1]//10):
#         restack.append(data[:,i,:])
        
#     data = np.stack(restack, axis=1)
    
    lstm_bin_class_net.cuda()
#     lstm_bin_class_net.zero_grad()

#     #loss_function = nn.L1Loss()#loss_fu 
#     loss_function = nn.CrossEntropyLoss()
#     lr = 0.0001
#     optimizer = optim.SGD(lstm_bin_class_net.parameters(), lr=lr)

#     for epoch in range(10):
#         train_ids = np.random.permutation(train_ids)
#         epoch_loss = []
#         for i in range(0, len(train_ids), 256):
#             batch_ids = train_ids[i:i+256]
#             inp = data[batch_ids]
#             lbl = labels[batch_ids]
             
#             inp = Variable(torch.from_numpy(inp).type(torch.FloatTensor).cuda())
#             lbl = Variable(torch.from_numpy(lbl).type(torch.LongTensor).cuda())
            
#             lstm_bin_class_net.zero_grad()
#             out = lstm_bin_class_net(inp)
#             loss = loss_function(out, lbl)
#             loss.backward()
#             optimizer.step()

#             #print (loss.data[0])
#             epoch_loss.append(loss.data[0])

#         print (city[:-4] + " epoch " + str(epoch) +  ": " + str(np.mean(epoch_loss)))

#     torch.save(lstm_bin_class_net, '../weights/lstm_32bin_class_hm_' + city[:-3] + 'pt')


    lstm_bin_class_net = torch.load('../weights/lstm_32bin_class_hm_' + city[:-3] + 'pt')

    output = []
    tot_out = []

    for i in range(0, data.shape[0], 50000):
        batch = data[i:i+50000, :, :]
        batch_t = Variable(torch.from_numpy(batch).type(torch.FloatTensor).cuda())
        out = lstm_bin_class_net(batch_t)
        _, preds = torch.max(out.data, 1)
        output += list(preds.data.cpu().numpy())
    
    output = np.asarray(output)
    print (output.shape)
    output = output.reshape(h_w[0], h_w[1])
    print (output.shape)
    
    cv2.imwrite('../datasets/onera/results/lstm_32bin_class_hm_' + city + '.png', output * 255)