# -*-coding:utf-8-*-
# -*-coding:euc-kr-*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.nn.functional import normalize
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset
import argparse

def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        print("Reading %d x %d flo file" % (h, w))

        #data2d = np.fromfile(f, np.float32, count=2 * w * h)
        data2d=np.fromfile(f,np.float32)
        # reshape data into 3D array (columns, rows, channels)
        #data2d = np.resize(data2d, (h[0], w[0], 2))
        print (data2d.shape)
        #print(np.max(data2d), np.min(data2d))

    f.close()
    return data2d

class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(28, 112, 2, stride=2),
            #nn.BatchNorm2d(112),
            nn.ReLU(True),
            nn.Conv2d(112, 256, 2, stride=2),
            #nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 2, stride=2),
            #nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 2, stride=2),
            #nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.AvgPool2d(7)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 7, stride=3),
            #nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 112, 5, stride=4),
            #nn.BatchNorm2d(113),
            nn.ReLU(True),
            nn.ConvTranspose2d(112, 28, 2, stride=4, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return x

def search_walk(dirName):
    file_list = []
    for dpath, dname, filename in os.walk(dirName):
        #print(f'{dpath}, {dname}, {filename}')
        #if dpath[-3:] == 'flo':
        filename.sort()
        for i in filename:
            file_list.append(os.path.join(dpath, i))

    file_list.sort()
    return file_list

def stack_np(list, idx):
    read1 = read_flow(list[idx])
    read2 = read_flow(list[idx + 1])
    read3 = read_flow(list[idx + 2])
    read4 = read_flow(list[idx + 3])
    read5 = read_flow(list[idx + 4])
    read6 = read_flow(list[idx + 5])
    read7 = read_flow(list[idx + 6])
    read8 = read_flow(list[idx + 7])
    read9 = read_flow(list[idx + 8])
    read10 = read_flow(list[idx + 9])
    read11 = read_flow(list[idx + 10])
    read12 = read_flow(list[idx + 11])
    read13 = read_flow(list[idx + 12])
    read14 = read_flow(list[idx + 13])

    np_st = np.stack(
        [read1, read2, read3, read4, read5, read6, read7, read8, read9, read10, read11, read12, read13, read14
         ], axis=0)
    np_view_st = np_st.reshape(-1, 112, 112)
    print(f'np_view_st:{np_view_st.shape}')

    return np_view_st

def read_np(list, flag):
    label = []
    r_stack = []
    
    print(f'len of list:{len(list)}')
    # file means idx
    for file in range(0, len(list)-13, 1):
        try:
            r_stack.append(stack_np(list, file))
            print('{}\n {}/{} : stacked for train'.format(list[file], file, len(list)))
        except Exception as e:
            print(str(e))

    flow_np = np.asarray(r_stack)
    
    print('flow_np.shape')
    print(flow_np.shape)

    for i in range(0, len(flow_np)):
        if flag == 1:
            label.append(1.0)
        else:
            label.append(0.0)

    label = np.asarray(label, dtype=np.float32)
    label = np.resize(label, (len(flow_np), 1))
    print('label.shape')
    print(label.shape)
    return flow_np, label

path = 'flo'
# path = '/usr/mnt3/Anomaly_Test_highdimension_fpn/{}_FPN'.format(args.dpath)
# l4 = search(path, 16)
l4 = search_walk(path)
rl4, rl4_label = read_np(l4, 1)
print('rl4')
print(rl4.shape)
# anomaly - x is datas y is labels
# anomaly flog : 1
trn_ax_torch = torch.from_numpy(rl4).type(torch.FloatTensor)
print('trn_ax_torch')
print(trn_ax_torch.shape)
print('trn_ax_torch+view')
trn_ay_torch = torch.from_numpy(rl4_label)
print(trn_ay_torch.shape)
print('anomaly trn torch')
trn = TensorDataset(trn_ax_torch, trn_ay_torch)
trn_data = torch.utils.data.DataLoader(trn, batch_size=50, shuffle=False, num_workers=0)
print('finished preprocessing')
ae = AutoEncoder()
print(ae)
# iterations = batch * epocs
Epocs = 100
# weights combine
loss_fn = nn.L1Loss()
#loss_fn = nn.MSELoss()

# optimizer = MultipleOptimizer(torch.optim.Adagrad(ae.parameters(), lr=0.005),
#                              torch.optim.Adam(ae.parameters(), lr=0.0001))
losses_a = []
optimizer = torch.optim.Adagrad(ae.parameters(), lr=0.005)
for epoch in range(Epocs):
    for data in trn_data:
        data, _ = data
        data = torch.autograd.Variable(data)
        pred = ae(data)
        optimizer.zero_grad()
        loss_a = loss_fn(pred, data)
        loss_a.backward()
        optimizer.step()

        if epoch % 10 == 0:
            torch.save(ae.state_dict(), 'output/flow-ae-feature-{}-{}.pth'.format('210428', epoch))

    print('epoch [{}/{}], loss : {:.6f}'.format(epoch + 1, Epocs, loss_a.data))

    losses_a.append(loss_a.data)
torch.save(ae.state_dict(), 'output/latest-flow-ae-feature-{}-{}.pth'.format('210428', epoch))

