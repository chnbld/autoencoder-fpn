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
from torch.utils.data import TensorDataset, Dataset, Subset,DataLoader
from os import listdir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        print('l28')
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        print('l33')
        print("Reading %d x %d flo file" % (h, w))

        #data2d = np.fromfile(f, np.float32, count=2 * w * h)
        data2d=np.fromfile(f,np.float32)
        # reshape data into 3D array (columns, rows, channels)
        #data2d = np.resize(data2d, (h[0], w[0], 2))
        #print (data2d.shape)
        #print(np.max(data2d), np.min(data2d))

    f.close()
    return data2d

class FrameFolderDataset(Dataset):
    def __init__(self, root_dir, clip_size=16, clip_stride=1):
        self.root_dir = root_dir
        self.clip_size = clip_size
        self.clip_stride = clip_stride
        self.data = self._generate_lists()

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        # return self.data[idx]
        return self._get_np_stack(idx)

    def _generate_lists(self):
        filenames = np.array(os.listdir(self.root_dir))
        filenames.sort()
        num_files = len(filenames)

        if num_files < self.clip_size:
            warnings.warn('Total number of images is insufficient')

        clip_start_ids = np.arange(0, num_files, self.clip_stride)
        clip_start_ids = clip_start_ids[clip_start_ids + self.clip_size <= num_files]
        clips_ids = np.array([np.arange(i, i + self.clip_size, dtype=np.int) for i in clip_start_ids])
        data = []


        for clip_id in clips_ids:
            img_paths = [os.path.join(self.root_dir, filename) for filename in filenames[clip_id]]
            data.append(img_paths)

        return data

    def _get_np_stack(self, idx):
        img_paths = self.data[idx]
        print("l82")
        print(img_paths)
        rl4, rl4_label = read_np(img_paths, 112, 1)
        # imgs = imgs[:, :, 8:120, 30:142]
        imgs = torch.tensor(rl4, dtype=torch.float32)
        labels = torch.tensor(rl4_label, dtype=torch.float32)

        return imgs,labels

#flag 1 - Anomaly
#flag 0 - Normal
def generate_train_test_list(input_path, output_path):
    train_list = []

    for folder in listdir(input_path):
        output_folder = os.path.join(output_path, folder)
        output_file = os.path.join(output_folder, folder + '.aenpy')
        print('l99')
        print('train {} {} {} {}'.format(folder, output_path, output_folder, output_file))
        train_list.append((os.path.join(input_path,folder), output_folder, output_file))

    train_list.sort()

    print('l105')
    print('\n\n{}\n\n'.format(len(train_list)))
    return train_list

class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(32, 112, 2, stride=2),
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
            nn.ConvTranspose2d(112, 32, 2, stride=4, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return x


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
    read15 = read_flow(list[idx + 14])
    read16 = read_flow(list[idx + 15])

    np_st = np.stack(
        [read1, read2, read3, read4, read5, read6, read7, read8, read9, read10, read11, read12, read13, read14, read15, read16], axis=0)

    np_view_st = np_st.reshape(-1, 112, 112)

    return np_view_st


def read_np(list, N, flag):
    label = []
    r_stack = []
    # file means idx
    try:
        r_stack.append(stack_np(list, 0))
    except Exception as e:
        print('l185')
        print(str(e))

    flow_np = np.asarray(r_stack)

    for i in range(0, len(flow_np)):
        if flag == 1:
            label.append(1.0)
        else:
            label.append(0.0)

    label = np.asarray(label, dtype=np.float32)
    label = np.resize(label, (len(flow_np), 1))
    print('198')
    print(label.shape)

    return flow_np, label


class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


def build_instances(features, output_file, out_shape, clip_stride, num_instances=20):
    instances_start_ids = np.round(np.linspace(0, len(features) - 1, num_instances + 1)).astype(np.int)

    segments_features = []
    for i in range(num_instances):
        start = instances_start_ids[i]
        end = instances_start_ids[i + 1]

        if start == end:
            instance_features = features[start, :]
        elif end < start:
            instance_features = features[start, :]
        else:
            instance_features = torch.mean(features[start:end, :], dim=0)

        instance_features = torch.nn.functional.normalize(instance_features, p=2, dim=0)
        segments_features.append(instance_features.numpy())

    segments_features = np.array(segments_features)
    print('l236')
    print(output_file)
    # instances, frames
    output_file = output_file + '_' + str(num_instances) + str(clip_stride) + out_shape
    print('l240')
    print(output_file)
    np.savetxt(output_file, segments_features, fmt='%.6f')

def train_dataset(input_path, output_file, clip_stride=16):
    images_data = FrameFolderDataset(input_path, clip_stride=clip_stride)
    dataloader = DataLoader(images_data, batch_size=16, shuffle=False, num_workers=0)

    items = []
    labels = []

    for sample in dataloader:
        item, label = sample
        items.append(item)
        labels.append(label)

    items = torch.cat(items, dim=0)
    labels = torch.cat(labels, dim=0)
    t = items.shape[2:]
    items = items.view(-1,int(t[0]),int(t[1]),int(t[2]))
    labels = labels.view(-1, 1)
    
    print('l262')
    print(items.shape)
    print(labels.shape)
    return items
    #build_instances(items, output_file, clip_stride)


input_path = '/autoencoder/normal/Test'
out_path = '/autoencoder/cabuse-ae-features/normal/Test'
trn_list = generate_train_test_list(input_path, out_path)
print(trn_list)
def get_active(name,activation):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook

def train(ae_datasets):
    print('l280')
    print(ae_datasets.shape)
    trn_data = torch.utils.data.DataLoader(ae_datasets, batch_size=50, shuffle=False, num_workers=0)
    # iterations = batch * epocs
    Epocs = 100
    # # weights combine
    loss_fn = nn.L1Loss()
    # loss_fn = nn.MSELoss()

    # # optimizer = MultipleOptimizer(torch.optim.Adagrad(ae.parameters(), lr=0.005),
    # #                              torch.optim.Adam(ae.parameters(), lr=0.0001))
    losses_a = []
    optimizer = torch.optim.Adagrad(ae.parameters(), lr=0.005)
    for epoch in range(Epocs):
        for data in trn_data:
            data = data
            data = torch.autograd.Variable(data).to(device)
            pred = ae(data)
            optimizer.zero_grad()
            loss_a = loss_fn(pred, data)
            loss_a.backward()
            optimizer.step()

            if epoch % 10 == 0:
                torch.save(ae.state_dict(), './tmp/conv_{}_2_Adagrad{}_ae.pth'.format('210429', epoch))
            print('l305')
            print('epoch [{}/{}], loss : {:.6f}'.format(epoch + 1, Epocs, loss_a.data))
            losses_a.append(loss_a.data)

    torch.save(ae.state_dict(), './tmp/conv_{}_2_Adagrad{}_ae.pth'.format('210429', epoch))

    # plt.plot(np.arange(len(losses_a)), losses_a)
    # plt.savefig('loss_{}_{}.png'.format('200518', args.dpath, dpi=300))
    # torch.save(ae.state_dict(), './tmp/conv_{}_{}_2_Adagrad{}_ae.pth'.format('200518',args.dpath, epoch))

def save_raw(features, output_file):
    features = features.numpy()
    np.savetxt(output_file, features)


@torch.no_grad()
def extract(input_path, output_file, activation, clip_stride=16):
    print('l322')
    print(input_path, output_file)

    images_data = FrameFolderDataset(input_path, clip_stride=clip_stride)
    dataloader = DataLoader(images_data, batch_size=50, shuffle=False, num_workers=0)

    outputs = []

    for data in dataloader:
        datas,_ = data
        t = datas.shape[2:]
        datas = datas.view(-1, int(t[0]), int(t[1]), int(t[2]))
        print('l334')
        print(datas.shape)
        output = ae(datas)
        b_out = activation['bottleneck']
        print(b_out.shape)
        view_fms = b_out.view(-1, 1024)
        print(view_fms.shape)
        outputs.append(view_fms.cpu())

    outputs = torch.cat(outputs, dim=0)
    print('l344')
    print(outputs.shape[0])
    save_raw(outputs, output_file)
    build_instances(outputs, output_file, str(outputs.shape[0]), clip_stride)


for row in trn_list:
    ae_datasets = []
    ae = AutoEncoder()
    ae = ae.to(device)
    source_path, output_folder, output_file = row
    print('l355')
    print(output_file)

    if not os.path.exists(output_folder):
        print(output_folder)
        os.makedirs(output_folder)

    if not os.path.exists(output_file):
        print(source_path)
        datas = train_dataset(source_path, output_file, clip_stride=1)
        ae_datasets.append(datas)

        ae_datasets = torch.cat(ae_datasets, dim=0)
        train(ae_datasets)
        modelpath = './tmp/conv_{}_2_Adagrad{}_ae.pth'.format('210429', 99)
        ae = AutoEncoder()
        ae.load_state_dict((torch.load(modelpath)))

        activation = {}
        ae.bottleneck.register_forward_hook(get_active('bottleneck',activation))
        extract(source_path, output_file, activation, 1)
            
