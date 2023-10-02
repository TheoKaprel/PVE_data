import time

import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import torch
from torch.utils.data import Dataset,DataLoader

folder = "/export/home/tkaprelian/Desktop/PVE/datasets/validation_dataset"




class npyDataset(Dataset):
    def __init__(self, folder):
        self.list_npy = glob.glob(os.path.join(folder,'?????_noisy_PVE_PVfree_rec_fp.npy'))
        self.nprojs = 120
        self.input_eq_angles = 4
        self.with_adj_angles = True
        self.nsrc = len(self.list_npy)
        self.build_channels_id()


    def __len__(self):
        return self.nprojs*self.nsrc

    def __getitem__(self, item):
        src_i = item % self.nsrc
        proj_i = item // self.nsrc
        channels = self.get_channels_id_i(proj_i=proj_i)
        array = np.load(self.list_npy[src_i])
        return array[0,channels,:,:],array[2,channels,:,:]


    def build_channels_id(self):
        # rotating channels id
        step = int(self.nprojs / (self.input_eq_angles))
        self.channels_id = np.array([0])
        if self.with_adj_angles:
            adjacent_channels_id = np.array([(-1) % self.nprojs, (1) % self.nprojs])
            self.channels_id = np.concatenate((self.channels_id, adjacent_channels_id))

        equiditributed_channels_id = np.array([(k * step) % self.nprojs for k in range(1, self.input_eq_angles)])
        self.channels_id = np.concatenate((self.channels_id, equiditributed_channels_id)) if len(
            equiditributed_channels_id) > 0 else self.channels_id


    def get_channels_id_i(self, proj_i):
        return (self.channels_id+proj_i)%120


class hd5Dataset1(Dataset):
    def __init__(self,folder):
        self.list_h5 = glob.glob(os.path.join(folder,'?????_data.h5'))
        self.nprojs = 120
        self.nsrc = len(self.list_h5)

    def __len__(self):
        return self.nprojs*self.nsrc

    def __getitem__(self, item):
        src_i = item % self.nsrc
        proj_i = item // self.nsrc
        f = h5py.File(os.path.join(folder,self.list_h5[src_i]), 'r')
        data = f['data']
        return np.array(data[0,proj_i,:,:],dtype=np.float32),np.array(data[2,proj_i,:,:],dtype=np.float32)


class hd5Dataset2(Dataset):
    def __init__(self,folder):
        self.list_h5 = glob.glob(os.path.join(folder,'?????_data2.h5'))
        self.nprojs = 120
        self.nsrc = len(self.list_h5)

    def __len__(self):
        return self.nprojs*self.nsrc

    def __getitem__(self, item):
        src_i = item % self.nsrc
        proj_i = item // self.nsrc
        f = h5py.File(os.path.join(folder,self.list_h5[src_i]), 'r')
        data_PVE_noisy,data_PVfree = np.array(f['PVE_noisy'][proj_i,:,:],dtype=np.float32),np.array(f['PVfree'][proj_i,:,:],dtype=np.float32)
        return data_PVE_noisy,data_PVfree

class hd5Dataset3(Dataset):
    def __init__(self,datasetfn):
        self.nprojs = 120
        self.input_eq_angles = 4
        self.with_adj_angles = True
        self.datasetfn = datasetfn
        self.dataset = h5py.File(self.datasetfn,'r')
        self.nsrc = len(self.dataset.keys())
        self.keys = list(self.dataset.keys())
        self.build_channels_id()

    def __len__(self):
        return self.nprojs*self.nsrc

    def build_channels_id(self):
        # rotating channels id
        step = int(self.nprojs / (self.input_eq_angles))
        self.channels_id = np.array([0])
        if self.with_adj_angles:
            adjacent_channels_id = np.array([(-1) % self.nprojs, (1) % self.nprojs])
            self.channels_id = np.concatenate((self.channels_id, adjacent_channels_id))

        equiditributed_channels_id = np.array([(k * step) % self.nprojs for k in range(1, self.input_eq_angles)])
        self.channels_id = np.concatenate((self.channels_id, equiditributed_channels_id)) if len(
            equiditributed_channels_id) > 0 else self.channels_id


    def get_channels_id_i(self, proj_i):
        return (self.channels_id+proj_i)%120

    def __getitem__(self, item):
        src_i = item % self.nsrc
        proj_i = item // self.nsrc
        channels = self.get_channels_id_i(proj_i=proj_i)
        id = np.argsort(channels)
        invid = np.argsort(id)

        with h5py.File(self.datasetfn, 'r') as f:
            data = f[self.keys[src_i]]['data']
            data_PVE_noisy,data_PVfree = np.array(data[0,channels[id],:,:],dtype=np.float32)[invid],np.array(data[2,proj_i:proj_i+1,:,:],dtype=np.float32)
            return data_PVE_noisy,data_PVfree


# print('*'*20)
# print('READ npy')
# t0 = time.time()
# ds = npyDataset(folder=folder)
# dl = DataLoader(dataset=ds,batch_size=12,num_workers=12)
# S = 0
# for (i,batch) in enumerate(dl):
#     S+=torch.sum(batch[0]).item()
# t1 = time.time()
# print(t1-t0)
# print(len(ds))
# print(S)


# print('*'*20)
# print('READ h5 opt1')
# t0 = time.time()
# ds = hd5Dataset1(folder=folder)
# dl = DataLoader(dataset=ds,batch_size=12,num_workers=12)
# S=0
# for (i,batch) in enumerate(dl):
#     S+=torch.sum(batch[0]).item()
# t1 = time.time()
# print(t1-t0)
# print(len(ds))
# print(S)
#
# print('*'*20)
# print('READ h5 opt2')
# t0 = time.time()
# ds = hd5Dataset2(folder=folder)
# dl = DataLoader(dataset=ds,batch_size=12,num_workers=12)
# S = 0
# for (i,batch) in enumerate(dl):
#     S+=torch.sum(batch[0]).item()
# t1 = time.time()
# print(t1-t0)
# print(len(ds))
# print(S)

print('*'*20)
print('READ h5 opt3')
t0 = time.time()
ds = hd5Dataset3(datasetfn=os.path.join(folder, 'dataset.h5'))
dl = DataLoader(dataset=ds,batch_size=12,num_workers=12)
S = 0
for (i,batch) in enumerate(dl):
    S+=torch.sum(batch[0]).item()
    # if i%10==0:
    #     fig,ax = plt.subplots(1,7)
    #     for k in range(6):
    #         ax[k].imshow(batch[0][0,k,:,:].numpy())
    #     ax[6].imshow(batch[1][0,0,:,:].numpy())
    #     plt.show()

t1 = time.time()
print(t1-t0)
print(len(ds))
print(S)
