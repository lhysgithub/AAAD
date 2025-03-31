import torch
import os
import random
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def get_dataset_np(dataset):
    if "HADES" in dataset:
        name = "hades"
        train = np.load(f'./data/data_for_tsad/{name}_kpi_train.npy')
        test = np.load(f'./data/data_for_tsad/{name}_kpi_test.npy')
        test_label = np.load(f'./data/data_for_tsad/label_{name}_kpi_test.npy')
        
        return train, test, test_label
    elif "YZH" in dataset:
        name = "yzh"
        train = np.load(f'./data/data_for_tsad/{name}_kpi_train.npy')
        test = np.load(f'./data/data_for_tsad/{name}_kpi_test.npy')
        test_label = np.load(f'./data/data_for_tsad/label_{name}_kpi_test.npy')
        
        return train, test, test_label
    elif "PSM" in dataset:
        name = "PSM"
        train_df = pd.read_csv(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/train.csv', sep=",", header=None, skiprows=1,
                               dtype=np.float32).fillna(0)
        test_df = pd.read_csv(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/test.csv', sep=",", header=None, skiprows=1,
                              dtype=np.float32).fillna(0)
        test_label_df = pd.read_csv(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/test_label.csv', sep=",", header=None, skiprows=1,
                              dtype=np.float32).fillna(0)
        train_df.drop(0, axis=1, inplace=True)
        test_df.drop(0, axis=1, inplace=True)
        test_label_df.drop(0, axis=1, inplace=True)
        
        return train_df.values, test_df.values, test_label_df.values
    elif "SMAP_ALL" in dataset:
        name = "SMAP"
        train = np.load(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/{name}_train.npy')
        test = np.load(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/{name}_test.npy')
        test_label = np.load(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/{name}_test_label.npy')
        return train, test, test_label
    elif "MSL_ALL" in dataset:
        name = "MSL"
        train = np.load(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/{name}_train.npy')
        test = np.load(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/{name}_test.npy')
        test_label = np.load(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/{name}_test_label.npy')
        return train, test, test_label
    elif "ZX_ALL" in dataset:
        name = "ZX"
        subname = "zte"
        train = np.load(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/{subname}_kpi_train.npy')
        test = np.load(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/{subname}_kpi_test.npy')
        test_label = np.load(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/label_{subname}_kpi_test.npy')
        return train, test, test_label
    elif "SMD_ALL" in dataset:
        name = "SMD"
        train = np.load(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/{name}_train.npy')
        test = np.load(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/{name}_test.npy')
        test_label = np.load(f'/home/hongyi/workspace/Anomaly-Transformer-main/dataset/{name}/{name}_test_label.npy')
        return train, test, test_label
    elif "ZTE" in dataset:
        name = "zte"
        train = np.load(f'/home/hongyi/workspace/hades_merge/new_hades/common/data_for_tsad/{name}_kpi_train.npy')
        test = np.load(f'/home/hongyi/workspace/hades_merge/new_hades/common/data_for_tsad/{name}_kpi_test.npy')
        test_label = np.load(f'/home/hongyi/workspace/hades_merge/new_hades/common/data_for_tsad/label_{name}_kpi_test.npy')
        
        return train, test, test_label
    elif "SWAT" in dataset:
        dim = 50
        train_df = pd.read_csv(f'/home/hongyi/workspace/TSAD/data/SWAT/A1_A2/train.csv', sep=",", header=None, skiprows=1,dtype=np.float32).fillna(0)
        test_df = pd.read_csv(f'/home/hongyi/workspace/TSAD/data/SWAT/A1_A2/test.csv', sep=",", header=None, skiprows=1, dtype=np.float32).fillna(0)
        # train_df["y"] = np.zeros(train_df.shape[0], dtype=np.float32)

        # Get test anomaly labels
        test_label = test_df.iloc[:, dim]
        train_df.drop(dim, axis=1, inplace=True)
        test_df.drop(dim, axis=1, inplace=True)
        return train_df.to_numpy(), test_df.to_numpy(), test_label.to_numpy()
    
class SWaTSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv', header=1)
        data = data.values[:, 1:-1]

        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = pd.read_csv(data_path + '/test.csv')

        y = test_data['Normal/Attack'].to_numpy()
        labels = []
        for i in y:
            if i == 'Attack':
                labels.append(1)
            else:
                labels.append(0)
        labels = np.array(labels)


        test_data = test_data.values[:, 1:-1]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)
        self.train = data
        self.test_labels = labels.reshape(-1, 1)

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class PSMSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class MSLSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class SMAPSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

def mask_data(data,removed_dims):
    # data2 = data.copy()
    # for i in removed_dims: # mask
    #     data2[:,i] = np.zeros_like(data2[:,i])
    return data

def mask_data_old(data,removed_dims):
    data2 = data.copy()
    for i in removed_dims: # mask
        data2[:,i] = np.zeros_like(data2[:,i])
    return data2

class SegLoader(object):
    def __init__(self,dataset,removed_dims, random_mask_rate, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.selected_dims = []
        self.removed_dims = removed_dims
        
        data, test_data, test_labels = get_dataset_np(dataset)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)
        
        self.random_mask_rate = random_mask_rate
        _ ,self.c = data.shape
        if len(removed_dims) > 0:
            data = mask_data(data,self.removed_dims) # mask
            self.test = mask_data(self.test,self.removed_dims) # mask
        
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_labels

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        mask_rate = self.random_mask_rate
        mask_num = int(mask_rate*self.c)
        mask_tabel = np.random.choice(self.c,mask_num,replace=False)
        if self.mode == "train":
            out_data = np.float32(self.train[index:index + self.win_size])
            out_label = np.float32(self.test_labels[0:self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        elif (self.mode == 'val'):
            out_data = np.float32(self.val[index:index + self.win_size])
            out_label = np.float32(self.test_labels[0:self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        elif (self.mode == 'test'):
            out_data = np.float32(self.test[index:index + self.win_size])
            out_label = np.float32(self.test_labels[index:index + self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        else:
            out_data = np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            out_label = np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        return out_data, out_label
    
def get_loader_segment(data_path, batch_size, win_size, step, mode='train', dataset='KDD', val_ratio=0.2):
    '''
    model : 'train' or 'test'
    '''
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'SWaT'):
        dataset = SWaTSegLoader(data_path, win_size, step, mode)
    else:
        dataset = SegLoader(dataset,[],0, data_path, win_size, step, mode)
    
    shuffle = False
    if mode == 'train':
        shuffle = True

        dataset_len = int(len(dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))

        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        
        indices = torch.arange(dataset_len)
        
        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(dataset, val_sub_indices)
        
        train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        k_use_len = int(train_use_len*0.1)
        k_sub_indices = indices[:k_use_len]
        k_subset = Subset(dataset, k_sub_indices)
        k_loader = DataLoader(dataset=k_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        return train_loader, val_loader, k_loader

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    
    return data_loader, data_loader