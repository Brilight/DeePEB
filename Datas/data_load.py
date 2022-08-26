#!/usr/bin/env python
# coding: utf-8
import time
import csv
import os

import numpy as np
import torch 
import torch.utils.data as data
from tqdm import trange


class DataGenerator(data.Dataset):
    def __init__(self, ini, label, batch_size = 1):
        self.ini = ini
        self.label = label
        self.batch_size = batch_size
        self.pick = np.arange(self.ini.shape[0])
        
    def __getitem__(self, index):
        """
        Return an item from the dataset one by one randomly
        Note: will not keep the sequence of the original datas
        """
        if self.pick.shape[0] == 0:
            self.pick = np.arange(self.ini.shape[0])
        np.random.shuffle(self.pick)
        idx = self.pick[:self.batch_size]
        self.pick = np.delete(self.pick, np.arange(self.batch_size))
        ini = self.ini[idx,...]
        label = self.label[idx,...]
        return ini, label

    
def list_all_files(rootdir): #列出文件夹下所有的目录与文件
    _files = []
    list_file = os.listdir(rootdir)
    for i in range(0,len(list_file)):
        path = os.path.join(rootdir,list_file[i])        
        if os.path.isdir(path): # 判断路径是否是一个文件目录或者文件;
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
             _files.append(path)
    return _files


def csv_to_np(filename, verbose=True):
    start = time.time()
    if verbose:
        print(filename, end=' ')
    if os.path.isfile(filename+'.npy'):
        concen = np.load(filename+'.npy')
    else:
        concen=[]
        with open(filename+".csv",'rt') as f:
            cr = csv.reader(f)
            for i,row in enumerate(cr):
                if i>5 and len(row)!=0:
                    concen.append(list(map(float,row[0].split("\t"))))
        concen = np.array(concen) #z,y，x=80,1000,1000 by default
        np.save(filename+'.npy', concen)
        print(".csv converted to .npy", end=';')
    if verbose:
        print("Read in {:.2f} s".format(time.time()-start))
    return concen


def dataset_generate(datapath, Masks, Resmax, res_trans, shuffle=False):
    if shuffle:
        np.random.shuffle(Masks) 
    Acds = Ihbs = None
    for mask in range(len(Masks)):
        Acd_ini = csv_to_np(datapath+str(Masks[mask]).zfill(4)+"Acid_ini").reshape((1,1,*Resmax))
        Ihb_final = csv_to_np(datapath+str(Masks[mask]).zfill(4)+"Inhibitor").reshape((1,1,*Resmax))
        Acds = Acd_ini if Acds is None else np.vstack((Acds, Acd_ini))
        Ihbs = res_trans(Ihb_final,True) if Ihbs is None else np.vstack((Ihbs, res_trans(Ihb_final,True)))
        print('--> Mask {} Read Complete, with dataset size: {}, {} masks left ...\n'.format(
            Masks[mask], Acds.shape, len(Masks)-mask-1))
    DataGenerator_train = DataGenerator(Acds, Ihbs)
    dataset_train = iter(DataGenerator_train)
    return dataset_train


def dataset_append(opt, Data_list, res_trans, data_per_set=10):
    """
    Return a list contains setnums elements (Dataset class)
    Each element contains datas of at most data_per_set 
    distributions and providing a random mask upon called
    """
    dataset_list = []
    setnums = len(Data_list)//data_per_set
    for setval in trange(setnums):
        Masks = Data_list[:data_per_set]
        dataset_tmp = dataset_generate(opt.datapath, Masks, opt.Resmax, res_trans)
        dataset_list.append(dataset_tmp)
        Data_list = np.delete(Data_list, np.arange(data_per_set))
    if len(Data_list)!=0:
        dataset_tmp = dataset_generate(opt.datapath, Data_list, opt.Resmax, res_trans)
        dataset_list.append(dataset_tmp)
    return dataset_list