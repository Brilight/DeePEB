#coding:utf8
import warnings
import os

import numpy as np
import torch

def res_trans(res, inv = False, trans = True):
    '''
    To (not) transform the value of inhibitor, 
    detailed reasons for the transformation are recommended to refer the paper
    trans: whether to trans
    inv: forward transform or inverse transform
    '''
    k_c = 0.9 #reaction coeff. in PEB simulation
    if not trans:
        return res
    if inv:
        return -np.log(-np.log(res)/k_c)
    if type(res) is torch.Tensor:
        return torch.exp(-k_c*torch.exp(-res))
    return np.exp(-k_c*np.exp(-res))


def RDevelop(Ihb):
    '''
    To transform the inhibitor distribution into that of development rate
    '''
    n, Mth = 30, 0.5
    Rmax, Rmin = 40, 0.0003
    a = (n+1)/(n-1)*(1-Mth)**n
    return Rmin+Rmax*(((a+1)*np.power(1-Ihb,n))/(a+np.power(1-Ihb,n)))


class DefaultConfig(object):
    env = 'default' # visdom env
    Path_list = ["../dataset_train_v1/", "./Ckpts/", "./Resist/", "./Results/", "./Figs/"]
    datapath, modelpath, resistpath, respath, figpath = Path_list
    for path in Path_list:
        if not os.path.exists(path):
            os.mkdir(path)
    debug_file = '/tmp/debug.log' # if os.path.exists(debug_file): enter ipdb
    load_model_path = None

    Resmax, CD_size, Full_size = [80,1000,1000], [60,1500,1500], [80,2000,2000]
    dataidx_train = np.arange(80)
    dataidx_test = np.arange(20)+80
    #Lists containing setnums items (Dataset class), each class containing datas of set_size masks
    #Each Dataset will provide a random mask upon called
    
    batch_size = 30 # batch size
    use_gpu = True # use GPU or not
    print_freq = 20 # print info every N batch
    len_test = 3
    TotLoss = None
    epochs = 500
    
    t_dev, Heights, scale = 60, np.arange(10)*8, 20
    dataidx_eval = np.arange(20)+80
    CD_seeds = 49

def parse(self,kwargs):
    '''
    Update config with kwargs
    '''
    for k,v in kwargs.iteritems():
        if not hasattr(self,k):
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self,k,v)
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k,getattr(self,k))


def help():
    '''
    Usage： python file.py help
    '''
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)
    
    
DefaultConfig.parse = parse
opt = DefaultConfig()