#!/usr/bin/env python
# coding: utf-8

import math
import time
import csv
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="4"
os.environ['CUDA_LAUNCH_BLOCKING']="1"

import numpy as np
import torch 
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
np.random.seed(3407)

import itertools
from tqdm import trange, tqdm
import importlib
import matplotlib.pyplot as plt

from configs import opt, res_trans, RDevelop
from Datas.data_load import dataset_append
from Model.DeePEB_v1 import SpectralConv3d, DeePEB
from utils.Save_and_load import save, load
from Train.train_v1 import train
from Test.Evaluate import evaluate


def Model_def(mode_xy, mode_z, channels, layers, hf_channels, ckptpath=None):
    model = DeePEB([device0, device1], mode_z, mode_xy, mode_xy, channels, layers, hf_channels)
    model.initialize()
    if ckptpath is not None:
        ckpt = torch.load(ckptpath)
        model.load_state_dict(ckpt)
    print(model.parameters)
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated()/1024/1024/1024, torch.cuda.memory_reserved()/1024/1024/1024)
    return model


mode_xy, mode_z = 50, 40
device0 = device1 = torch.device('cuda:0')
channels = 25
layers = 1
hf_channels = 10
batch_size = 30
ckptpath = opt.modelpath+"DeePEB_v1.pth"
model = Model_def(mode_xy, mode_z, channels, layers, hf_channels, ckptpath)

dataset_train = dataset_append(opt, opt.dataidx_train, res_trans)
dataset_test = dataset_append(opt, opt.dataidx_test, res_trans) 


def opt_strategy(model):
    LR = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,)# weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)
    return optimizer, scheduler


TotLoss = lambda pred,label: ((pred-label)**2).max()
optimizer, scheduler = opt_strategy(model)
train_log, test_log, lr_log = train(model, optimizer, scheduler, opt, dataset_train, device_train,
                                           dataset_test, res_trans, ckpt_name, TotLoss=TotLoss)
if not os.path.exists(opt.load_model_path):
    save(model, opt.modelpath, ckpt_name)

Runtime, RMSE, NMSE, _, CD_err_x, CD_err_y, CD_RMSE = evaluate(opt, model, device0, res_trans, RDevelop)
print("Mean runtime:", np.mean(Runtime))
print("RMSE(Ihb, rate):", np.mean(RMSE,axis=-1))
print("NRMSE(Ihb, rate):", np.mean(NMSE,axis=-1))
print("CD Error(x, y):", np.mean(CD_RMSE,axis=-1))

Err_x, Err_y = np.array([]), np.array([])
for tmp in range(len(CD_err_x)):
    Err_x = np.append(Err_x, np.array(CD_err_x[tmp]).flatten())
    Err_y = np.append(Err_y, np.array(CD_err_y[tmp]).flatten())

np.savetxt(opt.respath+"CDErr-x_deepeb.csv", Err_x.flatten(), delimiter=' ')
np.savetxt(opt.respath+"CDErr-y_deepeb.csv", Err_y.flatten(), delimiter=' ')

