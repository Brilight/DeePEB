import math
import time
import os

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

torch.manual_seed(42)
np.random.seed(42)

from tqdm import trange, tqdm
from utils.Save_and_load import save, load


def train(model, optimizer, scheduler, opt, dataset_train, device_train, 
          dataset_test, res_trans, RDevelop, modelname, save_check=0.3, 
          TotLoss = lambda pred,label: ((pred-label)**2).max()):

    np.set_printoptions(precision=5)
    TotLoss = opt.Totloss if opt.Totloss is not None else TotLoss
    lr_log, train_log, test_log = np.array([]), np.array([]), np.array([]).reshape(3,-1)
    
    for epoch in trange(opt.epochs):        
        for itercount in range(opt.batch_size):
            setidx = np.random.randint(len(dataset_train))
            ini, ihb = next(dataset_train[setidx])
            ini, s_label = torch.Tensor(ini).to(device_train), torch.Tensor(ihb).to(device_train)
            s_pred = model(ini)
            loss = (TotLoss(s_pred, s_label))/opt.batch_size
            loss.backward()
            train_log = np.append(train_log, loss.item())

        lr_log = np.append(lr_log, optimizer.state_dict()['param_groups'][0]['lr'])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        
        if epoch%opt.print_freq == 0:
            test_loss_max, test_loss_mean, test_loss_nrmse = 0, 0, 0
            len_test = opt.len_test
            for masklabel in range(len_test):
                ini, s_label = next(dataset_test[np.random.randint(len(dataset_test))])
                with torch.no_grad():
                    ini = torch.Tensor(ini).to(device_train)
                    s_pred = model(ini).detach().cpu()
                
                test_loss_max += ((s_pred-s_label)**2).max()/len_test                    
                s_pred, s_label = res_trans(s_pred), res_trans(s_label)
                test_loss_mean += (((s_pred-s_label)**2).mean())**0.5/len_test
                test_loss_nrmse += ((((s_pred-s_label)**2).mean()/(s_label**2).mean())**0.5)*100/len_test
            
            test_log = np.column_stack((test_log, np.array([[test_loss_max],
                                                            [test_loss_mean],
                                                            [test_loss_nrmse]])))
            print("Epoch {} | LR: {:.2e} | Train Loss: {:.2e} | Test Loss: {:.2e}, {:.2e}, {:.2f}%".format(
                epoch, lr_log[-1], train_log[-1], test_loss_max, test_loss_mean, test_loss_nrmse))
        
            if test_loss_nrmse < save_check:
                save_check = test_loss_nrmse*0.95
                info = "\n!Model saved at Epoch {}, with Train Loss {:.2e}(Total) | ".format(epoch, train_log[-1])+\
                    "Test Error(Ihb): {:.2e}(MSE), {:.2f}%(NRMSE)\n".format(test_loss_mean, test_loss_nrmse)
                print(info)
                save(model, opt.modelpath, modelname, info)
        
    return model, train_log, test_log, lr_log 