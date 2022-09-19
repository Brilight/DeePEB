import math
import time
import os

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from copy import deepcopy

torch.manual_seed(42)
np.random.seed(42)

from tqdm import trange, tqdm
from utils.Save_and_load import save, load


def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    

def train(netG, netD, D_size, optimizerG, optimizerD, schedulerG, schedulerD, opt, dataset_train, device_train, 
          dataset_test, res_trans, modelname, save_check=15):

    criterionGAN = nn.MSELoss()
    criterionL1 = nn.L1Loss()
    lambda_L1 = 10
    real_label, fake_label = 1, 0

    label_d = torch.FloatTensor(size=D_size).requires_grad_(True).to(device_train)
    lr_log, train_log, test_log = np.array([]), np.array([]).reshape(2,-1), np.array([]).reshape(3,-1)
    
    for epoch in trange(opt.epochs):        
        for itercount in range(opt.batch_size):
            setidx = np.random.randint(len(dataset_train))
            Acd, Ihb = next(dataset_train[setidx])
            Acd, Ihb = torch.Tensor(Acd).to(device_train), torch.Tensor(Ihb).to(device_train)
            fake_I = netG(Acd)  # G(A)
            
            # update D
            set_requires_grad(netD, True)  # enable backprop for D
            optimizerD.zero_grad()     # set D's gradients to zero
            
            real_AI = torch.cat((Acd, Ihb), 1)  
            pred_real = netD(real_AI)
            label_d.data.fill_(fake_label)
            loss_D_real = criterionGAN(pred_real, label_d)
            
            fake_AI = torch.cat((Acd, fake_I), 1)  
            pred_fake = netD(fake_AI.detach())
            label_d.data.fill_(real_label)
            loss_D_fake = criterionGAN(pred_fake, label_d)
            loss_D = (loss_D_fake + loss_D_real) * 0.5 /opt.batch_size # combine loss
            loss_D.backward()
        
            # update G
            set_requires_grad(netD, False)  # D requires no gradients when optimizing G
            optimizerG.zero_grad()        # set G's gradients to zero
            pred_fake = netD(fake_AI)
            label_d.data.fill_(real_label)
            loss_G_GAN = criterionGAN(pred_fake, label_d)
            loss_G_L1 = criterionL1(fake_I, Ihb) * lambda_L1
            # combine loss and calculate gradients
            loss_G = (loss_G_GAN + loss_G_L1) /opt.batch_size
            loss_G.backward()
            train_log = np.column_stack((train_log, np.array([[loss_G.item()], [loss_D.item()]])))
        
        lr_log = np.append(lr_log, optimizerG.state_dict()['param_groups'][0]['lr'])
        
        optimizerD.step()
        optimizerG.step()
        if schedulerG is not None:
            schedulerG.step()
        if schedulerD is not None:
            schedulerD.step()
        torch.cuda.empty_cache()
        
        if epoch%opt.print_freq == 0:
            test_loss_max, test_loss_mean, test_loss_nrmse = 0, 0, 0
            len_test = opt.len_test
            for masklabel in range(len_test):
                ini, s_label = next(dataset_test[np.random.randint(len(dataset_test))])
                with torch.no_grad():
                    ini = torch.Tensor(ini).to(device_train)
                    s_pred = netG(ini).detach().cpu()
                
                test_loss_max += ((s_pred-s_label)**2).max()/len_test                    
                s_pred, s_label = res_trans(s_pred), res_trans(s_label)
                test_loss_mean += (((s_pred-s_label)**2).mean())**0.5/len_test
                test_loss_nrmse += ((((s_pred-s_label)**2).mean()/(s_label**2).mean())**0.5)*100/len_test
            
            test_log = np.column_stack((test_log, np.array([[test_loss_max],
                                                            [test_loss_mean],
                                                            [test_loss_nrmse]])))
            print("Epoch {} | LR: {:.2e} | Train Loss: {:.2e}(G) {:.2e}(D) | Test Loss: {:.2e}, {:.2e}, {:.2f}%".format(
                epoch, lr_log[-1], train_log[0,-1], train_log[1,-1], test_loss_max, test_loss_mean, test_loss_nrmse))
        
            if test_loss_nrmse < save_check:
                save_check = test_loss_nrmse*0.95
                info = "\n!Model saved at Epoch {}, with Train Loss {:.2e}(G) | ".format(epoch, train_log[0,-1])+\
                    "Test Error(Ihb): {:.2e}(MSE), {:.2f}%(NRMSE)\n".format(test_loss_mean, test_loss_nrmse)
                print(info)
                save(netG, opt.modelpath, modelname[0], info+"\nGenerator\n")
                info = "\n!Model saved at Epoch {}, with Train Loss {:.2e}(D) | ".format(epoch, train_log[1,-1])
                print(info)
                save(netD, opt.modelpath, modelname[1], info+"\nDiscriminator\n")
        
    return train_log, test_log