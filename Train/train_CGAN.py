import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

import math
import time
import os

from tqdm import trange, tqdm
from modules.Predict import test_eval
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
                    
                    
def train(netG, netD, D_size, device, optimizerG, optimizerD, schedulerG, schedulerD, res_trans, RDevelop, 
          epochs, batch_size, dataset_train, Acd_test, Ihb_test, Resmax, device_test, modelPATH, modelname, 
          check = 0.3, verbose=10):
    np.set_printoptions(precision=3, suppress=False)

    test_log = lr_log = np.array([])
    train_log, err_log = np.array([]).reshape(2,-1), np.array([]).reshape(3,-1)
    
    criterionGAN = nn.MSELoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    lambda_L1 = 10
    real_label, fake_label = 1, 0

    label_d = Variable(torch.FloatTensor(size=D_size)).to(device)

    for epoch in trange(epochs):
        
        for itercount in range(batch_size):
            setidx = np.random.randint(len(dataset_train))
            Acd, Ihb = next(dataset_train[setidx])
            Acd, Ihb = torch.Tensor(Acd).to(device), torch.Tensor(Ihb).to(device)
            fake_I = netG(Acd, epoch)  # G(A)
            
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
            loss_D = (loss_D_fake + loss_D_real) * 0.5 /batch_size # combine loss
            loss_D.backward()
        
            # update G
            set_requires_grad(netD, False)  # D requires no gradients when optimizing G
            optimizerG.zero_grad()        # set G's gradients to zero
            pred_fake = netD(fake_AI)
            label_d.data.fill_(real_label)
            loss_G_GAN = criterionGAN(pred_fake, label_d)
            loss_G_L1 = criterionL1(fake_I, Ihb) * lambda_L1
            # combine loss and calculate gradients
            loss_G = (loss_G_GAN + loss_G_L1) /batch_size
            loss_G.backward()

            train_log = np.column_stack((train_log, np.array([[loss_D.item()], [loss_G.item()]])))
            if epoch+itercount<=2:
                print("After BP: {} (GiB)".format(torch.cuda.memory_reserved()/1024/1024/1024))
            
        lr_log = np.append(lr_log, optimizerG.state_dict()['param_groups'][0]['lr'])
        
        if epoch%(verbose*5) == 0:
            for name, parms in netG.named_parameters():
                if (parms.grad is not None):
                    print('-->name:{:s} -->grad_norm:{:.3e} -->value_norm:{:.2e}'.format(
                        name,parms.grad.norm(p=1).data,parms.norm(p=1).data))    
            
            for name, parms in netD.named_parameters():
                if (parms.grad is not None):
                    print('-->name:{:s} -->grad_norm:{:.3e} -->value_norm:{:.2e}'.format(
                        name,parms.grad.norm(p=1).data,parms.norm(p=1).data))    
        
        optimizerD.step()
        optimizerG.step()
        if schedulerG is not None:
            schedulerG.step()
        if schedulerD is not None:
            schedulerD.step()
        torch.cuda.empty_cache()
        
        if epoch%verbose == 0:
            test_loss, Ihb_err, Top_err, Rate_err = test_eval(netG, Acd_test, Ihb_test, Resmax, 
                                                              device_test, res_trans, RDevelop)
            test_log = np.append(test_log, test_loss)
            err_log = np.column_stack((err_log, np.array([[Ihb_err], [Top_err], [Rate_err]])))
            loss1, loss2 = train_log[0,-batch_size:].sum(), train_log[1,-batch_size:].sum()
            
            print("--->Epoch {} | LR: {:.2e} | Train Loss: {:.2e}(D) {:.2e}(G) | Test Loss: {:.2e}|".format(
                epoch, lr_log[-1], loss1, loss2, test_log[-1]))
            print("             | Pred Error: {:.2f}%(Ihb), {:.2f}%(Rate)|".format(
                epoch, Ihb_err*100, Rate_err*100))
            
        if Ihb_err < check:
            check = Ihb_err*0.95
            save(netG, modelPATH, modelname[0])
            save(netD, modelPATH, modelname[1])
        
    return train_log, test_log, err_log 