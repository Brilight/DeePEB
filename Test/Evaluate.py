import math
import time
import csv
import os

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

from tqdm import trange, tqdm

from Datas.data_load import csv_to_np
from utils.Res_plot import NRMSE_plot
from Test.CD import CD_Error, CD_measure
from development import development as Dev


def evaluate(opt, model, device, res_trans, RDevelop, 
              windows = 100, save_res=False, figpath=None):
    """
    To evaluate the model, including: runtime, overall RMSE&NRMSE of inhibitor and rate, 
    RMSE&NRMSE of inhibitor and rate at different heights (to check the learning ability of the model),
    and CD errors
    """
    np.set_printoptions(suppress=True, precision=3, threshold=40, linewidth=100, edgeitems=20)  
    Resmax, Full_size = opt.Resmax, opt.Full_size
    CD_size = Full_size if opt.CD_size is None else opt.CD_size
    CD_scale = Resmax[0]/CD_size[0]
    t_dev, scale = opt.t_dev, opt.scale
    
    test_num = len(opt.dataidx_eval)
    Runtime = np.zeros((test_num))
    RMSE, NMSE, NMSE_height = np.zeros((2, test_num)), np.zeros((2, test_num)), np.zeros((2, test_num, 80))
    CD_err_x, CD_err_y = [[] for i in range(test_num)], [[] for i in range(test_num)]
    CD_RMSE = np.zeros((2, test_num))
    
    for mask_idx in trange(test_num):
        masktest = opt.dataidx_eval[mask_idx]
        Acd_test = csv_to_np(opt.datapath+str(masktest).zfill(4)+"Acid_ini").reshape((1, 1, *Resmax))
        Ihb_test = csv_to_np(opt.datapath+str(masktest).zfill(4)+"Inhibitor").reshape((1, 1, *Resmax))

        with torch.no_grad():
            start = time.time()
            ini = torch.Tensor(Acd_test).to(device)
            s_pred = model(ini)
            s_pred = res_trans(s_pred.detach().cpu())
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
        Runtime[mask_idx] = time.time()-start

        s_pred, s_label = s_pred.numpy().reshape(*Resmax), Ihb_test.reshape(*Resmax)
        RMSE[0, mask_idx] = np.mean((s_pred-s_label)**2)**0.5
        NMSE[0, mask_idx] = (((s_pred-s_label)**2).sum()/(s_label**2).sum())**(1/2)*100

        pred, label = RDevelop(s_pred), RDevelop(s_label)
        RMSE[1, mask_idx] = np.mean((pred-label)**2)**0.5
        NMSE[1, mask_idx] = (((pred-label)**2).sum()/(label**2).sum())**(1/2)*100
        
        if save_res:
            np.savetxt(opt.resistpath+str(mask_idx).zfill(4)+'pred.csv', pred.flatten(), delimiter=' ')
            #resist_pred = np.loadtxt(resistpath+str(mask_idx).zfill(4)+"pred-arrT", float).reshape(80,1000,1000)
        
        for height in np.arange(80):
            a, b = s_pred[height,...], s_label[height,...]
            NMSE_height[0, mask_idx, height] = (((a - b)**2).sum()/(b**2).sum())**(1/2)*100
            a, b = pred[height,...], label[height,...]
            NMSE_height[1, mask_idx, height] = (((a - b)**2).sum()/(b**2).sum())**(1/2)*100
        
        print("RMSE for label {}:".format(mask_idx), RMSE[:,mask_idx])
        print("NMSE for label {}:".format(mask_idx), NMSE[:,mask_idx])
        print("NMSE_height for label {}:".format(mask_idx), np.mean(NMSE_height,axis=-1)[:,mask_idx])
        
        pred = np.array(F.interpolate(torch.Tensor(pred).reshape(1,1,*Resmax), size=CD_size, 
                                      mode='trilinear', align_corners=False,)).squeeze().transpose(2,1,0)
        T_pred = CD_scale*Dev(pred, opt.CD_seeds).transpose(2,1,0)
        T_pred = np.array(F.interpolate(torch.Tensor(T_pred).reshape(1,1,*CD_size), size=Full_size, 
                                      mode='trilinear', align_corners=False,)).squeeze()
        
        label = np.array(F.interpolate(torch.Tensor(label).reshape(1,1,*Resmax), size=CD_size, 
                                       mode='trilinear', align_corners=False,)).squeeze().transpose(2,1,0)
        T_label = CD_scale*Dev(label, opt.CD_seeds).transpose(2,1,0)
        T_label = np.array(F.interpolate(torch.Tensor(T_label).reshape(1,1,*CD_size), size=Full_size, 
                                       mode='trilinear', align_corners=False,)).squeeze()

        for height in opt.Heights:
            CD_x_label = CD_measure(T_label, t_dev, height, scale, 'X', Resmax, Full_size)
            CD_y_label = CD_measure(T_label, t_dev, height, scale, 'Y', Resmax, Full_size)

            CD_x_pred = CD_measure(T_pred, t_dev, height, scale, 'X', Resmax, Full_size)
            CD_y_pred = CD_measure(T_pred, t_dev, height, scale, 'Y', Resmax, Full_size)

            if CD_x_label.shape[0]>=CD_x_pred.shape[0]:
                Tmp = -CD_Error(CD_x_label, CD_x_pred, str(height)+"-CD_x_label")
            else:
                Tmp = CD_Error(CD_x_pred, CD_x_label, str(height)+"-CD_x_pred")
            print("CD_x error at height {}:\n".format(height), Tmp)
            CD_err_x[mask_idx].extend(list(Tmp))

            if CD_y_label.shape[0]>=CD_y_pred.shape[0]:
                Tmp = -CD_Error(CD_y_label, CD_y_pred, str(height)+"-CD_y_label")
            else:
                Tmp = CD_Error(CD_y_pred, CD_y_label, str(height)+"-CD_y_pred")
            print("CD_y error at height {}:\n".format(height), Tmp)
            CD_err_y[mask_idx].extend(list(Tmp))

        CD_RMSE[0, mask_idx] = np.mean(np.array([*CD_err_x[mask_idx]])**2)**0.5
        CD_RMSE[1, mask_idx] = np.mean(np.array([*CD_err_y[mask_idx]])**2)**0.5
        print("CD_RMSE for label {}:".format(mask_idx), CD_RMSE[:,mask_idx])

    print("CD Error:",np.mean(CD_RMSE,axis=-1))
    print("RMSE(Mean, Std):", np.mean(RMSE,axis=-1), np.std(RMSE,axis=-1))
    print("NMSE(Mean, Std):", np.mean(NMSE,axis=-1), np.std(NMSE,axis=-1))
    print("NMSE_height:", np.mean(NMSE_height,axis=(-2)))
    print("Top Error (mean):", np.mean(np.mean(NMSE_height,axis=-2)[:,-10:],axis=-1))
    
    NRMSE_plot(Resmax, np.mean(NMSE_height,axis=1), figpath=figpath)
    
    return Runtime, RMSE, NMSE, NMSE_height, CD_err_x, CD_err_y, CD_RMSE

    