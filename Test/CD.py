import math
import time
import csv
import os
import scipy
import shutil

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
np.set_printoptions(suppress=True, precision=3, 
                    threshold=40, linewidth=100, edgeitems=20)  


def CD_measure(resist, t_dev, height, scale, direc, Resmax, Full_size, resolution=5, cd_min=1, cd_max=50):
    """
    resist.size = Full_size, in the z,y,x direction
    Resmax is the geometric shape size
    """
    X, Y = np.meshgrid(np.linspace(-Resmax[2], Resmax[2], Full_size[2]*scale), 
                       np.linspace(-Resmax[1], Resmax[1], Full_size[1]*scale))
    Pos = np.stack((X, Y),axis=-1)
    Values = np.swapaxes(resist[height,...], 0, 1) if direc=='X' else resist[height,...]
    Values_new = F.interpolate(torch.Tensor(Values).reshape(1,1,Full_size[1],Full_size[2]), scale_factor=(scale, scale),
                           mode='bicubic', align_corners=True).detach().numpy().squeeze().reshape(scale*Full_size[1],-1)
    idx_contact = np.where((Values_new>=0.5*t_dev)&(Values_new<=1.5*t_dev))
    if len(idx_contact[0])<=1:
        return np.array([[0,0,0],])
    Contact_pos, Contact_val = Pos[idx_contact], Values_new[idx_contact]
    Y_fix = np.unique(Contact_pos[...,1])
    #print("Search Range in {}:".format(direc), Y_fix.min(), Y_fix.max())
    
    Keys, flag = [], [0,0]
    for y_cord in Y_fix:
        line = np.where(Contact_pos[...,1]==y_cord)
        X_cords, T_arr = Contact_pos[line][...,0], Contact_val[line]
        X_new = F.interpolate(torch.Tensor(X_cords).reshape(1,1,X_cords.shape[0]), scale_factor=resolution,
                             mode='linear').numpy().squeeze()
        T_new = F.interpolate(torch.Tensor(T_arr).reshape(1,1,T_arr.shape[0]), scale_factor=resolution,
                             mode='linear').numpy().squeeze()
        T_diff = T_new - t_dev
        thre_cord = np.where(T_diff[:-1]*T_diff[1:]<=0)[0]
        while thre_cord.shape[0]>1:
            #print("{} changing points for {}".format(thre_cord.shape, y_cord))
            left, right = X_new[thre_cord[0]], X_new[thre_cord[1]]
            thre_cord = thre_cord[2:]
            i_key, insert, key = 0, True if cd_max>right-left>cd_min else False, [left, y_cord, right-left]
            
            while i_key<len(Keys) and insert:
                cd_key = Keys[i_key]
                if cd_key != flag:
                    if np.sqrt((cd_key[0]-key[0])**2+(cd_key[1]-key[1])**2)<=((cd_key[2]+key[2])/2)**2**0.5:
                        if key[2]>cd_key[2]:
                            Keys[i_key] = flag
                        else:
                            insert = False
                i_key +=1
            if flag in Keys:
                Keys.remove(flag)
            if insert:
                Keys.append(key)
            #print("\n",*Keys,sep="\n")
    Keys = np.array(Keys) if direc=='X' else np.array(Keys)[:, [1,0,2]]
    return Keys

def CD_Error(CD_label, CD_pred, Long):
    Flag = np.ones(len(CD_label))
    CD_Err = np.zeros(len(CD_label))
    for idx1 in range(len(CD_label)):
        idx2 = 0
        while idx2<len(CD_pred) and Flag[idx1]:
            if ((CD_label[idx1, 0]-CD_pred[idx2, 0])**2+
                (CD_label[idx1, 1]-CD_pred[idx2, 1])**2)**0.5<=(CD_label[idx1, 2]+CD_pred[idx2, 2]):
                CD_Err[idx1] = CD_label[idx1, 2] - CD_pred[idx2, 2]
                Flag[idx1] = 0
            idx2+=1
        if Flag[idx1]:
            print("Contact {} in {} is redundant, with info {}".format(idx1, Long, CD_label[idx1,:]))
            CD_Err[idx1] = CD_label[idx1, 2]
    return CD_Err


def CD_eval(opt):
    t_dev, Heights, scale, testmasks = opt.t_dev, opt.Heights, opt.scale, len(opt.dataidx_eval)
    Modelname, respath, resistpath, ratepath = opt.Modelname, opt.respath, opt.resistpath, opt.ratepath 
    CD_err_x, CD_err_y = [[] for i in range(testmasks)], [[] for i in range(testmasks)]
    CD_RMSE = np.zeros((2, testmasks))
    f = open(respath+"out.txt", "a")
    
    for mask_idx in trange((testmasks)):
        
        mask_label = opt.dataidx_eval[mask_idx]
        resist_pred = np.loadtxt(resistpath+Modelname+str(mask_label).zfill(4)+"pred-arrT", float).reshape(80,1000,1000)
        resist_label = np.loadtxt(resistpath+str(mask_label).zfill(4)+'label-arrT', float).reshape(80,1000,1000)
        print("{} Read Complete".format(mask_label))

        for height in Heights:
            CD_x_label = CD_measure(resist_label, t_dev, height, scale, 'X', Resmax, Full_size)
            CD_y_label = CD_measure(resist_label, t_dev, height, scale, 'Y', Resmax, Full_size)

            CD_x_pred = CD_measure(resist_pred, t_dev, height, scale, 'X', Resmax, Full_size)
            CD_y_pred = CD_measure(resist_pred, t_dev, height, scale, 'Y', Resmax, Full_size)

            Tmp = -CD_Error(CD_x_label, CD_x_pred, str(height)+"-CD_x_label")
            print("CD_x error at height {}:\n".format(height), Tmp, file=f)
            CD_err_x[mask_idx].extend(list(Tmp))

            Tmp = -CD_Error(CD_y_label, CD_y_pred, str(height)+"-CD_y_label")
            print("CD_y error at height {}:\n".format(height), Tmp, file=f)
            CD_err_y[mask_idx].extend(list(Tmp))
        
        CD_RMSE[0, mask_idx] = np.mean(np.array([*CD_err_x[mask_idx]])**2)**0.5
        CD_RMSE[1, mask_idx] = np.mean(np.array([*CD_err_y[mask_idx]])**2)**0.5
        print("CD_RMSE for label {}:".format(mask_idx), CD_RMSE[:,mask_idx], file=f)
    
    f.close()
    return CD_err_x, CD_err_y, CD_RMSE