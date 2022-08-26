#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch 
import torch.utils.data as data
import torch.nn.init as init
import torch.nn.functional as F

import math
import time

import csv
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.rc('text', usetex=False) 
plt.rc('font', family='serif')
plt.rcParams['font.size'] = '25'

from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import mpl_toolkits.mplot3d as Axes3D


def CNN_plot(Sample_values, layerlabel=-1, attention=1000):
    Resmin = [-1000,-1000,0]
    Resmax = [1000,1000,80]
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams['font.size'] = '20'
    plt.rcParams['font.family'] = 'DejaVu Serif'

    plt.tick_params(labelsize=20)
    if type(Sample_values) is torch.Tensor:
        Sample_values = Sample_values.cpu().detach().numpy()
    while len(Sample_values.shape)>2:
        Sample_values = np.sum(Sample_values,0)
    X = np.linspace(Resmin[0],Resmax[0],Sample_values.shape[0])
    Y = np.linspace(Resmin[1],Resmax[1],Sample_values.shape[1])
    X, Y = np.meshgrid(X, Y)
    levels=np.linspace(Sample_values.min(), Sample_values.max(),50)
    cset1 = ax.contourf(X, Y, Sample_values, levels, cmap=cm.jet)    
    ax.set_title('Projected sum of Layer {}, Reso={:.0f}'.format(
        layerlabel,(Resmax[0]-Resmin[0])/Sample_values.shape[0]))
    plt.xlim([-attention, attention]); plt.ylim([-attention, attention])
    if attention==1000:
        plt.xticks([-1000,-500,0,500,1000]); plt.yticks([-500,0,500,1000])
    if attention==100:
        plt.xticks([-100,-50,0,50,100]); plt.yticks([-50,0,50,100])
    position = fig.add_axes([0.99, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset1, pad=0, fraction=0, cax=position)
    cbar.set_ticks(np.linspace(levels[0], levels[-1], 5).tolist())
    
    plt.show()

    
def hist_plot(A, B, title=None, hist_size=[40, 500, 500], figpath=None):
    '''
    A and B are all Tensors located in cpu, with shape [1,1,*Res]
    '''
    plt.rcParams['font.size'] = '20'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    
    A_new = F.interpolate(A, size=hist_size, mode='trilinear', align_corners=False,).detach().cpu().numpy()
    B_new = F.interpolate(B, size=hist_size, mode='trilinear', align_corners=False,).detach().cpu().numpy()
    scale = 1.1
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(131)
    ax.hist(A_new.flatten(),bins=100,)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    plt.xticks(np.linspace(A_new.min()/scale, A_new.max()*scale,4))
    plt.yscale('log'); plt.xscale('linear')
    if title is not None:
        ax.set_title("Hist plot of {}".format(title[0]))
    
    ax = fig.add_subplot(132)
    ax.hist(B_new.flatten(),bins=100,)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    plt.xticks(np.linspace(B_new.min()/scale, B_new.max()*scale,4))
    plt.yscale('log'); plt.xscale('linear')
    if title is not None:
        ax.set_title("Hist plot of {}".format(title[1]))
    
    ax = fig.add_subplot(133)
    scat_plot = ax.scatter(x=A_new, y=B_new )
    if title is not None:
        ax.set_title('{}-{}'.format(*title))
    plt.xscale('linear'); plt.yscale('linear')
    if figpath is not None:
        plt.savefig(figpath+"{}-{}-hist.pdf".format(*title), dpi = 400)
    plt.show()


def log_plot(log1, log2, log3 = None, label = None, figpath=None):
    if any(x in np.concatenate((log1, log2)) for x in [np.nan, np.inf]):
        print("There exists nan or inf the log datas...")
        return
    fig,ax1 = plt.subplots(figsize=(16,8))
    plt.rcParams['font.size'] = '20'
    plt.rcParams['font.family'] = 'DejaVu Serif'

    X = np.arange(log1.shape[0])
    plt.plot(X, log1, color='b', label=label[0])
    X = np.arange(log2.shape[0])*(log1.shape[0]-1)/(log2.shape[0]-1)
    plt.plot(X, log2, color='r', label=label[1])
    plt.xticks(np.linspace(0, log1.shape[0], 8)//50*50)
    plt.ylim([0.9*min(log1.min(),log2.min()),1.1*max(log1.max(),log2.max())])
    plt.yscale('log')
    ax1.set_ylabel("Loss",fontsize='25')
    plt.legend(loc='upper left')
    if log3 is not None:
        ax2=ax1.twinx()
        X = np.arange(log3.shape[0])*(log1.shape[0]-1)/(log3.shape[0]-1)
        plt.plot(X, log3, color='y',label=label[2])
        plt.ylim([log3.min()*0.9,log3.max()*1.1])
        plt.yscale('log')
        #ax2.set_ylabel("Learning rate")
        plt.legend(loc='lower right')
    if figpath is not None:
        plt.savefig(figpath+"{}-{}-log.pdf".format(label[0],label[1]), dpi = 400)
    plt.show()

    
def res_plot(Sample_points, s_pred, s_label, res_type, height=0, attention=1000, figpath=None):
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter  

    fig = plt.figure(figsize=(18, 9))    
    plt.rcParams['font.size'] = '20'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    levelmax = 1 if res_type=='Ihb' else 40
    
    heigt = 'bottom' if height==0 else 'top'
    Resmin, Resmax = [-1000,-1000,0], [1000,1000,80]
    Sample_points, s_pred, s_label = Sample_points[height,...], s_pred[height,...], s_label[height,...]
    X = np.linspace(Resmin[0], Resmax[0], Sample_points.shape[0])
    Y = np.linspace(Resmin[1], Resmax[1], Sample_points.shape[1])
    X, Y = np.meshgrid(X, Y)
    
    ax = fig.add_subplot(121)
    levels = np.linspace(min(0, s_pred.min(), s_label.min()),max(levelmax, s_pred.max(), s_label.max()),20)
    cset = ax.contourf(X, Y, s_pred, levels, cmap=cm.jet)    
    ax.set_title('predict at '+heigt,fontsize='25')
    plt.xlim(-attention, attention); plt.ylim(-attention, attention)
    if attention==1000:
        plt.xticks([-1000,-500,0,500,1000]); plt.yticks([-500,0,500,1000])
    if attention==100:
        plt.xticks([-100,-50,0,50,100]); plt.yticks([-50,0,50,100])
    #position = fig.add_axes([0.01, 0.2, 0.02, 0.6])
    
    ax = fig.add_subplot(122)
    cset = ax.contourf(X, Y, s_label, levels, cmap=cm.jet)    
    ax.set_title('label at '+heigt,fontsize='25')
    plt.xlim(-attention, attention); plt.ylim(-attention, attention)
    if attention==1000:
        plt.xticks([-1000,-500,0,500,1000]); plt.yticks([-500,0,500,1000])
    if attention==100:
        plt.xticks([-100,-50,0,50,100]); plt.yticks([-50,0,50,100])
    position = fig.add_axes([0.99, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, pad=0, fraction=0, cax=position)
    cbar.set_ticks(np.linspace(levels[0], levels[-1], 5).tolist(), FormatStrFormatter('%1.1f') )
    if figpath is not None:
        plt.savefig(figpath+"{}-{}-res.pdf".format(res_type, height), dpi = 400)
    
    plt.show()

    
def Cord_trans(Cord, Resmax):
    return np.stack(((2*Cord[:,0]-999.5)/Resmax[0],
                     (2*Cord[:,1]-999.5)/Resmax[1],
                    (Cord[:,2]+0.5)/Resmax[2]),axis=1)

def FFT_plot(Values, kmax=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams['font.size'] = '20'
    plt.rcParams['font.family'] = 'DejaVu Serif'

    plt.tick_params(labelsize=20)
    
    if isinstance(Values, torch.Tensor):
        Values = Values.cpu().detach().numpy()
    while len(Values.shape)>2:
        Values = np.sum(Values,0)
    
    X = np.fft.fftfreq(Values.shape[0], 1)
    Y = np.fft.fftfreq(Values.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    levels=np.linspace(min(0, Values.min()),max(1, Values.max()),20)
    cset1 = ax.contourf(X, Y, Values, levels, cmap=cm.jet)    
    ax.set_title('FFT Results')
    kmax = kmax if kmax is not None else X.max()
    plt.xlim(-kmax,kmax); plt.ylim(-kmax, kmax)
    cbar = fig.colorbar(cset1)
    cbar.set_ticks(np.linspace(levels[0], levels[-1], 5).tolist())
    plt.show()
    
    
def Err_plot(Err, err_type, attention=100, figpath=None):
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
    fig = plt.figure(figsize=(18, 9))    
    plt.rcParams['font.size'] = '20'
    plt.rcParams['font.family'] = 'DejaVu Serif'

    Resmin, Resmax = [0,-1000,-1000], [80,1000,1000]
    X, Y, Z = np.meshgrid(np.arange(1000), np.arange(1000), np.arange(80))
    Sample_points = np.swapaxes(np.stack((2*X-999.5, 2*Y-999.5, Z+0.5),axis=3), 0, 2)
    X, Y = np.meshgrid(np.linspace(-1000, 1000, 1000), np.linspace(-1000, 1000, 1000))
    
    levels = np.linspace(Err.min(), Err.max(), 20)
    ax = fig.add_subplot(121)
    Sample_points, err = Sample_points[0,...], Err[0,...]
    cset = ax.contourf(X, Y, err, levels, cmap=cm.jet)    
    ax.set_title('Error at bottom',fontsize='25')
    plt.xlim(-attention, attention); plt.ylim(-attention, attention)
    if attention==1000:
        plt.xticks([-1000,-500,0,500,1000]); plt.yticks([-500,0,500,1000])
    if attention==100:
        plt.xticks([-100,-50,0,50,100]); plt.yticks([-50,0,50,100])  
        
    ax = fig.add_subplot(122)
    Sample_points, err = Sample_points[-1,...], Err[-1,...]
    cset = ax.contourf(X, Y, err, levels, cmap=cm.jet)    
    ax.set_title('Error at top',fontsize='25')
    plt.xlim(-attention, attention); plt.ylim(-attention, attention)
    if attention==1000:
        plt.xticks([-1000,-500,0,500,1000]); plt.yticks([-500,0,500,1000])
    if attention==100:
        plt.xticks([-100,-50,0,50,100]); plt.yticks([-50,0,50,100])  
        
    position = fig.add_axes([0.99, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, pad=0, fraction=0, cax=position)
    if np.abs(err.max())<1:
        tmp = int(max(abs(Err.min()),abs(Err.max()))*10)/10.0
        cbar.set_ticks([-tmp,-tmp/2,0,tmp/2,tmp], FormatStrFormatter('%.1f') )
    else:
        cbar.set_ticks([-30,-20,-10,0,10,20,30], FormatStrFormatter('%.1f') )
    if figpath is not None:
        plt.savefig(figpath+"{}-Error.pdf".format(err_type), dpi = 400)
    
    plt.show()

    
def Errlog_plot(log1, log2, label=None, errorlim=None, figpath=None):
    fig,ax1 = plt.subplots(figsize=(16,8))
    plt.rcParams['font.size'] = '20'
    plt.rcParams['font.family'] = 'DejaVu Serif'

    X = np.arange(log1.shape[0])*5
    plt.plot(X, log1, color='b', label=label[0])
    plt.plot(X, log2, color='r', label=label[1])
    plt.xticks(np.arange(X[0],X[-1]*1.1,math.ceil(X.shape[0]/100)*100))
    if errorlim is not None:
        plt.ylim(errorlim)
    else:
        plt.ylim([0.9*min(log1.min(),log2.min()),1.1*max(log1.max(),log2.max())])
    plt.yscale('linear')
    ax1.set_ylabel("Relative Error",fontsize='25')
    plt.legend(loc='upper right')
    plt.ylim([0,100])
    if figpath is not None:
        plt.savefig(figpath+"{}-{}-errlog.pdf".format(*label), dpi = 400)
    plt.show()
    

def CD_errXY_plot(Err_x, Err_y, Modelname, figpath=None):
    plt.rcParams['font.size'] = '25'
    plt.rcParams['font.family'] = 'DejaVu Serif'

    Count = lambda Err:[Err[np.abs(Err)<1].size, 
                        Err[np.abs(np.abs(Err)-1.5)<0.5].size, 
                        Err[np.abs(np.abs(Err)-3)<1].size, 
                        Err[np.abs(np.abs(Err)-5)<1].size,
                        Err[np.abs(np.abs(Err)-7)<1].size,
                        Err[np.abs(np.abs(Err)-9)<1].size, Err[np.abs(Err)>=10].size]
    resx, resy = Count(Err_x), Count(Err_y)

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    width, n = 0.3, 1

    x = [0, 1, 2, 3, 4, 5, 6]
    name = ['0~1', '1~2', '2~4', '4~6', '6~8', '8~10', '>10']

    a = plt.bar(x, resx/np.sum(resx)*100, width = width, label = 'Error in X', fc='#5E9EE4')
    for i in range(len(x)):  
        x[i] = x[i] + width

    b = plt.bar(x, resy/np.sum(resy)*100, width = width, label = 'Error in Y', fc ='#F9F871')

    plt.xticks(ticks=np.arange(len(name))+width, labels=name, fontsize=20)
    plt.xlabel(Modelname+'CD Error in the X/Y Direction (nm)', fontsize=25)
    plt.yticks([0,20,40,60,80], fontsize=20)
    plt.ylabel('Percentage/%', fontsize=25)
    plt.legend(frameon=True, fontsize=20) 
    plt.grid(axis='y', linestyle='--')
    if figpath is not None:
        plt.savefig(figpath+"CD Error-XY.pdf", dpi = 400)
    plt.show()
    
    
def Resist_plot(resist, height, attention=100, window=None, figpath=None):
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
    fig = plt.figure(figsize=(8, 7))    
    plt.rcParams['font.size'] = '20'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    
    X, Y = np.meshgrid(np.linspace(-1000, 1000, 1000), np.linspace(-1000, 1000, 1000))
    levels = np.linspace(resist.min(), resist.max(),20)
    ax = fig.add_subplot(111)
    res = np.swapaxes(resist[height,...], 0, 1)
    cset = ax.contourf(X, Y, res, levels, cmap=cm.jet)    
    ax.set_title('Developed Profile at Height {}'.format(height),fontsize='25')
    if window is None:
        plt.xlim(-attention, attention); plt.ylim(-attention, attention)
        if attention==1000:
            plt.xticks([-1000,-500,0,500,1000]); plt.yticks([-500,0,500,1000])
        else:
            plt.xticks([-100,-50,0,50,100]); plt.yticks([-50,0,50,100]) 
    else:
        plt.xlim(window[0], window[1]); plt.ylim(window[2], window[3])
    
    position = fig.add_axes([0.99, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, pad=0, fraction=0, cax=position, label="Arrivail Time(ms)",)
    cbar.formatter.set_scientific(True)
    cbar.set_ticks([0, 10000, 30000, 50000, 70000, 90000,], FormatStrFormatter('%.0e') )
    if figpath is not None:
        plt.savefig(figpath+"resist-{}.pdf".format(height), dpi = 400)

    plt.show()
    
    
def NRMSE_plot(Resmax, NMSE_height, legend=["Inhibitor", "Rate"], figpath=None):
    fig, ax = plt.subplots(figsize=(16,8))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.rcParams['font.size'] = '20'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.scatter(np.arange(Resmax[0]), NMSE_height[0], color='b', label=legend[0])
    plt.xticks(np.arange(0, Resmax[0]+1, 10))
    plt.yticks([0,2,4,6,8,10])
    plt.ylabel('NRMSE/%'); plt.xlabel('Height/nm')
    
    Count = [np.mean(NMSE_height[0]), np.std(NMSE_height[0])]
    for i in range(int(Count[1]/0.01)):
        plt.axhline(y=Count[0]+i*0.01, linewidth=0.1, ls="-", c="skyblue", alpha=0.4)
        plt.axhline(y=Count[0]-i*0.01, linewidth=0.1, ls="-", c="skyblue", alpha=0.4)

    plt.scatter(np.arange(Resmax[0]), NMSE_height[1], color='r', label=legend[1])
    plt.legend(frameon=True, fontsize=20) 
    plt.yscale('linear'); plt.show()
    if figpath is not None:
        plt.savefig(figpath+"NRMSE.pdf", dpi = 400)
