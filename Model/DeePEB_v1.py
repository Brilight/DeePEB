import math
import time
import os

import numpy as np
import torch 
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt
from functools import partial

torch.manual_seed(42)
np.random.seed(42)

from tqdm import trange, tqdm

def compl_mul3d(a, b):
    # (batch, in_channel, z, y, x), (in_channel, out_channel, z, y, x) -> (batch, out_channel, z, y, x)
    op = partial(torch.einsum, "bizyx,iozyx->bozyx")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])], dim=-1)


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3 #Number of Fourier modes, max: floor(N/2)+1

        self.scale = math.sqrt(2 / in_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights3 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights4 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))

    def forward(self, x):
        batchsize, sizez, sizey, sizex = x.shape[0], x.size(-3), x.size(-2), x.size(-1)
        x = torch.fft.rfftn(x, s=(sizez, sizey, sizex), norm='ortho',)
        x = torch.stack([x.real, x.imag], dim=5)
        out_fft = torch.zeros(batchsize, self.out_channels, sizez, sizey, sizex//2 + 1, 2, device=x.device)
        out_fft[:, :, :self.modes1, :self.modes2, :self.modes3] = compl_mul3d(
            x[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_fft[:, :, -self.modes1:, :self.modes2, :self.modes3] = compl_mul3d(
            x[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_fft[:, :, :self.modes1, -self.modes2:, :self.modes3] = compl_mul3d(
            x[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_fft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = compl_mul3d(
            x[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        out_fft = torch.complex(out_fft[..., 0], out_fft[..., 1])
        x = torch.fft.irfftn(out_fft, s=(sizez, sizey, sizex), norm='ortho',)
        del out_fft; torch.cuda.empty_cache()
        return x
    

class DeePEB(nn.Module):
    def __init__(self, modes1, modes2, modes3, channels, layers, hf_channels):
        super(DeePEB, self).__init__()
        
        self.channels, self.modes1, self.modes2, self.modes3 = channels, modes1, modes2, modes3
        self.layers = layers
        
        self.dsp = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=3, stride=(1,2,2), padding=1, padding_mode='replicate'))
        
        self.fn0 = nn.Sequential(nn.Linear(1, self.channels))
        
        self.activation = nn.LeakyReLU(inplace=True)
        
        self.conv1 = SpectralConv3d(self.channels, self.channels, self.modes1, self.modes2, self.modes3)
        self.w1 = nn.Sequential(nn.Conv3d(self.channels, self.channels, kernel_size=1, stride=1,))
        
        self.fn1 = nn.Sequential(nn.Linear(self.channels, 64))
        self.fn2 = nn.Sequential(nn.Linear(64, 1))
        
        self.usp = nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear',align_corners=True))      
        
        self.hf0 = nn.Sequential(
            nn.Conv3d(1, hf_channels, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True), 
            nn.Conv3d(hf_channels, 1, kernel_size=1, stride=1))
        
        self.hf1 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=3, stride=1, 
                      dilation=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(1, 1, kernel_size=3, stride=1,
                      dilation=1, padding=1, padding_mode='replicate'))
        
        self.hf2 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=3, stride=1, 
                      dilation=2, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(1, 1, kernel_size=3, stride=1,
                      dilation=2, padding=2, padding_mode='replicate'))
        
        self.hf3 = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=(1,5,5), stride=1, 
                      dilation=1, padding=(0,2,2), padding_mode='replicate'),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(2, 1, kernel_size=(1,5,5), stride=1,
                      dilation=1, padding=(0,2,2), padding_mode='replicate'))
        
    def initialize(self):
        for m in self.children():
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                        if layer.bias is not None:
                            nn.init.uniform_(layer.bias, 0)
        print("Net Initialized")
        
    def forward(self, acd):
        x = self.dsp(acd)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fn0(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        for layer in range(self.layers):
            x = self.conv1(x) + self.w1(x)
            x = self.activation(x)
            
        x = self.fn1(x.permute(0, 2, 3, 4, 1))
        x = self.activation(x)
        x = self.fn2(x).permute(0, 4, 1, 2, 3)
        x = self.usp(x)
        x += self.hf0(acd) + self.hf1(acd) + self.hf2(acd) + self.hf3(acd)
        return x
    