import math
import time
import os

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)


class BasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super(BasicBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, 
                      padding=int(kernel_size//2), padding_mode='replicate', bias=bias),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, 
                      padding=int(kernel_size//2), padding_mode='replicate', bias=bias),)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),)
            
    def init(self):
        for m in self.children():
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Conv3d):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                        if layer.bias is not None:
                            nn.init.uniform_(layer.bias, 0)
    
    def forward(self, x):
        out = self.conv(x)
        out = out + self.shortcut(x)
        return F.leaky_relu(out)
    
    
class ResNet(nn.Module):
    def __init__(self, Channels, kernel_size=3, bias=False):
        super(ResNet, self).__init__()
        self.inchannel, self.kernel_size, self.bias = Channels[0], kernel_size, bias
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, Channels[0], kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(inplace=True),)
        
        reslayers = []
        for channels in Channels:
            reslayers.append(self.make_layer(channels, 2,))
        self.reslayers = nn.Sequential(*reslayers)
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(Channels[-1], 1, kernel_size=1, stride=1, padding=0, bias=bias),)
        
    def make_layer(self, channels, num_blocks):
        strides, layers = [1] * num_blocks, []
        for stride in strides:
            layers.append(BasicBlock(self.inchannel, channels, self.kernel_size, self.bias))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def initialize(self):
        for m in self.children():
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Conv3d):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                    if isinstance(layer, BasicBlock):
                        layer.init()
        print("Net Initialized")
    
    def forward(self, x, epoch=1, window=300):
        out = self.conv1(x)
        out = self.reslayers(out)
        out = self.conv2(out)
        return out
    