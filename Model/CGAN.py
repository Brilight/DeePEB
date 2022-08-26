import math
import time
import os

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

from tqdm import trange, tqdm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1, stride=stride, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),)
    
    def init(self):
        for m in self.children():
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Conv3d):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                        if layer.bias is not None:
                            nn.init.uniform_(layer.bias, 0)
    
    def forward(self, x):
        out = self.block(x)
        return out
    
    
class Generator(nn.Module):
    def __init__(self, Channels, kernel_conv=5, Ihb_size=[80, 1000, 1000]):
        super(Generator, self).__init__()
        self.Ihb_size = Ihb_size
        
        self.left_conv_1 = ConvBlock(
            in_channels=Channels[0], middle_channels=Channels[1], out_channels=Channels[1], stride=2)

        self.left_conv_2 = ConvBlock(
            in_channels=Channels[1], middle_channels=Channels[2], out_channels=Channels[2], stride=2)

        self.left_conv_3 = ConvBlock(
            in_channels=Channels[2], middle_channels=Channels[3], out_channels=Channels[3], stride=2)
        
        self.left_conv_4 = ConvBlock(
            in_channels=Channels[3], middle_channels=Channels[4], out_channels=Channels[4], stride=2)
        
        self.deconv_1 = nn.ConvTranspose3d(in_channels=Channels[4], out_channels=Channels[3], bias=True, stride=2,
                                           kernel_size=kernel_conv, padding=kernel_conv//2, output_padding=(1,0,0))
        self.right_conv_1 = ConvBlock(
            in_channels=Channels[4], middle_channels=Channels[3], out_channels=Channels[3], stride=1)

        self.deconv_2 = nn.ConvTranspose3d(in_channels=Channels[3], out_channels=Channels[2], bias=True, stride=2,
                                           kernel_size=kernel_conv, padding=kernel_conv//2, output_padding=1)
        self.right_conv_2 = ConvBlock(
            in_channels=Channels[3], middle_channels=Channels[2], out_channels=Channels[2], stride=1)

        self.deconv_3 = nn.ConvTranspose3d(in_channels=Channels[2], out_channels=Channels[1],  bias=True, stride=2,
                                           kernel_size=kernel_conv, padding=kernel_conv//2, output_padding=1)
        self.right_conv_3 = ConvBlock(
            in_channels=Channels[2], middle_channels=Channels[1], out_channels=Channels[1], stride=1)

        self.deconv_4 = nn.ConvTranspose3d(in_channels=Channels[1], out_channels=Channels[0],  bias=True, stride=2,
                                           kernel_size=kernel_conv, padding=kernel_conv//2, output_padding=1)
        
        self.regular = nn.Upsample(size=self.Ihb_size, mode='trilinear', align_corners=False)
    
    def initialize(self):
        for layer in self.children():
            if isinstance(layer, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                if layer.bias is not None:
                    nn.init.uniform_(layer.bias, 0)
            if isinstance(layer, ConvBlock):
                layer.init()
        print("Net Initialized")
    
    def forward(self, x):

        feature_1 = self.left_conv_1(x)        
        feature_2 = self.left_conv_2(feature_1)
        feature_3 = self.left_conv_3(feature_2)
        feature_4 = self.left_conv_4(feature_3)
        
        out = self.deconv_1(feature_4)
        out = torch.cat((feature_3, out), dim=1)
        out = self.right_conv_1(out)

        out = self.deconv_2(out)
        out = torch.cat((feature_2, out), dim=1)
        out = self.right_conv_2(out)
        
        out = self.deconv_3(out)
        out = torch.cat((feature_1, out), dim=1)
        out = self.right_conv_3(out)

        out = self.deconv_4(out)
        return self.regular(out)
    
    
class BasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super(BasicBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, 
                      padding=int(kernel_size//2), padding_mode='replicate', bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=2, 
                      padding=int(kernel_size//2), padding_mode='replicate', bias=bias),)
        
        self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),)
            
        self.norm = nn.GroupNorm(1, out_channels)
        
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
        out = self.norm(out)
        return F.leaky_relu(out)
    
    
class Descriminator(nn.Module):
    def __init__(self, nc, nf=10, kernel_size=3, bias=False):
        super(Descriminator, self).__init__()
        self.inchannel, self.kernel_size, self.bias = nf, kernel_size, bias
        
        reslayers = [nn.Conv3d(nc, nf, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.LeakyReLU(0.1, inplace=True),
                    self.make_layer(nf*2, 2,),
                    self.make_layer(nf, 2,),
                    nn.Conv3d(nf, 1, kernel_size=3, stride=(1,2,2), padding=1, bias=bias),
                    nn.Sigmoid(),]
        
        self.reslayers = nn.Sequential(*reslayers)
                
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
                        nn.init.kaiming_normal_(layer.weight, mode='fan_in')
                    if isinstance(layer, BasicBlock):
                        layer.init()
    
    def forward(self, x):        
        out = self.reslayers(x)
        return out
    
    