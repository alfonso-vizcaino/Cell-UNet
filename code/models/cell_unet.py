#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:40:57 2021

@author: alfonso
based on https://github.com/milesial/Pytorch-UNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,output_size,dropout=False):
        super().__init__()
        if dropout :    
            self.maxpool_conv = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size),
                DoubleConv(in_channels, out_channels),
                nn.Dropout(0.5)
            )
        else :
            self.maxpool_conv = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size),
                DoubleConv(in_channels, out_channels),
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.dropout = dropout
        self.dp = nn.Dropout(0.5)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        if self.dropout :
            return self.dp(self.conv(x))
        else :
            return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class CellUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(CellUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.ps1 = nn.PixelShuffle(2)
        self.down1 = Down(64, 128, 256)
        self.ps2 = nn.PixelShuffle(2)
        self.down2 = Down(128, 256, 128)
        self.ps3 = nn.PixelShuffle(2)
        self.down3 = Down(256, 512, 64)
        self.ps4 = nn.PixelShuffle(2)
        self.down4 = Down(512, 1024, 32, dropout=True)
        self.ps5 = nn.PixelShuffle(2)
        self.up1 = Up(256, 128, dropout=True)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)
        
        self.bl = nn.AdaptiveMaxPool2d(512)
        
        self.outc = OutConv(16, n_classes)
        
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x1 = self.ps1(x1)
        x3 = self.down2(x2)
        x2 = self.ps2(x2)
        x4 = self.down3(x3)
        x3 = self.ps3(x3)
        x5 = self.down4(x4)
        x4 = self.ps4(x4)
        x5 = self.ps5(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.bl(x)
        logits = self.outc(x)
        return logits

from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

model = CellUNet(n_channels=3, n_classes=1).to(device)
summary(model, (3,512,512))

# def get_n_params(model):
#     pp=0
#     for p in list(model.parameters()):
#         nn=1
#         for s in list(p.size()):
#             nn = nn*s
#         pp += nn
#     return pp

# print(get_n_params(model))