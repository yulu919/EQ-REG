from __future__ import print_function
import os
import time
import datetime
from My_module import MyModule
from My_sequential import MySequential
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F



class MyConv(MyModule):       
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride,  bias=True, tran_num=4, ifIni=0, dilation = 1, p=2):
        super(MyConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size//2), stride =stride, bias=bias)

        self.tranNum = tran_num
        self.sizeP = kernel_size

        if ifIni:
            self.expand = 1
        else:
            self.expand = self.tranNum

        self.p = p
        self.rot_eq_loss = 0
        self.layer_num = 0
    



    def forward(self, x):

        if self.training:
            # print("train ing num:", self.rot_num)
            y = self.conv(x)
            B, C, H, W = y.size() # 32 256 48 48
            z = y.reshape([2, B//2, C, H, W])
            self.rot_eq_loss = torch.norm(self.rot(z[0, :, :, :, :]) - self.shift(z[1, :, :, :, :]), p = self.p)


        else:
            y = self.conv(x)
        
        return y

    def shift(self,x):
        B, C, H, W = x.size()
        x = x.reshape([B,C//self.tranNum, self.tranNum, H, W])
        y = torch.cat((x[:,:,self.rot_num:,:,:],x[:,:,:self.rot_num,:,:] ), dim=2)
        return y.reshape([B, C, H, W])

    def rot(self,x):
        y = torch.rot90(x, self.rot_num, [2, 3])
        return y

    def rotateAny(self, images):
        images_clone = images.clone()
        step = 360 / 8
        angle = self.rot_num * step
        angle_rad = torch.deg2rad(torch.tensor(angle))

        theta = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad), 0],
                              [torch.sin(angle_rad), torch.cos(angle_rad), 0]])
        theta = theta.float()

        grid = F.affine_grid(theta.unsqueeze(0).expand(images_clone.size(0), -1, -1), images_clone.size(),
                                       align_corners=True).to(images.device)
        rotated_images = F.grid_sample(images_clone, grid, align_corners=True)

        return rotated_images

    def GetBasis_PCA(self, sizeP, tranNum=8, inP=None, Smooth=True):
        if inP == None:
            inP = sizeP
        inX, inY, Mask = self.MaskC(sizeP)
        X0 = np.expand_dims(inX, 2)
        Y0 = np.expand_dims(inY, 2)
        Mask = np.expand_dims(Mask, 2)
        theta = np.arange(tranNum) / tranNum * 2 * np.pi
        theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)
        #    theta = torch.FloatTensor(theta)
        X = np.cos(theta) * X0 - np.sin(theta) * Y0
        Y = np.cos(theta) * Y0 + np.sin(theta) * X0
        #    X = X.unsqueeze(3).unsqueeze(4)
        X = np.expand_dims(np.expand_dims(X, 3), 4)
        Y = np.expand_dims(np.expand_dims(Y, 3), 4)
        v = np.pi / inP * (inP - 1)
        p = inP / 2

        k = np.reshape(np.arange(inP), [1, 1, 1, inP, 1])
        l = np.reshape(np.arange(inP), [1, 1, 1, 1, inP])

        BasisC = np.cos((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)
        BasisS = np.sin((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)

        BasisC = np.reshape(BasisC, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)
        BasisS = np.reshape(BasisS, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)

        BasisC = np.reshape(BasisC, [sizeP * sizeP * tranNum, inP * inP])
        BasisS = np.reshape(BasisS, [sizeP * sizeP * tranNum, inP * inP])

        BasisR = np.concatenate((BasisC, BasisS), axis=1)

        U, S, VT = np.linalg.svd(np.matmul(BasisR.T, BasisR))

        Rank = np.sum(S > 0.0001)
        BasisR = np.matmul(np.matmul(BasisR, U[:, :Rank]), np.diag(1 / np.sqrt(S[:Rank] + 0.0000000001)))
        BasisR = np.reshape(BasisR, [sizeP, sizeP, tranNum, Rank])

        temp = np.reshape(BasisR, [sizeP * sizeP, tranNum, Rank])
        var = (np.std(np.sum(temp, axis=0) ** 2, axis=0) + np.std(np.sum(temp ** 2 * sizeP * sizeP, axis=0),
                                                                  axis=0)) / np.mean(
            np.sum(temp, axis=0) ** 2 + np.sum(temp ** 2 * sizeP * sizeP, axis=0), axis=0)
        Trod = 1
        Ind = var < Trod
        Rank = np.sum(Ind)
        Weight = 1 / np.maximum(var, 0.04) / 25
        if Smooth:
            BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight, 0), 0), 0) * BasisR

        return torch.FloatTensor(BasisR), Rank, Weight

    def Getini_reg(self, nNum, inNum, outNum, expand, weight=1):
        A = (np.random.rand(outNum, inNum, expand, nNum) - 0.5) * 2 * 2.4495 / np.sqrt((inNum) * nNum) * np.expand_dims(
            np.expand_dims(np.expand_dims(weight, axis=0), axis=0), axis=0)
        return torch.FloatTensor(A)

    def MaskC(self, SizeP):
        p = (SizeP - 1) / 2
        x = np.arange(-p, p + 1) / p
        X, Y = np.meshgrid(x, x)
        C = X ** 2 + Y ** 2
        Mask = np.ones([SizeP, SizeP]) # Mask[C>(1+1/(4*p))**2]=0
        Mask = np.exp(-np.maximum(C - 1, 0) / 0.2)
        return X, Y, Mask



def default_conv(in_channels, out_channels, kernel_size, bias=True, ifIni = 0):
    return MyConv(
        in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias, ifIni=0)

