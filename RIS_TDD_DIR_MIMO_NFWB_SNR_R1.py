

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:29:40 2021

@author: 5106
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
import einops
# from Transformer_model import *
# from NFBF_RIS import Channel_BS_RIS_LOS, Channel_BS_RIS_NLOS, Channel_RIS_UE_NLOS, Channel_RIS_UE_LOS, batch_Channel_RIS_UE_LOS, Channel_RIS_UE_MIMO_NLOS, Channel_RIS_UE_MIMO_LOS, Channel_BS_UE_MIMO_NLOS, Channel_BS_RIS_LOS_PLA
from NFBF_RIS_R3 import Channel_BS_RIS_NLOS, Channel_RIS_UE_MIMO_NLOS, Channel_RIS_UE_MIMO_LOS, Channel_BS_UE_MIMO_NLOS, Channel_BS_RIS_LOS_PLA
from scipy.linalg import dft
from GFNet import Block
from PolarizedSelfAttention import ParallelPolarizedSelfAttention,SequentialPolarizedSelfAttention
import random
from math import *
import datetime
import math
import scipy.io as sio # mat

max_time_delay = torch.tensor(5e-9)

bit = 1

random.seed(2023)

def dft_matrix(n):
    DFT = np.empty((n, n), dtype=np.complex128)
    w = np.exp(-2j * np.pi / n)
    for i in range(n):
        for j in range(n):
            DFT[i][j] = w ** (i * j)
    return DFT / np.sqrt(n)

# Convolutional module
class conv_block1(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, strides,pads, dilas):
        super(conv_block1, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=strides, padding=pads, dilation=dilas,bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.3)
            )

    def forward(self, x):

        x = self.conv(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x
class MLPMixer(nn.Module):
    def __init__(self, dim, num_patch,num_output, depth, token_dim, channel_dim):
        super().__init__()
        
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, num_patch, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_output)
        )
    def forward(self, x): # (batch, 8, 26, 32)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x) # (batch, patch, dim)
        #x = x.mean(dim=1)
        return self.mlp_head(x) 

class UL_CE(nn.Module): 
    def __init__(self, param_list, param_list1): 
        super(UL_CE,self).__init__()
        
        self.fc = param_list[0]
        self.B  = param_list[1]
        self.Nc = param_list[2]
        self.M1  = param_list[3]
        self.M2  = param_list[4]
        self.N1  = param_list[5]
        self.N2  = param_list[6]
        
        self.D_ant = param_list[7]
        self.R1 = param_list[8]
        
        self.R = param_list[9]
        
        self.h1 = param_list[10]
        self.h2 = param_list[11]
        
        self.aod_azi = param_list[12]
        self.aod_ele = param_list[13]
        
        self.aoa_azi = param_list[14]
        self.aoa_ele = param_list[15]
        
        self.tau = param_list[16]
        self.L1 = param_list[17]
        
        self.L2 = param_list[18]
        self.K = param_list[19]
        self.Nr = param_list[20]

        
        self.RF_chain = param_list1[0]
        self.BS_TDDs = param_list1[1]
        self.RIS_TDDs = param_list1[2]
        self.L = param_list1[3]
        self.UL = param_list1[4]
        
        self.Bit = param_list1[5]
        self.sigma = param_list1[6]
        self.batch = param_list1[7]
        
        self.upsnr = param_list1[8]
        self.downsnr = param_list1[9]

        
        self.M_ant = self.M1*self.M2
        self.N_ant = self.N1*self.N2
        L = self.UL
        
        # self.S_digital = torch.nn.Parameter(torch.randn(L, self.Nc, self.RF_chain, self.Nr)+1j*torch.randn(L, self.Nc, self.RF_chain, self.Nr)).cuda() #L个时隙BS导频,这里假设频率选择
        self.S_analog = torch.nn.Parameter(torch.randn(L,self.Nc,self.M_ant,self.RF_chain)).cuda() 
        self.S_analog_TDD = torch.nn.Parameter(torch.randn(L,self.Nc,self.BS_TDDs,self.RF_chain)).cuda() 

        # self.Phase = torch.nn.Parameter(torch.randn(L,Nc, N_ant)).cuda()   #L个时隙RIS相位
        # self.Phase1 = torch.nn.Parameter(torch.randn(L, self.N_ant)).cuda()   #L个时隙RIS相位
        # self.Phase2 = torch.nn.Parameter(torch.randn(L, self.N_ant)).cuda()
        self.Phase1 = torch.nn.Parameter(torch.rand(L, self.N_ant)).cuda()   #L个时隙RIS相位
        self.Phase2 = torch.nn.Parameter(torch.rand(L, self.N_ant)).cuda()
        
        self.Phase_TDD = torch.nn.Parameter(torch.randn(L, self.RIS_TDDs)).cuda()
        
        self.B_quan = 2
        # self.trans  = TRANS_BLOCK(self.K*self.RF_chain*self.Nr*2*L,4*L,256,1)
        
        # # self.linear = nn.Linear(2*L*self.Nc, self.Bit//self.B_quan) 
        # # self.bn = nn.BatchNorm1d(self.Bit//self.B_quan)
        # depth = 2
        # self.blocks = nn.ModuleList([
        #     Block(dim=self.Nc, mlp_ratio=2, h=self.K*self.RF_chain*self.Nr*2, w=L)
        #     for i in range(depth)
        # ])
        
        
        self.psa = ParallelPolarizedSelfAttention(channel=self.Nc)

        
        # self.QL = QuantizationLayer(self.B_quan)
        self.conv1 = conv_block1(self.Nr*self.RF_chain, 2,1,1,1)
        self.sig = nn.Sigmoid()
        
        
        
    def forward(self, param_list, param_list1, H_BR, H_RU, H_BU):

        
        M_ant = self.M1*self.M2
        N_ant = self.N1*self.N2
        L = self.UL
        
        n = (torch.arange(0, self.Nc)+1).cuda()
        nn = -(self.Nc+1)/2
        deta = self.B/self.Nc
        fm=(nn+n)*deta+self.fc
        
        Y = torch.zeros(H_RU.shape[0],self.Nc, self.Nr, self.RF_chain, L).cuda() + 1j
        Theta = torch.zeros(self.Nc, self.N_ant, self.N_ant).cuda() + 1j
        F_RF_TTDs = torch.zeros(self.Nc,self.M_ant,self.RF_chain).cuda() + 1j
        
        # self.upsnr = 0
        
        train_snr = param_list1[10]
        snr_index = random.randint(0, len(train_snr)-1)            
        self.upsnr = train_snr[snr_index]
        SNR_linear=10**(-1*self.upsnr/10.)
        seg = (2*pi)/(2**bit)
        data_normalized1 = 2*pi*self.Phase1
        Phase1_bit =   torch.floor(data_normalized1/seg)* seg
        
        data_normalized2 = 2*pi*self.Phase2
        Phase2_bit =   torch.floor(data_normalized2/seg)* seg
        

        for l in range(L):
            # for nr in range(self.Nr):
                
            # H_equ = (torch.zeros(batch,Nc,RF_chain,K) + 0j).cuda()
            # for i in range(K):
            #     H_equ[:,:,i] = (H_RU[:,i] @ Phi @ H_BR @ F_RF).reshape(-1,Nc,RF_chain)


            # data_normalized1 = 2*pi*(self.Phase1[l,:] - self.Phase1[l,:].min()) / (self.Phase1[l,:].max() - self.Phase1[l,:].min())
            
            # Phase1_bit =   torch.floor(data_normalized1/seg)* seg
            # data_normalized2 = 2*pi*(self.Phase2[l,:] - self.Phase2[l,:].min()) / (self.Phase2[l,:].max() - self.Phase2[l,:].min())
            # Phase2_bit =   torch.floor(data_normalized2/seg)* seg
            
            Theta1 = torch.exp(1j*Phase1_bit) # L,Nc, N_ant
            Theta2 = torch.exp(1j*Phase2_bit) # L,Nc, N_ant
            # Phase_TDD = torch.exp(1j*self.Phase_TDD[l,:]) # L,Nc, N_ant
            S_analog = torch.exp(1j*self.S_analog[l,:,:,:])/sqrt(self.M_ant)
            
            for nc in range(self.Nc):
                # self.Phase_TDD[l,:]= torch.abs(self.Phase_TDD[l,:])
                # self.Phase_TDD[l,:]= max_time_delay*self.Phase_TDD[l,:]/torch.max(self.Phase_TDD[l,:])
                
                a1 = max_time_delay * self.sig(self.Phase_TDD[l,:]) 
                Phi_T = torch.exp(1j*2*pi*fm[nc]*a1)
                
                # self.S_analog_TDD[l,nc,:,:]= torch.abs(self.S_analog_TDD[l,nc,:,:])
                # self.S_analog_TDD[l,nc,:,:]= max_time_delay*self.S_analog_TDD[l,nc,:,:]/torch.max(self.S_analog_TDD[l,nc,:,:])
                
                b = max_time_delay * self.sig(self.S_analog_TDD[l,nc,:,:]) 
                
                S_analog_T = torch.exp(1j*2*pi*fm[nc]*b)
                Phi_TDD = torch.kron(Phi_T.reshape(1,self.RIS_TDDs),torch.ones(1,self.N_ant//self.RIS_TDDs).cuda())
                
                F_RF_TDD = torch.kron(S_analog_T,torch.ones(M_ant//self.BS_TDDs,1).cuda())
                F_RF_TTDs[nc,:,:] = S_analog[nc,:,:]*F_RF_TDD
                
                Theta[nc,:,:] = torch.diag_embed(Theta1*Phi_TDD*Theta2)
            
            # Theta = torch.diag(self.Phase[l,:])
            a = H_RU[:,self.K-1,:,:,:] @ Theta# batch Nc 1 N_ant
            a = a @ H_BR + H_BU[:,self.K-1,:,:,:] # batch Nc 1 M
            # a = a @ torch.exp(1j*self.S_analog[l,:,:,:]) # batch Nc 1 RF_chain
            a = a @ F_RF_TTDs # batch Nc 1 RF_chain
            
            Power = (torch.sum(torch.abs(a)**2,[2,3])).reshape(-1,self.Nc,1,1)
            # Power = (torch.mean(torch.abs(a)**2,[1,2,3])).reshape(-1,1,1,1)

            
            # Power = torch.tensor(1)

            # a = a @ self.S_analog
            # F_SDMA = torch.exp(1j*self.S_analog[l,:,:,:]) @ self.S_digital[l,:,:,k]  # L Nc M 1
            # Power = (torch.sum(torch.abs(F_SDMA)**2,[2,3])).reshape(-1,Nc,1,1)
            # self.S_digital = self.S_digital[l,:,:,:] / torch.sqrt(Power)
            
            # a = a @ self.S_digital[l,:,:,k] # batch Nc 1 1

            n = (torch.randn(H_RU.shape[0],self.Nc,self.Nr,self.RF_chain) + 1j*torch.randn(H_RU.shape[0],self.Nc,self.Nr,self.RF_chain)).cuda()*torch.sqrt(Power*SNR_linear/2)
            y = a + n # batch Nc 1 RF_chain
            # y = a
            # y = y.reshape(self.batch, self.Nc, self.RF_chain)
            # y = y.permute(0,1,3,2)
            Y[:,:,:,:,l] = y
                
        Y = Y.reshape(H_RU.shape[0],  self.Nc, self.K*self.RF_chain*self.Nr, L)
            
        x = torch.cat((torch.real(Y),torch.imag(Y)), 3) #[batch, K*RF_chain, Nc, 2*L]
        
        x = self.psa(x)

        
        # # x = self.conv1(x)
        # # x = x.reshape(batch, Nc, K*RF_chain*Nr*2*L)
        
        # x = x.reshape(self.batch, self.K*self.RF_chain*self.Nr*2*self.L, self.Nc)
        
        # for blk in self.blocks:
        #     x = blk(x) 
        # x = x.reshape(self.batch, self.Nc, self.K*self.RF_chain*self.Nr*2*L)
        
        # x = self.trans(x)

        # x = x.reshape(-1,2*Nc*L)
        # x = self.linear(x)
        # x = self.bn(x)
        # x = torch.sigmoid(x)
        # x = self.QL(x)
        return x

class DL_BF_TTD(nn.Module): 
    def __init__(self, param_list, param_list1): 
        super(DL_BF_TTD,self).__init__()
        
        self.fc = param_list[0]
        self.B  = param_list[1]
        self.Nc = param_list[2]
        self.M1  = param_list[3]
        self.M2  = param_list[4]
        self.N1  = param_list[5]
        self.N2  = param_list[6]
        
        self.D_ant = param_list[7]
        self.R1 = param_list[8]
        
        self.R = param_list[9]
        
        self.h1 = param_list[10]
        self.h2 = param_list[11]
        
        self.aod_azi = param_list[12]
        self.aod_ele = param_list[13]
        
        self.aoa_azi = param_list[14]
        self.aoa_ele = param_list[15]
        
        self.tau = param_list[16]
        self.L1 = param_list[17]
        
        self.L2 = param_list[18]
        self.K = param_list[19]
        self.Nr = param_list[20]

        
        self.RF_chain = param_list1[0]
        self.BS_TDDs = param_list1[1]
        self.RIS_TDDs = param_list1[2]
        self.L = param_list1[3]
        self.UL = param_list1[4]
        
        self.Bit = param_list1[5]
        self.sigma = param_list1[6]
        self.batch = param_list1[7]
        
        self.upsnr = param_list1[8]
        self.downsnr = param_list1[9]

        
        self.M_ant = self.M1*self.M2
        self.N_ant = self.N1*self.N2
        self.L = self.UL
        
        # self.B_quan = 2
        # self.DQL = DequantizationLayer(self.B_quan)
        # self.FC1 = nn.Linear(self.Bit//self.B_quan,2*self.Nc*L)#全连接
        # self.bn1 = nn.BatchNorm1d(2*self.Nc*L)
        
        
        # self.trans  = TRANS_BLOCK(self.K*self.RF_chain*self.Nr*2*self.L,self.N_ant,256,2) # batch, Nc, K*N_ant
        self.FC = nn.Linear(self.K*self.RF_chain*self.Nr*2*self.L, self.N_ant)
        
        
        self.linear_H  = nn.Linear(self.N_ant, 32)
        self.linear_RIS = nn.Linear(self.Nc*32, self.N_ant)   #生成RIS相位
        self.linear_RIS2 = nn.Linear(self.Nc*32, self.N_ant)   #生成RIS相位
        self.LN_RIS = nn.LayerNorm(self.N_ant)

        
        self.linear_RIS_T  = nn.Linear(self.N_ant, self.RIS_TDDs)
        
        self.linear_BS_RF1  = nn.Linear(self.Nc, self.RF_chain)
        self.linear_BS_RF2  = nn.Linear(self.N_ant, self.M_ant)
        
        self.linear_BS_T  = nn.Linear(self.N_ant, self.RF_chain*self.BS_TDDs)
        
        self.linear_BS_BB  = nn.Linear(self.N_ant, 2*self.RF_chain*self.Nr)
        
        # self.trans_RIS1  = TRANS_BLOCK(self.N_ant,self.N_ant,256,1)
        
        # self.trans_RIS2  = TRANS_BLOCK(self.N_ant,self.N_ant,256,1)

        self.LN1 = nn.LayerNorm(self.N_ant)
        
        # self.trans_RIS_T  = TRANS_BLOCK(self.RIS_TDDs,self.RIS_TDDs,256,1)
        self.trans_RIS_T  = MLPMixer(dim= self.RIS_TDDs, num_patch = self.Nc, num_output = self.RIS_TDDs, depth=1, token_dim=128, channel_dim=64)
        
        self.LN_RIS_T = nn.LayerNorm(self.RIS_TDDs)
        # self.trans_BS_RF1  = TRANS_BLOCK(self.RF_chain,self.RF_chain,256,1)
        self.trans_BS_RF1  = MLPMixer(dim= self.RF_chain, num_patch = self.M_ant, num_output =self.RF_chain, depth=1, token_dim=128, channel_dim=64)
        
        self.LN_BS_RF1 = nn.LayerNorm(self.RF_chain)

        # self.trans_BS_T  = TRANS_BLOCK(self.RF_chain*self.BS_TDDs,self.RF_chain*self.BS_TDDs,256,1)
        self.trans_BS_T  = MLPMixer(dim= self.RF_chain*self.BS_TDDs, num_patch = self.Nc, num_output =self.RF_chain*self.BS_TDDs, depth=1, token_dim=128, channel_dim=64)
        
        self.LN_BS_T = nn.LayerNorm(self.RF_chain*self.BS_TDDs)

        # self.trans_BS_BB  = TRANS_BLOCK(2*self.RF_chain*self.Nr,2*self.RF_chain*self.Nr,256,1)
        self.trans_BS_BB  = MLPMixer(dim= 2*self.RF_chain*self.Nr, num_patch = self.Nc, num_output = 2*self.RF_chain*self.Nr, depth=1, token_dim=128, channel_dim=64)
        
        self.LN_BS_BB = nn.LayerNorm(2*self.RF_chain*self.Nr)
        self.sig = nn.Sigmoid()
        
        # self.tran  = TRANS_BLOCK(self.K*self.RF_chain*self.Nr*2*self.L,self.N_ant,256,2)
        
        # self.linear = nn.Linear(2*L*self.Nc, self.Bit//self.B_quan) 
        # self.bn = nn.BatchNorm1d(self.Bit//self.B_quan)
        depth = 2
        self.blocks = nn.ModuleList([
            Block(dim=self.Nc, mlp_ratio=2, h=self.K*self.RF_chain*self.Nr*2, w=self.L)
            for i in range(depth)
        ])
        
        d_model = self.K*self.RF_chain*self.Nr*2*self.L
        out_len = self.Nc
        
        
        self.projection2 = MLPMixer(dim= d_model, num_patch = out_len, num_output = self.N_ant, depth=2, token_dim=64, channel_dim=64)
        self.sig = nn.Sigmoid()

        # self.linear_RF = nn.Linear(Nc*32, M_ant*K)   
        
        # self.trans2  = TRANS_BLOCK(2*Nr*Nr,4*Nr+1,256,1) 
        
        
    def forward(self, param_list,  param_list1, x):

        
        # x = self.DQL(x)-0.5
        # x = self.FC1(x)
        # x = self.bn1(x)
        # x = torch.relu(x)
        
        # x = x.reshape(batch,Nc,K*2*L)
        
        
        
        x = x.reshape(x.shape[0], self.K*self.RF_chain*self.Nr*2*self.L, self.Nc)
        
        for blk in self.blocks:
            x = blk(x) 
        x = x.reshape(x.shape[0], self.Nc, self.K*self.RF_chain*self.Nr*2*self.L)
        
        # x_init = self.tran(x)
        
        x_init = self.projection2(x)
        # x_init = self.FC(x_init)

        # x_init = self.LN1(x_init)
        
        # x_init = self.trans(x) # batch, Nc, K*N_ant
        
        x_R = self.linear_H(x_init)   #[batch,Nc,32]
        out_RIS = x_R.reshape(-1,self.Nc*32)
        Phi_phase = self.linear_RIS(out_RIS)
        # Phi_phase = self.trans_RIS1(Phi_phase)

        Phi_phase = self.sig(self.LN_RIS(Phi_phase))
        
        Phi_phase2 = self.linear_RIS2(out_RIS)
        # Phi_phase2 = self.trans_RIS2(Phi_phase2)

        Phi_phase2 = self.sig(self.LN_RIS(Phi_phase2))
        
        # min_value, min_index = torch.min(Phi_phase,1)
        # max_value, max_index = torch.max(Phi_phase,1)

        # bit = 3
        seg = (2*pi)/(2**bit)

        data_normalized1 = 2 * pi * Phi_phase
        Phi_phase_qua = torch.floor(data_normalized1 / seg) * seg
        data_normalized2 = 2 * pi * Phi_phase2
        Phi_phase2_qua = torch.floor(data_normalized2 / seg) * seg


        # Phi_phase_qua = torch.zeros(Phi_phase.shape[0],self.N_ant).cuda()
        # Phi_phase2_qua = torch.zeros(Phi_phase.shape[0],self.N_ant).cuda()

        # for ba in range(Phi_phase.shape[0]):
        #     data_normalized1 = 2*pi*(Phi_phase[ba,:] - Phi_phase[ba,:].min()) / (Phi_phase[ba,:].max() - Phi_phase[ba,:].min())
        #     # data_normalized1 = torch.clamp(Phi_phase,0,2*pi)
        #     Phi_phase_qua[ba,:] =   torch.floor(data_normalized1/seg)* seg
        #
        #     data_normalized2 = 2*pi*(Phi_phase2[ba,:] - Phi_phase2[ba,:].min()) / (Phi_phase2[ba,:].max() - Phi_phase2[ba,:].min())
        #     # data_normalized2 = torch.clamp(Phi_phase2,0,2*pi)
        #     Phi_phase2_qua[ba,:] =   torch.floor(data_normalized2/seg)* seg
            
        # Phi_phase=torch.clamp(Phi_phase,0,2*pi)
        
        Phi_phase = torch.exp(1j*Phi_phase_qua.reshape(-1,1,self.N_ant))
        Phi_phase2 = torch.exp(1j*Phi_phase2_qua.reshape(-1,1,self.N_ant))

        # Phi_phase = torch.exp(1j*Phi_phase.reshape(-1,1,self.N_ant))
        # Phi_phase2 = torch.exp(1j*Phi_phase2.reshape(-1,1,self.N_ant))


        Phi = torch.diag_embed(Phi_phase2)  #[batch,1,M_ant,M_ant]
        
        x_T = self.linear_RIS_T(x_init)   #[batch,Nc,N_ant]
        x_T = self.trans_RIS_T(x_T)
        x_T = self.LN_RIS_T(x_T)
        
        n = (torch.arange(0, self.Nc)+1).cuda()
        nn = -(self.Nc+1)/2
        deta = self.B/self.Nc
        fm=(nn+n)*deta+self.fc
        Phi_WB = torch.zeros(Phi.shape[0],self.Nc,self.N_ant,self.N_ant).cuda() + 1j
        # for nc in range(Nc):
        #     Phi_T = torch.exp(1j*2*pi*fm[nc]*x_T[:,nc,:]).reshape(-1,1,N_ant) 
        #     Phi_WB[:,nc,:,:] = torch.diag_embed(Phi_phase*Phi_T).reshape(-1,N_ant,N_ant) 
        
        # for nc in range(Nc):
            
        #     Phi_T = torch.exp(1j*2*pi*fm[nc]*x_T[:,nc,:])
        #     for ba in range(Phi.shape[0]):
        #         Phi_TDD = torch.kron(Phi_T[ba,:].reshape(1,RIS_TDDs),torch.ones(1,N_ant//RIS_TDDs).cuda())
        #         Phi_WB[:,nc,:,:] = torch.diag_embed(Phi_phase*Phi_TDD*Phi_phase2).reshape(-1,N_ant,N_ant) 
        
        RF_phase = self.linear_BS_RF2(x_init) # batch N_ant RF_chains
        RF_phase = RF_phase.reshape(-1,self.M_ant,self.Nc)

        RF_phase = self.linear_BS_RF1(RF_phase) # batch N_ant RF_chains
        RF_phase = self.trans_BS_RF1(RF_phase)
        RF_phase = self.LN_BS_RF1(RF_phase)
        
        F_RF = torch.exp(1j*RF_phase.reshape(-1,1,self.M_ant,self.RF_chain)) #[batch,1,M_ant,K]
        
        RF_delay = self.linear_BS_T(x_init) # batch Nc BS_TDDs*RF_chains
        RF_delay = self.trans_BS_T(RF_delay)
        RF_delay = self.LN_BS_T(RF_delay)
        RF_delay = RF_delay.reshape(-1,self.Nc,self.BS_TDDs,self.RF_chain)
        # F_RF_delay = torch.exp(1j*2*pi*fm*RF_delay) #[batch,1,M_ant,K]
        
        F_RF_TTDs = torch.zeros(Phi.shape[0],self.Nc,self.M_ant,self.RF_chain).cuda() + 1j
        for nc in range(self.Nc):
            
            # RF_delay[:,nc,:,:]= torch.abs(RF_delay[:,nc,:,:])
            # RF_delay[:,nc,:,:]= max_time_delay*RF_delay[:,nc,:,:]/torch.max(RF_delay[:,nc,:,:])
            
            b = max_time_delay * self.sig(RF_delay[:,nc,:,:])
            a = max_time_delay * self.sig(x_T[:,nc,:])

            
            F_RF_delay = torch.exp(1j*2*pi*fm[nc]*b)
            
            # x_T[:,nc,:]= torch.abs(x_T[:,nc,:])
            # x_T[:,nc,:]= max_time_delay*x_T[:,nc,:]/torch.max(x_T[:,nc,:])
            
            Phi_T = torch.exp(1j*2*pi*fm[nc]*a)
            for ba in range(Phi.shape[0]):
                F_RF_TDD = torch.kron(F_RF_delay[ba,:,:],torch.ones(self.M_ant//self.BS_TDDs,1).cuda())
                F_RF_TTDs[ba,nc,:,:] = F_RF[ba,0,:,:]*F_RF_TDD
                Phi_TDD = torch.kron(Phi_T[ba,:].reshape(1,self.RIS_TDDs),torch.ones(1,self.N_ant//self.RIS_TDDs).cuda())
                Phi_WB[:,nc,:,:] = torch.diag_embed(Phi_phase*Phi_TDD*Phi_phase2).reshape(-1,self.N_ant,self.N_ant) 
            
        
        F_BB = self.linear_BS_BB(x_init) # batch Nc 2*K*RF_chains
        F_BB = self.trans_BS_BB(F_BB)
        F_BB = self.LN_BS_BB(F_BB)
        # F_BB = F_BB.reshape(-1,Nc,2*K,RF_chain)
        # F_BB_SDMA = (F_BB[:,:,0:K,:] + 1j*F_BB[:,:,K:K*2,:])
        F_BB = F_BB.reshape(-1,self.Nc,self.RF_chain,2*self.Nr)
        F_BB_SDMA = (F_BB[:,:,:,0:self.Nr] + 1j*F_BB[:,:,:,self.Nr:self.Nr*2])
        
        
        
        # F_RF_WB = F_RF*(torch.kron(F_RF_delay,torch.ones(F_RF_delay.size(0),Nc,BS_TDDs,1).cuda()))
        
        F_SDMA = F_RF_TTDs @ F_BB_SDMA  

        Power = (torch.sum(torch.abs(F_SDMA)**2,[2,3])).reshape(-1,self.Nc,1,1)

        F_BB_SDMA = F_BB_SDMA / torch.sqrt(Power)*sqrt(self.Nr)


        return Phi_WB,F_RF_TTDs,F_BB_SDMA,Phi  #Phi[batch,1,M_ant,M_ant]  F_RF[batch,1,M_ant,K]  F_BB_SDMA[batch,Nc,K,K] F_BB_RS[batch,Nc,K,1] 


class NFWB_RIS(nn.Module): 
    def __init__(self, param_list,param_list1): 
        super(NFWB_RIS,self).__init__()
        self.encoder = UL_CE(param_list,param_list1)
        self.decoder = DL_BF_TTD(param_list,param_list1)
        
        
    def forward(self, param_list,param_list1,H_BR, H_RU1, H_BU):
        
        out = self.encoder(param_list,param_list1,H_BR,H_RU1, H_BU)
        Phi,F_RF_WB,F_BB_SDMA,Phi_T = self.decoder(param_list,param_list1,out)


        return Phi,F_RF_WB,F_BB_SDMA,Phi_T
    

def SE_RS_BB(batch,Nc,K,Nt,N_RS1,N_RS2,RS_cor,H,F_SDMA,F_RS,sigma2):
    
    H_SDMA = H @ F_SDMA #[batch,Nc,K,K]

    R = 0
    R_SA = 0
    R_SDMA = 0
    
    Nr = H_SDMA.shape[3]
    
    R_k = torch.zeros(batch,Nr).cuda()
    for a in range(N_RS1): 
        for b in RS_cor[a]: 
            signal = torch.abs(H_SDMA[:,:,b,b]*H_SDMA[:,:,b,b])#[batch,Nc]
            Interfer = torch.zeros(batch,Nc).cuda()
            for c in range(N_RS1): 
                if c!=a:
                    Interfer = Interfer + torch.abs(H_RS[:,:,b,c]*H_RS[:,:,b,c])
            for c in range(K):
                if c!=b:
                    Interfer = Interfer + torch.abs(H_SDMA[:,:,b,c]*H_SDMA[:,:,b,c])
            SINR = signal/(Interfer+sigma2)#[batch,Nc]
            # SINR = signal/(Interfer)#[batch,Nc]

            R_k[:,b] = torch.mean(torch.log2(1+SINR),1)#[batch]
        # R_min,aa = torch.min(R_k,1) #[batch]
        R_min = torch.sum(R_k,1)
        R_SDMA = R_SDMA + torch.mean(R_min)
        Loss = -R_SDMA
            # print(torch.mean(torch.log2(1+SINR)))
    
    return R_SDMA,Loss
class SE_RS(torch.nn.Module):   
    def __init__(self):
        super(SE_RS, self).__init__()
    def forward(self, param_list, param_list1,H_BR,H_RU,H_BU,Phi,F_RF,F_BB_SDMA,F_BB_RS,dSNR):
        
        fc = param_list[0]
        B  = param_list[1]
        Nc = param_list[2]
        M1  = param_list[3]
        M2  = param_list[4]
        N1  = param_list[5]
        N2  = param_list[6]
        
        D_ant = param_list[7]
        R1 = param_list[8]
        
        R = param_list[9]
        
        h1 = param_list[10]
        h2 = param_list[11]
        
        aod_azi = param_list[12]
        aod_ele = param_list[13]
        
        aoa_azi = param_list[14]
        aoa_ele = param_list[15]
        
        tau = param_list[16]
        L1 = param_list[17]
        
        L2 = param_list[18]
        K = param_list[19]
        Nr = param_list[20]

        
        RF_chain = param_list1[0]
        BS_TDDs = param_list1[1]
        RIS_TDDs = param_list1[2]
        L = param_list1[3]
        UL = param_list1[4]
        
        Bit = param_list1[5]
        sigma = param_list1[6]
        batch = param_list1[7]
        
        upsnr = param_list1[8]
        downsnr = param_list1[9]

        batch = H_RU.shape[0]
        
        N_RS1 = 1
        N_RS2 = K
        # RS_cor = [[0,1,2,3]]
        RS_cor = [np.arange(Nr)]
        
        H_BR = H_BR.cuda()
        H_RU = H_RU.cuda()


        # H_equ = (torch.zeros(batch,Nc,RF_chain,Nr) + 0j).cuda()
        H_equ = (H_RU[:,K-1] @ Phi @ H_BR + H_BU[:,K-1])@ F_RF
        
        H_SDMA = H_equ @ F_BB_SDMA #[batch,Nc,K,K]
        
        Nr = H_SDMA.shape[3]
        
        # Pt_dB=0
        # sigma2 = -120
        period = 2e4
        # SNR_dB = Pt_dB-sigma2
        
        # noise_dB = -174
        # noise_dB = noise_dB + 10*np.log10(B/10) - 30
        # noise_power = 10**(noise_dB/10)
        
        # Pt = dSNR-30
        # Pt_p = 10**(Pt/10)
        # sigma = noise_power/Pt_p
        
        SNR_dB = downsnr
        sigma=10**(-1*SNR_dB/10.)
        
        L_CP = 4
        # UL = 0
        

        R_k = torch.zeros(batch).cuda()
        SINR = 0
        Loss = 0
        for a in range(H_SDMA.shape[0]): 
            
            for b in range(H_SDMA.shape[1]): 
                # signal = torch.abs(H_SDMA[:,:,b,b]*H_SDMA[:,:,b,b])#[batch,Nc]
                # signal = torch.det(torch.eye(Nr).cuda() + torch.abs(H_SDMA[a,b,:,:] * H_SDMA[a,b,:,:])/(Nr*sigma2))#[batch,Nc]

                # signal = torch.det(torch.eye(Nr).cuda() + torch.real(H_SDMA[a,b,:,:] @ torch.transpose(torch.conj(H_SDMA[a,b,:,:]), 0,1))/(Nr*sigma))#[batch,Nc]
                
                # # signal = torch.sum(torch.abs(H_SDMA[a,b,:,:] * H_SDMA[a,b,:,:])/(Nr*sigma),[0,1])
                
                # SINR = SINR + torch.log2(signal)
                
                signal = torch.det(torch.eye(Nr).cuda() + H_SDMA[a,b,:,:] @ torch.transpose(torch.conj(H_SDMA[a,b,:,:]), 0,1)/(Nr*sigma))#[batch,Nc]
                
                # signal = torch.sum(torch.abs(H_SDMA[a,b,:,:] * H_SDMA[a,b,:,:])/(Nr*sigma),[0,1])
                
                SINR = SINR + torch.real(torch.log2(signal))
                
        R = (period-UL*Nr)*SINR/(period*H_SDMA.shape[0]*(H_SDMA.shape[1]+L_CP))
        # R = SINR/(H_SDMA.shape[0]*(H_SDMA.shape[1]+L_CP))

        Loss = -R
        
        # R,Loss = SE_RS_BB(batch,Nc,K,K,N_RS1,N_RS2,RS_cor,H_equ,F_BB_SDMA,F_BB_RS,sigma)
        
        
        return R, Loss 
    
def adjust_learning_rate(optimizer, epoch,learning_rate_init,learning_rate_final, epochs):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    # learning_rate_init = 1e-4
    # learning_rate_final = 1e-7
    lr = learning_rate_final + 0.5*(learning_rate_init-learning_rate_final)*(1+math.cos((epoch*3.14)/epochs))
    # lr = 0.00003* (1+math.cos(float(epoch)/TOTAL_EPOCHS*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
            
def NFWB_RIS_TTD(param_list,param_list1, batch,Num_batchs,model_name):
    
    fc = param_list[0]
    B  = param_list[1]
    Nc = param_list[2]
    M1  = param_list[3]
    M2  = param_list[4]
    N1  = param_list[5]
    N2  = param_list[6]
    
    D_ant = param_list[7]
    R1 = param_list[8]
    
    R = param_list[9]
    
    h1 = param_list[10]
    h2 = param_list[11]
    
    aod_azi = param_list[12]
    aod_ele = param_list[13]
    
    aoa_azi = param_list[14]
    aoa_ele = param_list[15]
    
    tau = param_list[16]
    L1 = param_list[17]
    
    L2 = param_list[18]
    K = param_list[19]
    Nr = param_list[20]

    
    RF_chain = param_list1[0]
    BS_TDDs = param_list1[1]
    RIS_TDDs = param_list1[2]
    L = param_list1[3]
    UL = param_list1[4]
    
    Bit = param_list1[5]
    sigma = param_list1[6]
    batch = param_list1[7]
    
    upsnr = param_list1[8]
    downsnr = param_list1[9]
    
    train_snr = param_list1[10]
    
    M_ant = M1*M2
    N_ant = N1*N2

    loss_func = SE_RS()
    loss_func = loss_func.cuda()

    model = NFWB_RIS(param_list,param_list1).cuda()
    # BW = 7e9
    # model_name1 = './models/RIS_NFWB_TTD_'+str(floor(BW))+'band'+str(floor(BS_TDDs))+'BS_TTDs_'+str(floor(RIS_TDDs))+'RIS_TTDs'+str(UL)+'pilots_'+str(floor(upsnr))+'UdB'+str(floor(downsnr))+'DdB'+'.pth'

    # model_name1 = './models/RIS_NFWB_TTD_16BS_TTDs_64RIS_TTDs64pilots_10UdB20DdB.pth'
    # model = torch.load(model_name)
            
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=0.001)
    # optimizer = torch.optim.SGD(net_AE.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4,nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-07,weight_decay=1e-5)


    
    
    num_train = 0
    train_nmse = 0
    train_SE = 0
    best_SE = 0
    numStep = 0
    # warmup_steps = 4000
    d_model = 256
    LR0 = 1
    
    Gt_gain = 10**(25/10.)
    GRIS_gain = 10**(5/10.)

    Gr_gain = 10**(20/10.)
    
    # Gt_gain = 1
    # Gr_gain = 1
     
    flag =0
    
    # if flag == 1:
    #     H_RU = Channel_RIS_UE_MIMO_NLOS(param_list,batch,flag)
    #     H_RU1 = H_RU.reshape(batch,K,Nc,Nr,N_ant) 
    #     # H_BR = (Channel_BS_RIS_LOS(param_list,flag)+Channel_BS_RIS_NLOS(param_list,flag))
    #     H_BR = Channel_BS_RIS_NLOS(param_list,flag)

    # else:
    #     H_RU = Channel_RIS_UE_MIMO_NLOS(param_list,batch,flag)* 1e3 
    #     H_RU1 = H_RU.reshape(batch,K,Nc,Nr,N_ant) 
    #     H_BR = (Channel_BS_RIS_LOS(param_list,flag)+Channel_BS_RIS_NLOS(param_list,1))* 1e3 
    # H_BR = Channel_BS_RIS_LOS(param_list,flag)
    
    scaler = torch.cuda.amp.GradScaler()
    
    Num_epochs = 10
    num_epoch = 0
    
    test_ESE = []
    
    for num_batch in range(Num_batchs//Num_epochs):
        
        start = datetime.datetime.now()

        
        lr = adjust_learning_rate(optimizer, num_batch,5e-3,5e-5,Num_batchs//Num_epochs)
        
        model.train()
        
        H_RU1 = sqrt(Gr_gain*GRIS_gain)*(Channel_RIS_UE_MIMO_LOS(param_list,batch,flag)+Channel_RIS_UE_MIMO_NLOS(param_list,batch,flag))
        H_BU = sqrt(Gt_gain*Gr_gain)*Channel_BS_UE_MIMO_NLOS(param_list,batch,flag)
        H_BU = H_BU.reshape(batch,K,Nc,Nr,M_ant) 
        H_RU1 = H_RU1.reshape(batch,K,Nc,Nr,N_ant) 
        H_BR = sqrt(Gt_gain*GRIS_gain)*(Channel_BS_RIS_LOS_PLA(param_list,flag)+Channel_BS_RIS_NLOS(param_list,flag))
        
        for num_epoch in range(Num_epochs):
            
            snr_index = random.randint(0, len(train_snr)-1)            
            
            # if (num_epoch)%1==0:
            #     if flag == 1:
            #         H_RU1 = Channel_RIS_UE_MIMO_NLOS(param_list,batch,flag)
            #         H_BU = Channel_BS_UE_MIMO_NLOS(param_list,batch,flag)
            #         H_BU = H_BU.reshape(batch,K,Nc,Nr,M_ant) 
            #         H_RU1 = H_RU1.reshape(batch,K,Nc,Nr,N_ant) 
            #         # H_BR = (Channel_BS_RIS_LOS(param_list,flag)+Channel_BS_RIS_NLOS(param_list,flag))
                    

            #     else:
            #         H_RU1 = sqrt(Gr_gain*GRIS_gain)*(Channel_RIS_UE_MIMO_LOS(param_list,batch,flag)+Channel_RIS_UE_MIMO_NLOS(param_list,batch,flag))
            #         H_BU = sqrt(Gt_gain*Gr_gain)*Channel_BS_UE_MIMO_NLOS(param_list,batch,flag)
            #         H_BU = H_BU.reshape(batch,K,Nc,Nr,M_ant) 
            #         H_RU1 = H_RU1.reshape(batch,K,Nc,Nr,N_ant) 
            #         H_BR = sqrt(Gt_gain*GRIS_gain)*(Channel_BS_RIS_LOS_PLA(param_list,flag)+Channel_BS_RIS_NLOS(param_list,flag))
    
    
    
            # optimizer.param_groups[0]['lr'] = (d_model**(-0.5))*min(numStep**(-0.5),numStep*warmup_steps**(-1.5))*LR0
            numStep = numStep+1
            
            Phi,F_RF,F_BB_SDMA,F_BB_RS = model(param_list, param_list1, H_BR, H_RU1, H_BU)
    
            R_sum, Loss = loss_func(param_list, param_list1, H_BR,H_RU1,H_BU,Phi,F_RF,F_BB_SDMA,F_BB_RS, downsnr)
            loss = -R_sum
    
            
            num_train = num_train + 1
            # train_nmse = train_nmse + nmse.detach_()
            train_SE = train_SE + R_sum.detach_()
    
            # print(train_nmse,nmse.detach_(),num_train)
            
            #Adaptive mixed precision method
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            
            # print('num_epoch:',num_batch,'loss',Loss.detach_(),'SE',R_sum.detach_()) 
        # end = datetime.datetime.now()

        # print('run time: %s Seconds'%(end-start))

        if (num_batch+1)%1==0:
            with torch.no_grad():
                model.eval()
                test_SE = 0
                test_SDMA = 0
                test_loop = 5
                for i in range(test_loop):
                    if flag == 1:
                        H_RU1 = Channel_RIS_UE_MIMO_NLOS(param_list,batch,flag)
                        H_RU1 = H_RU1.reshape(batch,K,Nc,Nr,N_ant) 
                        H_BU = Channel_BS_UE_MIMO_NLOS(param_list,batch,flag)
                        H_BU = H_BU.reshape(batch,K,Nc,Nr,M_ant) 
                        # H_BR = (Channel_BS_RIS_LOS(param_list,flag)+Channel_BS_RIS_NLOS(param_list,flag))
                    else:
                        H_RU1 = sqrt(Gr_gain*GRIS_gain)*(Channel_RIS_UE_MIMO_LOS(param_list,batch,flag)+Channel_RIS_UE_MIMO_NLOS(param_list,batch,flag))
                        H_BU = sqrt(Gt_gain*Gr_gain)*Channel_BS_UE_MIMO_NLOS(param_list,batch,flag)
                        H_BU = H_BU.reshape(batch,K,Nc,Nr,M_ant) 
                        H_RU1 = H_RU1.reshape(batch,K,Nc,Nr,N_ant) 
                        H_BR = sqrt(Gt_gain*GRIS_gain)*(Channel_BS_RIS_LOS_PLA(param_list,flag)+Channel_BS_RIS_NLOS(param_list,flag))

                    Phi,F_RF,F_BB_SDMA,F_BB_RS = model(param_list, param_list1, H_BR, H_RU1, H_BU)

                    # H_RU = H_RU.reshape(batch,K,Nc,1,N_ant)

                    R_sum, Loss = loss_func(param_list, param_list1, H_BR,H_RU1,H_BU,Phi,F_RF,F_BB_SDMA,F_BB_RS, downsnr)
                    test_SE = test_SE + R_sum

                test_SE = test_SE/test_loop  

                

            train_nmse = train_nmse/num_train
            train_SE = train_SE/num_train
            time0 =  datetime.datetime.now()-start
            test_ESE.append(test_SE.cpu().numpy().item())
            
            print('num_batch:',num_batch,'time',time0,'train_SE %.5f' % train_SE.cpu(),'test_SE %.5f' % test_SE.cpu()) 
            if test_SE > best_SE:
                best_SE = test_SE
                # model_name = './models/RIS_NFWB_TTD_'+str(floor(B))+'band'+str(floor(BS_TDDs))+'BS_TTDs_'+str(floor(RIS_TDDs))+'RIS_TTDs'+str(UL)+'pilots_'+str(floor(upsnr))+'UdB'+str(floor(downsnr))+'DdB'+'.pth'

                torch.save(model, model_name)
                print('Model saved!')
            num_train = 0
            train_nmse = 0
            train_SE = 0
            start = datetime.datetime.now()
            vadilate_SE = './results/vadilate'+str(floor(BS_TDDs))+'BS_TTDs_'+str(floor(RIS_TDDs))+'RIS_TTDs'+'.mat'

            sio.savemat(vadilate_SE, {'a':test_ESE})
            
    return best_SE

import scipy.io as sio

def NFWB_RIS_TTD_test(param_list,param_list1, batch,Num_batchs,model_name):
    
    fc = param_list[0]
    B  = param_list[1]
    Nc = param_list[2]
    M1  = param_list[3]
    M2  = param_list[4]
    N1  = param_list[5]
    N2  = param_list[6]
    
    D_ant = param_list[7]
    R1 = param_list[8]
    
    R = param_list[9]
    
    h1 = param_list[10]
    h2 = param_list[11]
    
    aod_azi = param_list[12]
    aod_ele = param_list[13]
    
    aoa_azi = param_list[14]
    aoa_ele = param_list[15]
    
    tau = param_list[16]
    L1 = param_list[17]
    
    L2 = param_list[18]
    K = param_list[19]
    Nr = param_list[20]

    
    RF_chain = param_list1[0]
    BS_TDDs = param_list1[1]
    RIS_TDDs = param_list1[2]
    L = param_list1[3]
    UL = param_list1[4]
    
    Bit = param_list1[5]
    sigma = param_list1[6]
    batch = param_list1[7]
    upsnr = param_list1[8]
    downsnr = param_list1[9]
    
    M_ant = M1*M2
    N_ant = N1*N2

    loss_func = SE_RS()
    loss_func = loss_func.cuda()

    model = NFWB_RIS(param_list,param_list1).cuda()
    model = torch.load(model_name) 
    
    # model_name1 = './models/RIS_NFWB_TTD_16BS_TTDs_64RIS_TTDs64pilots_10UdB20DdB.pth'
    # model = torch.load(model_name1) 
            
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=0.001)
    # optimizer = torch.optim.SGD(net_AE.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4,nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-07,weight_decay=1e-5)
    
    # Gt_gain = 10**(25/10.)
    # Gr_gain = 10**(20/10.)
    
    
    
    Gt_gain = 10**(20/10.)
    GRIS_gain = 10**(5/10.)

    Gr_gain = 10**(20/10.)
     
    flag =0
    
    H_RU = torch.tensor(sio.loadmat('H_RU'+str(math.floor(B))+'band'+'.mat')['H_RU']).cuda()
    H_BU = torch.tensor(sio.loadmat('H_BU'+str(math.floor(B))+'band'+'.mat')['H_BU']).cuda()
    H_BR = torch.tensor(sio.loadmat('H_BR'+str(math.floor(B))+'band'+'.mat')['H_BR']).cuda()
    
    # batch = 100
    # H_RU1 = sqrt(Gr_gain*GRIS_gain)*(Channel_RIS_UE_MIMO_LOS(param_list,batch,flag)+Channel_RIS_UE_MIMO_NLOS(param_list,batch,flag))
    # H_BU = sqrt(Gt_gain*Gr_gain)*Channel_BS_UE_MIMO_NLOS(param_list,batch,flag)
    # H_BU = H_BU.reshape(batch,K,Nc,Nr,M_ant) 
    # H_RU = H_RU1.reshape(batch,K,Nc,Nr,N_ant) 
    # H_BR = sqrt(Gt_gain*GRIS_gain)*(Channel_BS_RIS_LOS_PLA(param_list,flag)+Channel_BS_RIS_NLOS(param_list,flag))
    
    batchsize = 10
    
    ebatch = H_RU.shape[0]//batchsize
    
            
    with torch.no_grad():
        model.eval()
        test_SE = 0
        for i in range(ebatch):
            H_BU1 = H_BU[i*batchsize:(i+1)*batchsize,:]
            H_RU1 = H_RU[i*batchsize:(i+1)*batchsize,:] 
            H_BR1 = H_BR
 

            Phi,F_RF,F_BB_SDMA,F_BB_RS = model(param_list, param_list1, H_BR1, H_RU1, H_BU1)

            # H_RU = H_RU.reshape(batch,K,Nc,1,N_ant)

            R_sum, Loss = loss_func(param_list, param_list1, H_BR1,H_RU1,H_BU1,Phi,F_RF,F_BB_SDMA,F_BB_RS,downsnr)
            test_SE = test_SE + R_sum

        test_SE = test_SE/ebatch  

    return test_SE

# def NFWB_RIS_TTD_test(param_list,param_list1, batch,Num_batchs,model_name):
    
#     fc = param_list[0]
#     B  = param_list[1]
#     Nc = param_list[2]
#     M1  = param_list[3]
#     M2  = param_list[4]
#     N1  = param_list[5]
#     N2  = param_list[6]
    
#     D_ant = param_list[7]
#     R1 = param_list[8]
    
#     R = param_list[9]
    
#     h1 = param_list[10]
#     h2 = param_list[11]
    
#     aod_azi = param_list[12]
#     aod_ele = param_list[13]
    
#     aoa_azi = param_list[14]
#     aoa_ele = param_list[15]
    
#     tau = param_list[16]
#     L1 = param_list[17]
    
#     L2 = param_list[18]
#     K = param_list[19]
#     Nr = param_list[20]

    
#     RF_chain = param_list1[0]
#     BS_TDDs = param_list1[1]
#     RIS_TDDs = param_list1[2]
#     L = param_list1[3]
#     UL = param_list1[4]
    
#     Bit = param_list1[5]
#     sigma = param_list1[6]
#     batch = param_list1[7]
#     upsnr = param_list1[8]
#     downsnr = param_list1[9]
    
#     M_ant = M1*M2
#     N_ant = N1*N2

#     loss_func = SE_RS()
#     loss_func = loss_func.cuda()

#     model = NFWB_RIS(param_list,param_list1).cuda()
#     model = torch.load(model_name) 
            
#     # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=0.001)
#     # optimizer = torch.optim.SGD(net_AE.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4,nesterov=True)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-07,weight_decay=1e-5)
    
#     # Gt_gain = 10**(25/10.)
#     # Gr_gain = 10**(20/10.)
    
    
    
#     Gt_gain = 10**(20/10.)
#     GRIS_gain = 10**(5/10.)

#     Gr_gain = 10**(20/10.)
     
#     flag =0
    
    
#     # batch = 100
#     # H_RU1 = sqrt(Gr_gain*GRIS_gain)*(Channel_RIS_UE_MIMO_LOS(param_list,batch,flag)+Channel_RIS_UE_MIMO_NLOS(param_list,batch,flag))
#     # H_BU = sqrt(Gt_gain*Gr_gain)*Channel_BS_UE_MIMO_NLOS(param_list,batch,flag)
#     # H_BU = H_BU.reshape(batch,K,Nc,Nr,M_ant) 
#     # H_RU = H_RU1.reshape(batch,K,Nc,Nr,N_ant) 
#     # H_BR = sqrt(Gt_gain*GRIS_gain)*(Channel_BS_RIS_LOS_PLA(param_list,flag)+Channel_BS_RIS_NLOS(param_list,flag))
    
#     # batchsize = 10
    
#     # ebatch = H_RU.shape[0]//batchsize
    
            
#     with torch.no_grad():
#         model.eval()
#         test_SE = 0
#         test_SDMA = 0
#         test_loop = 10
#         for i in range(test_loop):
#             if flag == 1:
#                 H_RU1 = Channel_RIS_UE_MIMO_NLOS(param_list,batch,flag)
#                 H_RU1 = H_RU1.reshape(batch,K,Nc,Nr,N_ant) 
#                 H_BU = Channel_BS_UE_MIMO_NLOS(param_list,batch,flag)
#                 H_BU = H_BU.reshape(batch,K,Nc,Nr,M_ant) 
#                 # H_BR = (Channel_BS_RIS_LOS(param_list,flag)+Channel_BS_RIS_NLOS(param_list,flag))
#             else:
#                 H_RU1 = sqrt(Gr_gain*GRIS_gain)*(Channel_RIS_UE_MIMO_LOS(param_list,batch,flag)+Channel_RIS_UE_MIMO_NLOS(param_list,batch,flag))
#                 H_BU = sqrt(Gt_gain*Gr_gain)*Channel_BS_UE_MIMO_NLOS(param_list,batch,flag)
#                 H_BU = H_BU.reshape(batch,K,Nc,Nr,M_ant) 
#                 H_RU1 = H_RU1.reshape(batch,K,Nc,Nr,N_ant) 
#                 H_BR = sqrt(Gt_gain*GRIS_gain)*(Channel_BS_RIS_LOS_PLA(param_list,flag)+Channel_BS_RIS_NLOS(param_list,flag))
    
#             Phi,F_RF,F_BB_SDMA,F_BB_RS = model(param_list, param_list1, H_BR, H_RU1, H_BU)
    
#             # H_RU = H_RU.reshape(batch,K,Nc,1,N_ant)
    
#             R_sum, Loss = loss_func(param_list, param_list1, H_BR,H_RU1,H_BU,Phi,F_RF,F_BB_SDMA,F_BB_RS, downsnr)
#             test_SE = test_SE + R_sum
    
#         test_SE = test_SE/test_loop

#     return test_SE

# NMSE function
# def NMSE(x, x_hat):
#     x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
#     x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
#     x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
#     x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
#     x_C = x_real  + 1j * (x_imag )
#     x_hat_C = x_hat_real  + 1j * (x_hat_imag )
#     power = np.sum(abs(x_C) ** 2, axis=1)
#     mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
#     nmse = np.mean(mse / power)
#     return nmse

# Data argumentation operations to avoid the network overfitting 
def _cutmix(im2, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = int(h*cut_ratio), int(w*cut_ratio)

    fcy = np.random.randint(0, h-ch+1)
    fcx = np.random.randint(0, w-cw+1)
    tcy, tcx = fcy, fcx
    rindex = torch.randperm(im2.size(0)).to(im2.device)

    return {
        "rindex": rindex, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
    }

def cutmixup(
    im1, im2,    
    mixup_prob=1.0, mixup_alpha=1.0,
    cutmix_prob=1.0, cutmix_alpha=1.0
):
    c = _cutmix(im2, cutmix_prob, cutmix_alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    v = np.random.beta(mixup_alpha, mixup_alpha)
    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        im2_aug = im2[rindex, :]
        im1_aug = im1[rindex, :]

    else:
        im2_aug = v * im2 + (1-v) * im2[rindex, :]
        im1_aug = v * im1 + (1-v) * im1[rindex, :]

    # apply mixup to inside or outside
    if np.random.random() > 0.5:
        im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2_aug[..., fcy:fcy+ch, fcx:fcx+cw]
        im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
    else:
        im2_aug[..., tcy:tcy+ch, tcx:tcx+cw] = im2[..., fcy:fcy+ch, fcx:fcx+cw]
        im1_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
        im2, im1 = im2_aug, im1_aug

    return im1, im2

def rgb(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2

    perm = np.random.permutation(2)

    im1 = im1[:,perm,:,:]
    im2 = im2[:,perm,:,:]

    return im1, im2

def rgb1(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2
    
    se = np.zeros(2)
    se[0]=1
    se[1]=-1
    
    r = np.random.randint(2)
    phase = se[r]
    im1[:,0,:,:] = phase*im1[:,0,:,:]
    im2[:,0,:,:] = phase*im2[:,0,:,:]
    r = np.random.randint(2)
    phase = se[r]
    im1[:,1,:,:] = phase*im1[:,1,:,:]
    im2[:,1,:,:] = phase*im2[:,1,:,:]

    return im1, im2

def cutmix(im1, im2, prob=1.0, alpha=1.0):
    c = _cutmix(im2, prob, alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2[rindex, :, fcy:fcy+ch, fcx:fcx+cw]
    im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[rindex, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]

    return im1, im2

def mixup(im1, im2, prob=1.0, alpha=1.2):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(im1.size(0)).to(im2.device)

    im1 = v * im1 + (1-v) * im1[r_index, :]
    im2 = v * im2 + (1-v) * im2[r_index, :]
    
    return im1, im2