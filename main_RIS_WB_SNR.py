# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:14:45 2021

@author: 5106
"""

# %reset -f
# import matplotlib.pyplot as plt
# plt.close("all")
# %clear

import math

import pylab
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import h5py
import torch.utils.data as Data
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from einops import rearrange
# from BFNet_position import channel_est
# from RIS_FDD1 import *
# from RIS_WB_FDD import *
# from RIS import *
# from RIS_TDD_WB_MIMO import *

flag_WB = 0
flag_TTD = 0
flag_bit = 1

if flag_TTD == 1:
    from RIS_TDD_DIR_MIMO_NFWB_SNR import *
    if flag_bit == 1:
        from RIS_TDD_DIR_MIMO_NFWB_SNR_R1 import *
else:
    from RIS_SUB_DIR_MIMO_NFWB_SNR import *
    if flag_bit == 1:
        from RIS_SUB_DIR_MIMO_NFWB_SNR_R1 import *


np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def NMSE(x, x_hat):
#     x_real = np.reshape(x[:, 0, :, :], (len(x), -1))
#     x_imag = np.reshape(x[:, 1, :, :], (len(x), -1))
#     x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
#     x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
#     x_C = x_real  + 1j * (x_imag )
#     x_hat_C = x_hat_real  + 1j * (x_hat_imag )
#     power = np.sum(abs(x_C) ** 2, axis=1)
#     mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
#     nmse = np.mean(mse / power)
#     return nmse


# model = CAN().to(device)

fc = 73e9 #150GHz  
BW = 7e9    # 总带宽10GHz  
Nc = 16    # 64个子载波
M1 = 128
M2 = 1
N1 = 16
N2 = 32
K = 1

M = M1*M2
N = N1*N2
Nr = 4

c = 3e8;  #光速
lambda_c = c/fc  


D_ant = lambda_c/2

Loc_BS = [0, 0, 5]
Loc_RIS = [0, 20, 5]
R_BR = np.sqrt((Loc_BS[0]-Loc_RIS[0])**2+(Loc_BS[1]-Loc_RIS[1])**2+(Loc_BS[2]-Loc_RIS[2])**2)

Loc_UE = [0, 15, 1]

# Loc_UE = np.zeros([3,K])

# rCell = 5
# for k in range (K):
#     r = rCell*(random.random()*2-1)
#     theta = random.random()*np.pi/2
#     Loc_UE[:,k] = [Loc_RIS[0]+r*np.cos(theta), Loc_RIS[1] + r*np.sin(theta), 1]

# R_RU = np.sqrt((Loc_UE[0]-Loc_RIS[0])**2+(Loc_UE[1]-Loc_RIS[1])**2+(Loc_UE[2]-Loc_RIS[2])**2)


# R_BR = 20
R_RU = 5


h1 = 5
h2 = 1

aod_azi_LOS = pi/4
aod_ele_LOS = pi/3

aoa_azi_LOS = pi/4
aoa_ele_LOS = pi/3

# tau = R/c
tau_max = 20e-9 
C_BR = 3
L_BR = 6
C_RU = 3
L_RU = 5
C_BU = 3
L_BU = 4


DL_overhead = N*2
DL_pilots = M*N//DL_overhead
UL_overhead = 8
UL_pilots = N//UL_overhead
feedback_ratio = 16
feedback_overhead = M*N*Nc*2//feedback_ratio

RF_chains = 4
BS_TTDs = M//8
RIS_TTDs = N//8

if flag_WB == 1:
    BS_TTDs = 0
    RIS_TTDs = 0

sigma = 1e0

batch = 16


channel_param_list = [fc,BW,Nc,M1,M2,N1,N2,D_ant,R_BR,R_RU,h1,h2,aod_azi_LOS,aod_ele_LOS, aoa_azi_LOS, aoa_ele_LOS, tau_max, L_BR, L_RU, K, Nr, L_BU, C_BU, C_BR, C_RU]
 


training_SNR = np.linspace(0,20,5)
 
uplink_snr = 10
downlink_snr = 20
sys_param_list = [RF_chains, BS_TTDs, RIS_TTDs, DL_pilots, UL_pilots, feedback_overhead, sigma, batch, uplink_snr, downlink_snr, training_SNR]
Num_batchs = 20000
# model_name = './models/RIS_NFWB_'+str(UL_pilots)+'pilots_'+str(floor(uplink_snr))+'UdB'+str(floor(downlink_snr))+'DdB'+'.pth'
# model_name = './models/RIS_NFWB_TTD_'+str(floor(BW))+'band'+str(floor(BS_TTDs))+'BS_TTDs_'+str(floor(RIS_TTDs))+'RIS_TTDs'+str(UL_pilots)+'pilots_'+str(floor(uplink_snr))+'UdB'+str(floor(downlink_snr))+'DdB'+'.pth'
#model_name = './models/RIS_NFWB_VIR_'+str(floor(BW))+'band'+str(floor(BS_TTDs))+'BS_TTDs_'+str(UL_pilots)+'pilots_'+str(floor(uplink_snr))+'UdB'+str(floor(downlink_snr))+'DdB'+'.pth'
# model_name = './models/RIS_NFWB_TTD_SNR_'+str(floor(BW))+'band'+str(floor(BS_TTDs))+'BS_TTDs_'+str(floor(RIS_TTDs))+'RIS_TTDs'+str(UL_pilots)+'pilots_'+str(floor(uplink_snr))+'UdB'+str(floor(downlink_snr))+'DdB'+'.pth'


D_snrs = np.linspace(-10,30,9)
SE = np.zeros([len(D_snrs),1])

if flag_TTD ==1:
    model_name = 'RIS_NFWB_TTD_'+str(R_RU)+'m'+str(floor(BW))+'band'+str(floor(BS_TTDs))+'BS_TTDs_'+str(floor(RIS_TTDs))+'RIS_TTDs'+str(UL_pilots)+'pilots_'+str(floor(uplink_snr))+'UdB'+str(floor(downlink_snr))+'DdB'+'.pth'
    best_SE = NFWB_RIS_TTD(channel_param_list,sys_param_list, batch,Num_batchs,model_name)
else:
    model_name = 'RIS_NFWB_VIR_'+str(R_RU)+'m'+str(floor(BW))+'band'+str(floor(BS_TTDs))+'BS_TTDs_'+str(UL_pilots)+'pilots_'+str(floor(uplink_snr))+'UdB'+str(floor(downlink_snr))+'DdB'+'.pth'
    best_SE = NFWB_RIS_VIR(channel_param_list,sys_param_list, batch,Num_batchs,model_name)

flag_DownSNR = 1

# training_SNR = np.linspace(20,20,1)

if flag_DownSNR ==1:
  
    for d_snr in range(len(D_snrs)):
    
        uplink_snr = [10]
        downlink_snr = D_snrs[d_snr]
        # downlink_snr = 20
        
        sys_param_list = [RF_chains, BS_TTDs, RIS_TTDs, DL_pilots, UL_pilots, feedback_overhead, sigma, batch, uplink_snr, downlink_snr, uplink_snr]
        if flag_TTD ==1:
            model_name_SE = 'SE1_'+str(R_RU)+'m'+str(floor(BW))+'band'+str(UL_pilots)+'pilots_'+str(floor(BS_TTDs))+'BS_TTDs_'+str(floor(RIS_TTDs))+'RIS_TTDs'+str(floor(uplink_snr[0]))+'UdB'+'.mat'
            SE_gpu = NFWB_RIS_TTD_test(channel_param_list,sys_param_list, batch, Num_batchs, model_name)
        else:
            model_name_SE = 'SE1_'+str(R_RU)+'m'+str(floor(BW))+'band'+str(UL_pilots)+'pilots_'+str(floor(BS_TTDs))+'BS_TTDs_'+str(floor(Nc))+'RIS_sub'+str(floor(uplink_snr[0]))+'UdB'+'.mat'
            SE_gpu = NFWB_RIS_VIR_test(channel_param_list,sys_param_list, batch, Num_batchs, model_name)
        
        loss_func = SE_RS()
        loss_func = loss_func.cuda()
        
        SE[d_snr] = SE_gpu.cpu().detach().numpy()
        
else:
    
    for d_snr in range(len(D_snrs)):
    
        uplink_snr = [D_snrs[d_snr]]
        downlink_snr = 20
        
        sys_param_list = [RF_chains, BS_TTDs, RIS_TTDs, DL_pilots, UL_pilots, feedback_overhead, sigma, batch, uplink_snr, downlink_snr, uplink_snr]
        if flag_TTD ==1:
            model_name_SE = 'SE1_UP_'+str(R_RU)+'m'+str(floor(BW))+'band'+str(UL_pilots)+'pilots_'+str(floor(BS_TTDs))+'BS_TTDs_'+str(floor(RIS_TTDs))+'RIS_TTDs'+str(floor(downlink_snr))+'DdB'+'.mat'
            SE_gpu = NFWB_RIS_TTD_test(channel_param_list,sys_param_list, batch, Num_batchs, model_name)
        else:
            model_name_SE = 'SE1_UP_'+str(R_RU)+'m'+str(floor(BW))+'band'+str(UL_pilots)+'pilots_'+str(floor(BS_TTDs))+'BS_TTDs_'+str(floor(Nc))+'RIS_sub'+str(floor(downlink_snr))+'DdB'+'.mat'
            SE_gpu = NFWB_RIS_VIR_test(channel_param_list,sys_param_list, batch, Num_batchs, model_name)
        SE[d_snr] = SE_gpu.cpu().detach().numpy()
            
import scipy.io as sio # mat
sio.savemat(model_name_SE, {'a':SE})

plt.plot(D_snrs, SE, ls='-', marker='+', c='black', label='NF-WB')
plt.legend()
plt.grid(True) 
plt.xlabel('SNR/dB')
plt.ylabel('Spectral efficiency (bit/s/Hz)')
plt.show()
