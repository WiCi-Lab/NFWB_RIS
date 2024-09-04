# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:18:39 2023

@author: WiCi
"""

import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二块GPU（从0开始）
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import torch.nn.functional as F
import torchvision
import numpy as np
from math import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
from IPython import display
import torch.utils.data as Data
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from scipy.linalg import block_diag
import datetime
from torch.nn.utils import *
from Transformer_model import *

import random


def Hermitian(X):#torch矩阵共轭转置
    X = torch.real(X) - 1j*torch.imag(X)
    return X.transpose(-1,-2)
def kron(a,b):
    #a与b维度为[batch,N],输出应为[batch,N*N]
    batch = a.shape[0]
    a = a.reshape(batch,-1,1)
    b = b.reshape(batch,1,-1)
    c = a @ b
    return c.reshape(batch,-1) #输出的维度a在前，b在后
def kron_add(a,b):
    #a与b维度为[batch,N],输出应为[batch,N*N]
    batch = a.shape[0]
    a = a.reshape(batch,-1,1)
    b = b.reshape(batch,1,-1)
    c = a + b
    return c.reshape(batch,-1) #输出的维度a在前，b在后

def radius(a,b):
    r = 1+(a-b)**2/(a+b)**2
    return r

def ModelAbs(f):
    
    f = f / 1e9
    
    gamma_o = (3.02e-4/(1+1.9e-5*f**1.5)+0.283/((f-118.75)**2+2.91))*f**2*1e-3 - 0.00306
    
    n1 = 0.955 + 0.006*7.5;
    n2 = 0.735 + 0.0353*7.5;   
    gamma_w = (3.98*n1*radius(f,22)/((f-22.235)**2 + 9.42*n1**2) + 11.96*n1/((f-183.31)**2 + 11.14*n1**2)+
               0.081*n1/((f-321.226)**2+6.29*n1**2) + 3.66*n1/((f-235.153)**2+9.22*n1**2) + 
               25.37*n1/(f-380)**2 + 17.4*n1/(f-448)**2 +844.6*n1*radius(f,557)/(f-557)**2 + 
               290*n1*radius(f,752)/(f-752)**2 + 8.3328e4*n2*radius(f,1780)/(f-1780)**2) * f**2 * 7.5e-4
    
    gamma = gamma_o + gamma_w

    
    return gamma

### BS-RIS信道
def Channel_BS_RIS_LOS(param_list,flag):
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
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
    
    L = param_list[18]
    K = param_list[19]
    
    m1 = np.arange(-(M1-1)/2,(M1-1)/2+1,1)
    m2 = np.arange(-(M2-1)/2,(M2-1)/2+1,1)
    
    n1 = np.arange(-(N1-1)/2,(N1-1)/2+1,1)
    n2 = np.arange(-(N2-1)/2,(N2-1)/2+1,1)
    
    M = M1 * M2    
    N = N1 * N2
    
    # G_UPA_LOS_WB = torch.zeros(Nc, M, N).cuda()
    # H_UPA2 = torch.zeros(N2,M2).cuda()
    # H_UPA1 = torch.zeros(N1,M1).cuda()
    
    G_UPA_LOS_WB = torch.zeros(Nc, N, M).to(torch.complex64)
    H_UPA2 = torch.zeros(N2,M2).to(torch.complex64)
    H_UPA1 = torch.zeros(N1,M1).to(torch.complex64)


    n = (torch.arange(0, Nc)+1)
    nn = -(Nc+1)/2
    deta = B/Nc
    fm=(nn+n)*deta+fc
    
    c = 3e8
    lambda_c = c/fc;
    d= D_ant
    d_BR = R
    
    
    for nc in range(Nc):
        response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m1*d*cos(aod_azi)*sin(aod_ele)
                                                    +m1**2*d**2*(1-cos(aod_azi)**2*sin(aod_ele)**2)/2*R))
        response_UPA2_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m2*d*cos(aod_ele )
                                                    +m2**2*d**2*sin(aod_ele)**2)/2*R)
    
        response_UPA_BR = torch.kron(response_UPA1_BR, response_UPA2_BR)
        
        response_UPA1_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n1*d*cos(aoa_azi )*sin(aoa_ele)
                                                    +n1**2*d**2*(1-cos(aoa_azi)**2*sin(aoa_ele)**2)/2*R))
        response_UPA2_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n2*d*cos(aoa_ele )
                                                    +n2**2*d**2*sin(aoa_ele)**2)/2*R)
    
        response_UPA_RIS = torch.kron(response_UPA1_RIS, response_UPA2_RIS)
        
        inx = 0;
        
        for nx in range(N2):
            H_UPA2[nx,:] = torch.exp(-1j*2*pi*fm[nc]/c/d_BR*m2*d*n2[nx]*d*(1-(cos(aoa_azi)**2)*sin(aoa_ele)**2))
            
        for ny in range(N1):
            H_UPA1[ny,:] = torch.exp(-1j*2*pi*fm[nc]/c/d_BR*m1*d*n1[ny]*d*sin(aoa_ele)**2)
            
        H_UPA = torch.kron(H_UPA1,H_UPA2)
        
        absorption = ModelAbs(fm[nc]);
        if flag ==1:
            beta_pk = torch.tensor(1)
        else:
            beta_pk = c/(4*pi*fm[nc]*R)*torch.exp(-1/2*absorption*R)*torch.exp(-1j*2*pi*tau*fm[nc]);
        
        # G_UPA_LOS_WB = torch.unsqueeze(response_UPA_RIS,1)@torch.unsqueeze(response_UPA_BR,0)
   
        G_UPA_LOS_WB[nc,:,:] = torch.sqrt(beta_pk)*torch.unsqueeze(response_UPA_RIS,1)@torch.unsqueeze(response_UPA_BR,0)*H_UPA

    return G_UPA_LOS_WB.cuda()

def Channel_BS_RIS_LOS_PLA(param_list,flag):
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
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
    
    L = param_list[18]
    K = param_list[19]
    
    m1 = np.arange(-(M1-1)/2,(M1-1)/2+1,1)
    m2 = np.arange(-(M2-1)/2,(M2-1)/2+1,1)
    
    n1 = np.arange(-(N1-1)/2,(N1-1)/2+1,1)
    n2 = np.arange(-(N2-1)/2,(N2-1)/2+1,1)
    
    M = M1 * M2    
    N = N1 * N2
    
    m = np.arange(-(M-1)/2,(M-1)/2+1,1)
    
    # G_UPA_LOS_WB = torch.zeros(Nc, M, N).cuda()
    # H_UPA2 = torch.zeros(N2,M2).cuda()
    # H_UPA1 = torch.zeros(N1,M1).cuda()
    
    G_UPA_LOS_WB = torch.zeros(Nc, N, M).to(torch.complex64)
    H_UPA2 = torch.zeros(N2,1).to(torch.complex64)
    H_UPA1 = torch.zeros(N1,M).to(torch.complex64)


    n = (torch.arange(0, Nc)+1)
    nn = -(Nc+1)/2
    deta = B/Nc
    fm=(nn+n)*deta+fc
    
    c = 3e8
    lambda_c = c/fc;
    d= D_ant
    d_BR = R
    
    
    for nc in range(Nc):
        
        # response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m1*d*cos(aod_azi)*sin(aod_ele)
        #                                             +m1**2*d**2*(1-cos(aod_azi)**2*sin(aod_ele)**2)/2*R))
        # response_UPA2_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m2*d*cos(aod_ele )
        #                                             +m2**2*d**2*sin(aod_ele)**2)/2*R)
    
        # response_UPA_BR = torch.kron(response_UPA1_BR, response_UPA2_BR)
        
        # response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m*d*cos(aod_azi)))
    
        response_ULA_BR = torch.exp(-1j*2*pi*fm[nc]/c*(m*d*cos(aod_azi)
                                                    +m**2*d**2*sin(aod_azi)**2)/2*R)
        
        response_UPA1_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n1*d*cos(aoa_azi )*sin(aoa_ele)
                                                    +n1**2*d**2*(1-cos(aoa_azi)**2*sin(aoa_ele)**2)/2*R))
        response_UPA2_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n2*d*cos(aoa_ele )
                                                    +n2**2*d**2*sin(aoa_ele)**2)/2*R)
    
        response_UPA_RIS = torch.kron(response_UPA1_RIS, response_UPA2_RIS)
        
        inx = 0;
        
        for nx in range(N2):
            H_UPA2[nx,:] = torch.exp(-1j*2*pi*fm[nc]/c/d_BR*n2[nx]*d*(1-(cos(aoa_azi)**2)*sin(aoa_ele)**2))
            
        for ny in range(N1):
            H_UPA1[ny,:] = torch.exp(-1j*2*pi*fm[nc]/c/d_BR*m*d*n1[ny]*d*sin(aoa_ele)**2)
            
        H_UPA = torch.kron(H_UPA1,H_UPA2)
        
        absorption = ModelAbs(fm[nc]);
        if flag ==1:
            beta_pk = torch.tensor(1)
        else:
            beta_pk = c/(4*pi*fm[nc]*R)*torch.exp(-1/2*absorption*R)*torch.exp(-1j*2*pi*tau*fm[nc]);
        
        # G_UPA_LOS_WB = torch.unsqueeze(response_UPA_RIS,1)@torch.unsqueeze(response_UPA_BR,0)
   
        G_UPA_LOS_WB[nc,:,:] = torch.sqrt(beta_pk)*torch.unsqueeze(response_UPA_RIS,1)@torch.unsqueeze(response_ULA_BR,0)*H_UPA

    return G_UPA_LOS_WB.cuda()

def Channel_BS_RIS_NLOS(param_list,flag):
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
    fc = param_list[0]
    B  = param_list[1]
    Nc = param_list[2]
    M1  = param_list[3]
    M2  = param_list[4]
    N1  = param_list[5]
    N2  = param_list[6]
    
    D_ant = param_list[7]
    R = param_list[8]
    
    R1 = param_list[9]
    
    h1 = param_list[10]
    h2 = param_list[11]
    
    aod_azi = param_list[12]
    aod_ele = param_list[13]
    
    aoa_azi = param_list[14]
    aoa_ele = param_list[15]
    
    tau = param_list[16]
    L_BR = param_list[17]
    
    L1 = param_list[18]
    K = param_list[19]
    
    m1 = np.arange(-(M1-1)/2,(M1-1)/2+1,1)
    m2 = np.arange(-(M2-1)/2,(M2-1)/2+1,1)
    
    n1 = np.arange(-(N1-1)/2,(N1-1)/2+1,1)
    n2 = np.arange(-(N2-1)/2,(N2-1)/2+1,1)
    
    M = M1 * M2    
    N = N1 * N2
    
    m = np.arange(-(M-1)/2,(M-1)/2+1,1)
    
    # G_UPA_LOS_WB = torch.zeros(Nc, M, N).cuda()
    # H_UPA2 = torch.zeros(N2,M2).cuda()
    # H_UPA1 = torch.zeros(N1,M1).cuda()
    
    G_UPA_NLOS_WB = torch.zeros(Nc, N, M).to(torch.complex64)


    n = (torch.arange(0, Nc)+1)
    nn = -(Nc+1)/2
    deta = B/Nc
    fm=(nn+n)*deta+fc
    
    c = 3e8
    lambda_c = c/fc;
    d= D_ant
    d_BR = R
    
    L_BU = param_list[21]
    C_BU = param_list[22]
    C_BR = param_list[23]
    C_RU = param_list[24]
    
    sigma_bs = 5 * pi / 180
    sigma_mt = 5 * pi / 180
    
    Theta1 = np.random.rand(C_BR,1)*pi - pi/2
    Theta2 = np.random.rand(C_BR,1)*pi - pi/2

    sigma=sigma_bs 
    b=sigma/np.sqrt(2)   
    a=np.random.rand(C_BR,L_BR)-0.5 
    dTheta1 = - b*np.sign(a)*np.log(1-2*abs(a))
    dTheta2 = - b*np.sign(a)*np.log(1-2*abs(a))

    
    sigma_mt = 5 * pi / 180
    Alpha = np.random.rand(C_BR,1)*pi - pi/2
    sigma=sigma_mt 
    b=sigma/np.sqrt(2)   
    a=np.random.rand(C_RU,L_BR)-0.5 
    dAlpha = - b*np.sign(a)*np.log(1-2*abs(a))
    
    
    for nc in range(Nc):
        G_UPA_NLOS_WB_lp = 0
        for cl in range(C_BR): 
            for lp in range(L_BR):
                
                # sector_angle = 10
                # aod_azi_lp = aod_azi + random.randint(-1*sector_angle, 1*sector_angle)*aod_azi/sector_angle/2
                # aod_ele_lp = aod_ele + random.randint(-1*sector_angle, 1*sector_angle)*aod_ele/sector_angle/2
                # aoa_azi_lp = aoa_azi + random.randint(-1*sector_angle, 1*sector_angle)*aoa_azi/sector_angle/2
                # aoa_ele_lp = aoa_ele + random.randint(-1*sector_angle, 1*sector_angle)*aoa_ele/sector_angle/2
                
                aod_azi_lp = Alpha[cl] + dAlpha[cl,lp]
                aoa_azi_lp = Theta1[cl] + dTheta1[cl,lp]
                aoa_ele_lp = Theta2[cl] + dTheta2[cl,lp]
                
                sector_R = R/3
                # R1_lp = R + random.uniform(-1, 1)*sector_R
                R1_lp = R - random.random()*sector_R
                R2_lp = R - sector_R*random.random()
                
                tau = c/(R1_lp+R2_lp)
    
                
                # response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m1*d*cos(aod_azi_lp)*sin(aod_ele_lp)
                #                                             +m1**2*d**2*(1-cos(aod_azi)**2*sin(aod_ele)**2)/2*R1_lp))
                # response_UPA2_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m2*d*cos(aod_ele_lp)
                #                                             +m2**2*d**2*sin(aod_ele_lp)**2)/2*R1_lp)
            
                # response_UPA_BR = torch.kron(response_UPA1_BR, response_UPA2_BR)
            
                response_UPA_BR = torch.exp(-1j*2*pi*fm[nc]/c*(m*d*cos(aod_azi_lp[0])
                                                            +m**2*d**2*sin(aod_azi_lp[0])**2)/2*R1_lp)
                
                response_UPA1_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n1*d*cos(aoa_azi_lp[0] )*sin(aoa_ele_lp[0])
                                                            +n1**2*d**2*(1-cos(aoa_azi_lp[0])**2*sin(aoa_ele_lp[0])**2)/2*R2_lp))
                response_UPA2_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n2*d*cos(aoa_ele_lp[0])
                                                            +n2**2*d**2*sin(aoa_ele_lp[0])**2)/2*R2_lp)
            
                response_UPA_RIS = torch.kron(response_UPA1_RIS, response_UPA2_RIS)
                
                absorption = ModelAbs(fm[nc]);
                if flag ==1:
                    beta_pk = torch.tensor(1)
                else:
                    beta_pk = c/(4*pi*fm[nc]*(R1_lp+R2_lp))*torch.exp(-1/2*absorption*(R1_lp+R2_lp))*torch.exp(-1j*2*pi*tau*fm[nc]);
                
                G_UPA_NLOS_WB_lp = G_UPA_NLOS_WB_lp+torch.sqrt(beta_pk)*torch.unsqueeze(response_UPA_RIS,1)@torch.unsqueeze(response_UPA_BR,0)
           
        G_UPA_NLOS_WB[nc,:,:] = G_UPA_NLOS_WB_lp/sqrt(C_BR*L_BR)

    return G_UPA_NLOS_WB.cuda()

def Channel_BS_RIS_NLOS_FF(param_list,flag):
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
    fc = param_list[0]
    B  = param_list[1]
    Nc = param_list[2]
    M1  = param_list[3]
    M2  = param_list[4]
    N1  = param_list[5]
    N2  = param_list[6]
    
    D_ant = param_list[7]
    R = param_list[8]
    
    R1 = param_list[9]
    
    h1 = param_list[10]
    h2 = param_list[11]
    
    aod_azi = param_list[12]
    aod_ele = param_list[13]
    
    aoa_azi = param_list[14]
    aoa_ele = param_list[15]
    
    tau = param_list[16]
    L = param_list[17]
    
    L1 = param_list[18]
    K = param_list[19]
    Nr = param_list[20]
    L_FF = param_list[21]
    
    m1 = np.arange(-(M1-1)/2,(M1-1)/2+1,1)
    m2 = np.arange(-(M2-1)/2,(M2-1)/2+1,1)
    
    n1 = np.arange(-(N1-1)/2,(N1-1)/2+1,1)
    n2 = np.arange(-(N2-1)/2,(N2-1)/2+1,1)
    
    M = M1 * M2    
    N = N1 * N2
    
    # G_UPA_LOS_WB = torch.zeros(Nc, M, N).cuda()
    # H_UPA2 = torch.zeros(N2,M2).cuda()
    # H_UPA1 = torch.zeros(N1,M1).cuda()
    
    G_UPA_NLOS_WB = torch.zeros(Nc, N, M).to(torch.complex64)


    n = (torch.arange(0, Nc)+1)
    nn = -(Nc+1)/2
    deta = B/Nc
    fm=(nn+n)*deta+fc
    
    c = 3e8
    lambda_c = c/fc;
    d= D_ant
    d_BR = R
    
    
    for nc in range(Nc):
        G_UPA_NLOS_WB_lp = 0
        for lp in range(L_FF):
            
            sector_angle = 10
            aod_azi_lp = aod_azi + random.randint(-1*sector_angle, 1*sector_angle)*aod_azi/sector_angle/2
            aod_ele_lp = aod_ele + random.randint(-1*sector_angle, 1*sector_angle)*aod_ele/sector_angle/2
            aoa_azi_lp = aoa_azi + random.randint(-1*sector_angle, 1*sector_angle)*aoa_azi/sector_angle/2
            aoa_ele_lp = aoa_ele + random.randint(-1*sector_angle, 1*sector_angle)*aoa_ele/sector_angle/2
            
            sector_R = R/3
            # R1_lp = R + random.uniform(-1, 1)*sector_R
            R1_lp = R - random.random()*sector_R
            R2_lp = R - sector_R*random.random()
            
            tau = c/(R1_lp+R2_lp)

            
            response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(m1*d*cos(aod_azi_lp)*sin(aod_ele_lp)))
            response_UPA2_BR = torch.exp(-1j*2*pi*fm[nc]/c*(m2*d*cos(aod_ele_lp)))
        
            response_UPA_BR = torch.kron(response_UPA1_BR, response_UPA2_BR)
            
            response_UPA1_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n1*d*cos(aoa_azi_lp )*sin(aoa_ele_lp)))
            response_UPA2_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(n2*d*cos(aoa_ele_lp)))
        
            response_UPA_RIS = torch.kron(response_UPA1_RIS, response_UPA2_RIS)
            
            absorption = ModelAbs(fm[nc]);
            if flag ==1:
                beta_pk = torch.tensor(1)
            else:
                beta_pk = c/(4*pi*fm[nc]*(R1_lp+R2_lp))*torch.exp(-1/2*absorption*(R1_lp+R2_lp))*torch.exp(-1j*2*pi*tau*fm[nc]);
            
            G_UPA_NLOS_WB_lp = G_UPA_NLOS_WB_lp+torch.sqrt(beta_pk)*torch.unsqueeze(response_UPA_RIS,1)@torch.unsqueeze(response_UPA_BR,0)
       
        G_UPA_NLOS_WB[nc,:,:] = G_UPA_NLOS_WB_lp/sqrt(L)

    return G_UPA_NLOS_WB.cuda()

### UE-RIS信道

def Channel_RIS_UE_MIMO_LOS(param_list,batch,flag):
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
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
    
    L = param_list[18]
    K = param_list[19]
    Nr = param_list[20]
    
    L_BU = param_list[21]
    C_BU = param_list[22]
    C_BR = param_list[23]
    C_RU = param_list[23]

    
    m1 = np.arange(-(Nr-1)/2,(Nr-1)/2+1,1)
    
    n1 = np.arange(-(N1-1)/2,(N1-1)/2+1,1)
    n2 = np.arange(-(N2-1)/2,(N2-1)/2+1,1)
    
    M = M1 * M2    
    N = N1 * N2
    
    # G_UPA_LOS_WB = torch.zeros(Nc, M, N).cuda().to(torch.complex64)
    H_UPA2 = torch.zeros(N2,1).to(torch.complex64)
    H_UPA1 = torch.zeros(N1,Nr).to(torch.complex64)
    
    G_ULA_LOS_WB = torch.zeros(batch,Nc, Nr, N ).to(torch.complex64)


    n = (torch.arange(0, Nc)+1)
    nn = -(Nc+1)/2
    deta = B/Nc
    fm=(nn+n)*deta+fc
    
    c = 3e8
    lambda_c = c/fc;
    d= D_ant
    d_BR = R
    
    for ib in range(batch):
        
        sector_angle = 10
        # aod_azi_lp = aod_azi + random.randint(-1*sector_angle, 1*sector_angle)*aod_azi/sector_angle/2
        # aod_ele_lp = aod_ele + random.randint(-1*sector_angle, 1*sector_angle)*aod_ele/sector_angle/2
        # aoa_azi_lp = aoa_azi + random.randint(-1*sector_angle, 1*sector_angle)*aoa_azi/sector_angle/2
        # aoa_ele_lp = aoa_ele + random.randint(-1*sector_angle, 1*sector_angle)*aoa_ele/sector_angle/2
        # sector_R = R/5
        # R = R + random.uniform(-1, 1)*sector_R
        aod_azi_lp = aod_azi
        aod_ele_lp = aod_ele
        aoa_azi_lp = aoa_azi
        aoa_ele_lp = aoa_ele
        
        tau = c/R
        
        for nc in range(Nc):
            
            
            # response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m1*d*cos(aod_azi_lp)
            #                                             +m1**2*d**2*sin(aod_ele)**2/2*R1_lp))
            
            # response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m1*d*cos(aoa_azi_lp)))
            
            response_ULA_BR = torch.exp(-1j*2*pi*fm[nc]/c*(m1*d*cos(aod_azi_lp)
                                                        +m1**2*d**2*sin(aod_azi_lp)**2)/2*R)
        
            response_ULA_BR = response_ULA_BR
            
            response_UPA1_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n1*d*cos(aoa_azi_lp )*sin(aoa_ele_lp)
                                                        +n1**2*d**2*(1-cos(aoa_azi_lp)**2*sin(aoa_ele_lp)**2)/2*R))
            response_UPA2_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n2*d*cos(aoa_ele_lp )
                                                        +n2**2*d**2*sin(aoa_ele_lp)**2)/2*R)
        
            response_UPA_RIS = torch.kron(response_UPA1_RIS, response_UPA2_RIS)
            
            inx = 0;
            
            for nx in range(N2):
                H_UPA2[nx,:] = torch.exp(-1j*2*pi*fm[nc]/c/d_BR*n2[nx]*d*(1-(cos(aoa_azi_lp)**2)*sin(aoa_ele_lp)**2))
                
            for ny in range(N1):
                H_UPA1[ny,:] = torch.exp(-1j*2*pi*fm[nc]/c/d_BR*m1*d*n1[ny]*d*sin(aoa_ele_lp)**2)
                
            H_UPA = torch.kron(H_UPA1,H_UPA2)
            
            absorption = ModelAbs(fm[nc]);
            if flag ==1:
                beta_pk = torch.tensor(1)
            else:
                beta_pk = c/(4*pi*fm[nc]*R)*torch.exp(-1/2*absorption*R)*torch.exp(-1j*2*pi*tau*fm[nc]);
            
            # G_UPA_LOS_WB = torch.unsqueeze(response_UPA_RIS,1)@torch.unsqueeze(response_UPA_BR,0)
       
            G_ULA_LOS_WB[ib,nc,:,:] = torch.sqrt(beta_pk)*torch.unsqueeze(response_ULA_BR,1)@torch.unsqueeze(response_UPA_RIS,0)*torch.transpose(H_UPA,0,1)


    return G_ULA_LOS_WB.cuda()

# def Channel_RIS_UE_MIMO_NLOS(param_list,batch,flag):
#     #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
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
    
#     L = param_list[18]
#     K = param_list[19]
#     Nr = param_list[20]
    
#     L_BU = param_list[21]
#     C_BU = param_list[22]
#     C_BR = param_list[23]
#     C_RU = param_list[24]
    
#     m1 = np.arange(-(Nr-1)/2,(Nr-1)/2+1,1)
    
#     n1 = np.arange(-(N1-1)/2,(N1-1)/2+1,1)
#     n2 = np.arange(-(N2-1)/2,(N2-1)/2+1,1)
    
#     M = M1 * M2    
#     N = N1 * N2
    
#     # G_UPA_LOS_WB = torch.zeros(Nc, M, N).cuda()
#     # H_UPA2 = torch.zeros(N2,M2).cuda()
#     # H_UPA1 = torch.zeros(N1,M1).cuda()
    
#     G_ULA_NLOS_WB = torch.zeros(batch,Nc, Nr, N ).to(torch.complex64)


#     n = (torch.arange(0, Nc)+1)
#     nn = -(Nc+1)/2
#     deta = B/Nc
#     fm=(nn+n)*deta+fc
    
#     c = 3e8
#     lambda_c = c/fc;
#     d= D_ant
#     d_BR = R
    
#     # pi = math.pi
#     tau_max = c/(R)
#     sigma_t = 0.06e-9
#     # C_BR = 4
#     # L_BR = 10
#     # C_RU = 4
#     # L_RU = 8
#     # C_BU = 4
#     # L_BU = 6
    
#     sigma_bs = 5 * pi / 180
#     sigma_mt = 5 * pi / 180
    
#     Theta1 = np.random.rand(C_RU,1)*pi - pi/2
#     Theta2 = np.random.rand(C_RU,1)*pi - pi/2

#     sigma=sigma_bs 
#     b=sigma/np.sqrt(2)   
#     a=np.random.rand(C_RU,L_RU)-0.5 
#     dTheta1 = - b*np.sign(a)*np.log(1-2*abs(a))
#     dTheta2 = - b*np.sign(a)*np.log(1-2*abs(a))

    
#     sigma_mt = 5 * pi / 180
#     Alpha = np.random.rand(C_RU,1)*pi - pi/2
#     sigma=sigma_mt 
#     b=sigma/np.sqrt(2)   
#     a=np.random.rand(C_RU,L_RU)-0.5 
#     dAlpha = - b*np.sign(a)*np.log(1-2*abs(a))
    
#     Delay = np.random.rand(C_RU,1)*tau_max
#     sigma=sigma_t 
#     b=sigma/np.sqrt(2)   
#     a=np.random.rand(C_RU,L_RU)-0.5 
#     dDelay = - b*np.sign(a)*np.log(1-2*abs(a))
    
#     for ib in range(batch):
#         for nc in range(Nc):
#             G_ULA_NLOS_WB_lp = 0
#             for lp in range(L):
                
#                 sector_angle = 10
#                 aod_azi_lp = aod_azi + random.randint(-1*sector_angle, 1*sector_angle)*aod_azi/sector_angle/2
#                 aod_ele_lp = aod_ele + random.randint(-1*sector_angle, 1*sector_angle)*aod_ele/sector_angle/2
#                 aoa_azi_lp = aoa_azi + random.randint(-1*sector_angle, 1*sector_angle)*aoa_azi/sector_angle/2
#                 aoa_ele_lp = aoa_ele + random.randint(-1*sector_angle, 1*sector_angle)*aoa_ele/sector_angle/2
                
#                 sector_R = R/3
#                 # R1_lp = R + random.uniform(-1, 1)*sector_R
#                 R1_lp = R - random.random()*sector_R
#                 R2_lp = R - sector_R*random.random()
                
#                 tau = c/(R1_lp+R2_lp)
    
                
#                 # response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m1*d*cos(aod_azi_lp)
#                 #                                             +m1**2*d**2*sin(aod_ele)**2/2*R1_lp))
                
#                 response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m1*d*cos(aoa_azi_lp)))
            
#                 response_ULA_BR = response_UPA1_BR
                
#                 response_UPA1_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n1*d*cos(aod_azi_lp )*sin(aod_ele_lp)
#                                                             +n1**2*d**2*(1-cos(aod_azi_lp)**2*sin(aod_ele_lp)**2)/2*R2_lp))
#                 response_UPA2_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n2*d*cos(aod_ele_lp)
#                                                             +n2**2*d**2*sin(aod_ele_lp)**2)/2*R1_lp)
            
#                 response_UPA_RIS = torch.kron(response_UPA1_RIS, response_UPA2_RIS)
                
#                 absorption = ModelAbs(fm[nc]);
#                 if flag ==1:
#                     beta_pk = torch.tensor(1)
#                 else:
#                     beta_pk = c/(4*pi*fm[nc]*(R1_lp+R2_lp))*torch.exp(-1/2*absorption*(R1_lp+R2_lp))*torch.exp(-1j*2*pi*tau*fm[nc]);
                
#                 G_ULA_NLOS_WB_lp = G_ULA_NLOS_WB_lp+torch.sqrt(beta_pk)*torch.unsqueeze(response_ULA_BR,1)@torch.unsqueeze(response_UPA_RIS,0)
       
#             G_ULA_NLOS_WB[ib, nc,:,:] = G_ULA_NLOS_WB_lp/sqrt(L)

#     return G_ULA_NLOS_WB.cuda()

def Channel_RIS_UE_MIMO_NLOS(param_list,batch,flag):
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
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
    
    L_RU = param_list[18]
    K = param_list[19]
    Nr = param_list[20]
    
    L_BU = param_list[21]
    C_BU = param_list[22]
    C_BR = param_list[23]
    C_RU = param_list[24]
    
    m1 = np.arange(-(Nr-1)/2,(Nr-1)/2+1,1)
    
    n1 = np.arange(-(N1-1)/2,(N1-1)/2+1,1)
    n2 = np.arange(-(N2-1)/2,(N2-1)/2+1,1)
    
    M = M1 * M2    
    N = N1 * N2
    
    # G_UPA_LOS_WB = torch.zeros(Nc, M, N).cuda()
    # H_UPA2 = torch.zeros(N2,M2).cuda()
    # H_UPA1 = torch.zeros(N1,M1).cuda()
    
    G_ULA_NLOS_WB = torch.zeros(batch,Nc, Nr, N ).to(torch.complex64)


    n = (torch.arange(0, Nc)+1)
    nn = -(Nc+1)/2
    deta = B/Nc
    fm=(nn+n)*deta+fc
    
    c = 3e8
    lambda_c = c/fc;
    d= D_ant
    d_BR = R
    
    # pi = math.pi
    tau_max = c/(R)
    sigma_t = 0.06e-9
    # C_BR = 4
    # L_BR = 10
    # C_RU = 4
    # L_RU = 8
    # C_BU = 4
    # L_BU = 6
    
    sigma_bs = 5 * pi / 180
    sigma_mt = 5 * pi / 180
    
    Theta1 = np.random.rand(C_RU,1)*pi - pi/2
    Theta2 = np.random.rand(C_RU,1)*pi - pi/2

    sigma=sigma_bs 
    b=sigma/np.sqrt(2)   
    a=np.random.rand(C_RU,L_RU)-0.5 
    dTheta1 = - b*np.sign(a)*np.log(1-2*abs(a))
    dTheta2 = - b*np.sign(a)*np.log(1-2*abs(a))

    
    sigma_mt = 5 * pi / 180
    Alpha = np.random.rand(C_RU,1)*pi - pi/2
    sigma=sigma_mt 
    b=sigma/np.sqrt(2)   
    a=np.random.rand(C_RU,L_RU)-0.5 
    dAlpha = - b*np.sign(a)*np.log(1-2*abs(a))
    
    # Delay = np.random.rand(C_RU,1)*tau_max
    # sigma=sigma_t 
    # b=sigma/np.sqrt(2)   
    # a=np.random.rand(C_RU,L_RU)-0.5 
    # dDelay = - b*np.sign(a)*np.log(1-2*abs(a))
    
    for ib in range(batch):
        for nc in range(Nc):
            G_ULA_NLOS_WB_lp = 0
            for cl in range(C_RU):
                for lp in range(L_RU):
                    
                    sector_angle = 10
                    aod_azi_lp = Theta1[cl] + dTheta1[cl,lp]
                    aod_ele_lp = Theta2[cl] + dTheta2[cl,lp]
                    aoa_azi_lp = Alpha[cl] + dAlpha[cl,lp]
                    aoa_ele_lp = aoa_ele + random.randint(-1*sector_angle, 1*sector_angle)*aoa_ele/sector_angle/2
                    
                    sector_R = R/3
                    # R1_lp = R + random.uniform(-1, 1)*sector_R
                    R1_lp = R - random.random()*sector_R
                    R2_lp = R - sector_R*random.random()
                    
                    tau = c/(R1_lp+R2_lp)
        
                    
                    # response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m1*d*cos(aod_azi_lp)
                    #                                             +m1**2*d**2*sin(aod_ele)**2/2*R1_lp))
                    
                    response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m1*d*cos(aoa_azi_lp[0])))
                
                    response_ULA_BR = response_UPA1_BR
                    
                    response_UPA1_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n1*d*cos(aod_azi_lp[0] )*sin(aod_ele_lp[0])
                                                                +n1**2*d**2*(1-cos(aod_azi_lp[0])**2*sin(aod_ele_lp[0])**2)/2*R2_lp))
                    response_UPA2_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n2*d*cos(aod_ele_lp[0])
                                                                +n2**2*d**2*sin(aod_ele_lp[0])**2)/2*R1_lp)
                
                    response_UPA_RIS = torch.kron(response_UPA1_RIS, response_UPA2_RIS)
                    
                    absorption = ModelAbs(fm[nc]);
                    if flag ==1:
                        beta_pk = torch.tensor(1)
                    else:
                        beta_pk = c/(4*pi*fm[nc]*(R1_lp+R2_lp))*torch.exp(-1/2*absorption*(R1_lp+R2_lp))*torch.exp(-1j*2*pi*tau*fm[nc]);
                    
                    G_ULA_NLOS_WB_lp = G_ULA_NLOS_WB_lp+torch.sqrt(beta_pk)*torch.unsqueeze(response_ULA_BR,1)@torch.unsqueeze(response_UPA_RIS,0)
           
                G_ULA_NLOS_WB[ib, nc,:,:] = G_ULA_NLOS_WB_lp/sqrt(C_RU*L_RU)

    return G_ULA_NLOS_WB.cuda()

def batch_Channel_RIS_UE_MIMO_NLOS(param_list,batch):
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
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
    
    L = param_list[18]
    K = param_list[19]
    Nr = param_list[20]
    
    m1 = np.arange(-(Nr-1)/2,(Nr-1)/2+1,1)
    
    n1 = np.arange(-(N1-1)/2,(N1-1)/2+1,1)
    n2 = np.arange(-(N2-1)/2,(N2-1)/2+1,1)
    
    M = M1 * M2    
    N = N1 * N2
    
    # G_UPA_LOS_WB = torch.zeros(Nc, M, N).cuda()
    # H_UPA2 = torch.zeros(N2,M2).cuda()
    # H_UPA1 = torch.zeros(N1,M1).cuda()
    
    G_ULA_NLOS_WB = torch.zeros(batch,Nc, Nr, N ).to(torch.complex64)


    n = (torch.arange(0, Nc)+1)
    nn = -(Nc+1)/2
    deta = B/Nc
    fm=(nn+n)*deta+fc
    
    c = 3e8
    lambda_c = c/fc;
    d= D_ant
    d_BR = R
    
    for nc in range(Nc):
        G_ULA_NLOS_WB_lp = 0
        for lp in range(L):
            
            sector_angle = 10
            aod_azi_lp = aod_azi + random.randint(-1*sector_angle, 1*sector_angle)*aod_azi/sector_angle/2
            aod_ele_lp = aod_ele + random.randint(-1*sector_angle, 1*sector_angle)*aod_ele/sector_angle/2
            aoa_azi_lp = aoa_azi + random.randint(-1*sector_angle, 1*sector_angle)*aoa_azi/sector_angle/2
            aoa_ele_lp = aoa_ele + random.randint(-1*sector_angle, 1*sector_angle)*aoa_ele/sector_angle/2
            
            sector_R = R/3
            # R1_lp = R + random.uniform(-1, 1)*sector_R
            R1_lp = R - random.random()*sector_R
            R2_lp = R - sector_R*random.random()
            
            tau = c/(R1_lp+R2_lp)

            
            response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-m1*d*cos(aod_azi_lp)
                                                        +m1**2*d**2*sin(aod_ele)**2/2*R1_lp))
        
            response_ULA_BR = response_UPA1_BR
            
            response_UPA1_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n1*d*cos(aoa_azi_lp )*sin(aoa_ele_lp)
                                                        +n1**2*d**2*(1-cos(aoa_azi_lp)**2*sin(aoa_ele_lp)**2)/2*R2_lp))
            response_UPA2_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n2*d*cos(aoa_ele_lp)
                                                        +n2**2*d**2*sin(aoa_ele_lp)**2)/2*R2_lp)
        
            response_UPA_RIS = torch.kron(response_UPA1_RIS, response_UPA2_RIS)
            
            absorption = ModelAbs(fm[nc]);
            if flag ==1:
                beta_pk = torch.tensor(1)
            else:
                beta_pk = c/(4*pi*fm[nc]*(R1_lp+R2_lp))*torch.exp(-1/2*absorption*(R1_lp+R2_lp))*torch.exp(-1j*2*pi*tau*fm[nc]);
            
            G_ULA_NLOS_WB_lp = G_ULA_NLOS_WB_lp+torch.sqrt(beta_pk)*torch.unsqueeze(response_ULA_BR,1)@torch.unsqueeze(response_UPA_RIS,0)
   
        G_ULA_NLOS_WB[:, nc,:,:] = G_ULA_NLOS_WB_lp

    return G_ULA_NLOS_WB.cuda()

### UE-BS信道
def Channel_BS_UE_MIMO_NLOS(param_list,batch,flag):
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
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
    
    L = param_list[18]
    K = param_list[19]
    Nr = param_list[20]
    
    u1 = np.arange(-(Nr-1)/2,(Nr-1)/2+1,1)
    
    m1 = np.arange(-(M1-1)/2,(M1-1)/2+1,1)
    m2 = np.arange(-(M2-1)/2,(M2-1)/2+1,1)
    
    n1 = np.arange(-(N1-1)/2,(N1-1)/2+1,1)
    n2 = np.arange(-(N2-1)/2,(N2-1)/2+1,1)
    
    M = M1 * M2    
    N = N1 * N2
    
    m = np.arange(-(M-1)/2,(M-1)/2+1,1)

    
    # G_UPA_LOS_WB = torch.zeros(Nc, M, N).cuda()
    # H_UPA2 = torch.zeros(N2,M2).cuda()
    # H_UPA1 = torch.zeros(N1,M1).cuda()
    
    G_ULA_NLOS_WB = torch.zeros(batch,Nc, Nr, M ).to(torch.complex64)


    n = (torch.arange(0, Nc)+1)
    nn = -(Nc+1)/2
    deta = B/Nc
    fm=(nn+n)*deta+fc
    
    c = 3e8
    lambda_c = c/fc;
    d= D_ant
    R = R1 + R - 3
    
    L_BU = param_list[21]
    C_BU = param_list[22]
    C_BR = param_list[23]
    C_RU = param_list[24]
    
    sigma_bs = 5 * pi / 180
    sigma_mt = 5 * pi / 180
    
    Theta1 = np.random.rand(C_BU,1)*pi - pi/2

    sigma=sigma_bs 
    b=sigma/np.sqrt(2)   
    a=np.random.rand(C_BU,L_BU)-0.5 
    dTheta1 = - b*np.sign(a)*np.log(1-2*abs(a))

    
    sigma_mt = 5 * pi / 180
    Alpha = np.random.rand(C_BU,1)*pi - pi/2
    sigma=sigma_mt 
    b=sigma/np.sqrt(2)   
    a=np.random.rand(C_BU,L_BU)-0.5 
    dAlpha = - b*np.sign(a)*np.log(1-2*abs(a))
    
    for ib in range(batch):
        for nc in range(Nc):
            G_ULA_NLOS_WB_lp = 0
            for cl in range(C_BU):
                for lp in range(L_BU):
                    
                    # sector_angle = 10
                    # aod_azi_lp = aod_azi + random.randint(-1*sector_angle, 1*sector_angle)*aod_azi/sector_angle/2
                    # aod_ele_lp = aod_ele + random.randint(-1*sector_angle, 1*sector_angle)*aod_ele/sector_angle/2
                    # aoa_azi_lp = aoa_azi + random.randint(-1*sector_angle, 1*sector_angle)*aoa_azi/sector_angle/2
                    # aoa_ele_lp = aoa_ele + random.randint(-1*sector_angle, 1*sector_angle)*aoa_ele/sector_angle/2
                    
                    aod_azi_lp = Alpha[cl] + dAlpha[cl,lp]
                    aoa_azi_lp = Theta1[cl] + dTheta1[cl,lp]
                    
                    sector_R = R/3
                    # R1_lp = R + random.uniform(-1, 1)*sector_R
                    R1_lp = R - random.random()*sector_R
                    R2_lp = R - sector_R*random.random()
                    
                    tau = c/(R1_lp+R2_lp)
        
                    
                    # response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-u1*d*cos(aod_azi_lp)
                    #                                             +u1**2*d**2*sin(aod_ele)**2/2*R1_lp))
                    
                    response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-u1*d*cos(aoa_azi_lp[0])))
                
                    response_ULA_BR = response_UPA1_BR
                    
                    # if UPA
                    # response_UPA1_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-m1*d*cos(aod_azi_lp )*sin(aod_ele_lp)
                    #                                             +m1**2*d**2*(1-cos(aod_azi_lp)**2*sin(aod_ele_lp)**2)/2*R2_lp))
                    # response_UPA2_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-m2*d*cos(aod_ele_lp)
                    #                                             +m2**2*d**2*sin(aod_ele_lp)**2)/2*R2_lp)
                
                    # response_UPA_RIS = torch.kron(response_UPA1_RIS, response_UPA2_RIS)
                    
                    # if ULA
                    response_UPA_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-m*d*cos(aod_azi_lp[0])
                                                                +m**2*d**2*sin(aod_azi_lp[0])**2)/2*R1_lp)
                    
                    absorption = ModelAbs(fm[nc]);
                    if flag ==1:
                        beta_pk = torch.tensor(1)
                    else:
                        beta_pk = c/(4*pi*fm[nc]*(R1_lp+R2_lp))*torch.exp(-1/2*absorption*(R1_lp+R2_lp))*torch.exp(-1j*2*pi*tau*fm[nc]);
                    
                    G_ULA_NLOS_WB_lp = G_ULA_NLOS_WB_lp+torch.sqrt(beta_pk)*torch.unsqueeze(response_ULA_BR,1)@torch.unsqueeze(response_UPA_RIS,0)
           
            G_ULA_NLOS_WB[ib, nc,:,:] = G_ULA_NLOS_WB_lp/sqrt(C_BU*L_BU)

    return G_ULA_NLOS_WB.cuda()

def Channel_BS_UE_MIMO_NLOS_ULA(param_list,batch,flag):
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
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
    
    L = param_list[18]
    K = param_list[19]
    Nr = param_list[20]
    
    u1 = np.arange(-(Nr-1)/2,(Nr-1)/2+1,1)
    
    m1 = np.arange(-(M1-1)/2,(M1-1)/2+1,1)
    m2 = np.arange(-(M2-1)/2,(M2-1)/2+1,1)
    
    n1 = np.arange(-(N1-1)/2,(N1-1)/2+1,1)
    n2 = np.arange(-(N2-1)/2,(N2-1)/2+1,1)
    
    M = M1 * M2    
    N = N1 * N2
    
    m = np.arange(-(M-1)/2,(M-1)/2+1,1)
    
    # G_UPA_LOS_WB = torch.zeros(Nc, M, N).cuda()
    # H_UPA2 = torch.zeros(N2,M2).cuda()
    # H_UPA1 = torch.zeros(N1,M1).cuda()
    
    G_ULA_NLOS_WB = torch.zeros(batch,Nc, Nr, M ).to(torch.complex64)


    n = (torch.arange(0, Nc)+1)
    nn = -(Nc+1)/2
    deta = B/Nc
    fm=(nn+n)*deta+fc
    
    c = 3e8
    lambda_c = c/fc;
    d= D_ant
    R = R1 + R - 3
    
    for ib in range(batch):
        for nc in range(Nc):
            G_ULA_NLOS_WB_lp = 0
            for lp in range(L):
                
                sector_angle = 10
                aod_azi_lp = aod_azi + random.randint(-1*sector_angle, 1*sector_angle)*aod_azi/sector_angle/2
                aod_ele_lp = aod_ele + random.randint(-1*sector_angle, 1*sector_angle)*aod_ele/sector_angle/2
                aoa_azi_lp = aoa_azi + random.randint(-1*sector_angle, 1*sector_angle)*aoa_azi/sector_angle/2
                aoa_ele_lp = aoa_ele + random.randint(-1*sector_angle, 1*sector_angle)*aoa_ele/sector_angle/2
                
                sector_R = R/3
                # R1_lp = R + random.uniform(-1, 1)*sector_R
                R1_lp = R - random.random()*sector_R
                R2_lp = R - sector_R*random.random()
                
                tau = c/(R1_lp+R2_lp)
    
                
                # response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-u1*d*cos(aod_azi_lp)
                #                                             +u1**2*d**2*sin(aod_ele)**2/2*R1_lp))
                
                response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-u1*d*cos(aod_azi_lp)))
            
                response_ULA_BR = response_UPA1_BR
                
                # response_UPA1_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-m1*d*cos(aoa_azi_lp )*sin(aoa_ele_lp)
                #                                             +m1**2*d**2*(1-cos(aoa_azi_lp)**2*sin(aoa_ele_lp)**2)/2*R2_lp))
                # response_UPA2_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-m2*d*cos(aoa_ele_lp)
                #                                             +m2**2*d**2*sin(aoa_ele_lp)**2)/2*R2_lp)
            
                response_UPA_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-m*d*cos(aoa_azi_lp)
                                                                +(m**2*d**2*sin(aoa_ele_lp)**2)/2*R2_lp)/2*R2_lp)
                
                absorption = ModelAbs(fm[nc]);
                if flag ==1:
                    beta_pk = torch.tensor(1)
                else:
                    beta_pk = c/(4*pi*fm[nc]*(R1_lp+R2_lp))*torch.exp(-1/2*absorption*(R1_lp+R2_lp))*torch.exp(-1j*2*pi*tau*fm[nc]);
                
                G_ULA_NLOS_WB_lp = G_ULA_NLOS_WB_lp+torch.sqrt(beta_pk)*torch.unsqueeze(response_ULA_BR,1)@torch.unsqueeze(response_UPA_RIS,0)
       
            G_ULA_NLOS_WB[ib, nc,:,:] = G_ULA_NLOS_WB_lp/sqrt(L)

    return G_ULA_NLOS_WB.cuda()

def Channel_BS_UE_MIMO_NLOS_FF(param_list,batch,flag):
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
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
    
    L = param_list[18]
    K = param_list[19]
    Nr = param_list[20]
    
    u1 = np.arange(-(Nr-1)/2,(Nr-1)/2+1,1)
    
    m1 = np.arange(-(M1-1)/2,(M1-1)/2+1,1)
    m2 = np.arange(-(M2-1)/2,(M2-1)/2+1,1)
    
    n1 = np.arange(-(N1-1)/2,(N1-1)/2+1,1)
    n2 = np.arange(-(N2-1)/2,(N2-1)/2+1,1)
    
    M = M1 * M2    
    N = N1 * N2
    
    # G_UPA_LOS_WB = torch.zeros(Nc, M, N).cuda()
    # H_UPA2 = torch.zeros(N2,M2).cuda()
    # H_UPA1 = torch.zeros(N1,M1).cuda()
    
    G_ULA_NLOS_WB = torch.zeros(batch,Nc, Nr, M ).to(torch.complex64)


    n = (torch.arange(0, Nc)+1)
    nn = -(Nc+1)/2
    deta = B/Nc
    fm=(nn+n)*deta+fc
    
    c = 3e8
    lambda_c = c/fc;
    d= D_ant
    R = R1 + R + 2
    
    for ib in range(batch):
        for nc in range(Nc):
            G_ULA_NLOS_WB_lp = 0
            for lp in range(L):
                
                sector_angle = 10
                aod_azi_lp = aod_azi + random.randint(-1*sector_angle, 1*sector_angle)*aod_azi/sector_angle/2
                aod_ele_lp = aod_ele + random.randint(-1*sector_angle, 1*sector_angle)*aod_ele/sector_angle/2
                aoa_azi_lp = aoa_azi + random.randint(-1*sector_angle, 1*sector_angle)*aoa_azi/sector_angle/2
                aoa_ele_lp = aoa_ele + random.randint(-1*sector_angle, 1*sector_angle)*aoa_ele/sector_angle/2
                
                sector_R = R/3
                # R1_lp = R + random.uniform(-1, 1)*sector_R
                R1_lp = R - random.random()*sector_R
                R2_lp = R - sector_R*random.random()
                
                tau = c/(R1_lp+R2_lp)
    
                
                response_UPA1_BR = torch.exp(-1j*2*pi*fm[nc]/c*(-u1*d*cos(aod_azi_lp)))
            
                response_ULA_BR = response_UPA1_BR
                
                response_UPA1_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-m1*d*cos(aoa_azi_lp )*sin(aoa_ele_lp)))
                response_UPA2_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-m2*d*cos(aoa_ele_lp)))
            
                response_UPA_RIS = torch.kron(response_UPA1_RIS, response_UPA2_RIS)
                
                absorption = ModelAbs(fm[nc])
                if flag ==1:
                    beta_pk = torch.tensor(1)
                else:
                    beta_pk = c/(4*pi*fm[nc]*(R1_lp+R2_lp))*torch.exp(-1/2*absorption*(R1_lp+R2_lp))*torch.exp(-1j*2*pi*tau*fm[nc]);
                
                G_ULA_NLOS_WB_lp = G_ULA_NLOS_WB_lp+torch.sqrt(beta_pk)*torch.unsqueeze(response_ULA_BR,1)@torch.unsqueeze(response_UPA_RIS,0)
       
            G_ULA_NLOS_WB[ib, nc,:,:] = G_ULA_NLOS_WB_lp/sqrt(L)

    return G_ULA_NLOS_WB.cuda()

def Channel_RIS_UE_LOS(param_list,batch,flag):
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
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
    
    L = param_list[18]
    K = param_list[19]
    
    m1 = np.arange(-(M1-1)/2,(M1-1)/2+1,1)
    m2 = np.arange(-(M2-1)/2,(M2-1)/2+1,1)
    
    n1 = np.arange(-(N1-1)/2,(N1-1)/2+1,1)
    n2 = np.arange(-(N2-1)/2,(N2-1)/2+1,1)
    
    M = M1 * M2    
    N = N1 * N2
    
    # G_UPA_LOS_WB = torch.zeros(Nc, M, N).cuda()
    # H_UPA2 = torch.zeros(N2,M2).cuda()
    # H_UPA1 = torch.zeros(N1,M1).cuda()
    
    h_UPA_NLOS_WB = torch.zeros(batch*K, Nc, N).to(torch.complex64)


    n = (torch.arange(0, Nc)+1)
    nn = -(Nc+1)/2
    deta = B/Nc
    fm=(nn+n)*deta+fc
    
    c = 3e8
    lambda_c = c/fc;
    d= D_ant
    d_BR = R
    
    for ib in range(batch):
    
        for k in range(K):
    
            for nc in range(Nc):
                
                sector_angle = 10
                aoa_azi_k = aoa_azi + (k-K/2)*aoa_azi/K
                aoa_ele_k = aoa_ele + (k-K/2)*aoa_azi/K
                
                sector_R = R/K
                R1_lp = R + (k-K/2)*sector_R/2
                # R2_lp = R - sector_R*random.random()
                
                tau = c/R
                
                response_UPA1_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n1*d*cos(aoa_azi_k )*sin(aoa_ele_k)
                                                            +n1**2*d**2*(1-cos(aoa_azi)**2*sin(aoa_ele)**2)/2*R1_lp))
                response_UPA2_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n2*d*cos(aoa_ele_lp)
                                                            +n2**2*d**2*sin(aoa_ele_k)**2)/2*R1_lp)
            
                response_UPA_RIS = torch.kron(response_UPA1_RIS, response_UPA2_RIS)
                
                absorption = ModelAbs(fm[nc])
                if flag ==1:
                    beta_pk = torch.tensor(1)
                else:
                    beta_pk = c/(4*pi*fm[nc]*(R1_lp))*torch.exp(-1/2*absorption*(R1_lp))*torch.exp(-1j*2*pi*tau*fm[nc]);
                
                h_UPA_NLOS_WB[ib*K+k,nc,:] = torch.sqrt(beta_pk)*response_UPA_RIS

    return h_UPA_NLOS_WB.cuda()

def Channel_RIS_UE_NLOS(param_list,batch,flag):
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
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

    L = param_list[18]
    K = param_list[19]
    
    m1 = np.arange(-(M1-1)/2,(M1-1)/2+1,1)
    m2 = np.arange(-(M2-1)/2,(M2-1)/2+1,1)
    
    n1 = np.arange(-(N1-1)/2,(N1-1)/2+1,1)
    n2 = np.arange(-(N2-1)/2,(N2-1)/2+1,1)
    
    M = M1 * M2    
    N = N1 * N2
    
    # G_UPA_LOS_WB = torch.zeros(Nc, M, N).cuda()
    # H_UPA2 = torch.zeros(N2,M2).cuda()
    # H_UPA1 = torch.zeros(N1,M1).cuda()
    
    h_UPA_NLOS_WB = torch.zeros(batch*K, Nc, N).to(torch.complex64)


    n = (torch.arange(0, Nc)+1)
    nn = -(Nc+1)/2
    deta = B/Nc
    fm=(nn+n)*deta+fc
    
    c = 3e8
    lambda_c = c/fc;
    d= D_ant
    d_BR = R
    
    for ib in range(batch):
        for k in range(K):
            for nc in range(Nc):
                G_UPA_NLOS_WB_lp = 0
                for lp in range(L):
                    
                    sector_angle = 10
                    aod_azi_lp = aod_azi + random.randint(-1*sector_angle, 1*sector_angle)*aod_azi/sector_angle/2
                    aod_ele_lp = aod_ele + random.randint(-1*sector_angle, 1*sector_angle)*aod_ele/sector_angle/2
                    aoa_azi_lp = aoa_azi + random.randint(-1*sector_angle, 1*sector_angle)*aoa_azi/sector_angle/2
                    aoa_ele_lp = aoa_ele + random.randint(-1*sector_angle, 1*sector_angle)*aoa_ele/sector_angle/2
                    
                    sector_R = R/5
                    # R1_lp = R + random.uniform(-1, 1)*sector_R
                    R1_lp = R - random.random()*sector_R
                    R2_lp = R - sector_R*random.random()
                    
                    tau = c/(R1_lp+R2_lp)
                    
                    response_UPA1_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n1*d*cos(aoa_azi_lp )*sin(aoa_ele_lp)
                                                                +n1**2*d**2*(1-cos(aoa_azi_lp)**2*sin(aoa_ele_lp)**2)/2*R2_lp))
                    response_UPA2_RIS = torch.exp(-1j*2*pi*fm[nc]/c*(-n2*d*cos(aoa_ele_lp)
                                                                +n2**2*d**2*sin(aoa_ele_lp)**2)/2*R2_lp)
                
                    response_UPA_RIS = torch.kron(response_UPA1_RIS, response_UPA2_RIS)
                    
                    absorption = ModelAbs(fm[nc])
                    if flag ==1:
                        beta_pk = torch.tensor(1)
                    else:
                        beta_pk = c/(4*pi*fm[nc]*(R1_lp+R2_lp))*torch.exp(-1/2*absorption*(R1_lp+R2_lp))*torch.exp(-1j*2*pi*tau*fm[nc]);
                    
                    G_UPA_NLOS_WB_lp = G_UPA_NLOS_WB_lp+torch.sqrt(beta_pk)*response_UPA_RIS
               
                h_UPA_NLOS_WB[ib*K+k,nc,:] = G_UPA_NLOS_WB_lp/sqrt(L)

    return h_UPA_NLOS_WB.cuda()

def batch_Channel_RIS_UE_LOS(param_list,batch):
    
    K = param_list[19]
    Nc = param_list[2]
    M1  = param_list[3]
    M2  = param_list[4]
    N1  = param_list[5]
    N2  = param_list[6]
    
    M = M1 * M2    
    N = N1 * N2
    
    H_RU = (torch.zeros(batch*K,Nc,N) + 0j).cuda()
    
    for i in range(batch):
        H_RU[i*K:(i+1)*K,:,:] = Channel_RIS_UE_LOS(param_list)
        
    return H_RU

if  __name__ == '__main__':

    
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
    
    
    batch = 100
    
    
    channel_param_list = [fc,BW,Nc,M1,M2,N1,N2,D_ant,R_BR,R_RU,h1,h2,aod_azi_LOS,aod_ele_LOS, aoa_azi_LOS, aoa_ele_LOS, tau_max, L_BR, L_RU, K, Nr, L_BU, C_BU, C_BR, C_RU]
        
    Gt_gain = 10**(20/10.)
    GRIS_gain = 10**(5/10.)

    Gr_gain = 10**(20/10.)
     
    flag =0
    Loop = 100
    
    # H_RU = Channel_RIS_UE_MIMO_LOS(channel_param_list,batch,flag) + Channel_RIS_UE_MIMO_NLOS(channel_param_list,batch,flag)
    
    # H_RU = Channel_RIS_UE_MIMO_NLOS(channel_param_list,batch,flag)
    # H_RU = H_RU.reshape(batch,K,Nc,Nr,N) 
    # H_BU = Channel_BS_UE_MIMO_NLOS(channel_param_list,batch,flag)
    # H_BU = H_BU.reshape(batch,K,Nc,Nr,M) 
    # H_BR = Channel_BS_RIS_LOS_PLA(channel_param_list,flag) + Channel_BS_RIS_NLOS(channel_param_list,flag)
    # H_BR = Channel_BS_RIS_LOS(channel_param_list,flag)
    
    H_RU1 = sqrt(Gr_gain*GRIS_gain)*(Channel_RIS_UE_MIMO_LOS(channel_param_list,batch,flag)+Channel_RIS_UE_MIMO_NLOS(channel_param_list,batch,flag))
    H_BU = sqrt(Gt_gain*Gr_gain)*Channel_BS_UE_MIMO_NLOS(channel_param_list,batch,flag)
    H_BU = H_BU.reshape(batch,K,Nc,Nr,M) 
    H_RU = H_RU1.reshape(batch,K,Nc,Nr,N) 
    H_BR = sqrt(Gt_gain*GRIS_gain)*(Channel_BS_RIS_LOS_PLA(channel_param_list,flag)+Channel_BS_RIS_NLOS(channel_param_list,flag))
    
    import scipy.io as sio
    h_ru = 'H_RU'+str(floor(BW))+'band'+'mat'
    sio.savemat('H_RU'+str(math.floor(BW))+'band'+'.mat', {'H_RU': H_RU.cpu().detach().numpy()})
    sio.savemat('H_BU'+str(math.floor(BW))+'band'+'.mat', {'H_BU': H_BU.cpu().detach().numpy()})
    sio.savemat('H_BR'+str(math.floor(BW))+'band'+'.mat', {'H_BR': H_BR.cpu().detach().numpy()})


# a = Channel_BS_RIS_LOS(param_list)

# b= Channel_BS_RIS_NLOS(param_list)

# d= Channel_RIS_UE_LOS(param_list)

# c= Channel_RIS_UE_NLOS(param_list)

# import matplotlib.pyplot as plt
# plt.imshow(np.real(d[1,:,:].cpu().detach().numpy()))
# plt.show()


