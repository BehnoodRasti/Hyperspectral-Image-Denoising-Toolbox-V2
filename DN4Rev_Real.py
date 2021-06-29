# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:07:47 2021

@author: behnood
"""


from __future__ import print_function
import matplotlib.pyplot as plt
#%matplotlib inline

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from skimage.measure import compare_mse
from utils.denoising_utils import *

from skimage._shared import *
from skimage.util import *
from skimage.metrics.simple_metrics import _as_floats
from skimage.metrics.simple_metrics import mean_squared_error


# from UtilityMine import compare_snr
# from UtilityMine import find_endmember
# from UtilityMine import add_noise
#from UtilityMine import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.
#%% Load image
import scipy.io
#%%
fname2  = "C:/Users/behnood/OneDrive - Háskóli Íslands/RS_IR/Indian.mat"
# ensure dimensions [0][1] are divisible by 32 (or 2^depth)!
mat2 = scipy.io.loadmat(fname2)
img_np_gt = mat2["Indian"]
img_np_gt = img_np_gt.transpose(2,0,1)
[p1, nr1, nc1] = img_np_gt.shape
#%%
import pymf
from pymf.sivm import *
from pymf.chnmf import *
import matplotlib.pyplot as plt
npar=np.zeros((1,3))
npar[0,0]=10
npar[0,1]=31.8
npar[0,2]=100
#npar[0,3]=1090
tol1=npar.shape[1]
tol2=5
Metric=np.zeros((tol2,10,tol1))
#mse_E=np.zeros((tol1,tol2))
save_result=False
import time
from tqdm import tqdm
img_noisy_np=img_np_gt
img_resh=np.reshape(img_noisy_np,(p1,nr1*nc1))
#%% Set up Simulated 
#%%
INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 0.0# 1./30. # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER='adam' # 'LBFGS'
show_every = 200
exp_weight=0.99
# if fi==0:
#     num_iter = 650
# elif fi==1:
#     num_iter = 1330
# elif fi==2:
num_iter = 2000

input_depth = img_resh.shape[0] 
figsize = 5 
net = skip(input_depth, img_resh.shape[0],  
   # num_channels_down = [64, 64, 64, 128,128], 
   # num_channels_up   = [64, 64, 64, 128,128],
   # num_channels_skip = [4, 4, 4, 4, 4], 
num_channels_down = [256],
num_channels_up =   [256],
num_channels_skip =    [4],  
   filter_size_up = 3,filter_size_down = 3,  filter_skip_size=1,
   upsample_mode='bilinear', # downsample_mode='avg',
   need1x1_up=True,
   need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
"""
net = skip(
            input_depth, img_np.shape[0], 
            num_channels_down = [8, 16, 32, 64, 128], 
            num_channels_up   = [8, 16, 32, 64, 128],
            num_channels_skip = [0, 0, 0, 4, 4], 
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
"""
net = net.type(dtype)
#net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
net_input = get_noise(input_depth, INPUT, (nr1, nc1)).type(dtype).detach()

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
#%%
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0

i = 0
def closure():
    
    global i, out, out_np,out_avg_np, out_avg, psrn_noisy_last, last_net, net_input
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out = net(net_input)
    
    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            
    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()
        
    
    # out_np = out.detach().cpu().squeeze().numpy()
    # out_avg_np = out_avg.detach().cpu().squeeze().numpy()
    # psrn_noisy = compare_psnr(img_np_gt.astype(np.float32), img_noisy_np.astype(np.float32))  
    # psrn_gt    = compare_psnr(img_np_gt.astype(np.float32), np.clip(out_np, 0, 1)) 
    # psrn_gt_sm = compare_psnr(img_np_gt.astype(np.float32), np.clip(out_avg_np, 0, 1))    
    # # Note that we do not have GT for the "snail" example
    # # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    print ('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
    if  PLOT and i % show_every == 0:
        # out_np = torch_to_np(out)
        # out_avg_np = torch_to_np(out_avg)
        # plot_image_grid([np.clip(out_np, 0, 1), 
        #                 np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)
        
        # out_np = np.clip(out_np, 0, 1)
        # out_avg_np = np.clip(out_avg_np, 0, 1)
        out_np = out.detach().cpu().squeeze().numpy()
        out_avg_np = out_avg.detach().cpu().squeeze().numpy()
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15,15))
        ax1.imshow(np.stack((out_np[3,:,:],out_np[2,:,:],out_np[1,:,:]),2))
        ax2.imshow(np.stack((out_avg_np[3,:,:],out_avg_np[2,:,:],out_avg_np[1,:,:]),2))
        plt.show()
        
    
    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)   
#%%
if  save_result is True:
    scipy.io.savemat("C:/Users/behnood/Desktop/Result_Rev_DN/Real/out_avg_np.mat",
                    {'out_avg_np':out_avg_np.transpose(1,2,0)})
