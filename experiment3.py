# -*- coding: utf-8 -*-
"""
Created on Fri May  7 22:44:10 2021

@author: Phuc Hau
"""

import os
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from train import train, train_v2, train_v3
from get_model import get_dc_model, get_model, get_dc_model_v3
from matplotlib import pyplot as plt
import random

tf.keras.backend.set_floatx('float64')

params_adamax = [0.0001,0.9,0.999,1e-7]

eps2 = 0.1
rho2 = 1e-6
rho3 = 1e-8
kappa2 = 1e-8
kappa3 = 0.0

eps1_max = 1/eps2
rho1_max = 4/rho2+1/rho3
kappa1_max = 1/kappa2

eps1 = 1.
rho1 = 1.
kappa1 = 1.

rate_eps1 = 2.
rate_rho1 = 2.
rate_kappa1 = 2.

params_decom = [eps1,eps2,rho1,rho2,rho3,kappa1,kappa2,kappa3]

dc_model = get_dc_model_v3(params_decom)

num_iter_cnvx_list = [1,5,10,20]

for num_iter_cnvx in num_iter_cnvx_list:
    train_time,train_loss,train_accuracy,val_accuracy = \
    train_v3(dc_model,params_decom,reg_param = 0.0,epochs=20,num_iter_cnvx=num_iter_cnvx,\
             max_iter_cnvx=50,convex_optimizer='Adamax',learning_rate=params_adamax,ld=0)
    
    with open('osDCAlike_numitercnvx_'+str(num_iter_cnvx)+'.pkl','wb') as f:
        pickle.dump([train_loss,train_time,train_accuracy,val_accuracy],f)
        
with open('osDCAlike_numitercnvx_'+str(num_iter_cnvx)+'.pkl','rb') as f:
    [train_loss,train_time,train_accuracy,val_accuracy] = pickle.load(f)
    
plt.plot(train_time,train_accuracy,label='train_accuracy')
plt.plot(train_time,val_accuracy,label='val_accuracy')

plt.xlabel('time (s)')
plt.ylabel('accuracy')

plt.legend()
