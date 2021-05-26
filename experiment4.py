# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:20:27 2021

@author: luu2
"""

import os
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from train import train, train_v2
from get_model import get_dc_model, get_model, get_dc_model_v3
from matplotlib import pyplot as plt
import random

tf.keras.backend.set_floatx('float64')


learning_rate = 0.0001 # increase lr from 1e-5 up to 1e-4

eps2 = 0.1
rho2 = 1e-6
rho3 = 1e-5
kappa2 = 1e-9
eps1 = 1/eps2
rho1 = 4/rho2+1/rho3
kappa1 = 1/kappa2
kappa3 = 0.
params_decom = [eps1,eps2,rho1,rho2,rho3,kappa1,kappa2,kappa3]

dc_model = get_dc_model_v3(params_decom)

train_time,train_loss,train_accuracy,val_accuracy = \
train_v2(dc_model=dc_model, reg_param = 0.000001,epochs=20,num_iter_cnvx=1,\
      max_iter_cnvx=80,convex_optimizer='Adamax',learning_rate=learning_rate,ld=0)

# dc_model = get_dc_model_v3(params_decom)

# train_time1,train_loss1,train_accuracy1,val_accuracy1 = \
# train_v2(dc_model=dc_model, reg_param = 0.0,epochs=10,num_iter_cnvx=1,\
#       max_iter_cnvx=80,convex_optimizer='Adamax',learning_rate=learning_rate,ld=0)


plt.plot(train_time,train_accuracy)

# with open('osDCA_reg_0.0_epochs_20.pkl','wb') as f:
#     pickle.dump([train_loss,train_time,train_accuracy,val_accuracy],f)



    
 