# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 22:06:31 2021

@author: Phuc Hau
"""

import os
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from train import train
from get_model import get_dc_model, get_model
from matplotlib import pyplot as plt
import random

tf.keras.backend.set_floatx('float64')
# set path


lr_list = [1e-2,1e-3,1e-4,1e-5]
beta1_list = [0.6,0.7,0.8,0.9]
beta2_list = [0.6,0.7,0.8,0.9, 0.99]
epsilon_list = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]

ITERATIONS = 50


for i in range(ITERATIONS):
    lr = random.sample(lr_list,1)[0]
    beta1 = random.sample(beta1_list,1)[0]
    beta2 = random.sample(beta2_list,1)[0]
    eps = random.sample(epsilon_list,1)[0]
    params = [lr,beta1,beta2,eps]
    
    dc_model = get_dc_model(activation='relu')
    
    train_time,train_loss,train_accuracy,val_accuracy = \
    train(dc_model=dc_model, reg_param = 0.0,epochs=20,num_iter_cnvx=1,\
          max_iter_cnvx=20,damping_const_init=5000,cg_eps=1e-5,\
              convex_optimizer='Adamax',learning_rate=params,ld=0)
    with open(str(lr)+'_'+str(beta1)+'_'+str(beta2)+'_'+str(eps)+'.pkl','wb') as f:
        pickle.dump([train_loss,train_time,train_accuracy,val_accuracy],f)
    
    del dc_model
    
    params_list.append(params)

with open('params_list','wb') as f:
    pickle.dump([params_list],f)
    

# for params in params_list:
#     lr, beta1, beta2, eps = params
#     with open(str(lr)+'_'+str(beta1)+'_'+str(beta2)+'_'+str(eps)+'.pkl','rb') as f:
#         train_loss,train_time,train_accuracy,val_accuracy = pickle.load(f)
#     plt.plot(train_time,train_loss)

