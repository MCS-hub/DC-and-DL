# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:39:46 2021

@author: Phuc Hau
"""


import os
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from train import train, train_v3
from get_model import get_dc_model, get_model, get_dc_model_v3
from matplotlib import pyplot as plt
import random

tf.keras.backend.set_floatx('float64')

params_adamax = [0.000001,0.9,0.999,1e-7]

eps2 = 1.
eps1 = 1/eps2

rho2 = 4.
rho3 = 1.
rho1 = 4/rho2+1/rho3

kappa2 = 1.
kappa1 = 1/kappa2
kappa3 = 0.

params_decom = [eps1,eps2,rho1,rho2,rho3,kappa1,kappa2,kappa3]

dc_model = get_dc_model_v3(params_decom)

train_time,train_loss,train_accuracy,val_accuracy = \
train(dc_model=dc_model, reg_param = 0.0,epochs=20,num_iter_cnvx=1,\
      max_iter_cnvx=20,damping_const_init=5000,cg_eps=1e-5,\
          convex_optimizer='Adamax',learning_rate=params_adamax,ld=0)



# for i in range(len(model.trainable_weights)):
#     dc_model.trainable_weights[i].assign(model.trainable_weights[i])
    
# inputs = tf.random.uniform(shape=(3,784))
# output1 = model(inputs)
# tmp = dc_model(inputs)
# output2 = tmp[:,:10] - tmp[:,10:]
# print(output1)
# print(output2)