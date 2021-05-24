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
from train import train, train_v2
from get_model import get_dc_model, get_model, get_dc_model_v3
from matplotlib import pyplot as plt
import random

tf.keras.backend.set_floatx('float64')

ITERATIONS = 20

params_adamax = [0.0001,0.9,0.999,1e-7]  # increase lr from 1e-5 up to 1e-4

params_decom_list = []

eps2_list = [1.,0.1,0.001]
rho2_list = [0.001,0.0001,0.00001,1e-6,1e-7,1e-8,1e-9]
rho3_list = [0.0001,0.00001,0.000001,1e-7,1e-8,1e-9]
kappa2_list = [0.001,0.0001,0.00001,0.000001,1e-7,1e-8,1e-9]


for i in range(ITERATIONS):
    eps2 = random.sample(eps2_list,1)[0]
    rho2 = random.sample(rho2_list,1)[0]
    rho3 = random.sample(rho3_list,1)[0]
    kappa2 = random.sample(kappa2_list,1)[0]
    eps1 = 1/eps2
    rho1 = 4/rho2+1/rho3
    kappa1 = 1/kappa2
    kappa3 = 0.
    params_decom = [eps1,eps2,rho1,rho2,rho3,kappa1,kappa2,kappa3]

    dc_model = get_dc_model_v3(params_decom)

    train_time,train_loss,train_accuracy,val_accuracy = \
    train_v2(dc_model=dc_model, reg_param = 0.0,epochs=20,num_iter_cnvx=1,\
          max_iter_cnvx=80,convex_optimizer='Adamax',learning_rate=params_adamax,ld=0)
    
        
    with open('rho_dc_'+str(i)+'.pkl','wb') as f:
        pickle.dump([train_loss,train_time,train_accuracy,val_accuracy],f)
    
    del dc_model
    
    params_decom_list.append(params_decom)

with open('params_decom_list.pkl','wb') as f:
    pickle.dump([params_decom_list],f)


# plot figures

for i in range(11):
    with open('rho_dc_'+str(i)+'.pkl','rb') as f:
        [train_loss,train_time,train_accuracy,val_accuracy] = pickle.load(f)
    
    plt.plot(train_time,train_accuracy,label='params_decom_'+str(i))

plt.xlabel('time (s)')
plt.ylabel('train_accuracy')

plt.legend()

with open('params_decom_list.pkl','rb') as f:
    params_decom_list = pickle.load(f)
# # test set_weights
# model = get_model()

# params_decom = [1000.0, 0.001, 400100000.0, 1e-08, 1e-05, 999999999.9999999, 1e-09, 0.0]
# dc_model = get_dc_model_v3(params_decom)

# dc_model.set_weights(model.get_weights())
    
# inputs = tf.random.uniform(shape=(3,784))
# output1 = model(inputs)
# tmp = dc_model(inputs)
# output2 = tmp[:,:10] - tmp[:,10:]
# print(output1)
# print(output2)
