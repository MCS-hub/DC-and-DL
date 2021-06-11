# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:56:11 2021

@author: luu2
"""

import pickle
import tensorflow as tf
from train import train, train_v2, train_normal_model
from evaluation_func import test
from get_model import get_dc_model, get_model, get_dc_model_v3, get_dc_model_v4
from matplotlib import pyplot as plt
import random

tf.keras.backend.set_floatx('float64')

Lu_sigmoid = 1.
Lv_sigmoid = 1.
L_sigmoid = Lu_sigmoid + Lv_sigmoid

Lu_leakyrelu = 1.
Lv_leakyrelu = 0.001
L_leakyrelu = Lu_leakyrelu + Lv_leakyrelu

learning_rate = 0.0000001 # increase lr from 1e-5 up to 1e-4

ITERATIONS = 10


rho12_list = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
kappa12_list = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
rho22_list = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
rho23_list = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
kappa22_list = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
kappa23_list = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]

params_list = []

for i in range(ITERATIONS):
    rho12 = random.sample(rho12_list,1)[0]
    kappa12 = random.sample(kappa12_list,1)[0]
    rho22 = random.sample(rho22_list,1)[0]
    rho23 = random.sample(rho23_list,1)[0]
    kappa22 = random.sample(kappa22_list,1)[0]
    kappa23 = random.sample(kappa23_list,1)[0]
    
    rho11 = Lu_sigmoid**2/rho12
    kappa11 = Lv_sigmoid**2/kappa12
    rho21 = (L_sigmoid+Lu_sigmoid)**2/rho22 + Lu_sigmoid**2/rho23
    kappa21 = (L_sigmoid+Lv_sigmoid)**2/kappa22 + Lv_sigmoid**2/kappa23
    
    
    params_decom = [rho11,rho12,kappa11,kappa12,rho21,rho22,rho23,kappa21,kappa22,kappa23]
    params_list.append(params_decom)

    dc_model = get_dc_model_v4(params_decom,activation='sigmoid',init_weights=None)

    train_time,train_loss,train_accuracy,val_accuracy = \
    train_v2(dc_model=dc_model, reg_param = 0.000001,epochs=20,num_iter_cnvx=1,\
          max_iter_cnvx=80,convex_optimizer='Adamax',learning_rate=learning_rate,ld=0)
    
    solution = dc_model.get_weights()
    model = get_model(activation='sigmoid')
    model.set_weights(solution)
    test_accuracy = test(model)    
    
    with open('osDCA_sigmoid_'+str(i)+'.pkl','wb') as f:
        pickle.dump([train_time,train_loss,train_accuracy,val_accuracy,test_accuracy,solution],f)
    
    del dc_model, model
    
    


# =============================================================================
# # verify model
# dc_model = get_dc_model_v4(params_decom,activation='sigmoid',init_weights=None)
# 
# inputs = tf.random.normal(shape=(2,784))
# outputs = dc_model(inputs)
# delta = outputs[:,:10]
# psi = outputs[:,10:]
# 
# print(delta-psi)
# 
# model = get_model(activation='sigmoid')
# model.set_weights(dc_model.get_weights())
# 
# print(model(inputs))
# =============================================================================
