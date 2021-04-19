# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 11:26:48 2021

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


#%%
tf.keras.backend.set_floatx('float64')


# set path
path = os.getcwd()


convex_optimizer_names = ['Adamax', 'Adagrad', 'SGD', 'Nadam', 'Adadelta']
convex_optimizer_list = {'Adamax': 0.000001, 'Adagrad': 0.000001, 'SGD': 0.000001,\
                    'Nadam': 0.0000002, 'Adadelta': 0.00003}

for cnv_opt in convex_optimizer_names:
    
    dc_model = get_dc_model(activation='relu')
    
    train_time,train_loss,train_accuracy,val_accuracy = \
        train(dc_model=dc_model, reg_param = 0.0,epochs=20,num_iter_cnvx=50,\
              max_iter_cnvx=20,damping_const_init=5000,cg_eps=1e-5,\
                  convex_optimizer=cnv_opt,learning_rate=convex_optimizer_list[cnv_opt])
    
    with open('osDCA_'+cnv_opt + '_.pkl','wb') as f:  
            pickle.dump([train_loss,train_time,train_accuracy,val_accuracy],f)
    del dc_model
    

for cnv_opt in convex_optimizer_names:
    with open('osDCA_'+cnv_opt + '_.pkl','rb') as f:
        train_loss,train_time,train_accuracy,val_accuracy = pickle.load(f)
    plt.plot(train_time,val_accuracy,label='osDCA with ' + cnv_opt)
    plt.xlabel('time (s)')
    plt.ylabel('validation accuracy')
plt.legend()


#%%
with open('result_April_15.pkl','rb') as f:
    train_loss,train_time,train_accuracy,val_accuracy = pickle.load(f)

plt.figure()
plt.plot(train_time,train_accuracy,label='train accuracy')
plt.plot(train_time,val_accuracy,label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('time (s)')
plt.legend()

plt.figure()
plt.plot(train_time,train_loss,label='train loss')
plt.ylabel('loss')
plt.xlabel('time (s)')
plt.legend()
#%%
with open('Adamax_chosen_params.pkl','rb') as f:
    params_list = pickle.load(f)[0]

for params in params_list:
    lr = params[0]
    beta_1 = params[1]
    beta_2 = params[2]
    epsilon = params[3]
    num_iter_cnvx = params[4]
    
    with open('osDCA_Adamax_'+str(lr)+'_'+str(beta_1)+'_'+str(beta_2)+'_'\
              +str(epsilon)+'_'+str(num_iter_cnvx)+'_.pkl','rb') as f:
        train_loss,train_time,train_accuracy,val_accuracy = pickle.load(f)
    
    plt.plot(train_loss,label=str(lr)+'_'+str(beta_1)+'_'+str(beta_2)+'_'\
              +str(epsilon)+'_'+str(num_iter_cnvx))

plt.ylabel('train loss')
plt.xlabel('time (s)')
plt.legend()
    
