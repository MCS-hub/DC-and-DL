# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 02:00:42 2021

@author: Phuc Hau
"""

import pickle
import tensorflow as tf
from train import train, train_v2, train_normal_model
from evaluation_func import test
from get_model import get_dc_model, get_model, get_dc_model_v3
from matplotlib import pyplot as plt

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


# with open('initial_point.pkl','wb') as f:
#     pickle.dump(initial_point,f)

with open('initial_point.pkl','rb') as f:
    initial_point = pickle.load(f)

dc_model.set_weights(initial_point)

train_time,train_loss,train_accuracy,val_accuracy = \
train_v2(dc_model=dc_model, reg_param = 0.000001,epochs=20,num_iter_cnvx=1,\
      max_iter_cnvx=80,convex_optimizer='Adamax',learning_rate=learning_rate,ld=0,small_dataset=True)

solution = dc_model.get_weights()

model = get_model()
model.set_weights(solution)
test_accuracy, pre_macro, rec_macro, pre_micro, rec_micro = test(model)

with open('osDCA_small.pkl','wb') as f:
    pickle.dump([train_time,train_loss,train_accuracy,val_accuracy,test_accuracy,solution,\
                 pre_macro, rec_macro, pre_micro, rec_micro],f)
        
#%%
model = get_model()
with open('initial_point.pkl','rb') as f:
    initial_point = pickle.load(f)
    
model.set_weights(initial_point)
train_time, train_loss, train_accuracy, val_accuracy = train_normal_model(model, reg_param = 0.000001,epochs=20,optimizer='Adam',learning_rate=0.0001,small_dataset=True)

solution = model.get_weights()
test_accuracy, pre_macro, rec_macro, pre_micro, rec_micro = test(model)

with open('adam_0.0001small.pkl','wb') as f:
    pickle.dump([train_time,train_loss,train_accuracy,val_accuracy,test_accuracy,solution,\
                 pre_macro, rec_macro, pre_micro, rec_micro],f)
        
#%%
name_list = ['osDCA','adam_0.0001','adam_0.001','adam_0.01','osDCA_small','adam_0.0001small','adam_0.001small','adam_0.01small']

for name in name_list:
    with open(name+'.pkl','rb') as f:
        train_time,train_loss,train_accuracy,val_accuracy,test_accuracy,solution,\
                     pre_macro, rec_macro, pre_micro, rec_micro = pickle.load(f)
    print(round(train_time[-1],2),round(train_loss[-1],2),round(train_accuracy[-1],2),\
          round(val_accuracy[-1],2),round(test_accuracy,2),round(pre_macro,2),round(rec_macro,2))
