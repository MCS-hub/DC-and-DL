# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:20:27 2021

@author: luu2
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

initial_point = dc_model.get_weights()

train_time,train_loss,train_accuracy,val_accuracy = \
train_v2(dc_model=dc_model, reg_param = 0.000001,epochs=20,num_iter_cnvx=1,\
      max_iter_cnvx=80,convex_optimizer='Adamax',learning_rate=learning_rate,ld=0)

solution = dc_model.get_weights()

model = get_model()
model.set_weights(solution)
test_accuracy = test(model)

with open('osDCA.pkl','wb') as f:
    pickle.dump([train_time,train_loss,train_accuracy,val_accuracy,test_accuracy,initial_point,solution],f)
# dc_model = get_dc_model_v3(params_decom)

# train_time1,train_loss1,train_accuracy1,val_accuracy1 = \
# train_v2(dc_model=dc_model, reg_param = 0.0,epochs=10,num_iter_cnvx=1,\
#       max_iter_cnvx=80,convex_optimizer='Adamax',learning_rate=learning_rate,ld=0)


plt.plot(train_time,train_accuracy)

# with open('osDCA_reg_0.0_epochs_20.pkl','wb') as f:
#     pickle.dump([train_loss,train_time,train_accuracy,val_accuracy],f)

#%%
# train the normal model for 
with open('osDCA.pkl','rb') as f:
    [train_time,train_loss,train_accuracy,val_accuracy,test_accuracy,initial_point,solution] = pickle.load(f)
    
model = get_model()
model.set_weights(initial_point)

[train_time, train_loss, train_accuracy, val_accuracy] = train_normal_model(model,reg_param = 0.000001,epochs=50,optimizer='Adam',learning_rate=0.001)

solution = model.get_weights()


plt.plot(train_time,val_accuracy)


    
 