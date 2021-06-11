# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 23:51:31 2021

@author: Phuc Hau
"""


import pickle
import tensorflow as tf
from train import train_v4
from evaluation_func import test
from get_model import get_dc_model, get_model, get_dc_model_v3
from matplotlib import pyplot as plt

tf.keras.backend.set_floatx('float64')


eta = 2
delta = 0.9
mu0_list = [0.1,1,10,100]
n_linesearch = 10
epochs = 20

for mu0 in mu0_list:
    model = get_model()
    train_time, train_loss, train_accuracy, val_accuracy = train_v4(model,eta,delta,mu0,n_linesearch,epochs)
    
    with open('onlineDCA_like_mu0_'+str(mu0)+'.pkl','wb') as f:
        pickle.dump([train_time, train_loss, train_accuracy, val_accuracy],f)

#%%
# plot figures
for mu0 in mu0_list:
    with open('onlineDCA_like_mu0_'+str(mu0)+'.pkl','rb') as f:
        [train_time, train_loss, train_accuracy, val_accuracy] = pickle.load(f)
        plt.plot(train_time,train_accuracy,label='online_DCA_like, mu0 = '+str(mu0))
        print('time',train_time[-1],'loss',train_loss[-1],'train_acc',train_accuracy[-1],'val_acc',val_accuracy[-1])
        
plt.xlabel('time (s)')
plt.ylabel('accuracy')

plt.legend()