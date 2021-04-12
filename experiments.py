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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.keras import layers
from special_layers import *
from train import train
from utils.utils import *
from config import *
from get_model import get_model, get_dc_model
from matplotlib import pyplot as plt
tf.keras.backend.set_floatx('float64')

# set path
path = os.getcwd()
#-----------------------------------------------------------------------------
# Get a DC neural network

dc_model = get_dc_model(activation='relu')

#-----------------------------------------------------------------------------
# Initialized the solution of the normal model as the initial point for the DC model
model_dir = os.getcwd()+'\\results\\validation hyperparameters\\Adam_reg_0.0001_lr_0.0001'
normal_model = keras.models.load_model(model_dir)

#------------------------------------------------------------------------------
# =============================================================================
# # Experiment 1
# # Train the DC neural network
# for damping_const_init in [5000,10000,15000]:
#     for num_iter_cnvx in [2,3,5,8]:
#         
#         # assign DC model's weights to the trained weights
#         
#         for i in range(len(dc_model.trainable_weights)):
#             dc_model.trainable_weights[i].assign(normal_model.trainable_weights[i]) 
# 
#         
#         train_time,train_loss,train_accuracy,val_accuracy = train(dc_model,reg_param = 0.0001,epochs=3,num_iter_cnvx=num_iter_cnvx,damping_const_init=damping_const_init,\
#           cg_eps=1e-5,cg_k_max=100)
#         
#         # get a temporary model
#         tmp_model = get_model()
#         
#         for i in range(len(dc_model.trainable_weights)):
#             tmp_model.trainable_weights[i].assign(dc_model.trainable_weights[i])    
# 
#             
#         tmp_model.save('/tmp/dc_model/'+'train_dc_damping'+str(damping_const_init)+'_cnv_'+str(num_iter_cnvx))
#         del tmp_model
#         
#         with open('train_dc_damping'+str(damping_const_init)+'_cnv_'+str(num_iter_cnvx)+'_.pkl','wb') as f:  
#             pickle.dump([train_loss,train_time,train_accuracy,val_accuracy],f)
# =============================================================================

#------------------------------------------------------------------------------
# Experiment 2
for i in range(len(dc_model.trainable_weights)):
    dc_model.trainable_weights[i].assign(normal_model.trainable_weights[i]) 

train_time,train_loss,train_accuracy,val_accuracy = train(dc_model,reg_param = 0.0,epochs=1,num_iter_cnvx=3,damping_const_init=15000,\
    cg_eps=1e-5,cg_k_max=100)
    
#-----------------------------------------------------------------------------

