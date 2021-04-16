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
tf.keras.backend.set_floatx('float64')

# set path
path = os.getcwd()


dc_model = get_dc_model(activation='relu')

train_time,train_loss,train_accuracy,val_accuracy = train(dc_model=dc_model, reg_param = 0.0,epochs=10,num_iter_cnvx=8,max_iter_cnvx=20,damping_const_init=5000,cg_eps=1e-5)
    
