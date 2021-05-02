# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 13:43:25 2021

@author: luu2
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from special_layers import DC_V2, DC_V3_type1, DC_V3_type2
from utils.utils import log_sum_exp, norm_square

# normal model
def get_model():
        
    inputs = tf.keras.Input(shape=(784,))
    
    dense1 = layers.Dense(64,activation='relu')
    dense2 = layers.Dense(64,activation='relu')
    dense3 = layers.Dense(10)
    
    x = dense1(inputs)
    x =  dense2(x)
    outputs = dense3(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs,name='normal_model')
    return model


def get_dc_model(activation='relu'):
    
    inputs = tf.keras.Input(shape=(784,),name='input')
    Dense = layers.Dense(64,activation=activation,name='dense')
    DC1 = DC_V2(units=64,activation=activation)
    DC2 = DC_V2(units=10)

    # stack layers together
    x = Dense(inputs)
    x = tf.concat([x,0.0*x],axis=1)
    x = DC1(x)
    x = DC2(x)
    
    dc_model = keras.Model(inputs=inputs,outputs=x,name='dc_model')
    
    return dc_model

def get_DC_component(dc_model,x_batch,y_one_hot, component='both',reg_param = None):
    
    outputs = dc_model(x_batch)
    delta = outputs[:,:10]
    psi = outputs[:,10:]
    
    if component == 'both':
        
        if reg_param is None:
            raise NameError('reg_param is required.')
            
        H = tf.reduce_sum(psi,axis=1) + tf.reduce_sum(delta*y_one_hot,1)
        H = tf.reduce_mean(H)
                
        G = tf.squeeze(log_sum_exp(delta-psi))\
            +tf.reduce_sum(psi,axis=1)+tf.reduce_sum(psi*y_one_hot,1)
        G = tf.reduce_mean(G)+reg_param*norm_square(dc_model.trainable_weights)
        return G, H
    elif component == 'G':
        if reg_param is None:
            raise NameError('reg_param is required.')
            
        G = tf.squeeze(log_sum_exp(delta-psi))\
            +tf.reduce_sum(psi,axis=1)+tf.reduce_sum(psi*y_one_hot,1)
        G = tf.reduce_mean(G)+reg_param*norm_square(dc_model.trainable_weights)
        return G
    elif component == 'H':
        H = tf.reduce_sum(psi,axis=1) + tf.reduce_sum(delta*y_one_hot,1)
        H = tf.reduce_mean(H)
        return H


def get_dc_model_v3(params):
    
    inputs = tf.keras.Input(shape=(784,),name='input')
    Dense = layers.Dense(64,name='dense')  
    DC1 = DC_V3_type1(units=64,rho1=params[0],rho2=params[1])
    DC2 = DC_V3_type2(units=10,rho1=params[2],rho2=params[3],rho3=params[4],\
                      kappa1=params[5],kappa2=params[6],kappa3=params[7])

    # stack layers together
    x = Dense(inputs)
    x = DC1(x)
    x = DC2(x)
    
    dc_model = keras.Model(inputs=inputs,outputs=x,name='dc_model')
    
    return dc_model