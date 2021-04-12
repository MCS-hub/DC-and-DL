# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 13:43:25 2021

@author: luu2
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from special_layers import DC_V2

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
    x = tf.concat([x,0*x],axis=1)
    x = DC1(x)
    x = DC2(x)
    
    dc_model = keras.Model(inputs=inputs,outputs=x,name='dc_model')
    
    return dc_model