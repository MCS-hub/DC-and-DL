# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:42:05 2021

@author: luu2
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.keras import layers

    
    
class DC(keras.layers.Layer):
    def __init__(self,units=32,activation=None):
        super(DC,self).__init__()
        self.units=units
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.activation = activation
    
    def build(self,input_shape):
        self.w = self.add_weight(
            shape=(int(input_shape[-1]/2),self.units),
            initializer = self.initializer,
            trainable = True,
            dtype='float64'
            )
        self.b = self.add_weight(
            shape = (self.units,), initializer=self.initializer,trainable = True,
            dtype='float64'
            )
        
    def call(self,inputs):
        k = inputs.shape[-1]
        varphi = inputs[:,:int(k/2)]
        vartheta = inputs[:,int(k/2):]
        
    
        
        ReLu_w = tf.keras.activations.relu(self.w)
        ReLu_b = tf.keras.activations.relu(self.b)
        ReLu_nw = ReLu_w - self.w
        ReLu_nb = ReLu_b - self.b
        
        
        temp_term = 0.5*(tf.expand_dims(tf.reduce_sum(tf.square(varphi),1)+tf.reduce_sum(tf.square(vartheta),1),1) 
                         + tf.expand_dims(tf.reduce_sum(tf.square(self.w),0),0))

        G = tf.matmul(varphi,ReLu_w) + tf.matmul(vartheta,ReLu_nw) + ReLu_b + temp_term 
        H = tf.matmul(varphi,ReLu_nw) + tf.matmul(vartheta,ReLu_w) + ReLu_nb + temp_term
        
        
        if self.activation:  #for a moment, consider ReLU only
            varphi = keras.activations.relu(G-H) + G+H
            vartheta = G+H
            return tf.concat([varphi,vartheta],1)
        else: 
           return tf.concat([G,H],1)


class DC_V2(keras.layers.Layer):
    def __init__(self,units=32,activation=None):
        super(DC_V2,self).__init__()
        self.units=units
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.activation = activation
    
    def build(self,input_shape):
        self.w = self.add_weight(
            shape=(int(input_shape[-1]/2),self.units),
            initializer = self.initializer,
            trainable = True,
            dtype='float64'
            )
        self.b = self.add_weight(
            shape = (self.units,), initializer=self.initializer,trainable = True,
            dtype='float64'
            )
        
    def call(self,inputs):
        k = inputs.shape[-1]
        varphi = inputs[:,:int(k/2)]
        vartheta = inputs[:,int(k/2):]
        
        ReLu_w = tf.keras.activations.relu(self.w)
        ReLu_b = tf.keras.activations.relu(self.b)
        ReLu_nw = ReLu_w - self.w
        ReLu_nb = ReLu_b - self.b
           
        temp_term = 0.5*(tf.expand_dims(tf.reduce_sum(tf.square(varphi),1)+tf.reduce_sum(tf.square(vartheta),1),1) 
                         + tf.expand_dims(tf.reduce_sum(tf.square(self.w),0),0))

        G = tf.matmul(varphi,ReLu_w) + tf.matmul(vartheta,ReLu_nw) + ReLu_b + temp_term 
        H = tf.matmul(varphi,ReLu_nw) + tf.matmul(vartheta,ReLu_w) + ReLu_nb + temp_term
        
        if self.activation=='relu':  #for a moment, consider ReLU only
            varphi = keras.activations.relu(G-H) + H
            vartheta = H
            return tf.concat([varphi,vartheta],1)
        elif self.activation=='sigmoid':
            e = tf.math.exp(G-H)
            varphi = tf.math.log(1+e) + 2*H
            vartheta = tf.math.log(1+e) - e/(1+e) + 2*H
            return tf.concat([varphi,vartheta],1)
        else: 
           return tf.concat([G,H],1)
