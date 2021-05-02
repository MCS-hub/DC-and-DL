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

class DC_V3_type1(keras.layers.Layer):
    def __init__(self,units=32,rho1=1,rho2=1):
        super(DC_V3_type1,self).__init__()
        self.units = units
        self.rho1 = rho1
        self.rho2 = rho2
        self.initializer = tf.keras.initializers.GlorotUniform()
    
    def build(self,input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],self.units),
            initializer = self.initializer,
            trainable = True,
            dtype='float64'
            )
        self.b = self.add_weight(
            shape = (self.units,), initializer=self.initializer,trainable = True,
            dtype='float64'
            )
        
    def call(self,inputs):
        
        ReLu_w = tf.keras.activations.relu(self.w)
        ReLu_b = tf.keras.activations.relu(self.b)
        ReLu_nw = ReLu_w - self.w
        ReLu_nb = ReLu_b - self.b        
        
        varphi = tf.keras.activations.relu(inputs)
        temp_term = self.rho1/2*tf.reduce_sum(tf.square(self.w),0,keepdims=True)+\
            self.rho2/2*tf.reduce_sum(tf.square(inputs),1,keepdims=True)
        G = tf.matmul(varphi,ReLu_w) + ReLu_b + temp_term
        H = tf.matmul(varphi,ReLu_nw) + ReLu_nb + temp_term
        
        return tf.concat([G,H],1)

class DC_V3_type2(keras.layers.Layer):
    def __init__(self,units=32,rho1=1,rho2=1,rho3=1,kappa1=1,kappa2=1,kappa3=1):
        super(DC_V3_type2,self).__init__()
        self.units = units
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.kappa3 = kappa3
        self.initializer = tf.keras.initializers.GlorotUniform()
        
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
        preG = inputs[:,:int(k/2)]
        preH = inputs[:,int(k/2):]
        
        ReLu_w = tf.keras.activations.relu(self.w)
        ReLu_b = tf.keras.activations.relu(self.b)
        ReLu_nw = ReLu_w - self.w
        ReLu_nb = ReLu_b - self.b 
        
        varphi = tf.keras.activations.relu(preG-preH) + preH
        vartheta = preH
        
        temp_term = (self.kappa1+self.rho1)/2*tf.reduce_sum(tf.square(self.w),0,keepdims=True)+\
            (self.kappa2+self.rho2)/2*tf.reduce_sum(tf.square(preH),1,keepdims=True)+\
                (self.kappa3+self.rho3)/2*tf.reduce_sum(tf.square(preG),1,keepdims=True)
        G = tf.matmul(varphi,ReLu_w) + tf.matmul(vartheta,ReLu_nw) + ReLu_b + temp_term
        H = tf.matmul(vartheta,ReLu_w) + tf.matmul(varphi,ReLu_nw) + ReLu_nb + temp_term
        
        return tf.concat([G,H],1)
        