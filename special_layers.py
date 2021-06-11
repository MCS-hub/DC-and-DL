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

    
# Consider activations: sigma = u - v with u', v' are not necessarily nonnegative
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
        
        
        temp_term = 0.5*(tf.reduce_sum(tf.square(varphi),1,keepdims=True)+tf.reduce_sum(tf.square(vartheta),1,keepdims=True)
                         +tf.reduce_sum(tf.square(self.w),0,keepdims=True))

        G = tf.matmul(varphi,ReLu_w) + tf.matmul(vartheta,ReLu_nw) + ReLu_b + temp_term 
        H = tf.matmul(varphi,ReLu_nw) + tf.matmul(vartheta,ReLu_w) + ReLu_nb + temp_term
        
        
        if self.activation:  #for a moment, consider ReLU only
            varphi = keras.activations.relu(G-H) + G+H
            vartheta = G+H
            return tf.concat([varphi,vartheta],1)
        else: 
           return tf.concat([G,H],1)


# Consider activations: sigma = u-v where u',v'>=0
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
           
        temp_term = 0.5*(tf.reduce_sum(tf.square(varphi),1,keepdims=True)+tf.reduce_sum(tf.square(vartheta),1,keepdims=True) 
                         + tf.reduce_sum(tf.square(self.w),0,keepdims=True))

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


# DC layers with ``rho decompositions" for the multiplication between the previous 
# layer and the weight matrix of the current layer

# The first DC layer: right after the first hidden layer
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


# the second DC layer: applicable for all other layers except for the first DC layer
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


# DC layers for different types of activations: relu, sigmoid, leaky_relu
# The first DC layer: right after the first hidden layer
class DC_V4_type1(keras.layers.Layer):
    def __init__(self,units=32,rho1=1,rho2=1,kappa1=0,kappa2=0,activation=None):
        super(DC_V4_type1,self).__init__()
        self.units = units
        self.rho1 = rho1
        self.rho2 = rho2
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.activation = activation
    
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
        
        if self.activation == None or self.activation == 'relu':
            varphi = tf.keras.activations.relu(inputs)
            vartheta = tf.multiply(0.0,inputs)
            
        elif self.activation == 'sigmoid':
            varphi = tf.math.log(1+tf.math.exp(inputs))
            vartheta = tf.math.log(1+tf.math.exp(inputs)) - (tf.math.exp(inputs))/(tf.math.exp(inputs)+1)

        elif self.activation == 'leaky_relu':
            varphi = tf.math.maximum(inputs,0)
            vartheta = tf.math.maximum(tf.multiply(-0.001,inputs),0)
        else:
            raise NameError('Activation is not valid.')

        temp_term = (self.rho1+self.kappa1)/2*tf.reduce_sum(tf.square(self.w),0,keepdims=True)+\
            (self.rho2+self.kappa2)/2*tf.reduce_sum(tf.square(inputs),1,keepdims=True)
        G = tf.matmul(varphi,ReLu_w) + tf.matmul(vartheta,ReLu_nw)+ ReLu_b + temp_term
        H = tf.matmul(varphi,ReLu_nw) + tf.matmul(vartheta,ReLu_w)+ ReLu_nb + temp_term
            
        return tf.concat([G,H],1)


# the second DC layer: applicable for all other layers except for the first DC layer
class DC_V4_type2(keras.layers.Layer):
    def __init__(self,units=32,rho1=1,rho2=1,rho3=1,kappa1=1,kappa2=1,kappa3=1,activation=None):
        super(DC_V4_type2,self).__init__()
        self.units = units
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.kappa3 = kappa3
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
        preG = inputs[:,:int(k/2)]
        preH = inputs[:,int(k/2):]
        
        ReLu_w = tf.keras.activations.relu(self.w)
        ReLu_b = tf.keras.activations.relu(self.b)
        ReLu_nw = ReLu_w - self.w
        ReLu_nb = ReLu_b - self.b 
        
        if self.activation == None or self.activation == 'relu':
            varphi = tf.keras.activations.relu(preG-preH) + preH
            vartheta = preH
        elif self.activation == 'sigmoid':
            Lipschitz = tf.constant(2.,dtype=tf.float64)
            varphi = tf.math.log(1+tf.math.exp(preG-preH)) + tf.multiply(Lipschitz,preH)
            vartheta = tf.math.log(1+tf.math.exp(preG-preH)) \
                - (tf.math.exp(preG-preH))/(tf.math.exp(preG-preH)+1) + tf.multiply(Lipschitz,preH)
        elif self.activation == 'leaky_relu':
            Lipschitz = tf.constant(1. + 0.001,dtype=tf.float64)
            varphi = tf.math.maximum(preG-preH,0.) + tf.multiply(Lipschitz,preH)
            vartheta = tf.math.maximum(tf.multiply(-0.001,preG-preH),0.) + tf.multiply(Lipschitz,preH)
        else:
            raise NameError('Activation is not valid.')
        
        temp_term = (self.kappa1+self.rho1)/2*tf.reduce_sum(tf.square(self.w),0,keepdims=True)+\
            (self.kappa2+self.rho2)/2*tf.reduce_sum(tf.square(preH),1,keepdims=True)+\
                (self.kappa3+self.rho3)/2*tf.reduce_sum(tf.square(preG),1,keepdims=True)
        G = tf.matmul(varphi,ReLu_w) + tf.matmul(vartheta,ReLu_nw) + ReLu_b + temp_term
        H = tf.matmul(vartheta,ReLu_w) + tf.matmul(varphi,ReLu_nw) + ReLu_nb + temp_term
        
        return tf.concat([G,H],1)
        