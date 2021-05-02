# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:51:31 2021

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
from utils.utils import *
from get_model import get_model
from matplotlib import pyplot as plt
from config import *

tf.keras.backend.set_floatx('float64')


# get a model
model = get_model()

# number of epochs
epochs = 1

#create lists to contain histories
train_loss = []
train_time = [0]
train_accuracy = []
val_accuracy = []
opt = tf.keras.optimizers.SGD(learning_rate=0.01)

# compute the accuracy in the validation set at the beginning
y_model_prob_val = tf.keras.activations.softmax(model(x_val),axis=-1)
y_model_label_val = y_model_prob_val.numpy().argmax(axis=-1)
acc_metric.reset_states()
acc_metric.update_state(y_model_label_val,y_val)
val_accuracy.append(acc_metric.result().numpy())

# compute the loss and accuracy in the training set at the beginning
y_model_prob_train = tf.keras.activations.softmax(model(x_train),axis=-1)
y_model_label_train = y_model_prob_train.numpy().argmax(axis=-1)
acc_metric.reset_states()
acc_metric.update_state(y_model_label_train,y_train)
train_accuracy.append(acc_metric.result().numpy())

loss = loss_fn(y_train_tensor,y_model_prob_train)
train_loss.append(loss.numpy())


time0 = time.time()
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    for step, (x_batch,y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            output = tf.keras.activations.softmax(model(x_batch), axis=-1)
            loss = loss_fn(y_batch,output)
        grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grads,model.trainable_weights))
   
        if step%20==0:
            d_time = time.time() - time0
            train_time.append(d_time+train_time[-1])
            
            # compute the accuracy in the validation set
            y_model_prob_val = tf.keras.activations.softmax(model(x_val),axis=-1)
            y_model_label_val = y_model_prob_val.numpy().argmax(axis=-1)
            acc_metric.reset_states()
            acc_metric.update_state(y_model_label_val,y_val)
            val_accuracy.append(acc_metric.result().numpy())
            
            # compute the loss and accuracy 
            y_model_prob_train = tf.keras.activations.softmax(model(x_train),axis=-1)
            y_model_label_train = y_model_prob_train.numpy().argmax(axis=-1)
            acc_metric.reset_states()
            acc_metric.update_state(y_model_label_train,y_train)
            train_accuracy.append(acc_metric.result().numpy())
            
            loss = loss_fn(y_train_tensor,y_model_prob_train)
            train_loss.append(loss.numpy())
            
            print("train accuracy:",train_accuracy[-1],"val accuracy:",
                  val_accuracy[-1],"train loss:",train_loss[-1])
            
            time0 = time.time()
        
# save the normal model

model.save('./normal_model_SGD.h5')

with open('normal_model_SGD.pkl','wb') as f:  
    pickle.dump([train_loss,train_time,train_accuracy,val_accuracy],f)
del model
