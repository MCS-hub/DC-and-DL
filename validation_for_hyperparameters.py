# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:18:14 2021

@author: luu2
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
tf.keras.backend.set_floatx('float64')

# set path
path = os.getcwd()

# parameters for the normal model
batch_size=64
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()  #cross-entropy loss
acc_metric = tf.keras.metrics.Accuracy()
# reg_param_list = [0.01,0.001,0.0001] #1e-4 #0.001
# learning_rate_list = [0.01,0.001,0.0001]
# epochs = 100

reg_param_list = [0.0001]
learning_rate_list = [0.0001]
epochs = 50


# get data --------------------------------------------------------------------
# load data
(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()

# flatten images
x_train = x_train.reshape(60000,784).astype("float64")/255
x_test = x_test.reshape(10000, 784).astype("float64") / 255

# split the training set into training-validation sets
x_val = x_train[50000:,:]
y_val = y_train[50000:]
x_train = x_train[:50000,:]
y_train = y_train[:50000]

# compute the y_train in tensor form to be used later
y_train_tensor = tf.convert_to_tensor(y_train,dtype=tf.float64)

# create, shuffle and batch the training set
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.batch(batch_size)



# find hyperparamters----------------------------------------------------------

for reg_param in reg_param_list:
    for learning_rate in learning_rate_list:
        
        # get a model
        model = get_model()
        
        #create lists to contain histories
        train_loss = []
        train_time = [0]
        train_accuracy = []
        val_accuracy = []
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
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
        
        loss = loss_fn(y_train_tensor,y_model_prob_train)+\
            reg_param*norm_square(model.trainable_weights)
        train_loss.append(loss.numpy())


        time0 = time.time()
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            for step, (x_batch,y_batch) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    output = tf.keras.activations.softmax(model(x_batch), axis=-1)
                    loss = loss_fn(y_batch,output) + reg_param*norm_square(model.trainable_weights)
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
                    
                    loss = loss_fn(y_train_tensor,y_model_prob_train)+\
                        reg_param*norm_square(model.trainable_weights)
                    train_loss.append(loss.numpy())
                    
                    print("train accuracy:",train_accuracy[-1],"val accuracy:",
                          val_accuracy[-1],"train loss:",train_loss[-1])
                    
                    time0 = time.time()
                
        # save the normal model
        model.save('/tmp/model/'+'Adam_reg_'+str(reg_param)+'_lr_'+str(learning_rate)+'sigmoid') 
        with open('objs'+'_Adam_reg_'+str(reg_param)+'_lr_'+str(learning_rate)+'sigmoid'+'_.pkl','wb') as f:  
            pickle.dump([train_loss,train_time,train_accuracy,val_accuracy],f)
        del model
#------------------------------------------------------------------------------

# plot graphs to choose hyperparameters
for reg_param in reg_param_list:
    for learning_rate in learning_rate_list:
        with open('objs'+'_Adam_reg_'+str(reg_param)+'_lr_'+str(learning_rate)+'_.pkl','rb') as f:  
            train_loss, train_time, train_accuracy, val_accuracy = pickle.load(f)
        
        plt.plot(val_accuracy,label='reg_'+str(reg_param)+'_lr_'+str(learning_rate))

plt.legend()
      
        