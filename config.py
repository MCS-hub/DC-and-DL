# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:50:56 2021

@author: luu2
"""

import tensorflow as tf
from tensorflow import keras

# parameters for the normal model
batch_size=64
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()  #cross-entropy loss
acc_metric = tf.keras.metrics.Accuracy()

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