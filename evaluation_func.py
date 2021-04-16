# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:13:06 2021

@author: luu2
"""
from tensorflow import keras
from utils.utils import norm_square
from config import x_train, y_train, x_val, y_val, y_train_tensor, loss_fn, acc_metric

def evaluate(dc_model,reg_param):
    
    # compute and save accuracy on the validation set
    outputs = dc_model(x_val)
    delta = outputs[:,:10]
    psi = outputs[:,10:]
    y_model_prob_val = keras.layers.Softmax()(delta-psi)
    y_model_label_val = y_model_prob_val.numpy().argmax(axis=-1)
    acc_metric.reset_states()
    acc_metric.update_state(y_model_label_val,y_val)
    val_accuracy = acc_metric.result().numpy()
    
    # compute and save loss and accuracy on the train set
    # accuracy
    outputs = dc_model(x_train)
    delta = outputs[:,:10]
    psi = outputs[:,10:]
    y_model_prob_train = keras.layers.Softmax()(delta-psi)
    y_model_label_train = y_model_prob_train.numpy().argmax(axis=-1)
    acc_metric.reset_states()
    acc_metric.update_state(y_model_label_train,y_train).numpy()
    train_accuracy = acc_metric.result().numpy()
    
    # loss
    loss = (loss_fn(y_train_tensor,y_model_prob_train) + \
        reg_param*norm_square(dc_model.trainable_weights)).numpy()

    
    return val_accuracy, train_accuracy, loss