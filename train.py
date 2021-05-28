# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:18:36 2021
@author: luu2
"""
import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from utils.utils import norm_square, scalar_product
from get_model import get_DC_component
from evaluation_func import evaluate, evaluate_normal_model
from optimizers import osDCA, osDCA_V2, osDCAlike_V2


from config import *


# train using osDCA with Hessian-free
def train(dc_model, reg_param = 0.0001,epochs=3,num_iter_cnvx=3,max_iter_cnvx=20,damping_const_init=1000,\
          cg_eps=1e-3,convex_optimizer='Adam',learning_rate=0.001,ld=0):
    
    global loss_fn, acc_metric, x_val, y_val, x_train, y_train, train_dataset
        
    model_len = len(dc_model.trainable_weights)

    # to save histories
    train_time = [0]
    train_loss = []
    train_accuracy = []
    val_accuracy = []
    
    val_acc, train_acc, loss = evaluate(dc_model,reg_param)
    
    val_accuracy.append(val_acc)
    train_accuracy.append(train_acc)
    train_loss.append(loss)
    
    time0 = time.time()
    pre_t = 1.0
    for epoch in range(epochs):
           
        for step, (x_batch,y_batch) in enumerate(train_dataset):


            # osDCA(dc_model=dc_model,reg_param=reg_param,damping_const_init=damping_const_init,
            #       num_iter_cnvx=num_iter_cnvx,max_iter_cnvx=max_iter_cnvx,cg_eps=cg_eps,
            #       x_batch=x_batch,y_batch=y_batch)
            
            t = (1+np.sqrt(1+4*pre_t**2))/2
            Nesterov_coef = (pre_t-1)/t
            pre_t = t
            #print('Nesterov_coef: ',Nesterov_coef)
            
            osDCA_V2(dc_model=dc_model,reg_param=reg_param,learning_rate=learning_rate,
                     num_iter_cnvx=num_iter_cnvx,max_iter_cnvx=max_iter_cnvx,
                     x_batch=x_batch,y_batch=y_batch,convex_optimizer=convex_optimizer,Nesterov_coef=Nesterov_coef,ld=ld)
            
            if step%20==0:
                
                d_time = time.time() - time0
                
                train_time.append(train_time[-1]+d_time)
                
                val_acc, train_acc, loss = evaluate(dc_model,reg_param)
    
                val_accuracy.append(val_acc)
                train_accuracy.append(train_acc)
                train_loss.append(loss)
                
                print("train accuracy: ",round(train_accuracy[-1],8),"--val accuracy: ",
                      round(val_accuracy[-1],8),"--train loss: ",round(train_loss[-1],5))
                
                time0 = time.time()
    
    return train_time, train_loss, train_accuracy, val_accuracy


# train using osDCA with first-order optimizers
def train_v2(dc_model, reg_param = 0.0001,epochs=3,num_iter_cnvx=3,max_iter_cnvx=20,convex_optimizer='Adam',learning_rate=0.001,ld=0):
    
    global loss_fn, acc_metric, x_val, y_val, x_train, y_train, train_dataset
        
    model_len = len(dc_model.trainable_weights)

    # to save histories
    train_time = [0]
    train_loss = []
    train_accuracy = []
    val_accuracy = []
    
    val_acc, train_acc, loss = evaluate(dc_model,reg_param)
    
    val_accuracy.append(val_acc)
    train_accuracy.append(train_acc)
    train_loss.append(loss)
    
    time0 = time.time()
    
    for epoch in range(epochs):
        
        pre_t = 1.0   
        for step, (x_batch,y_batch) in enumerate(train_dataset):

            t = (1+np.sqrt(1+4*pre_t**2))/2
            Nesterov_coef = (pre_t-1)/t
            pre_t = t
            #print('Nesterov_coef: ',Nesterov_coef)
            
            osDCA_V2(dc_model=dc_model,reg_param=reg_param,learning_rate=learning_rate,
                     num_iter_cnvx=num_iter_cnvx,max_iter_cnvx=max_iter_cnvx,
                     x_batch=x_batch,y_batch=y_batch,convex_optimizer=convex_optimizer,Nesterov_coef=Nesterov_coef,ld=ld)
            
            if step%20==0:
                
                d_time = time.time() - time0
                
                train_time.append(train_time[-1]+d_time)
                
                val_acc, train_acc, loss = evaluate(dc_model,reg_param)
    
                val_accuracy.append(val_acc)
                train_accuracy.append(train_acc)
                train_loss.append(loss)
                
                print("train accuracy: ",round(train_accuracy[-1],8),"--val accuracy: ",
                      round(val_accuracy[-1],8),"--train loss: ",round(train_loss[-1],5))
                
                time0 = time.time()
    
    return train_time, train_loss, train_accuracy, val_accuracy


# train using osDCA-like
def train_v3(dc_model,params_decom,reg_param = 0.0001,epochs=3,num_iter_cnvx=3,max_iter_cnvx=20,convex_optimizer='Adam',learning_rate=0.001,ld=0):
    
    global loss_fn, acc_metric, x_val, y_val, x_train, y_train, train_dataset
        
    model_len = len(dc_model.trainable_weights)

    # to save histories
    train_time = [0]
    train_loss = []
    train_accuracy = []
    val_accuracy = []
    
    val_acc, train_acc, loss = evaluate(dc_model,reg_param)
    
    val_accuracy.append(val_acc)
    train_accuracy.append(train_acc)
    train_loss.append(loss)
    
    time0 = time.time()
    
    for epoch in range(epochs):
        
        pre_t = 1.0   
        for step, (x_batch,y_batch) in enumerate(train_dataset):

            
            t = (1+np.sqrt(1+4*pre_t**2))/2
            Nesterov_coef = (pre_t-1)/t
            pre_t = t
            #print('Nesterov_coef: ',Nesterov_coef)
            
            updated_weights = osDCAlike_V2(dc_model.get_weights(),params_decom=params_decom,reg_param=reg_param,\
                         learning_rate=learning_rate,num_iter_cnvx=num_iter_cnvx,max_iter_cnvx=max_iter_cnvx,\
                             x_batch=x_batch,y_batch=y_batch,convex_optimizer=convex_optimizer,Nesterov_coef=Nesterov_coef,ld=ld)
            
            dc_model.set_weights(updated_weights)
                
            if step%20==0:
                
                d_time = time.time() - time0
                
                train_time.append(train_time[-1]+d_time)
                
                val_acc, train_acc, loss = evaluate(dc_model,reg_param)
    
                val_accuracy.append(val_acc)
                train_accuracy.append(train_acc)
                train_loss.append(loss)
                
                print("train accuracy: ",round(train_accuracy[-1],8),"--val accuracy: ",
                      round(val_accuracy[-1],8),"--train loss: ",round(train_loss[-1],5))
                
                time0 = time.time()
    
    return train_time, train_loss, train_accuracy, val_accuracy


def train_normal_model(model, reg_param = 0.0001,epochs=3,optimizer='Adam',learning_rate=0.001):
    
    global loss_fn, acc_metric, x_val, y_val, x_train, y_train, train_dataset
        
    model_len = len(model.trainable_weights)

    # to save histories
    train_time = [0]
    train_loss = []
    train_accuracy = []
    val_accuracy = []
    
    if optimizer == 'SGD':
        opt =  tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'Adamax':
        opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer == 'Adam':
        opt =  tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'Nadam':
        opt =  tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        opt =  tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'Ftrl':
        opt =  tf.keras.optimizers.Ftrl(learning_rate=learning_rate)
    elif optimizer == 'Adagrad':
        opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer == 'Adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    else:
        raise NameError('The convex optimizer is not valid.')
    
    val_acc, train_acc, loss = evaluate_normal_model(model,reg_param)
    
    val_accuracy.append(val_acc)
    train_accuracy.append(train_acc)
    train_loss.append(loss)
    
    time0 = time.time()
    
    for epoch in range(epochs):
        for step, (x_batch,y_batch) in enumerate(train_dataset):
            
            with tf.GradientTape() as tape:
                output = tf.keras.activations.softmax(model(x_batch), axis=-1)
                loss = loss_fn(y_batch,output) + reg_param*norm_square(model.trainable_weights)
            grads = tape.gradient(loss, model.trainable_weights)
            opt.apply_gradients(zip(grads,model.trainable_weights))
            
            if step%20==0:
                
                d_time = time.time() - time0
                
                train_time.append(train_time[-1]+d_time)
                
                val_acc, train_acc, loss = evaluate_normal_model(model,reg_param)
    
                val_accuracy.append(val_acc)
                train_accuracy.append(train_acc)
                train_loss.append(loss)
                
                print("train accuracy: ",round(train_accuracy[-1],8),"--val accuracy: ",
                      round(val_accuracy[-1],8),"--train loss: ",round(train_loss[-1],5))
                
                time0 = time.time()
    
    return train_time, train_loss, train_accuracy, val_accuracy