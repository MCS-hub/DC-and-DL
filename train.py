# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:18:36 2021

@author: luu2
"""
import tensorflow as tf
from tensorflow import keras
import time
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from utils.utils import norm_square, scalar_product
from get_model import get_DC_component
from evaluation_func import evaluate
from optimizers import osDCA, osDCA_V2


from config import *


def train(dc_model, reg_param = 0.0001,epochs=3,num_iter_cnvx=3,max_iter_cnvx=20,damping_const_init=1000,\
          cg_eps=1e-3,convex_optimizer='Adam',learning_rate=0.001):
    
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
        
        for step, (x_batch,y_batch) in enumerate(train_dataset):

            # osDCA(dc_model=dc_model,reg_param=reg_param,damping_const_init=damping_const_init,
            #       num_iter_cnvx=num_iter_cnvx,max_iter_cnvx=max_iter_cnvx,cg_eps=cg_eps,
            #       x_batch=x_batch,y_batch=y_batch)
            
            osDCA_V2(dc_model=dc_model,reg_param=reg_param,learning_rate=learning_rate,
                     num_iter_cnvx=num_iter_cnvx,max_iter_cnvx=max_iter_cnvx,
                     x_batch=x_batch,y_batch=y_batch,convex_optimizer=convex_optimizer)
            
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