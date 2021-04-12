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

from config import *


@tf.custom_gradient
def log_sum_exp(x):
    def grad(upstream):
        X = tf.expand_dims(x,1)-tf.expand_dims(x,2)
        return upstream*(1/tf.reduce_sum(tf.math.exp(X),2))
    z = tf.reduce_logsumexp(x,axis=1,keepdims=True)
    return z, grad



def train(dc_model,reg_param = 0.0001,epochs=3,num_iter_cnvx=3,damping_const_init=1000,\
          cg_eps=1e-3,cg_k_max=1000):
    
    global loss_fn, acc_metric, x_val, y_val, x_train, y_train, train_dataset
        
    model_len = len(dc_model.trainable_weights)

    # to save histories
    train_time = [0]
    train_loss = []
    train_accuracy = []
    val_accuracy = []
    
    # compute and save accuracy on the validation set
    outputs = dc_model(x_val)
    delta = outputs[:,:10]
    psi = outputs[:,10:]
    y_model_prob_val = keras.layers.Softmax()(delta-psi)
    y_model_label_val = y_model_prob_val.numpy().argmax(axis=-1)
    acc_metric.reset_states()
    acc_metric.update_state(y_model_label_val,y_val)
    val_accuracy.append(acc_metric.result().numpy())
    
    # compute and save loss and accuracy on the train set
    # accuracy
    outputs = dc_model(x_train)
    delta = outputs[:,:10]
    psi = outputs[:,10:]
    y_model_prob_train = keras.layers.Softmax()(delta-psi)
    y_model_label_train = y_model_prob_train.numpy().argmax(axis=-1)
    acc_metric.reset_states()
    acc_metric.update_state(y_model_label_train,y_train)
    train_accuracy.append(acc_metric.result().numpy())
    
    # loss
    loss = loss_fn(y_train_tensor,y_model_prob_train) + \
        reg_param*norm_square(dc_model.trainable_weights)
    train_loss.append(loss.numpy())
    
    time0 = time.time()
    
    for epoch in range(epochs):
        for step, (x_batch,y_batch) in enumerate(train_dataset):
            
            current_batch_size = y_batch.shape[0]
            y_one_hot = tf.one_hot(y_batch,depth=10,dtype=tf.float64)
            with tf.GradientTape() as tape:
                outputs = dc_model(x_batch)
                delta = outputs[:,:10]
                psi = outputs[:,10:]
                H = tf.reduce_sum(delta + psi,axis=1) + tf.reduce_sum(delta*y_one_hot,1)
                H = tf.reduce_sum(H)/current_batch_size
                
            G = tf.squeeze(log_sum_exp(delta-psi))\
                +tf.reduce_sum(delta+psi,axis=1)+tf.reduce_sum(psi*y_one_hot,1)
            G = tf.reduce_sum(G)/current_batch_size+reg_param*norm_square(dc_model.trainable_weights)
                
            gradH = tape.gradient(H,dc_model.trainable_weights)
            
            H0 = H
            G0 = G
            subloss0 = G0 - H0
            gradH0x0 = scalar_product(gradH,dc_model.trainable_weights)
            
            #con_grad0 = [0*elem for elem in gradH]  #xem lai
            con_grad = [0*elem for elem in gradH]
            damping_const = damping_const_init
            k_max =100
            i_count = 0
            while True:
                i_count += 1
                with tf.GradientTape() as tape:
                    outputs = dc_model(x_batch)
                    delta = outputs[:,:10]
                    psi = outputs[:,10:]
                    G = tf.squeeze(log_sum_exp(delta-psi))\
                        +tf.reduce_sum(delta+psi,axis=1)+tf.reduce_sum(psi*y_one_hot,1)
                    G = tf.reduce_sum(G)/current_batch_size +\
                        reg_param*norm_square(dc_model.trainable_weights)
                
                if i_count>=num_iter_cnvx:
                    if i_count > 10:
                        print("break the convex optimizer.")
                        break
                    gradH0x = scalar_product(gradH,dc_model.trainable_weights)
                    subloss = G-(H0+gradH0x-gradH0x0)
                    if subloss < subloss0:
                        print("break the convex optimizer.")
                        break
                
                
                gradG = tape.gradient(G, dc_model.trainable_weights)
                ngrad = [-gradG[idx]+gradH[idx] for idx in range(model_len)]
                
                #con_grad = con_grad0   #can be improved?
                with tf.GradientTape() as out_tape:
                    with tf.GradientTape() as in_tape:
                        outputs = dc_model(x_batch)
                        delta = outputs[:,:10]
                        psi = outputs[:,10:]
                        G = tf.squeeze(log_sum_exp(delta-psi))\
                            +tf.reduce_sum(delta+psi,axis=1)+tf.reduce_sum(psi*y_one_hot,1)
                        G = tf.reduce_sum(G)/current_batch_size +\
                              reg_param*norm_square(dc_model.trainable_weights)
                    gradG = in_tape.gradient(G,dc_model.trainable_weights)
                    elemwise_products = [
                        math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
                        for grad_elem, v_elem in zip(gradG, con_grad)
                        if grad_elem is not None]
                Hess_vec = out_tape.gradient(elemwise_products,dc_model.trainable_weights)
                reg_Hess_vec = [Hess_vec[idx]+damping_const*con_grad[idx] for idx in range(model_len)]
                r = [ngrad[idx]-reg_Hess_vec[idx] for idx in range(model_len)]
                p = r
                k = 0
                if norm_square(r) >= cg_eps:
                    while True:
                        with tf.GradientTape() as out_tape:
                            with tf.GradientTape() as in_tape:
                                outputs = dc_model(x_batch)
                                delta = outputs[:,:10]
                                psi = outputs[:,10:]
                                G = tf.squeeze(log_sum_exp(delta-psi))\
                                    +tf.reduce_sum(delta+psi,axis=1)+tf.reduce_sum(psi*y_one_hot,1)
                                G = tf.reduce_sum(G)/current_batch_size +reg_param*norm_square(dc_model.trainable_weights)
                            gradG = in_tape.gradient(G,dc_model.trainable_weights)
                            elemwise_products = [
                                math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
                                for grad_elem, v_elem in zip(gradG, p)
                                if grad_elem is not None]
                        #print("elemwise",elemwise_products)
                        Hess_vec = out_tape.gradient(elemwise_products,dc_model.trainable_weights)
                        reg_Hess_vec = [Hess_vec[idx]+damping_const*p[idx] for idx in range(model_len)]
                        rdotr = scalar_product(r,r)
                        alpha = rdotr/scalar_product(p,reg_Hess_vec)
                        con_grad = [con_grad[idx]+alpha*p[idx] for idx in range(model_len)]
                        r_next = [r[idx]-alpha*reg_Hess_vec[idx] for idx in range(model_len)]
                        
                        # if norm_square(r_next) < cg_eps or k>k_max:
                        #     break
                    
                        if norm_square(r_next)<min(0.25,tf.sqrt(norm_square(ngrad)))*norm_square(ngrad):
                            break
                        
                        rnextdotrnext = scalar_product(r_next,r_next)
                        beta = rnextdotrnext/rdotr
                        r = r_next
                        p = [r[idx]+beta*p[idx] for idx in range(model_len)]
                        k += 1
                else:
                    print("solution found or NaN in r!")
                
                gradH0x = scalar_product(gradH,dc_model.trainable_weights)
                subloss = G-(H0+gradH0x-gradH0x0)
                print("subloss",subloss.numpy())

                
                for idx in range(model_len):
                    dc_model.trainable_weights[idx].assign_add(con_grad[idx])
                
                damping_const = damping_const*1.1
    
            if step%20==0:
                d_time = time.time() - time0
                train_time.append(train_time[-1]+d_time)
                
                # compute and save accuracy on the validation set
                outputs = dc_model(x_val)
                delta = outputs[:,:10]
                psi = outputs[:,10:]
                y_model_prob_val = keras.layers.Softmax()(delta-psi)
                y_model_label_val = y_model_prob_val.numpy().argmax(axis=-1)
                acc_metric.reset_states()
                acc_metric.update_state(y_model_label_val,y_val)
                val_accuracy.append(acc_metric.result().numpy())
                
                # compute and save loss and accuracy on the train set
                # accuracy
                outputs = dc_model(x_train)
                delta = outputs[:,:10]
                psi = outputs[:,10:]
                y_model_prob_train = keras.layers.Softmax()(delta-psi)
                y_model_label_train = y_model_prob_train.numpy().argmax(axis=-1)
                acc_metric.reset_states()
                acc_metric.update_state(y_model_label_train,y_train)
                train_accuracy.append(acc_metric.result().numpy())
                
                # loss
                loss = loss_fn(y_train_tensor,y_model_prob_train) + \
                    reg_param*norm_square(dc_model.trainable_weights)
                train_loss.append(loss.numpy())
                
                print("train accuracy: ",round(train_accuracy[-1],4),"--val accuracy: ",
                      round(val_accuracy[-1],4),"--train loss: ",round(train_loss[-1],5))
                
                time0 = time.time()
    
    return train_time,train_loss,train_accuracy,val_accuracy