# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:57:38 2021

@author: luu2
"""
import tensorflow as tf


def norm_square(list_of_tensors):
    return sum([tf.reduce_sum(tf.square(tensor)) for tensor in list_of_tensors])

def scalar_product(x,y):
    return sum([tf.reduce_sum(x[idx]*y[idx]) for idx in range(len(x))])


@tf.custom_gradient
def log_sum_exp(x):
    def grad(upstream):
        X = tf.expand_dims(x,1)-tf.expand_dims(x,2)
        return upstream*(1/tf.reduce_sum(tf.math.exp(X),2))
    z = tf.reduce_logsumexp(x,axis=1,keepdims=True)
    return z, grad