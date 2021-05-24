# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:13:15 2021

@author: Phuc Hau
"""

import tensorflow as tf
from get_model import get_DC_component
from utils.utils import norm_square, scalar_product
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from get_model import get_dc_model_v3
     

# osDCA with Hessian-free for solving convex subproblems
def osDCA(dc_model,reg_param,damping_const_init,num_iter_cnvx,max_iter_cnvx,cg_eps,x_batch,y_batch):
    
    y_one_hot = tf.one_hot(y_batch,depth=10,dtype=tf.float64)
    model_len = len(dc_model.trainable_weights)
    
    with tf.GradientTape() as tape:
        G, H = get_DC_component(dc_model,x_batch,y_one_hot, component='both',reg_param=reg_param)
        
    gradH = tape.gradient(H,dc_model.trainable_weights)
    
    H0 = H
    G0 = G
    subloss0 = G0 - H0
    print("subloss0: ",subloss0.numpy())
    gradH0x0 = scalar_product(gradH,dc_model.trainable_weights)
    
    con_grad = [0*elem for elem in gradH]
    damping_const = damping_const_init
    i_count = 0
    
    while True:
        i_count += 1
        
        with tf.GradientTape() as tape:
            G = get_DC_component(dc_model,x_batch,y_one_hot, component='G',reg_param=reg_param)
        
        # check condition to break the convex solver
        if i_count > num_iter_cnvx:
            if i_count > max_iter_cnvx:
                print("break the convex solver after 10 iterations.")
                break
            
            gradH0x = scalar_product(gradH,dc_model.trainable_weights)
            subloss = G-(H0+gradH0x-gradH0x0)
            print("subloss:  ", subloss.numpy())
            if subloss < subloss0:
                print("break the convex solver after finding better solution.")
                break
        
        gradG = tape.gradient(G, dc_model.trainable_weights)
        ngrad = [-gradG[idx]+gradH[idx] for idx in range(model_len)]
        
        con_grad = CGD(dc_model,ngrad,con_grad,reg_param,damping_const,cg_eps,x_batch,y_one_hot)

        
        for idx in range(model_len):
            dc_model.trainable_weights[idx].assign_add(con_grad[idx])
        
        #damping_const = damping_const*1.1
        
        
# Conjugate Gradient Descent       
def CGD(dc_model,ngrad,con_grad,reg_param,damping_const,cg_eps,x_batch,y_one_hot):
    
    model_len = len(dc_model.trainable_weights)
    
    with tf.GradientTape() as out_tape:
        with tf.GradientTape() as in_tape:
            G = get_DC_component(dc_model,x_batch,y_one_hot, component='G',reg_param=reg_param)
            
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
                    G = get_DC_component(dc_model,x_batch,y_one_hot, component='G',reg_param=reg_param)
                
                gradG = in_tape.gradient(G,dc_model.trainable_weights)
                elemwise_products = [
                    math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
                    for grad_elem, v_elem in zip(gradG, p)
                    if grad_elem is not None]

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
        
    return con_grad


# osDCA with first order methods for solving convex subproblems: especially Adamax
def osDCA_V2(dc_model,reg_param,learning_rate,num_iter_cnvx,max_iter_cnvx,x_batch,y_batch,convex_optimizer,Nesterov_coef=None,ld=0):
    #subhistory = []
    
    y_one_hot = tf.one_hot(y_batch,depth=10,dtype=tf.float64)
    model_len = len(dc_model.trainable_weights)
    
    with tf.GradientTape() as tape:
        G, H = get_DC_component(dc_model,x_batch,y_one_hot, component='both',reg_param=reg_param)
        G =  G + ld*norm_square(dc_model.trainable_weights)
        H = H + ld*norm_square(dc_model.trainable_weights)
        
    gradH = tape.gradient(H,dc_model.trainable_weights)
    
    H0 = H
    G0 = G
    subloss0 = G0 - H0
    print("subloss0:",subloss0.numpy())
    #subhistory.append(subloss0.numpy())
    
    gradH0x0 = scalar_product(gradH,dc_model.trainable_weights)
    
    i_count = 0
    
    if convex_optimizer == 'SGD':
        cnv_opt =  tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif convex_optimizer == 'Adamax':
        if type(learning_rate) == float:
            cnv_opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
        else:
            cnv_opt =  tf.keras.optimizers.Adamax(learning_rate=learning_rate[0],beta_1=learning_rate[1],
                                                  beta_2=learning_rate[2],epsilon=learning_rate[3])
    elif convex_optimizer == 'Adam':
        cnv_opt =  tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif convex_optimizer == 'Nadam':
        cnv_opt =  tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif convex_optimizer == 'RMSprop':
        cnv_opt =  tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif convex_optimizer == 'Ftrl':
        cnv_opt =  tf.keras.optimizers.Ftrl(learning_rate=learning_rate)
    elif convex_optimizer == 'Adagrad':
        cnv_opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif convex_optimizer == 'Adadelta':
        cnv_opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    else:
        raise NameError('The convex optimizer is not valid.')
        
        
    if Nesterov_coef is not None:
        pre_weights = dc_model.get_weights()
    
    weights0 = dc_model.get_weights()
    
    while True:
        i_count += 1
        
        with tf.GradientTape() as tape:
            G = get_DC_component(dc_model,x_batch,y_one_hot, component='G',reg_param=reg_param)
            G = G + ld*norm_square(dc_model.trainable_weights)
                
        # check condition to break the convex solver
        if i_count > num_iter_cnvx:
            if i_count > max_iter_cnvx:
                dc_model.set_weights(weights0)
                print("break the convex solver after max iterations.")
                break
            
            gradH0x = scalar_product(gradH,dc_model.trainable_weights)
            subloss = G-(H0+gradH0x-gradH0x0)
            #print("subloss:  ", subloss.numpy())
            if subloss < subloss0:
                print("break the convex solver after finding better solution.")
                break
        
        gradG = tape.gradient(G, dc_model.trainable_weights)
        grads = [gradG[idx]-gradH[idx] for idx in range(model_len)]
        
        cnv_opt.apply_gradients(zip(grads,dc_model.trainable_weights))
        
        # gradH0x = scalar_product(gradH,dc_model.trainable_weights)
        # G = get_DC_component(dc_model,x_batch,y_one_hot, component='G',reg_param=reg_param)\
        #     + ld*norm_square(dc_model.trainable_weights)
        # subloss = G-(H0+gradH0x-gradH0x0)
        # print("subloss: ", subloss.numpy())
        #subhistory.append(subloss.numpy())
    
    #return subhistory
    
    if Nesterov_coef is not None:
        d = [Nesterov_coef*(dc_model.get_weights()[elem] - pre_weights[elem]) for elem in range(model_len)]
        
        for idx in range(model_len):                            # can accelerate using set_weights
            dc_model.trainable_weights[idx].assign_add(d[idx])
            


def osDCAlike_V2(init_weights,params_decom,reg_param,learning_rate,num_iter_cnvx,max_iter_cnvx,x_batch,y_batch,convex_optimizer,Nesterov_coef=None,ld=0):

    y_one_hot = tf.one_hot(y_batch,depth=10,dtype=tf.float64)
    model_len = len(init_weights)
    
    while True:
        
        dc_model = get_dc_model_v3(params_decom,init_weights=init_weights)
        
        with tf.GradientTape() as tape:
            G, H = get_DC_component(dc_model,x_batch,y_one_hot, component='both',reg_param=reg_param)
            G =  G + ld*norm_square(dc_model.trainable_weights)
            H = H + ld*norm_square(dc_model.trainable_weights)
            
        gradH = tape.gradient(H,dc_model.trainable_weights)
        
        H0 = H
        G0 = G
        subloss0 = G0 - H0
        print("subloss0:",subloss0.numpy())
        
        gradH0x0 = scalar_product(gradH,dc_model.get_weights())
        
        i_count = 0
        
        if convex_optimizer == 'SGD':
            cnv_opt =  tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif convex_optimizer == 'Adamax':
            if type(learning_rate) == float:
                cnv_opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
            else:
                cnv_opt =  tf.keras.optimizers.Adamax(learning_rate=learning_rate[0],beta_1=learning_rate[1],
                                                      beta_2=learning_rate[2],epsilon=learning_rate[3])
        elif convex_optimizer == 'Adam':
            cnv_opt =  tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif convex_optimizer == 'Nadam':
            cnv_opt =  tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        elif convex_optimizer == 'RMSprop':
            cnv_opt =  tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif convex_optimizer == 'Ftrl':
            cnv_opt =  tf.keras.optimizers.Ftrl(learning_rate=learning_rate)
        elif convex_optimizer == 'Adagrad':
            cnv_opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif convex_optimizer == 'Adadelta':
            cnv_opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        else:
            raise NameError('The convex optimizer is not valid.')
             
            
        while True:
            i_count += 1
            
            with tf.GradientTape() as tape:
                G = get_DC_component(dc_model,x_batch,y_one_hot, component='G',reg_param=reg_param)
                G = G + ld*norm_square(dc_model.trainable_weights)
                    
            # check condition to break the convex solver
            if i_count > num_iter_cnvx:
                if i_count > max_iter_cnvx:
                    print("break the convex solver after max iterations.")
                    break
                
                gradH0x = scalar_product(gradH,dc_model.trainable_weights)
                subloss = G-(H0+gradH0x-gradH0x0)
                #print("subloss:  ", subloss.numpy())
                if subloss < subloss0:
                    print("break the convex solver after finding better solution.")
                    break
            
            gradG = tape.gradient(G, dc_model.trainable_weights)
            grads = [gradG[idx]-gradH[idx] for idx in range(model_len)]
            
            cnv_opt.apply_gradients(zip(grads,dc_model.trainable_weights))
            
            gradH0x = scalar_product(gradH,dc_model.trainable_weights)
            G = get_DC_component(dc_model,x_batch,y_one_hot, component='G',reg_param=reg_param)\
                + ld*norm_square(dc_model.trainable_weights)
            subloss = G-(H0+gradH0x-gradH0x0)
            print("subloss: ", subloss.numpy())
        
        H_tmp = get_DC_component(dc_model,x_batch,y_one_hot,component='H')+ld*norm_square(dc_model.get_weights())
        diff_x = [dc_model.get_weights()[i]-init_weights[i] for i in range(model_len)]
        H_linear = H0 + scalar_product(gradH,diff_x)
        if H_tmp >= H_linear:
            print('osDCA-like: a better solution found.')
            break
        else:
            params_decom[0] = 2*params_decom[0]
            params_decom[2] = 2*params_decom[2]
            params_decom[5] = 2*params_decom[5]
            print('osDCAlike: increase convexity.')
 
        
   # accelerate
    if Nesterov_coef is not None:
        d = [Nesterov_coef*(dc_model.get_weights()[idx] - init_weights[idx]) for idx in range(model_len)]
        updated_weights = [dc_model.get_weights()[idx] + d[idx] for idx in range(model_len)]
        return updated_weights
    else:
        return dc_model.get_weights()

