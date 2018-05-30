# coding: utf-8
'''
Created on 2018. 5. 25.

@author: Insup Jung
'''

import tensorflow as tf
import tensorboard as tb
import numpy as np
from tensorflow.python.ops import control_flow_ops

class CNN:
    def __init__(self):
        
        learning_rate = 0.0001
        training_epochs = 20
        batch_size = 100
        display_step = 10
        
        
        with tf.Graph().as_default():
            sess = tf.Session()
            x = tf.placeholder(tf.float32, shape=[None, 784]) # define input value
            t = tf.placeholder(tf.float32, shape=[None, 4]) # target Value 4개의 blood cell
            keep_prob = tf.placeholder(tf.float32) # drop_out 비율
            phase_train = tf.placeholder(tf.bool) #train phase인지 아니면 validation이나 test phase 인지 확인
            
    
    def conv2d(self, input, weight_shape, bias_shape, keep_prob):
        weightX = weight_shape[0]*weight_shape[1]*weight_shape[2]
        
        W_initializer = tf.random_normal_initializer(stddev=(2.0/weightX)**0.5)
        W = tf.get_variable('W', shape=weight_shape, initializer=W_initializer)
        b_initializer = tf.constant_initializer(value=0)
        b = tf.get_variable('b', shape=bias_shape, initializer=b_initializer)
        conv_out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
        logits = tf.nn.bias_add(conv_out, b)
        
        return tf.nn.relu(self.conv2d(logits, weight_shape[3], bias_shape, keep_prob))
    
    def max_pool(self, input, k=2):
        return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
    def conv_batch_norm(self, x, n_out, phase_train):
        beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
        gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)
        
        beta = tf.get_variable("beta", [n_out], initializer=beta_init)
        gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)
        
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = control_flow_ops.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))
        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, True)
        
        return normed
    
    def layer(self, input, weight_shape, bias_shape, keep_prob):
        W_initializer = tf.random_normal_initializer(stddev = (2.0/weight_shape[0])**0.5)
        b_initialzier = tf.constant_initializer(value=0)
        
        W = tf.get_variable("W", shape=weight_shape, initializer=W_initializer)
        b = tf.get_variable("b", shape=bias_shape, initialzier = b_initialzier)
        
        logits = tf.matmul(input, W) + b
        return tf.nn.relu(self.layer_batch_norm(logits, weight_shape[1], keep_prob))
    
    def layer_batch_norm(self, x, n_out, phase_train):
        beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
        gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)
        
        beta = tf.get_variable("beta", [n_out], initializer=beta_init)
        gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)
        
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = control_flow_ops.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))
        x_r = tf.reshape(x, [-1, 1, 1, n_out])
        print('n_out', n_out)
        normed = tf.nn.batch_norm_with_global_normalization(x_r, mean, var, beta, gamma, 1e-3, True)
        
        return tf.reshape(normed, [-1, n_out])
    
    def lastlayer(self, input, weight_shape, bias_shape, keep_prob):
        W_initializer = tf.random_normal_initializer(stddev = (2.0/weight_shape[0])**0.5)
        b_initialzier = tf.constant_initializer(value=0)
        
        W = tf.get_variable("W", shape=weight_shape, initializer=W_initializer)
        b = tf.get_variable("b", shape=bias_shape, initialzier = b_initialzier)
        
        logits = tf.matmul(input, W) + b
        return tf.nn.softmax(logits)
    
    def inference(self, x, keep_prob, phase_train):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        with tf.variable_scope("conv_1"):
            conv_1 = self.conv2d(x, [5, 5, 1, 32], [32], phase_train) #[가로, 세로, 입력수, 출력수] 입력수는 1이다. 채널이 1(흑백)이기 때문
            pool_1 = self.max_pool(conv_1)
        
        with tf.variable_scope("conv_2"):
            conv_2 = self.conv2d(pool_1, [5, 5, 32, 64], [64], phase_train)
            pool_2 = self.max_pool(conv_2)
        
        with tf.variable_scope("fc"):
            pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64]) #4차원 배열을 2차원 배열로 만들어줌
            fc_1 = self.layer(pool_2_flat, [7*7*64, 1024], [1024], phase_train)
            fc_1_drop = tf.nn.dropout(fc_1, keep_prob)
            
        with tf.variable_scope("output"):
            output = self.lastlayer(fc_1_drop, [1024, 10], [10])
        
        return output
    
    def loss(self, output, t):
        loss = tf.reduce_mean(-tf.reduce_sum(t*tf.log(tf.clip_by_value(output, 1e-10, 1.0)), reduction_indices=[1]))
        return loss
    
    def evaluate(self, output, t):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
        return accuracy
    
    def training(self, cost, global_step, learning_rate):
        tf.summary.scalar("cost", cost)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(cost)        
        return train_op
        
            