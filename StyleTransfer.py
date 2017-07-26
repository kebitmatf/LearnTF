# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:12:35 2017

@author: SEA
"""

import tensorflow as tf
import numpy as np
import os 
import scipy.misc as misc
import matplotlib.pyplot as plt
from cs231n.classifiers.squeezenet import SqueezeNet
from cs231n.image_utils import load_image, preprocess_image, deprocess_image

tf.reset_default_graph()
session = tf.Session()

#%% Load SqueezeNet model
model_path = 'cs231n/datasets/squeezenet.ckpt'
model = SqueezeNet(model_path, session)

content_imag_test = preprocess_image(load_image('styles/tubingen.jpg', size=192))[None]
style_imag_test = preprocess_image(load_image('styles/starry_night.jpg', size=192))[None]
answers = np.load('style-transfer-checks-tf.npz')
#%% Define loss functions

# loss content
def loss_content(content_imag, mix_imag, layer):
    content = session.run(model.extract_features()[layer], {model.image: content_imag})
    mix = model.extract_features(mix_imag)[layer]
    loss = tf.reduce_sum(tf.pow(content - mix, 2))
    return loss

# loss style
def loss_style(style_imag, mix_imag, layers):
    
    def gram_matrix(F): # F: filter output size ()        
        F = tf.reshape(F, [-1, tf.shape(F)[3]])
        gram = tf.matmul(tf.transpose(F), F)
        return gram
    
    def loss_gram (a, x):
        shape = tf.shape(a)
        N = shape[3]
        M = shape[1] * shape[2]        
        A = gram_matrix(a)
        X = gram_matrix(x)
        loss_gram = tf.cast(1/(4 * N**2 * M**2), tf.float32) * tf.reduce_sum(tf.pow(A - X, 2))  
#        loss_gram =  tf.reduce_sum(tf.pow(A - X, 2))  
        return loss_gram
    
    loss_style = 0
    for layer, weight in layers:
        style = session.run(model.extract_features()[layer], {model.image: style_imag})
        mix = model.extract_features(mix_imag)[layer]
        loss_style += weight * loss_gram(style, mix)
    
    return loss_style

# Test loss_content
mix_imag_test = tf.zeros(content_imag_test.shape)
t1 = 6e-2 * session.run(loss_content(content_imag_test, mix_imag_test, 3))
t2 = answers['cl_out']

# Test loss_style

mix_imag_test = tf.zeros(content_imag_test.shape)

layers = [(1,300000), (4,1000), (6,15), (7,3)]
t1 = session.run(loss_style(style_imag_test, content_imag_test, layers))
t2 = answers['sl_out']

#%%

    



