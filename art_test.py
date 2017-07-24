# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:47:11 2017

@author: SEA
"""

import os
import tensorflow as tf
import scipy.io as io
import scipy.misc as misc
import matplotlib.pyplot as plt
import numpy as np
import time

path_vggmodel = 'imagenet-vgg-verydeep-19.mat'
path_style = 'images/guernica.jpg'
path_content = 'images/hongkong.jpg'
path_out = 'test_output'

imag_height = 600
imag_width = 800
color_channels = 3
mean_value = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

alpha = 100
beta = 5
noise_ratio = 0.6

def imag_load (path):
    imag = misc.imread(path)
    if  imag.shape[0:2] != (imag_height, imag_width):
        imag = misc.imresize(imag, (imag_height, imag_width), 'bilinear')
#    plt.imshow(imag)
#    plt.show
    imag = imag.reshape((1,)+imag.shape) - mean_value
    return imag

def imag_save(path, name, imag):
    file_name = path+'/'+name 
    imag = imag + mean_value
    imag = imag.reshape(imag[0])
    imag = np.clip(imag, 0, 255)
    
#    plt.imshow(imag)
#    plt.show
    misc.imsave(file_name, imag)

def imag_noise(imag, noise_ratio):
    noise = np.random.uniform(-20, 20, (1, imag_height, imag_width, color_channels))
    imag = noise * noise_ratio + imag * (1-noise_ratio)
    imag = np.clip(imag, 0, 255)
    
#    plt.imshow(imag[0])
#    plt.show
    return imag

# Get params in VGG and modify the model maxpool -> avgpool
def vgg_model(path):
    """
    0 is conv1_1 (3, 3, 3, 64)
    1 is relu
    2 is conv1_2 (3, 3, 64, 64)
    3 is relu    
    4 is maxpool
    5 is conv2_1 (3, 3, 64, 128)
    6 is relu
    7 is conv2_2 (3, 3, 128, 128)
    8 is relu
    9 is maxpool
    10 is conv3_1 (3, 3, 128, 256)
    11 is relu
    12 is conv3_2 (3, 3, 256, 256)
    13 is relu
    14 is conv3_3 (3, 3, 256, 256)
    15 is relu
    16 is conv3_4 (3, 3, 256, 256)
    17 is relu
    18 is maxpool
    19 is conv4_1 (3, 3, 256, 512)
    20 is relu
    21 is conv4_2 (3, 3, 512, 512)
    22 is relu
    23 is conv4_3 (3, 3, 512, 512)
    24 is relu
    25 is conv4_4 (3, 3, 512, 512)
    26 is relu
    27 is maxpool
    28 is conv5_1 (3, 3, 512, 512)
    29 is relu
    30 is conv5_2 (3, 3, 512, 512)
    31 is relu
    32 is conv5_3 (3, 3, 512, 512)
    33 is relu
    34 is conv5_4 (3, 3, 512, 512)
    35 is relu
    36 is maxpool
    37 is fullyconnected (7, 7, 512, 4096)
    38 is relu
    39 is fullyconnected (1, 1, 4096, 4096)
    40 is relu
    41 is fullyconnected (1, 1, 4096, 1000)
    42 is softmax
    """
    vgg = io.loadmat(path)
    vgg_layers = vgg['layers']
    
    def _weight(layer, expected_layer_name):
        W = vgg_layers[0][layer][0][0][0][0][0]
        b = vgg_layers[0][layer][0][0][0][0][1]
        layer_name = vgg_layers[0][layer][0][0][-2]
        assert layer_name == expected_layer_name
        return W, b
    
    def _relu(layer):
        return tf.nn.relu(layer)
    
    def _conv2d(prev_layer, layer, layer_name):
        W, b = _weight(layer, layer_name)
        b = b.reshape(b.size)
        W = tf.constant(W)
        b = tf.constant(b)
        layer = tf.nn.conv2d(input = prev_layer, filter = W, strides = [1,1,1,1], padding= 'SAME') + b
        return layer
   
    def _avgpool(layer):
        return tf.nn.avg_pool(layer, ksize = [1,2,2,1], strides = [1,1,1,1], padding = 'SAME')
    
    # Build graph model
    graph ={}
    graph['input']      = tf.Variable(np.zeros((1, imag_height, imag_width, color_channels), dtype = 'float32'))
    
    graph['conv1_1']    = _conv2d(graph['input'], 0, 'conv1_1')
    graph['relu1_1']    = _relu(graph['conv1_1'])
    graph['conv1_2']    = _conv2d(graph['relu1_1'], 2, 'conv1_2')
    graph['relu1_2']    = _relu(graph['conv1_2'])
    graph['avgpool1']   = _avgpool(graph['relu1_2'])
    
    graph['conv2_1']    = _conv2d(graph['avgpool1'], 5, 'conv2_1')
    graph['relu2_1']    = _relu(graph['conv2_1'])
    graph['conv2_2']    = _conv2d(graph['relu2_1'], 7, 'conv2_2')
    graph['relu2_2']    = _relu(graph['conv2_2'])
    graph['avgpool2']   = _avgpool(graph['relu2_2'])
        
    graph['conv3_1']    = _conv2d(graph['avgpool2'], 10, 'conv3_1')
    graph['relu3_1']    = _relu(graph['conv3_1'])
    graph['conv3_2']    = _conv2d(graph['relu3_1'], 12, 'conv3_2')
    graph['relu3_2']    = _relu(graph['conv3_2'])
    graph['conv3_3']    = _conv2d(graph['relu3_2'], 14, 'conv3_3')
    graph['relu3_3']    = _relu(graph['conv3_3'])
    graph['conv3_4']    = _conv2d(graph['relu3_3'], 16, 'conv3_4')
    graph['relu3_4']    = _relu(graph['conv3_4'])
    graph['avgpool3']    = _avgpool(graph['relu3_4'])

    graph['conv4_1']    = _conv2d(graph['avgpool3'], 19, 'conv4_1')
    graph['relu4_1']    = _relu(graph['conv4_1'])
    graph['conv4_2']    = _conv2d(graph['relu4_1'], 21, 'conv4_2')
    graph['relu4_2']    = _relu(graph['conv4_2'])
    graph['conv4_3']    = _conv2d(graph['relu4_2'], 23, 'conv4_3')
    graph['relu4_3']    = _relu(graph['conv4_3'])
    graph['conv4_4']    = _conv2d(graph['relu4_3'], 25, 'conv4_4')
    graph['relu4_4']    = _relu(graph['conv4_4'])
    graph['avgpool4']   = _avgpool(graph['relu4_4'])

    graph['conv5_1']    = _conv2d(graph['avgpool4'], 28, 'conv5_1')
    graph['relu5_1']    = _relu(graph['conv5_1'])
    graph['conv5_2']    = _conv2d(graph['relu5_1'], 30, 'conv5_2')
    graph['relu5_2']    = _relu(graph['conv5_2'])
    graph['conv5_3']    = _conv2d(graph['relu5_2'], 32, 'conv5_3')
    graph['relu5_3']    = _relu(graph['conv5_3'])
    graph['conv5_4']    = _conv2d(graph['relu5_3'], 34, 'conv5_4')
    graph['relu5_4']    = _relu(graph['conv5_4'])
    graph['avgpool5']   = _avgpool(graph['relu5_4'])
    
    return graph

def content_loss_func(sess, model):
    p = sess.run(model['conv4_2'])
    x = model['conv4_2']
    
    N = p.shape[3] # No. filters
    M = p.shape[1] * p.shape[2] # No. feature map = filter size
    
    loss = 1/(4*N*M) * tf.reduce_mean(tf.pow(x - p, 2))
    return loss

def style_loss_func(sess, model):
    
    def gram_matrix (F, N, M):
        # Reshape to NxM matrix of F
        F = tf.reshape(F, (N, M))
        G = tf.matmul(tf.transpose(F), F)
        return G
    def _style_loss(a, x):
        N = a.shape[3]
        M = a.shape[1] * a.shape[2]
        
        A = gram_matrix(a, N, M)
        G = gram_matrix(x, N, M)
        loss = 1/(4 * N**2 * M**2) * tf.reduce_mean(tf.pow(G - A, 2))
        return loss
    
    layer_filters = [
                    ('conv1_1', 0.5),
                    ('conv2_1', 1.0),
                    ('conv3_1', 2.0),
                    ('conv4_1', 3.0),
                    ('conv5_1', 4.0)
                    ]
    
    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in layer_filters]
    W = [w for _, w in layer_filters]
    L = sum(W[l] * E[l] for l in range(len(layer_filters)))
    return L

with tf.device('/cpu:0'):
    # Start calculation
    imag_content = imag_load(path_content)
    imag_style = imag_load(path_style)
    imag_input = imag_noise(imag_content, noise_ratio)
    
    graph = vgg_model(path_vggmodel)
    
    # tf.InteractiveSession is same with tf.Session but not need to call with tf.Session.
    # It wokrs like 'evaluation' functions
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer()) # Initial all variables
    
    # Content loss
    sess.run(graph['input'].assign(imag_content))
    content_loss = content_loss_func(sess, graph)
    
    # Style loss
    sess.run(graph['input'].assign(imag_style))
    style_loss = style_loss_func(sess, graph)
    
    # Total loss
    loss_total = alpha * content_loss + beta * style_loss
    optimizer = tf.train.AdamOptimizer(learning_rate = 2.0).minimize(loss_total)

    start_time = time.time()
    if __name__ == '__main__':
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(graph['input'].assign(imag_input))
            
            if not os.path.exists(path_out):
                os.mkdir(path_out)
                
            iteration = 2        
            for i in range(0, iteration):
                sess.run(optimizer)
                print(i)
                if i % 2 == 0:
                    imag_mixed = sess.run(graph['input'])
                    file_name = 'train {}.jpg'.format(i)
                    imag_save(path_out, file_name, imag_mixed )    
            
    end_time = time.time()           
    
    print("Time usage: {0:>6.2}".format(end_time-start_time))
        

    

       
        
        
        
        
        
