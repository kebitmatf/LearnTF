# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 01:45:24 2017

@author: PeaceSea
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load and save MNIST data-set
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST',one_hot = True)

train_data = data.train
train_labels = data.train.labels

test_data = data.test
test_labels = data.test.labels

validation_data = data.validation
validation_labels = data.validation.labels

print("Train-set: \t\t{}".format(len(train_labels)))
print("Test-set: \t\t{}".format(len(test_labels)))
print("Validation-set: \t{}".format(len(validation_labels)))

# MNIST data-set
img_size = 28
imag_shape = (img_size, img_size)
num_channels = 1 # only gray color
img_flat = img_size * img_size * num_channels

# Model params
# input(:,28,28,1) 
# -> conv1(5,5,16) -> maxpooling(2,2,16) -> relu => (:,14,14,16)
# -> conv2(5,5,16,32) -> maxpooling(2,2,16,32) -> relu =>(:,7,7,16,32)
# -> flatten() -> fc(:,128) => (:,128)
# -> fc2(:,10) -> softmax() => (:,10)

filter_size1 = 5
filter_num1 = 16

filter_size2 = 5
filter_num2 = 32

fc_size = 128
out_size = 10

# Build model
x = tf.placeholder(tf.float32, [None, img_flat], name = 'x')
x_img = tf.reshape(x, [-1, img_size, img_size, num_channels])
w1 = tf.Variable(tf.truncated_normal(shape = [img_size, img_size, num_channels, filter_num1], stddev = 0.05))
b1 = tf.Variable(tf.constant(0.05, shape = [filter_num1]))

layer_conv1 = tf.nn.conv2d(input = x_img, filter = w1, strides = [1,1,1,1], padding = 'SAME')
layer_conv1 += b1
layer_conv1 = tf.nn.max_pool(layer_conv1, ksize = [1,1,1,1], strides = [1,2,2,1], padding = 'SAME')
layer_conv1 = tf.nn.relu(layer_conv1)
print(layer_conv1)




