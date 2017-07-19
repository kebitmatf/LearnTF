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
import time
from datetime import timedelta

# Load and save MNIST data-set
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST',one_hot = True)

print("Train-set: \t\t{}".format(len(data.train.labels)))
print("Test-set: \t\t{}".format(len(data.test.labels)))
print("Validation-set: \t{}".format(len(data.validation.labels)))

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
filter_num2 = 36

fc_size = 128
out_size = 10

# Build model

x = tf.placeholder(tf.float32, shape = [None, img_flat], name = 'x')
x_img = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape = [None, out_size], name = 'y_true')
y_true_cls = tf.arg_max(y_true, dimension = 1)


w1 = tf.Variable(tf.truncated_normal(shape = [filter_size1, filter_size1, num_channels, filter_num1], stddev = 0.05))
b1 = tf.Variable(tf.constant(0.05, shape = [filter_num1]))

w2 = tf.Variable(tf.truncated_normal(shape = [filter_size2, filter_size2, filter_num1, filter_num2], stddev = 0.05))
b2 = tf.Variable(tf.constant(0.05, shape = [filter_num2]))

wfc = tf.Variable(tf.truncated_normal(shape = [7*7*36, fc_size], stddev = 0.05))
bfc = tf.Variable(tf.constant(0.05, shape = [fc_size]))

wout = tf.Variable(tf.truncated_normal(shape = [fc_size, out_size], stddev = 0.05))
bout = tf.Variable(tf.constant(0.05, shape = [out_size]))

layer_conv1 = tf.nn.conv2d(input = x_img, filter = w1, strides = [1,1,1,1], padding = 'SAME')
layer_conv1 += b1
layer_conv1 = tf.nn.max_pool(layer_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
layer_conv1 = tf.nn.relu(layer_conv1)
print(layer_conv1)

layer_conv2 = tf.nn.conv2d(input = layer_conv1, filter = w2, strides = [1,1,1,1], padding = 'SAME')
layer_conv2 += b2
layer_conv2 = tf.nn.max_pool(layer_conv2, ksize =[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
layer_conv2 = tf.nn.relu(layer_conv2)
print(layer_conv2)

flat_shape = layer_conv2.get_shape()
flat_features = flat_shape[1:4].num_elements()
layer_fc = tf.reshape(layer_conv2, [-1,flat_features])

layer_fc = tf.matmul(layer_fc, wfc) + bfc

layer_out = tf.matmul(layer_fc, wout) + bout

# Get predicted labels
y_pred = tf.nn.softmax(layer_out)
y_pred_cls = tf.arg_max(y_pred, dimension = 1)

# Cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = layer_out,
                                                        labels = y_true)
cost = tf.reduce_mean(cross_entropy)

# Trainning method
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)

# Calculate accuracy
correct_predictions = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype = tf.float32))

# Start a TF session and running
session = tf.Session()
session.run(tf.global_variables_initializer())

batch_size = 64
iterations = 1000
start_time = time.time()

for i in range(0, iterations):    
    x_batch, y_batch = data.train.next_batch(batch_size)
    session.run(optimizer, feed_dict = {x: x_batch, y_true: y_batch})
    
    if i % 100 ==0:
        acc = session.run(accuracy, feed_dict = {x: x_batch, y_true: y_batch})
        print("Iterations: {0:>6}, Accuracy: {1:>6.1%}".format(i+1, acc))
end_time = time.time()
delta_time = end_time - start_time
print("Time usage: " + str(timedelta(seconds = int(round(delta_time)))))

# Check accuracy of test-set
test_batch_size = 64
test_total_size = len(data.test.labels)
test_pred = np.zeros(test_total_size, dtype = np.int)
while i < test_total_size:
    j = min(i+test_batch_size, test_total_size)
    x_test = data.test.images[i:j,:]
    y_test = data.test.labels[i:j,:]
    test_pred[i:j] = session.run(y_pred_cls, feed_dict = {x: x_test, y_true: y_test})
    i = j  

data_test_cls = np.argmax(data.test.labels, axis = 1)
test_correction = (test_pred == data_test_cls)
test_accuracy = test_correction.sum()/test_total_size
print("Test accuracy: {0:0.2%}".format(test_accuracy))





