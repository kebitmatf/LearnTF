# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 12:12:19 2017

@author: PeaceSea
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot = True)
y_train_true_cls = np.argmax(data.train.labels, axis = 1)
y_test_true_cls = np.argmax(data.test.labels, axis = 1)

imag_size = 28
imag_shape = (imag_size, imag_size)
color_channels = 1

# Plot function for several images
def plot_images(images, y_true_cls, y_predicted_cls = None):
    assert len(y_true_cls) == 9
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(imag_size, imag_size), cmap = 'binary')
        if y_predicted_cls is None:
            xlabel = "True: {}".format(y_true_cls[i])
        else:
            xlabel = "True: {0}, Predicted: {1}".format(y_true_cls[i], y_predicted_cls[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.plot()
    
plot_images(data.train.images[0:9,:], y_train_true_cls[0:9])

#%% 1. Initial variables
filter_size1 = 5
filter_number1 = 6
filter_size2 = 5
filter_number2 = 16
flat_size = filter_size2**2*filter_number2
fc_number1 = 120
fc_number2  = 84
output_size = 10

W1 = tf.Variable(tf.truncated_normal( shape = (filter_size1, filter_size1, color_channels, filter_number1), stddev = 0.01))
b1 = tf.Variable(tf.zeros(shape = filter_number1))
W2 = tf.Variable(tf.truncated_normal(shape = (filter_size2, filter_size2, filter_number1, filter_number2), stddev = 0.01))
b2 = tf.Variable(tf.zeros(shape = filter_number2))
W3 = tf.Variable(tf.truncated_normal(shape = (flat_size, fc_number1), stddev = 0.01))
b3 = tf.Variable(tf.zeros(shape = fc_number1))
W4 = tf.Variable(tf.truncated_normal(shape = (fc_number1, fc_number2), stddev = 0.01))
b4 = tf.Variable(tf.zeros(shape = fc_number2))
W5 = tf.Variable(tf.truncated_normal(shape = (fc_number2, output_size), stddev = 0.01))

#%% 2. Build model: size of each layer: 784 - 300 - 100 - 10
def model(x):
    X = tf.reshape(x, [-1, imag_size, imag_size, color_channels])
    
    conv1 = tf.nn.conv2d(input = X, filter = W1, strides = [1,1,1,1], padding = 'SAME')
    conv1 += b1
    conv1 = tf.nn.max_pool(value = conv1, ksize = [1,2,2,1], strides =[1,2,2,1], padding = 'SAME')
    conv1 = tf.nn.relu(conv1)
    
    conv2 = tf.nn.conv2d(input = conv1, filter = W2, strides = [1,1,1,1], padding = 'VALID')
    conv2 += b2
    conv2 = tf.nn.max_pool(value = conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    conv2 = tf.nn.relu(conv2)
    
    fc_flat = tf.reshape(conv2, shape = (-1,flat_size))
    fc1 = tf.matmul(fc_flat, W3)
    fc1 += b3
    
    fc2 = tf.matmul(fc1, W4)
    fc2 += b4
    
    net_out = tf.matmul(fc2, W5)
    return net_out    
    
#%% 3. Define input params. Here are input images and classes
x = tf.placeholder("float", shape = [None, imag_size**2], name = 'x')
y = tf.placeholder("float", shape = [None, output_size], name = 'y')

#%% 4. Call model and calculate total loss function
score = model(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y)
total_loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-2).minimize(total_loss)

#%% 5. Use softmax and predict classes
y_pred = tf.nn.softmax(score)
y_pred_cls = tf.arg_max(y_pred, dimension = 1)

#%% 6. Create a tf.Session, feed the params (x, y) then run gradient descent
total_samples = len(data.train.labels)
iterations = 20
print_iterations = 10
batch_size = 64
loss_train = np.zeros(np.int(iterations/print_iterations))
loss_test = np.zeros(np.int(iterations/print_iterations))
j = 0
start_time = time.time()
with tf.Session() as sess:
    
    # sess = tf.Session()
    sess.run(tf.global_variables_initializer())        
    
    for i in range(iterations):
        X_batch, Y_batch = data.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {x: X_batch, y: Y_batch})
        if i % print_iterations == 0:
            
            loss_train[j] = sess.run(total_loss, feed_dict = {x: X_batch, y: Y_batch})            
            j += 1
            print (j, "..Train..", loss_train[j-1])
            # Print updated value of b1
            # print(sess.run(b1))
    
#    # Calculate some interesting data after finish training
#    # Because data.train.next_batch suffle the order of images in data.train
#    # We need to reload the data.train in order with true classes
#    data = input_data.read_data_sets('data/MNIST', one_hot = True)
#    y_train_true_cls = np.argmax(data.train.labels, axis = 1)
#    y_test_true_cls = np.argmax(data.test.labels, axis = 1)
#               
#    # Print traing results
#    y_train_predicted = sess.run(y_pred_cls, feed_dict = {x: data.train.images})            
#    accuracy_train = np.mean(y_train_predicted == y_train_true_cls)
#    print("Train-set \t Iteration: {0}, Accuracy: {1:>6.1%}". format(i+1, accuracy_train))
#    
#    # Print test results                        
#    y_test_predicted = sess.run(y_pred_cls, feed_dict = {x: data.test.images})            
#    accuracy_test = np.mean(y_test_predicted == y_test_true_cls)
#    print("Test-set \t Iteration: {0}, Accuracy: {1:>6.1%}". format(i+1, accuracy_test))     

end_time = time.time()
print("Usage time: ", end_time - start_time)

#plot_images(data.train.images[0:9,:], y_train_true_cls[0:9], y_train_predicted[0:9])
#
#plt.plot(loss_train,'b')
#plt.plot(loss_test,'r')
#plt.show()            

