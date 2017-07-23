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
imag_features = imag_size * imag_size
hidden_layer_size1 = 300
hidden_layer_size2 = 100
output_size = 10

# Plot function for several images
def plot_images(images, cls_true, cls_pred = None):
    assert len(images) == len(cls_true) == 9
    
    fig, axes = plt.subplots(3,3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(imag_size, imag_size), cmap = 'binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)        
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

plot_images(data.train.images[0:9,:], y_train_true_cls[0:9])

#%% 1. Initial variables
W1 = tf.Variable(tf.random_normal((imag_features, hidden_layer_size1), stddev = 0.01))
b1 = tf.Variable(tf.zeros(hidden_layer_size1))
W2 = tf.Variable(tf.random_normal((hidden_layer_size1, hidden_layer_size2), stddev = 0.01))
b2 = tf.Variable(tf.zeros(hidden_layer_size2))   
W3 = tf.Variable(tf.random_normal((hidden_layer_size2, output_size), stddev = 0.01))

#%% 2. Build model: size of each layer: 784 - 300 - 100 - 10
def model (X):

    h1 = tf.sigmoid(tf.matmul(X, W1)) + b1 
    h2 = tf.sigmoid(tf.matmul(h1, W2)) + b2
    return tf.matmul(h2, W3)

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
iterations = 2000
print_iterations = 100
batch_size = 64
loss_train = np.zeros(np.int(iterations/print_iterations))
loss_test = np.zeros(np.int(iterations/print_iterations))
j = 0
start_time = time.time()
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())        
    
    for i in range(iterations):
        X_batch, Y_batch = data.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {x: X_batch, y: Y_batch})
        if i % print_iterations == 0:
            
            loss_train[j] = sess.run(total_loss, feed_dict = {x: data.train.images, y: data.train.labels})
            loss_test[j] = sess.run(total_loss, feed_dict = {x: data.test.images, y: data.test.labels})
            j += 1
            print (j, "..Train..", loss_train[j-1], "..Test..", loss_test[j-1])
            # Print updated value of b1
            # print(sess.run(b1))
    
    # Calculate some interesting data after finish training
    # Because data.train.next_batch suffle the order of images in data.train
    # We need to reload the data.train in order with true classes
    data = input_data.read_data_sets('data/MNIST', one_hot = True)
    y_train_true_cls = np.argmax(data.train.labels, axis = 1)
    y_test_true_cls = np.argmax(data.test.labels, axis = 1)
               
    # Print traing results
    y_train_predicted = sess.run(y_pred_cls, feed_dict = {x: data.train.images})            
    accuracy_train = np.mean(y_train_predicted == y_train_true_cls)
    print("Train-set \t Iteration: {0}, Accuracy: {1:>6.1%}". format(i+1, accuracy_train))
    
    # Print test results                        
    y_test_predicted = sess.run(y_pred_cls, feed_dict = {x: data.test.images})            
    accuracy_test = np.mean(y_test_predicted == y_test_true_cls)
    print("Test-set \t Iteration: {0}, Accuracy: {1:>6.1%}". format(i+1, accuracy_test))     

end_time = time.time()
print("Usage time: ", end_time - start_time)

plot_images(data.train.images[0:9,:], y_train_true_cls[0:9], y_train_predicted[0:9])

plt.plot(loss_train,'b')
plt.plot(loss_test,'r')
plt.show()            

