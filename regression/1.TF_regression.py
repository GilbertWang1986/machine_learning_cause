from __future__ import print_function
from __future__ import division

import cPickle

import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_function=None):
    weight = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size])+0.2)

    wx_plus_b = tf.matmul(inputs, weight)+bias 
    if activation_function == None:
        return wx_plus_b
    else:
        return activation_function(wx_plus_b)


train_data = cPickle.load(open('reg'))
train_x = train_data[:, 0].reshape(-1, 1)
train_y = train_data[:, 1].reshape(-1, 1)

plt.subplot(121).scatter(train_x, train_y, color='', marker='o', edgecolors='r', s=50)


# Build the network 
# 1(input)--->4(hiddend)--->4(hidden)--->1(output)
# input layer
input_layer = tf.placeholder(tf.float32, [None, 1])
# hidden layer 1
hidden_layer_1 = add_layer(input_layer, 1, 8, activation_function=tf.nn.tanh)
# hidden layer 2
hidden_layer_2 = add_layer(hidden_layer_1, 8, 8, activation_function=tf.nn.tanh)
# output layer 
output_layer = add_layer(hidden_layer_2, 8, 1, activation_function=None)
# ref for the real classification
ref = tf.placeholder(tf.float32, [None, 1])


# loss function
loss= tf.reduce_mean( tf.square(output_layer-ref) )


# # trainer
trainer = tf.train.AdamOptimizer(1.0e-3).minimize(loss)


# start a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_idx = 0
    while train_idx <= 5000:
        train_idx = train_idx + 1
        sess.run(trainer, feed_dict={input_layer: train_x,
                                     ref: train_y})

    
    to_predicted = np.linspace(2.0, 10.0, 100).reshape(-1, 1)
    prediction = sess.run(output_layer, feed_dict={input_layer: to_predicted})





# plot
plt.subplot(122).scatter(train_x, train_y, color='', marker='o', edgecolors='r', s=50)
plt.subplot(122).plot(to_predicted, prediction, 'b')

plt.show()
