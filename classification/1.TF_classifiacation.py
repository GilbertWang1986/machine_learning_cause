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


train_A = cPickle.load(open('train_A'))
train_B = cPickle.load(open('train_B'))
to_predicted = cPickle.load(open('pre'))


# prepare training data
# [1, 0] for class A
# [0, 1] for class B
train_data_input = np.vstack( (train_A, train_B) )
classification_A = np.hstack((np.ones((train_A.shape[0], 1)), np.zeros((train_A.shape[0], 1))))
classification_B = np.hstack((np.zeros((train_B.shape[0], 1)), np.ones((train_B.shape[0], 1))))
train_data_output = np.vstack( (classification_A, classification_B) )


# Build the network 
# 2(input)--->4(hiddend)--->4(hidden)--->2(output)
# input layer
input_layer = tf.placeholder(tf.float32, [None, 2])
# hidden layer 1
hidden_layer_1 = add_layer(input_layer, 2, 4, activation_function=tf.nn.relu)
# hidden layer 2
hidden_layer_2 = add_layer(hidden_layer_1, 4, 4, activation_function=tf.nn.relu)
# output layer 
output_layer = add_layer(hidden_layer_2, 4, 2, activation_function=tf.nn.softmax)
# ref for the real classification
ref = tf.placeholder(tf.float32, [None, 2])


# loss function
cross_entropy=-tf.reduce_mean(ref*tf.log(tf.clip_by_value(output_layer,1e-10,1.0)))


# trainer
trainer = tf.train.AdamOptimizer(1.0e-3).minimize(cross_entropy)


# start a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_idx = 0
    while train_idx <= 5000:
        train_idx = train_idx + 1
        sess.run(trainer, feed_dict={input_layer: train_data_input,
                                     ref: train_data_output})

    prediction = sess.run(output_layer, feed_dict={input_layer: to_predicted})





# plot
plt.scatter(train_A[:, 0], train_A[:, 1], color='', marker='o', edgecolors='r', s=50)
plt.scatter(train_B[:, 0], train_B[:, 1], color='', marker='o', edgecolors='b', s=50)

for i in range(to_predicted.shape[0]):
    if prediction[i, 0] >= prediction[i, 1]:
        plt.scatter(to_predicted[i, 0], to_predicted[i, 1], color='', marker='s', edgecolors='r', s=50)
    else:
        plt.scatter(to_predicted[i, 0], to_predicted[i, 1], color='', marker='s', edgecolors='b', s=50)


plt.xlim([0, 40])
plt.ylim([0, 40])
plt.legend(['Class A', 'Class B'])
plt.savefig('classification.pdf')
plt.show()
