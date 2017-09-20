from config import *
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def lerp(a,b,w):
    return w*a + (1.-w)*b

class QNet(object):
    def __init__(self, in_dim):
        self.scope = tf.get_variable_scope().name
        shape = (None,) + in_dim
        self.inputs = tf.placeholder(shape=shape, dtype=tf.float32)
        self.L = []

    def append(self, l):
        self.L.append(l)

    def setup(self):
        X = self.inputs
        for l in self.L:
            X = l.apply(X)
        self._Q = X 

        self.predict = tf.argmax(self._Q, 1)
        self.actions = tf.placeholder(shape=[None,1], dtype=tf.int32)
        self.actions_one_hot = tf.one_hot(self.actions, 3, dtype=tf.float32) #up-down-left-right

        self.Q = tf.reduce_sum(tf.multiply(self._Q, self.actions_one_hot), reduction_indices=1)
        self.Qn = tf.placeholder(shape=[None,1], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.Qn - self.Q))
        #trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        self.update = trainer.minimize(loss)

    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    def copyTo(self, net, tau):
        src = self.vars()
        dst = net.vars()
        return map(lambda (s,d): d.assign(lerp(s,d,tau)), zip(src,dst))

class ACNet(object):
    def __init__(self, in_dim, out_dim):
        self.scope = tf.get_variable_scope().name
        self.in_dim, self.out_dim = in_dim, out_dim
        shape = (None,) + in_dim
        self.inputs = tf.placeholder(shape=shape, dtype=tf.float32)
    def setup(self):
        conv = slim.stack(self.inputs, slim.conv2d, [(32,[3,3]), (16,[3,3]), (4,[1,1])], scope='conv')
        flat = slim.flatten(conv)
        fc = slim.stack(flat, slim.fully_connected,[256,self.out_dim], scope='fc')
        self._Q = fc

        self.predict = tf.argmax(self._Q, 1)
        self.actions = tf.placeholder(shape=[None,1], dtype=tf.int32)
        self.actions_one_hot = tf.one_hot(self.actions, 3, dtype=tf.float32) #up-down-left-right

        self.Q = tf.reduce_sum(tf.multiply(self._Q, self.actions_one_hot), reduction_indices=1)
        self.Qn = tf.placeholder(shape=[None,1], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.Qn - self.Q))
        #trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        self.update = trainer.minimize(loss)

    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    def copyTo(self, net, tau):
        src = self.vars()
        dst = net.vars()
        return map(lambda (s,d): d.assign(lerp(s,d,tau)), zip(src,dst))
