#!/usr/bin/python

import sys
import numpy as np
import tensorflow as tf

from config import *
from memory import Memory, MultiMemory
from qnet import *
from layers import *
from gravitron import Gravitron

## Initialize Tensorflow
tf.reset_default_graph()
session = tf.Session()

env = Gravitron()

def train(net, target_net, memory, episodes):

    eps = EPS_START
    step = 0
    rewards = []

    def train_once(eps, step):
        s0 = env.reset()
        net_reward = 0
        d = False
        while not d:
            if np.random.rand(1) < eps or step < pre_train_steps:
                a = env.action_space.sample()
            else:
                a = session.run(net.predict, feed_dict={net.inputs:[s0]})
                a = a[0]

            s1,r,d,_ = env.step(a)
            memory.add({'s0':s0, 'a':a, 'r':r, 's1':s1, 'd':d})
            if step > PRE_TRAIN_STEPS:
                if eps > EPS_END:
                    eps += EPS_DELTA 
                if step % 5 == 0:
                    input_batch = memory.sample(BATCH_SIZE)
                    _s0 = input_batch['s0']
                    _a = input_batch['a']
                    _r = input_batch['r']
                    _s1 = input_batch['s1']
                    _d = input_batch['d']

                    # s1 = (4,32), --> (x,32)
                    a_s1 = session.run(net.predict, feed_dict={net.inputs : _s1})
                    q_s1 = session.run(target_net._Q, feed_dict={target_net.inputs : _s1})
                    q = q_s1[range(batch_size), a_s1].reshape((batch_size,-1))
                    target_q = _r + GAMMA * q * (1 - _d)
                    _ = session.run(net.update, feed_dict = {net.inputs : _s0, net.Qn:target_q, net.actions:_a})
                    session.run(copy_ops)
            net_reward += r
            s0 = s1
            step += 1
        return net_reward, eps, step

    r_mean = 0
    i = 0
    
    while i < episodes: 
        net_reward, eps, step = train_once(eps, step)
        rewards.append(net_reward)

        if i % 50 == 0 and i > 0:
            r_mean = np.mean(rewards[-100:])
            print "Epoch : %d, Mean Reward: %f; Step : %d, Epsilon: %f; Test: %f" % (i, r_mean, step, eps, test(target_net,1)[0])
        i += 1

    if raw_input('Train Until Convergence?\n').lower() == 'y':
        while r_mean < 999:
            net_reward, eps, step = train_once(eps, step)
            rewards.append(net_reward)

            if i % 10 == 0 and i > 0:
                r_mean = np.mean(rewards[-100:])
                print "Epoch : %d, Mean Reward: %f; Step : %d, Epsilon: %f; Test: %f" % (i, r_mean, step, eps, test(target_net,1)[0])
            i += 1


    return rewards

def test(net, episodes):
    rewards = []

    # test
    for i in range(episodes):
        s = env.reset()
        d = False
        net_reward = 0
        while not d:
            #env.render()
            a = session.run(net.predict, feed_dict={net.inputs:[s]})
            a = a[0]
            s,r,d,_ = env.step(a)
            net_reward += r
        rewards.append(net_reward)
    return rewards

def setup():
    ## Start Environment
    state_size = reduce(lambda x,y:x*y, env.observation_space.shape)

    # initialize memory
    memory = MultiMemory(MEM_SIZE)
    def create_network():
        net = QNet((WIN_H,WIN_W,1))
        net.append(ConvolutionLayer((3,3,1,4)))
        net.append(ActivationLayer('relu'))
        net.append(DenseLayer((WIN_H*WIN_W*4,256)))
        net.append(ActivationLayer('relu'))
        net.append(DenseLayer((256,3)))
        #net.append(ActivationLayer('relu'))
        net.setup()
        return net


    # get networks
    with tf.variable_scope('net') as scope:
        net = create_network()
    with tf.variable_scope('target') as scope:
        target_net = create_network()

    return net, target_net, memory

def main():
    global copy_ops

    net, target_net, memory = setup()
    copy_ops = net.copyTo(target_net, TAU)

    # get this started...
    session.run(tf.global_variables_initializer())
    session.run(copy_ops)
    saver = tf.train.Saver()

    if len(sys.argv) > 1 and sys.argv[1].lower() == 'load':
        saver.restore(session, '/tmp/model.ckpt')
        print '[loaded]'
    else:
        train_rewards = train(net, target_net, memory, NUM_EPISODES)
        np.savetxt('train.csv', train_rewards, delimiter=',', fmt='%f')
        save_path = saver.save(session, '/tmp/model.ckpt')
        print("Model saved in file: %s" % save_path) 

    test_rewards = test(net,TEST_EPISODES)
    np.savetxt('test.csv', test_rewards, delimiter=',', fmt='%f')

if __name__ == "__main__":
    main()
    
