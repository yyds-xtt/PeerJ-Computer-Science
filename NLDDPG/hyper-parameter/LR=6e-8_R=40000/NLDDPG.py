'''
--------------------- NLDDPG --------------------
-------- Autor: Xi Hu and Yang Huang ---------
--- Northeastern University at Qinhuangdao ---
'''

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from MEC_environment import MEC_env

MAX_EPISODES = 600 
LR_A = 0.00000006
LR_C = 0.00000006
GAMMA = 0  
TAU = 0.01  
MEMORY_CAPACITY = 40000  
BATCH_SIZE = 64  

class StateNormalization(object):
    def __init__(self):
        env = MEC_env()
        M = env.M
        self.high_state = np.array([10 * 1024 * 1048576 * 8])
        self.high_state = np.append(self.high_state, np.ones(M * 2) * env.ground_length)
        self.high_state = np.append(self.high_state, np.ones(M) * 250 * 1048576 * 8)
        self.high_state = np.append(self.high_state, np.ones(M))
        self.low_state = np.zeros(4 * M + 1)

    def state_normal(self, state):
        return state / (self.high_state - self.low_state)

class NLDDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):   
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  
        self.pointer = 0
        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
        q_target = self.R + GAMMA * q_
        td_error = tf.losses.absolute_difference(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        a_loss = - tf.reduce_mean(q)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        s = s[np.newaxis, :]  
        c = self.sess.run(self.a, feed_dict={self.S: s})[0]
        return c  

    def learn(self):
        self.sess.run(self.soft_replace)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            x_train = tf.reshape(s, (-1, s.shape[1].value, 1))
            net = tf.keras.layers.LSTM(100, input_shape=(x_train.shape[1].value, x_train.shape[2].value),
                                       return_sequences=True, name='l1')(x_train)
            net = tf.keras.layers.LSTM(10, return_sequences=False,name='l2')(net)
            net = tf.keras.layers.Dropout(0.2)(net)
            net = tf.keras.layers.Dense(300, activation='relu', name='l3', trainable=trainable)(net)
            net = tf.keras.layers.Dense(10, activation='relu', name='l4', trainable=trainable)(net)
            actions = tf.keras.layers.Dense(self.a_dim, activation='tanh', name='a')(net)
            scaled_a = tf.multiply(actions, self.a_bound, name='scaled_a')
            return scaled_a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 100
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', shape=(1, n_l1), trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.reshape(net, (-1, net.shape[1].value, 1))
            net = tf.keras.layers.LSTM(100, input_shape=(-1, net.shape[1].value, 1), return_sequences=True, name='l1')(
                net)
            net = tf.keras.layers.LSTM(10, return_sequences=False, name='l2')(net)
            net = tf.keras.layers.Dropout(0.2)(net)
            net = tf.keras.layers.Dense(300, activation='relu', name='l3', trainable=trainable)(net)
            net = tf.keras.layers.Dense(10, activation='relu', name='l4', trainable=trainable)(net)
            net = tf.keras.layers.Dense(self.a_dim, activation='relu', name='a')(net)
            net = tf.keras.layers.Dense(1, trainable=trainable)(net)
            return   net

np.random.seed(1)
tf.set_random_seed(1)
env = MEC_env()
MAX_EP_STEPS = env.slot_num   
s_dim = env.state_dim  
a_dim = env.action_dim  
a_bound = env.action_bound  
NLddpg = NLDDPG(a_dim, s_dim, a_bound)
var = 0.01
t1 = time.time()
ep_reward_list = []
s_normal = StateNormalization()  
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    j = 0
    while j < MAX_EP_STEPS:
        a = NLddpg.choose_action(s_normal.state_normal(s))  
        a = np.clip(np.random.normal(((a+1)/2), var), *a_bound)  
        s_, r, is_terminal, step_redo = env.step(a)
        if step_redo:
            continue  
        NLddpg.store_transition(s_normal.state_normal(s), a, r, s_normal.state_normal(s_))  
        if NLddpg.pointer > MEMORY_CAPACITY:
            NLddpg.learn()
        s = s_
        ep_reward += r*100
        if j == MAX_EP_STEPS - 1 or is_terminal:
            print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward)
            ep_reward_list = np.append(ep_reward_list, ep_reward)
            
            file_name = 'NLDDPG_Episode-Reward_LR=6e-8_R=40000.txt'
            with open(file_name, 'a') as file_obj:
                file_obj.write("\nEpisode: " + '{:d}'.format(i) + " , Reward: " + '{:f}'.format(ep_reward))
            break
        j = j + 1

print('Running time: ', time.time() - t1)
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
