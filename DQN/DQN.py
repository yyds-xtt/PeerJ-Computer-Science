'''
--------------------- DQN --------------------
-------- Autor: Xi Hu and Yang Huang ---------
--- Northeastern University at Qinhuangdao ---
'''

import time
import numpy as np
import tensorflow as tf
from MEC_environment import MEC_env
import matplotlib.pyplot as plt

MAX_EPISODES = 600
MEMORY_CAPACITY = 40000
BATCH_SIZE = 64

class DeepQNetwork:
    def __init__(self, n_actions, n_features, learning_rate=0.1, reward_decay=0.001, e_greedy=0.99, replace_target_iter=200, memory_size=MEMORY_CAPACITY, batch_size=BATCH_SIZE):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = 0.99
        self.learn_step_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + 2), dtype=np.float32)
        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e3 = tf.layers.dense(e1, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e3')
            self.q_eval = tf.layers.dense(e3, self.n_actions, tf.nn.softmax, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 100, tf.nn.relu6, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t3 = tf.layers.dense(t1, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t3')
            self.q_next = tf.layers.dense(t3, self.n_actions, tf.nn.softmax, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t4')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })
        self.cost_his.append(cost)
        self.learn_step_counter += 1

env = MEC_env()
np.random.seed(1)
tf.set_random_seed(1)
s_dim = env.state_dim
n_actions = env.n_actions
DQN = DeepQNetwork(n_actions, s_dim)
t1 = time.time()
ep_reward_list = []
MAX_EP_STEPS = env.slot_num
for i in range(MAX_EPISODES):

    s = env.reset()
    ep_reward = 0
    j = 0
    while j < MAX_EP_STEPS:
        a = DQN.choose_action(s)
        s_, r, is_terminal, step_redo, reset_offload_ratio = env.step(a)
        if step_redo:
            continue
        if reset_offload_ratio:
            t1 = a % 11
            a = a - t1
        DQN.store_transition(s, a, r, s_)

        if DQN.memory_counter > MEMORY_CAPACITY:
            DQN.learn()
        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS - 1 or is_terminal:
            print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward)
            ep_reward_list = np.append(ep_reward_list, ep_reward)

            file_name = 'DQN_Episode-Reward.txt'
            with open(file_name, 'a') as file_obj:
                file_obj.write("\nEpisode: " + '{:d}'.format(i) + " , Reward: " + '{:f}'.format(ep_reward))
            break
        j = j+1
print('Running time: ', time.time() - t1)
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
