'''
--------------------- DLDQN --------------------
-------- Autor: Xi Hu and Yang Huang ---------
--- Northeastern University at Qinhuangdao ---
'''

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from MEC_environment import MEC_env

MAX_EPISODES = 600
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
            s = self.s
            x_train = tf.reshape(s, (-1, s.shape[1].value, 1))
            net = tf.keras.layers.LSTM(100, input_shape=(x_train.shape[1].value, x_train.shape[2].value),
                                       return_sequences=True, name='l1')(x_train)
            net = tf.keras.layers.LSTM(10, return_sequences=False, name='l2')(net)
            net = tf.keras.layers.Dropout(0.2)(net)
            net = tf.keras.layers.Dense(100, activation='relu', name='l3')(s)
            net = tf.keras.layers.Dense(20, activation='relu', name='l4')(net)
            self.q_eval = tf.layers.dense(net, self.n_actions, tf.nn.softmax, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='l5')

        with tf.variable_scope('target_net'):
            s_ = self.s_
            net = tf.reshape(s_, (-1, s_.shape[1].value, 1))
            net = tf.keras.layers.LSTM(100, input_shape=(-1, net.shape[1].value, 1), return_sequences=True, name='l1')(
                net)
            net = tf.keras.layers.LSTM(10, return_sequences=False, name='l2')(net)
            net = tf.keras.layers.Dropout(0.2)(net)
            net = tf.keras.layers.Dense(100, activation='relu', name='l3')(s_)
            net = tf.keras.layers.Dense(20, activation='relu', name='l4')(net)
            self.q_next = tf.layers.dense(net, self.n_actions, tf.nn.softmax, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='l5')

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
Normal = StateNormalization()
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
        a = DQN.choose_action(Normal.state_normal(s))
        s_, r, is_terminal, step_redo, reset_offload_ratio = env.step(a)
        if step_redo:
            continue
        if reset_offload_ratio:
            t1 = a % 10
            a = a - t1
        DQN.store_transition(Normal.state_normal(s), a, r, Normal.state_normal(s_))
        if DQN.memory_counter > MEMORY_CAPACITY:
            DQN.learn()
        s = s_
        ep_reward += r*10

        if j == MAX_EP_STEPS - 1 or is_terminal:
            print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward)
            ep_reward_list = np.append(ep_reward_list, ep_reward)
            file_name = 'DLDQN_Episode-Reward.txt'
            with open(file_name, 'a') as file_obj:
                file_obj.write("\nEpisode: " + '{:d}'.format(i) + " , Reward: " + '{:f}'.format(ep_reward))

            break
        j = j+1

print('Running time: ', time.time() - t1)
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
