import gym
import numpy as np
import tensorflow as tf
from collections import deque

env = gym.make('CartPole-v1')

#HYPERPARAMETERS

train_episodes = 1000
max_steps = 200
gamma = .99						# future reward discount

#exploration parameters
explore_start = 1.0				# exploration prob at start
explore_stop = .01				# minimum exp prob
decay_rate = .0001				# exponential decay rate

#network parameters
hidden_size = 64
learning_rate = .0001

#memory parameters
memory_size = 10000
batch_size = 20
pretrain_length = batch_size	# num of experiences to pretrain the memory


class Memory():
	def __init__(self, max_size = 1000):
		self.buffer = deque(maxlen = max_size)

	def add(self, experience):
		self.buffer.append(experience)

	def sample(self, batch_size):
		idx = np.random.choice(np.arange(len(self.buffer)), size = batch_size, replace = False)
		return [self.buffer[ii] for ii in idx]

class QNetwork:
	def __init__(self, learning_rate = .01, state_size = 4, action_size = 2 , hidden_size= 10, name = 'QNetwork'):

		#Q()
		with tf.variable_scope(name):

			self.inputs = tf.placeholder(tf.float32, [None, state_size], name = 'inputs') # [?, 4]
			# [
			# 	[s, s ,s ,s], state 1
			# 	[s, s ,s ,s], state 2
			# 	[s, s ,s ,s], state 3
			# 	...
			# ]

			#Relu
			self.fc1 = tf.contrib.layers.fully_connected(self.inputs, hidden_size) # [?, 10]
			self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size) # [?, 10]

			#Linear
			self.output = tf.contrib.layers.fully_connected(self.fc2, action_size, activation_fn = None) # [?, 2]

			# ----------------------------

			self.actions = tf.placeholder(tf.int32, [None], name = 'actions') # [?]
			# [
			#  a1 = 1, 
			#  a2 = 0,
			#  a3 = 1,
			#  a4 = 1,
			#  ...
			# ]


			one_hot_actions = tf.one_hot(self.actions, action_size) # [?, 2]
			# [
			#	[0, 1],
			#	[1, 0],
			#	[0, 1],
			#	[0, 1],
			#   ...
			# ]

			self.targetsQs = tf.placeholder(tf.float32, [None], name = 'target')

			self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis = 1)
			self.loss = tf.reduce_mean(tf.square(self.targetsQs - self.Q))
			self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

tf.reset_default_graph()
mainQn = QNetwork(name = 'main', hidden_size = hidden_size, learning_rate = learning_rate)
memory = Memory(max_size = memory_size)

env.reset()
#random step to get the pole-cart moving
state, reward, done, _ = env.step(env.action_space.sample())
#Make a bunch of random actions and store them in the memory

for ii in range(pretrain_length):
	action = env.action_space.sample()
	next_state, reward, done, _ = env.step(action)

	if done:
		next_state = np.zeros(state.shape)
		memory.add((state, action, reward, next_state))
		env.reset()
		#random step to get the pole-cart moving
		state, reward, done, _ = env.step(env.action_space.sample())
	else:
		memory.add((state, action, reward, next_state))
		state = next_state


# TRAINING

saver = tf.train.Saver()
rewards_list = []

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	step = 0
	for ep in range(1, train_episodes):
		total_reward = 0
		t = 0
		while t < max_steps:
			step += 1
			env.render()

			explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * step)
			if explore_p > np.random.rand():
				action = env.action_space.sample()
			else:
				# run the state through the NN and get the prediction Q values for each action
				# get the action with highest Q
				feed = {mainQn.inputs : state.reshape((1, *state.shape))}
				Qs = sess.run(mainQn.output, feed_dict = feed)
				action = np.argmax(Qs)

			next_state, reward, done, _ = env.step(action)

			total_reward += reward

			if done:
				next_state = np.zeros(state.shape)
				t = max_steps

				print(
					'Episode: {}'.format(ep),
					'Total reward: {}'.format(total_reward),
				 	'Training loss: {}'.format(loss),
				 	'Explore p: {}'.format(explore_p)
				 	)
				rewards_list.append((ep, total_reward))

				memory.add((state, action, reward, next_state))

				env.reset()
				state, reward, done, _ = env.step(env.action_space.sample())
			else:
				memory.add((state, action, reward, next_state))
				state = next_state
				t += 1

			batch = memory.sample(batch_size)
			states = np.array([each[0] for each in batch])
			actions = np.array([each[1] for each in batch])
			rewards = np.array([each[2] for each in batch])
			next_states = np.array([each[3] for each in batch])

			#Train
			# run the next_states through the NN and get the prediction Q values for each action
			targets_Qs = sess.run(mainQn.output, feed_dict = {mainQn.inputs : next_states})

			episode_ends = (next_states == np.zeros(states[0].shape)).all(axis = 1)
			targets_Qs[episode_ends] = (0, 0)

			targets = rewards + gamma * np.max(targets_Qs, axis = 1)

			feed_dict = { mainQn.inputs: states, mainQn.targetsQs: targets, mainQn.actions: actions }
			loss, _ = sess.run([mainQn.loss, mainQn.opt], feed_dict = feed_dict)
			saver.save(sess, "checkpoints/cartpole.ckpt")
