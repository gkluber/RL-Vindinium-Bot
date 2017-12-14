from random import choice
from enum import Enum
from utils import Pos
from utils import Player
from utils import Tile
from utils import GameState
from client import Client
import utils
import threading

import time

import tensorflow as tf, numpy as np

class ModelParametersCopier(object):
    def __init__(self, online_scope, target_scope):
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(online_scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(target_scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)
            
    def make(self, sess):
        sess.run(self.update_ops)

class Bot:
	def __init__(self, sess):
		self.sess = sess

#TODO
class RandomBot(Bot):
	def train(self):
		self.connection = Client(self, timeout, url, key, mode, self.depth)

class QBot(Bot):
	def __init__(self, sess, save, learning_rate, momentum, timeout, n_turns, save_iterations, mode, key, url):
		super().__init__(sess)
		self.save = save #boolean
		self.learning_rate = learning_rate
		self.momentum = momentum
		
		self.depth = 1
		self.connection = Client(self, timeout, url, key, mode, self.depth)
		self.connection_thread = threading.Thread(target=open_connection)
		self.replay_memory = []
		build_model()
		
		#ensures that concurrency problems are eliminated
		self.lock = RLock()
	
	#is called when the client receives the next board state
	def callback(self, state):
		with self.lock:
			#add to training pool
			self.replay_memory.append(to_state_matrix(state))
	
	def to_state_matrix(self, state):
		pass
	
	'''  
	TODO -- implement Swish activation function
	Reference: 
	Searching for Activation Functions (2017) 
	https://arxiv.org/abs/1710.05941
	'''
	@staticmethod
	def swish(x):
		return x*tf.nn.sigmoid(x)
	
	def build_model(self):
		self.input_height = 12
		self.input_width = 12
		self.input_channels = 23*self.depth
		self.conv_n_maps = [32, 64, 64]
		self.conv_kernel_sizes = [(3,3), (3,3), (2,2)]
		self.conv_strides = [2,1,1]
		self.conv_paddings = ["SAME"]*3
		self.conv_activation = [tf.nn.relu] * 3
		self.n_hidden_in = 100*input_channels #with one stride of 2
		n_hidden = 256
		hidden_activation = tf.nn.relu
		n_outputs = len(Action)
		initializer = tf.contrib.layers.variance_scaling_initializer() #He initialization

		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		
		if not self.ignore_checkpoint:
			self.load_model()
	
		def q_network(X_state, name):
			prev_layer = tf.cast(X_state, tf.int8)
			with tf.variable_scope(name) as scope:
				for n_maps, kernel_size, strides, padding, activation in zip(
						conv_n_maps, conv_kernel_sizes, conv_strides,
						conv_paddings, conv_activation):
					prev_layer = tf.layers.conv2d(
						prev_layer, filters=n_maps, kernel_size=kernel_size,
						strides=strides, padding=padding, activation=activation,
						kernel_initializer=initializer)
				last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
				hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
										 activation=hidden_activation,
										 kernel_initializer=initializer)
				outputs = tf.layers.dense(hidden, n_outputs,
										  kernel_initializer=initializer)
			trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
											   scope=scope.name)
			trainable_vars_by_name = {var.name[len(scope.name):]: var
									  for var in trainable_vars}
			return outputs
		
		X_state = tf.placeholder(tf.bool, shape=[None, self.input_height, self.input_width,
                                            self.input_channels])
		
		online_scope = "q_networks/online"
		target_scope = "q_networks/target"
		
		online_q_values = q_network(X_state, name=online_scope)
		target_q_values = q_network(X_state, name=target_scope)
		
		copier = ModelParametersCopier(online_scope,target_scope)

		with tf.variable_scope("train"):
			X_action = tf.placeholder(tf.int32, shape=[None])
			y = tf.placeholder(tf.float32, shape=[None, 1])
			q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
									axis=1, keep_dims=True)
			error = tf.abs(y - q_value)
			clipped_error = tf.clip_by_value(error, 0.0, 1.0)
			linear_error = 2 * (error - clipped_error)
			loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

			global_step = tf.Variable(0, trainable=False, name='global_step')
			optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum, use_nesterov=True)
			training_op = optimizer.minimize(loss, global_step=global_step)

		q_placeholder = tf.placeholder(tf.float32, shape=())
		loss_placeholder = tf.placeholder(tf.float32, shape=())
		#self.baseline_placeholder = tf.placeholder(tf.float32, shape=())
		tf.summary.scalar("Mean Max Q Value",q_placeholder)
		tf.summary.scalar("Loss value",loss_placeholder)
		#tf.summary.scalar("baseline",self.baseline_placeholder)
		summary = tf.summary.merge_all()
	
	def epsilon_greedy(self, q_values, step):
		epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
		if np.random.rand() < epsilon:
			return np.random.randint(n_outputs) # random action
		else:
			return np.argmax(q_values) # optimal action
		
	def sample_memories(self, batch_size):
		indices = np.random.permutation(len(self.replay_memory))[:batch_size]
		cols = [[], [], [], [], []] # state, action, reward, next_state, continue
		for idx in indices:
			memory = self.replay_memory[idx]
			for col, value in zip(cols, memory):
				col.append(value)
		cols = [np.array(col) for col in cols]
		return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)
	
	
	def train(self):
		#training phase
		#hyperparams
		#n_iterations = 1 #iters to train on
		#n_max_steps = 2000 #max steps per episode (to prevent infinite loop)
		#n_games_per_update = 10 #10 games per iter
		#save_iterations = 50 #save every 10 iters
		#discount_rate = 0.95
		#n_test_games = 50
		train_writer = tf.summary.FileWriter('summary/train', sess.graph)
		test_writer = tf.summary.FileWriter('summary/test', sess.graph)
		if os.path.isfile(checkpoint_path + ".index"):
			saver.restore(sess, checkpoint_path)
		else:
			init.run()
			copier.make(sess)
			#copy_online_to_target.run()
		
		#open the connection and start playing the game. retry until the connection opens
		open_connection()
		
		while True:
			
			step = global_step.eval()
			if step >= n_steps:
				break
			iteration += 1
			print("\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   ".format(
				iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q), end="")
			if done: # game over, start again
				obs = env.reset()
				for skip in range(skip_start): # skip the start of each game
					obs, reward, done, info = env.step(0)
				state = preprocess_observation(obs)

			# Online DQN evaluates what to do
			q_values = online_q_values.eval(feed_dict={X_state: [state]})
			action = epsilon_greedy(q_values, step)

			# Online DQN plays
			obs, reward, done, info = env.step(action)
			next_state = preprocess_observation(obs)

			# Let's memorize what happened
			replay_memory.append((state, action, reward, next_state, 1.0 - done))
			state = next_state

			# Compute statistics for tracking progress (not shown in the book)
			total_max_q += q_values.max()
			game_length += 1
			if done:
				mean_max_q = total_max_q / game_length
				total_max_q = 0.0
				game_length = 0

			if iteration < training_start or iteration % training_interval != 0:
				continue # only train after warmup period and at regular intervals
			
			# Sample memories and use the target DQN to produce the target Q-Value
			X_state_val, X_action_val, rewards, X_next_state_val, continues = (
				sample_memories(batch_size))
			next_q_values = target_q_values.eval(
				feed_dict={X_state: X_next_state_val})
			max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
			y_val = rewards + continues * discount_rate * max_next_q_values

			# Train the online DQN
			_, loss_val = sess.run([training_op, loss], feed_dict={
				X_state: X_state_val, X_action: X_action_val, y: y_val})

			# Regularly copy the online DQN to the target DQN
			if step % copy_steps == 0:
				copier.make(sess)

			# And save regularly
			if step % save_steps == 0:
				saver.save(sess, checkpoint_path)
			
			#write stats regularly
			if step % 20 == 0:
				sum = sess.run(summary, feed_dict={q_placeholder:mean_max_q,loss_placeholder:loss_val})
				train_writer.add_summary(sum,step)
			
			#check the status of the game
			
	#walls is matrix
	def legal(self, pos, walls: np.ndarray, action) -> bool:
		direction_vec = Pos.get_pos_vector(action.value)
		candidate = pos.clone().add(direction_vec)
		if walls[candidate.x][candidate.y] == 0:
			return True
		return False
		
	def open_connection(self):
		while True:
			self.connection.start()
		
	#state given as a JSON object
	#method predicts the next optimal move
	def move(self, state):
		self.game = Game(state)
		#dirs = ['Stay', 'North', 'South', 'East', 'West']
		