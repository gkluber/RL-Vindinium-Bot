from random import choice
import time

from game import Game
import tensorflow as tf, import numpy as np

class Bot:
	def __init__(self, sess):
		self.sess = sess

class QBot(Bot):
	def __init__(self, sess, save, learning_rate, momentum, timeout, n_turns, save_iterations, mode, key, url):
		super().__init__(sess)
		self.save = save #boolean
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.connection = Client(timeout, url, key, mode)
		build_model()
		
	def build_model(self):
		self.input_height = 20
		self.input_width = 20
		self.input_channels = 11
		self.conv_n_maps = [32, 64, 64]
		self.conv_kernel_sizes = [(5,5), (3,3), (3,3)]
		self.conv_strides = [2,1,1]
		self.conv_paddings = ["SAME"]*3
		self.conv_activation = [tf.nn.relu] * 3
		self.n_hidden_in = 100*input_channels #with one stride of 2
		n_hidden = 512
		hidden_activation = tf.nn.relu
		
		
		n_hidden = 4
		n_outputs = 1
		initializer = tf.contrib.layers.variance_scaling_initializer() #He initialization

		self.X = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width, self.input_channels])
		
		
		
		hidden = tf.layers.dense(self.X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
		logits = tf.layers.dense(hidden, n_outputs,kernel_initializer=initializer)
		outputs = tf.nn.sigmoid(logits) #probability of moving left

		p_left_and_right = tf.concat(axis=1, values=[outputs,1-outputs])
		threshold = tf.constant(0.5, shape=[1,1])
		
		self.action = tf.multinomial(tf.log(p_left_and_right),num_samples=1) #samples action from probability distribution
		
		y = 1 - tf.to_float(self.action) #probability of right
		
		#learning_rate = 0.1 
		cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2)
		grads_and_vars = self.optimizer.compute_gradients(cross_entropy)
		self.gradients = [grad for grad, var in grads_and_vars]
		self.gradient_placeholders = []
		grads_and_vars_feed = []
		for grad, var in grads_and_vars:
			gradient_placeholder = tf.placeholder(tf.float32,shape=grad.get_shape())
			self.gradient_placeholders.append(gradient_placeholder)
			grads_and_vars_feed.append((gradient_placeholder, var))
		self.training_op = self.optimizer.apply_gradients(grads_and_vars_feed)

		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		
		if not self.ignore_checkpoint:
			self.load_model()
	
	def train(self):
		#training phase
		#hyperparams
		#n_iterations = 1 #iters to train on
		#n_max_steps = 2000 #max steps per episode (to prevent infinite loop)
		#n_games_per_update = 10 #10 games per iter
		#save_iterations = 50 #save every 10 iters
		#discount_rate = 0.95
		#n_test_games = 50
		if self.ignore_checkpoint:
			self.init.run()
		for iteration in range(self.epochs):
			all_rewards = [] #all sequences of raw rewards for each episode
			all_gradients = [] #gradients saved at each step of each episode
			for game in range(self.games_per_update):
				print("Running game #{}".format(game))
				current_rewards = []
				current_gradients = []
				obs = self.env.reset()
				for step in range(self.max_steps):
					action_val, gradients_val = self.sess.run(
							[self.action,self.gradients],
							feed_dict={self.X:obs.reshape(1,self.n_inputs)}) #one observation
					obs, reward, done, info = self.env.step(action_val[0][0])
					current_rewards.append(reward) #raw reward
					current_gradients.append(gradients_val) #raw grads
					if done:
						print("Finished game #{} in {} steps".format(game,step+1))
						break
					elif step==self.max_steps-1:
						print("Hit max num of steps at game #{}".format(game))
				
				all_rewards.append(current_rewards) #adds to the history of rewards
				all_gradients.append(current_gradients) #gradient history
			#all games executed--time to perform policy gradient ascent
			print("Performing gradient ascent at iteration {}".format(iteration))
			all_rewards = self.discount_and_normalize_rewards(all_rewards, self.discount_rate)
			feed_dict = {}
			for var_index, grad_placeholder in enumerate(self.gradient_placeholders):
				#multiplication by the "action scores" obtained from discounting the future events appropriately--meaned to average the signals
				mean_gradients = np.mean(
					[reward*all_gradients[game_index][step][var_index] #iterates through each variable in the gradient (var_index)
						for game_index, rewards in enumerate(all_rewards)
						for step,reward in enumerate(rewards)],
					axis=0)
				feed_dict[grad_placeholder] = mean_gradients
			self.sess.run(self.training_op,feed_dict=feed_dict)
			if (iteration +1)% self.save_iterations == 0 and self.save:
				print("Saving model...")
				self.saver.save(self.sess,self.get_checkpoint_file())
	
	#returns true if legal, false otherwise
	def legal(self, direction):
		
	
	#state given as a JSON object
	def move(self, state):
		self.game = Game(state)
		dirs = ['Stay', 'North', 'South', 'East', 'West']
		