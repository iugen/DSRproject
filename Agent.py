import numpy as np
from keras import models, layers, optimizers
from collections import deque

class AgentClass(object):
	"""docstring for agent"""
	def __init__(self,exploration):
		self.actions_list = [0,1]   # stay/change
		self.training_memory = deque(maxlen=5000)
		self.epsilon = exploration
		self.eps_decay = 0.99
		self.min_eps = 0.1
		self.model = self.createNN()
		self.target_model = self.createNN()
		self.gamma = 0.9

	def createNN(self):
		model = models.Sequential()
		model.add(layers.Dense(64,activation='relu',input_shape=(17,)))
		model.add(layers.Dense(16,activation='relu',))
		model.add(layers.Dense(2,activation='linear')) 
		# I have only one traffic light. Simple strategy: stay/change
		opt = optimizers.Adam(lr=0.001)
		model.compile(optimizer = opt, loss='mse')
		return model

	def update_target_network(self):
		#self.target_model = self.model
		self.target_model.set_weights(self.model.get_weights()) 


	def select_action(self,state):
		action = None
		if self.epsilon > np.random.rand():
			action = np.random.choice(self.actions_list)  
		else:
			# action = self.model.predict(state)
			action = np.argmax(self.model.predict(state))
		return action

	
	def memory(self,state,reward,next_s,a,done):
		self.training_memory.append([state,reward,next_s,a,done])


	def fit_NN(self):
		data = np.array(self.training_memory)  
		batch = data[np.random.randint(data.shape[0], size=32), :]

		Q_target = np.empty((batch.shape[0], 2))
		state_vector = np.empty((batch.shape[0], 17))
		i=0
		for state,reward,next_s,a,done in batch:
			Q1 = reward + self.gamma*np.max(self.target_model.predict(next_s))*done
			Q = self.target_model.predict(state)
			Q[0][a] = Q1
			state_vector[[i]] = state
			Q_target[[i]] = Q
			i += 1
			self.model.fit(state_vector,Q_target, epochs=1, batch_size=batch.shape[0])
			if self.epsilon > self.min_eps:
				self.decay()

	def decay(self):
		self.epsilon = self.epsilon*self.eps_decay

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)

