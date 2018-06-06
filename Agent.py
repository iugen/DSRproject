import numpy as np
from keras import models, layers, optimizers
from collections import deque

class agent_class(object):
	"""docstring for agent"""
	def __init__(self):
		self.actions_list = [0,1]   # stay/change
		self.training_batch = deque(maxlen=500)
		self.epsilon = 1
		self.eps_decay = 0.95
		self.min_eps = 0.1
		self.model = self.createNN()
		self.gamma = 0.9

	def createNN(self):
		model = models.Sequential()
		model.add(layers.Dense(16,activation='relu',input_shape=(8,)))
		#model.add(layers.Dense(32,activation='relu',))
		model.add(layers.Dense(2,activation='linear')) 
		# I have only one traffic light. Simple strategy: stay/change
		opt = optimizers.RMSprop(lr=0.001)
		model.compile(optimizer = opt, loss='mse')
		return model


	def select_action(self,state):
		action = None
		if self.epsilon > np.random.rand():
			action = np.random.choice(self.actions_list)  
		else:
			# action = self.model.predict(state)
			action = np.argmax(self.model.predict(state))
		return action

	
	def memory(self,state,reward,next_s,a):
		self.training_batch.append([state,reward,next_s,a])


	def fit_NN(self):
		'''data = np.array(self.training_batch)
		for state,reward,next_s,a in data:
			Q1 = reward + self.gamma*np.max(self.model.predict(next_s))
			Q = self.model.predict(state)
			Q[0][a] = Q1            
			self.model.fit(state,Q, epochs=1)
		if self.epsilon > self.min_eps:
			self.decay()'''
		data = np.array(self.training_batch)    
		Q_target = np.empty((data.shape[0], 2))
		state_vector = np.empty((data.shape[0], 8))
		i=0
		for state,reward,next_s,a in data:
			Q1 = reward + self.gamma*np.max(self.model.predict(next_s))
			Q = self.model.predict(state)
			Q[0][a] = Q1
			state_vector[[i]] = state
			Q_target[[i]] = Q
			i += 1
		self.model.fit(state_vector,Q_target, epochs=5)
		if self.epsilon > self.min_eps:
			self.decay()


	def decay(self):
		self.epsilon = self.epsilon*self.eps_decay



