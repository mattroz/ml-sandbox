import numpy as np

# Adaline implementation using batch gradient descent
class AdalineGD:
	
	def __init__(self, learning_rate=0.1, n_iters=10):
		self.learning_rate = learning_rate
		self.n_iters = n_iters

	def fit(self, X, y):
		n_features = X.shape[1]
		self.weights_ = np.random.randn(n_features + 1)
		self.cost_ = []
		# Start to iterate through epochs
		for _ in range(self.n_iters):
			errors = y - self.net_input(X)
			# Weights update implemented using gradient descent,
			# where cost function is J(w) = SSE: 
			# 1/2 * sum(y - sum(w[j]*x[i][j])^2
			self.weights_[1:] += self.learning_rate * X.T.dot(errors)
			self.weights_[0] += self.learning_rate * errors.sum()
			cost = (errors**2).sum() / 2.0
			self.cost_.append(cost)
		return self
	
	def net_input(self, X):
		return (self.weights_[0] + np.dot(X, self.weights_[1:]))
	
	def activation(self, X):
		return self.net_input(X)

	def predict(self, X):
		return np.where(self.activation(X) >= 0.0, -1, 1)



# Adaline implementation with stochastic gradien descent
class AdalineSGD:

	# It's essential that data should be shuffled for SGD
	def __init__(self, learning_rate=0.01, n_iters=10, 
						shuffle=True):
		self.learning_rate = learning_rate
		self.n_iters = n_iters
		self.weights_initialized = False
		self.shuffle = shuffle

	# Single underscore means that this method is private
	def _shuffle(self, X, y):
		p = np.random.permutation(len(y))
		return X[p], y[p]

	# Update neuron weights and return error value
	def _update_weights(self, x, target):
		error = target - self.net_input(x)
		self.weights_[1:] += self.learning_rate * x.dot(error)
		self.weights_[0] += self.learning_rate * error
		return ((error ** 2) / 2)
	
	def _initialize_weights(self, n):
		self.weights_initialized = True
		self.weights_ = np.random.randn(n + 1)

	def fit(self, X, y):
		n_features = X.shape[1]
		self._initialize_weights(n_features)
		self.cost_ = []
		for _ in range(self.n_iters):
			epoch_cost = []
			# Shuffle samples
			if self.shuffle:
				X, y = self._shuffle(X,y)
			# Main difference between BGD and SGD is that
			# the last one update weights incrementally with
			# every sample:
			for x, target in zip(X, y):
				'''error = target - self.net_input(x)
				self.weights_[1:] += self.learning_rate * x.dot(error)
				self.weights_[0] += self.learning_rate * error
				'''
				epoch_cost.append(self._update_weights(x, target))
			# Calculate average cost in the current epoch
			avg_epoch_cost = sum(epoch_cost) / len(epoch_cost)
			self.cost_.append(avg_epoch_cost)
		return self

	# Partial fit is using for online learning, 
	# e.g. we can feed additional data after the main fit
	# have been called. It means that weights wouldn't be 
	# re-initialized and neuron wouldn't be re-fitted
	def partial_fit(self, X, y):
		if not self.weights_initialized:
			self._initialize_weights(X.shape[1])
		# If we have more than one sample have been fed to this method
		if y.ravel().shape[0] > 1:
			# Walk through each sample and update weights 
			# incrementally according to SGD  
			for x, target in zip(X,y):
				self._update_weights(x, target)
		# And if we have only one sample, just update weights once
		else:
			self._update_weights(X, y)
		return self

	def net_input(self, X):
		return (np.dot(X, self.weights_[1:]) + self.weights_[0])

	def activation(self, X):
		return self.net_input(X)	

	def predict(self, X):
		return (np.where(self.activation(X) >= 0, 1, -1))
