import numpy as np

class Adaline:
	
	def __init__(self, learning_rate=0.1, n_iters=10):
		self.learning_rate = learning_rate
		self.n_iters = n_iters
	
	
	def net_input(self, X):
		return (self.weights_[0] + np.dot(X, self.weights_[1:]))


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
			self.weight_[1:] += self.learning_rate * X.T.dot(errors)
			self.weights[0] += self.learning_rate * errors.sum()
			cost = (errors**2).sum() / 2.0
			self.cost_.append(cost)
		return self

	
	def activation(self, X):
		pass


	def predict(self, X):
		return np.where(self.net_input(X) >= 0.0, -1, 1)
