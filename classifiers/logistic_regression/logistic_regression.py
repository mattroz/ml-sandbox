import numpy as np

# Adaline implementation using batch gradient descent
class LogisticRegression:
	
	def __init__(self, learning_rate=0.1, n_iters=10):
		self.learning_rate = learning_rate
		self.n_iters = n_iters

	def fit(self, X, y):
		n_features = X.shape[1]
		self.weights_ = np.random.randn(n_features + 1)
		self.cost_ = []
		# Start to iterate through epochs
		for _ in range(self.n_iters):
			phi_z = self.net_input(X)
			print(phi_z)
		#	errors = y * np.log(phi_z) + (1 - y)*np.log(1 - phi_z)
			update = (y / self.net_input(X)) - (1 - y)/(1 - self.net_input(X))
			self.weights_[1:] += self.learning_rate * X.T.dot(update)
			self.weights_[0] += self.learning_rate * update.sum()
		#	cost = -(errors).sum()
		#	self.cost_.append(cost)
		return self
	
	def net_input(self, X):
		return (self.weights_[0] + np.dot(X, self.weights_[1:]))
	
	def activation(self, X):
		return self.net_input(X)

	def predict(self, X):
		return np.where(self.activation(X) >= 0.0, -1, 1)
