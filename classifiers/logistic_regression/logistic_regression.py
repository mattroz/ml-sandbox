import numpy as np

# Logistic regression implementation using batch gradient descent
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
			phi_z = self.sigmoid(X)
			errors = - y * np.log(phi_z) - (1 - y)*np.log(1 - phi_z)
			# Cost function derivative have been 
			# calculated by hand, but in case of fire: 
			# https://stats.stackexchange.com/questions/261692/gradient-descent-for-logistic-regression-partial-derivative-doubt
			update = -(y - self.sigmoid(X))
			self.weights_[1:] += self.learning_rate * X.T.dot(update)
			self.weights_[0] += self.learning_rate * update.sum()
			# Costs are NaNs at the end of training, need to
			# figure this issue out
			cost = -(errors).sum()
			self.cost_.append(cost)
		return self	

	# Z = w0*x0 + w1*x1 + ... + wm*xm
	def net_input(self, X):
		return (self.weights_[0] + np.dot(X, self.weights_[1:]))
	
	# Sigmoid function
	def sigmoid(self, X):
		return 1 / (1 + np.exp(-self.net_input(X)))

	def activation(self, X):
		return self.sigmoid(X)

	def predict(self, X):
		return np.where(self.activation(X) >= 0.5, -1, 1)
