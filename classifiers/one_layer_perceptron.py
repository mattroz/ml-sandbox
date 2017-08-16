import numpy as np

class Perceptron:

	def __init__(self, learning_rate=0.1, n_iters=10):
		self.learning_rate = learning_rate
		self.n_iters = n_iters

	def fit(self, X, y):
		# Create weights array with (1 + n_features) size
		n_features = X.shape[1]
		# Set random weights with expected value = 0 
		# and std deviation = 0.01 
		rand = np.random.RandomState(1)
		self.weights = rand.normal(loc=1.0, scale=.01, size=(1 + n_features))
		# TODO: finish fit method, implement 
		# prediction method, net function calculation
