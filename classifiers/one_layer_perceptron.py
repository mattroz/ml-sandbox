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
		# Errors list will show learning process through epoches
		self.errors = []
		# Start n_iters epoches
		for _ in range(n_iters):
			# Errors in current epoch
			errors = 0
			for _x, _y in zip(X,y):
				# Update variable need for learning process
				update = self.learning_rate * (_y - self.predict(x))
				# Update all weights (weight[0] is
				# so-called 'bias unit', doesn't depend from x
				self.weights[1:] += _x * update
				self.weights[0] += update
				# If update equals 0 - predictet label is correct
				# and there's no error, 1 otherwise
				errors += int(update == 0.0)
			
