import numpy as np
import matplotlib.pyplot as plt

class Perceptron:

	def __init__(self, learning_rate=0.1, n_iters=10):
		self.learning_rate = learning_rate
		self.n_iters = n_iters
	
	def plot_decision_boundary(self, X):
        plt.ylim(ymin=-2)
        plt.ylim(ymax=4)
        halfsize = len(X) // 2
        plt.scatter(X[:halfsize, 0], X[:halfsize, 1], color='red', marker='x')
        plt.scatter(X[halfsize:, 0], X[halfsize:, 1], color='blue', marker='o')
        w = self.weights
        x = X[:, 0]
        y = -(w[1]/w[2]) * x - w[0]/w[2]
        plt.plot(x,y)


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
		for _ in range(self.n_iters):
			# Errors in current epoch
			_errors = 0
			for _x, _y in zip(X,y):
				# Update variable need for learning process
				update = self.learning_rate * (_y - self.predict(_x))
				# Update all weights (weight[0] is
				# so-called 'bias unit', doesn't depend from x
				self.weights[1:] += _x * update
				self.weights[0] += update
				# If update equals 0 - predictet label is correct
				# and there's no error, 1 otherwise
				_errors += int(update != 0.0)
			self.plot_decision_boundary(X)
			self.errors.append(_errors)
		return self

	# Net input is a function z(X) = w0 + w1*x1 + w2*x2 + ... + wn*xn
	def net_input(self, X):
		inp = self.weights[0] + np.dot(X, self.weights[1:])
		return inp
	

	def predict(self, X):
			return np.where(self.net_input(X) >= 0.0, 1, -1)
