from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
import random
import math


class SimpleKNN():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train
	
	def predict(self, X_test):
		predictions = []
		for sample in X_test:
			# search for the closest neghbour for current sample
			# in the whole training set
			target_label = self.closest_neighbour(sample)
			predictions.append(target_label)
		return predictions

	# Find the closest neighbour to current sample row.
	# It's one-NN classifier for now
	def closest_neighbour(sample):
		shortest_dist = euclidean_distance(sample, self.X_train[0])
		shortest_dist_idx = 0
		for i in range(1, len(self.X_train)):
			dist = euclidean_distance(sample, self.X_train[i])
			if dist < shortest_dist:
				shortest_dist = dist
				shortest_dist_idx = i
		return self.y_train[shortest_dist_idx]		


def euclidean_distance(point_A, point_B):
	return distance.euclidean(point_A, point_B)


# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# train tree classifier
clf = SimpleKNN()
clf.fit(X_train, y_train)

# print out prediction accuracy
print(accuracy_score(y_test, clf.predict(X_test)))
