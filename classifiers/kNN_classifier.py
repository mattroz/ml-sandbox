# simple kNN classifier using artificial 2D-data

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import random
import math


class SimpleKNN:
    def fit(self, X_train, y_train, k=1):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

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
    def closest_neighbour(self, sample):
        class_labels = []
        X_train_copy = self.X_train
        # Iterate through k closest neighbours
        for i in range(0, self.k):
            shortest_dist = euclidean_distance(sample, X_train_copy[0])
            shortest_dist_idx = 0
            # Find the closest one at the current iteration
            j = 1
            while j < len(X_train_copy):
                dist = euclidean_distance(sample, X_train_copy[j])
                if dist < shortest_dist:
                    shortest_dist = dist
                    shortest_dist_idx = j
                j += 1
            # add the closes neighbour's label to list
            class_labels.append(self.y_train[shortest_dist_idx])
            # delete this neighbour from train set
            X_train_copy = np.delete(X_train_copy, shortest_dist_idx, 0)
        return self.most_common_class(class_labels)

    def most_common_class(self, labels):
        # count number of all elements' occurences
        counts = np.bincount(labels)
        # return the most common class
        return np.argmax(counts)


def euclidean_distance(point_A, point_B):
    return distance.euclidean(point_A, point_B)


# Create synthetic data of crosses and circles
labels = np.array([])

# This data is just for plotting
cross_x = 0.5 + 0.4 * np.random.randn(200)
cross_y = 3 + 0.3 * np.random.randn(200)
labels = np.concatenate((labels, np.ones(200).tolist()))

circle_x = 2 + 0.4 * np.random.randn(200)
circle_y = 2 + 0.4 * np.random.randn(200)
labels = np.concatenate((labels, np.zeros(200).tolist()))

# Join x's and y's
_x = np.concatenate((cross_x, circle_x))
_y = np.concatenate((cross_y, circle_y))
X = np.transpose(np.vstack((_x, _y)))
y = labels

# Split data into training and test sets
X_train, X_test, y_train, y_test = X[::2], X[1::2], y[::2], y[1::2]

# train tree classifier
clf = SimpleKNN()
clf.fit(X_train, y_train, k=6)

# Print out prediction accuracy.
predication = clf.predict(X_test)
print(accuracy_score(y_test, predication))

# This point is laying between two sets,
# so it's just interesting to see which 
# class will be assigned to this point by the classifier.
middle_point = [1.36, 2.56]
print(clf.predict([middle_point]))

# plot data
plt.scatter(cross_x, cross_y, color='red', marker='x')
plt.scatter(circle_x, circle_y, color='blue', marker='o')
plt.scatter(middle_point[0], middle_point[1], color='green', marker='^')
plt.show()
