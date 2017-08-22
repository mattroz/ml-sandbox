import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from one_layer_perceptron import Perceptron

# Download Iris dataset and extract two features
iris_df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Exctracting 'sepal length', 'petal length' features,
# class labels
sepal_idx = 0
petal_idx = 2
class_idx = 4
X = iris_df.iloc[:100, [sepal_idx, petal_idx]].values
# Divide class labels to digit representation:
# if Iris-setosa 		=> -1,
# if Iris-versicolor =>  1
y = iris_df.iloc[:100, class_idx].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Train perceptron
perceptron = Perceptron()
perceptron.fit(X, y)

# Plot perceptron error progress through epochs
plt.subplot(2,1,1)
plt.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Num of missclassif-s')

# Plot data and decision border
plt.subplot(2,1,2)
plt.scatter(X[:50, 0], X[:50, 1], marker='o', color='red', label='setosa')
plt.scatter(X[50:, 0], X[50:, 1], marker='x', color='blue', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')

# Decision border is a line w0 + w1*x + w2*y = 0,
# Hence y = -w1/w2*x - w0/w2
w = perceptron.weights
x_ = X[:,0]
y_ = (-w[1]/w[2] * x_) - (w[0]/w[2])
plt.plot(x_, y_, color='black')
plt.show()

