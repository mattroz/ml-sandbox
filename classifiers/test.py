import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
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

# Add setosa data to the plot
plt.scatter(X[:50, 0], X[:50, 1], marker='o', color='red', label='setosa')
# Add versicolor data to the plot
plt.scatter(X[50:, 0], X[50:, 1], marker='x', color='blue', label='versicolor')
# Add labels to axis
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()
