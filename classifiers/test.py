import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from one_layer_perceptron import Perceptron

# Download Iris dataset and extract two features
iris_df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

