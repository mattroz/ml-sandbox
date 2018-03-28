# Helpers for deepNN implementation
import numpy as np

def relu(Z):
	return np.maximum(0, Z)

def relu_derivative(Z):
	return np.where(Z >= 0, 1, 0)

def sigmoid(Z):
	return (1/(1 + np.exp(-Z)))

def sigmoid_derivative(Z):
	return(sigmoid(Z) * (1 - sigmoid(Z)))
