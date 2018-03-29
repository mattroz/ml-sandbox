import numpy as np
from helpers import *
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def initialize_parameters(layers_size):
    """
    Returns parameters ('W1', 'b1', W2', 'b2', ...)
    """
    
    parameters = {}
    for l in range(1, len(layers_size)):
        squash_coeff = np.sqrt(2/layers_size[l-1])
        #print(squash_coeff)
        parameters['W' + str(l)] = np.random.randn(layers_size[l], layers_size[l-1]) * squash_coeff
        parameters['b' + str(l)] = np.zeros((layers_size[l], 1))
    
    return parameters


def linearize_and_activate_forward_unit(A_previous, W, b, activation_function='relu'):
    """
    Returns A, layer_parameters(W, b, Z, A)
    """
    
    Z = np.dot(W, A_previous) + b
    if activation_function == 'relu':
        A = relu(Z)
    elif activation_function == 'sigmoid':
        A = sigmoid(Z)
    
    layer_parameters = {
        'W': W,
        'b': b,
        'Z': Z,
        'A': A_previous
    }
    
    return A, layer_parameters

def deep_forward(X, parameters):
    n_of_layers = len(parameters) // 2
    A = X
    layers_parameters = []
    
    W = parameters
    
    for l in range(1, n_of_layers):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A, params = linearize_and_activate_forward_unit(A, W, b, 'relu')
        layers_parameters.append(params)
    
    # Last layer
    W = parameters['W' + str(n_of_layers)]
    b = parameters['b' + str(n_of_layers)]
    A_last, params = linearize_and_activate_forward_unit(A, W, b, 'sigmoid')
    layers_parameters.append(params)
    
    return A_last, layers_parameters


def calculate_cost(Y_hat, Y):
    """
    Returns cost
    """
    m = Y.shape[1]
    
    log_probs = np.multiply(Y, np.log(Y_hat)) + np.multiply(1 - Y, np.log(1 - Y_hat))
    cost = -(1/m) * np.sum(log_probs)
    cost = np.squeeze(cost)
    
    return cost


def propagate_back_activation(dA, Z, activation):
    """
    Returns dZ[L]
    """
    if activation == 'relu':
        dZ = np.multiply(dA, relu_derivative(Z))
    elif activation == 'sigmoid':
        dZ = np.multiply(dA, sigmoid_derivative(Z))
    
    return dZ


def propagate_back_linear(layer_parameters, dZ):
    """
    Returns dA, dW, db
    """
    W = layer_parameters['W']
    b = layer_parameters['b']
    A_previous = layer_parameters['A']
    m = A_previous.shape[1]
    
    dW = (1/m) * np.dot(dZ, A_previous.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_previous = np.dot(W.T, dZ)
    
    return dA_previous, dW, db


def backward_propagation_unit(dA, layer_parameters, activation):
    W = layer_parameters['W']
    b = layer_parameters['b']
    Z = layer_parameters['Z']
    A_previous = layer_parameters['A']
    dZ = propagate_back_activation(dA, Z, activation)
    dA_previous, dW, db = propagate_back_linear(layer_parameters, dZ)
    
    return dA_previous, dW, db


def deep_backward(AL, Y, layers_parameters):
    """
    layer_parameters: W, b, Z, A
    """
    m = Y.shape[1]
    layers_number = len(layers_parameters)
    Y = Y.reshape(AL.shape)
    gradients = {}
    
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    gradients['dA' + str(layers_number)], \
    gradients['dW' + str(layers_number)], \
    gradients['db' + str(layers_number)] = backward_propagation_unit(dAL, layers_parameters[-1], 'sigmoid')
    
    for l in reversed(range(layers_number-1)):
        current_layer_params = layers_parameters[l]
        gradients['dA' + str(l+1)], \
        gradients['dW' + str(l+1)], \
        gradients['db' + str(l+1)] = backward_propagation_unit(gradients['dA' + str(l+2)], 
                                                               current_layer_params, 
                                                               'relu') 
    return gradients


def update_weights(parameters, gradients, learning_rate):
    n_of_layers = len(parameters) // 2
    
    for l in range(n_of_layers):
        parameters['W' + str(l+1)] -= learning_rate * gradients['dW' + str(l+1)]
        parameters['b' + str(l+1)] -= learning_rate * gradients['db' + str(l+1)]
    
    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))

    # Forward propagation
    probas, params = deep_forward(X, parameters)


    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print("Accuracy: "  + str(np.sum((p == y)/m)))

    return p



def deep_NN_model(X, Y, layers_size, learning_rate=0.0075, n_iters=2500):
    costs = []
    parameters = initialize_parameters(layers_size)
    
    for i in range(n_iters):
        A_last, layers_parameters = deep_forward(X, parameters)
        cost = calculate_cost(A_last, Y)
        gradients = deep_backward(A_last, Y, layers_parameters)
        parameters = update_weights(parameters, gradients, learning_rate)
        
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
