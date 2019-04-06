import numpy as np
import random

# NN class 
class NeuralNet(object):

    def __init__(self, layer_sizes, reg_lambda=0.01, dropout_p=0.5):
        '''        
        Arguments:
            layer_sizes {list} -- Initialize NN with number of layers and number of units per layer.
            Takes list of numbers. The size of the list indicate number of layers.
            Each number in the list indicates number of units in each layer.
        
        Keyword Arguments:
            reg_lambda {float} -- regularization lambda value (default: {0.01})
            dropout_p {float} -- probability value for dropouts  (default: {0.5})
        '''
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.reg_lambda = reg_lambda
        self.dropout_p = dropout_p
        
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def activation(self, z, **kwargs):
        '''Placeholder for activation funciton. It takes a string of predefined activation function, or you can pass your own custome function.
        
        Arguments:
            z {Numpy Array} -- Input from previous layer. Should be a Numpy vector
        
        Keyward Arguments:
            func {string or function} -- Either pass a string to select predefined activation function 
                eg: func='sigmoid' or pass a custome function, eg: func=<function type>
            
            List of predefined activation functions:
            - Sigmoid
        '''
        funcs = {'sigmoid': 1 / 1 + np.exp(-z), 
                }
        return funcs[kwargs['func']]

    def loss_function(self):
        pass

    def feed_forward(self, input, activation_func):
        '''Initiate feed forward process to compute the output.
        
        Arguments:
            input {Numpy vector}} -- the output from the previous layer or the input layer
        '''
        for b, w in zip(self.biases, self.weights):
            input = self.activation(np.dot(w, input) + b, func=activation_func)
        return input

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        '''Train the neural network using stochastic gradient descent. 
        
        Arguments:
            training_data {list} -- training data should be list of tuples in the format (x, y)
            epochs {int} -- number of iteration NN will be running
            mini_batch_size {int} -- size of the mini batch that SGD will train 
            learning_rate {float} -- 
        
        Keyword Arguments:
            test_data {list} -- if provided, the network will be evaluated against the test data and progress will be printed out (default: {None})
        '''
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            #shuffle the training data in every iteration
            random.shuffle(training_data) 
            # split the training data to batches according to the parameter mini_batch_size
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            # iterate over every mini batch to calculate the gradient and perform backpropagation
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, training_data)
            
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, None, n_test))
            
            else:
                print("Epoch {0} completed".format(j))

    def update_mini_batch(self, mini_batch, learning_rate):
        '''Update network's weight and biases by applying gradient descent using backpropagation 
            to a single mini batch
        
        Arguments:
            mini_batch {list} -- list of tuples (x, y)
            learning_rate {int} -- learning rate
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_propogation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # updating the biases and weights after running the packpropagation
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(learning_rate/len(mini_batch))* nw for w, nw in zip(self.weights, nabla_w)]



    
    
    
    def back_propogation(self, x, y):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

# method to compute forward pass
# method to compute packpropgation
# method that computes the activation fundtion
# method that computes the loss function
# method that compute the gradient?? (may be this is the same as backward pass stage?)
