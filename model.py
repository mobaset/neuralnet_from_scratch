import numpy as np
import random
import activations
import cost_funcs

# NN class 
class NeuralNet(object):

    def __init__(self, layer_sizes, activation, reg_lambda=0.01, dropout_p=0.5):
        '''        
        Arguments:
            layer_sizes {list} -- Initialize NN with number of layers and number of units per layer.
            Takes list of numbers. The size of the list indicate number of layers.
            Each number in the list indicates number of units in each layer.
        
        Keyword Arguments:
            reg_lambda {float} -- regularization lambda value (default: {0.01})
            dropout_p {float} -- probability value for dropouts  (default: {0.5}) 
        '''
        #TODO implement the dropout technique 
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.activation = activations.get(activation)
        self.reg_lambda = reg_lambda
        self.dropout_p = dropout_p
        
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def feed_forward(self, input):
        '''Initiate feed forward process to compute the output.
        
        Arguments:
            input {Numpy vector} -- the output from the previous layer or the input layer
        '''
        for b, w in zip(self.biases, self.weights): # loop over each layer
            input = self.activation(np.dot(w, input) + b)
        return input

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, cost_func, test_data=None):
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
                self.update_mini_batch(mini_batch, learning_rate, cost_func)
            
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            
            else:
                print("Epoch {0} completed".format(j))

    def update_mini_batch(self, mini_batch, learning_rate, cost_func):
        '''Update network's weight and biases by applying gradient descent using backpropagation 
            to a single mini batch
        
        Arguments:
            mini_batch {list} -- list of tuples (x, y)
            learning_rate {int} -- learning rate
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_propogation(x, y, cost_func)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # updating the biases and weights after running the packpropagation
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(learning_rate/len(mini_batch))* nw for w, nw in zip(self.weights, nabla_w)]

    def back_propogation(self, x, y, cost_func):
                
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward pass 
        activation = x
        activations = [x] #list of activations per layer
        zs = [] # list of weighted inputs per layer
        
        for w, b in zip(self.weights, self.biases): #loop over each layer
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation(z)
            activations.append(activation)

        #backward pass
        output_error = cost_funcs.get(cost_func)(activations[-1], y) * self.activation(zs[-1], is_derivative=True)
        nabla_b[-1] = output_error
        nabla_w[-1] = np.dot(output_error, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            output_error = np.dot(self.weights[-l+1].transpose(), output_error) * self.activation(z, is_derivative=True)
            nabla_b[-l] = output_error
            nabla_w[-l] = np.dot(output_error, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_result = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_result)

    def fit(self):
        #TODO implement fit method
        pass

    def predict(self):
        #TODO implement predict method
        pass
        