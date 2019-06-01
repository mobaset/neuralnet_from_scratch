import numpy as np
import random
import activations
import cost_funcs
import time
import logging
from logging.config import dictConfig

log_config_path = './config/log_config.py'
# dictConfig(log_config_path)

# logger = logging.getLogger(__name__)

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

        
        # logger.info('Number of layers %d', self.num_layers)
        # logger.info('')

    @staticmethod
    def shuffle_data(data):
        idxs = list(range(len(data)))
        random.shuffle(idxs)
        return np.array([data[idx] for idx in idxs])

    def feed_forward(self, input):
        '''Initiate feed forward process to compute the output.
        
        Arguments:
            input {Numpy vector} -- the output from the previous layer or the input layer
        '''
        for b, w in zip(self.biases, self.weights): # loop over each layer
            input = self.activation(np.dot(w, input) + b)
        return input

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, cost_func, test_data=None, is_vectorized=False):
        '''Train the neural network using stochastic gradient descent. 
        
        Arguments:
            training_data {list} -- training data should be list of tuples in the format (x, y)
            epochs {int} -- number of iteration NN will be running
            mini_batch_size {int} -- size of the mini batch that SGD will train 
            learning_rate {float} -- 
        
        Keyword Arguments:
            test_data {list} -- if provided, the network will be evaluated against the test data and progress will be printed out (default: {None})
        '''
        if test_data is not None: 
            n_test = test_data[0].shape[0]
        n = training_data[0].shape[0]

        for j in range(epochs):
            t = time.time()
            #shuffle the training data in every iteration
            # x_shuffled = self.shuffle_data(training_data[0])
            # y_shuffled = self.shuffle_data(training_data[1])
            random.shuffle(training_data[0])
            random.shuffle(training_data[1])
            # split the training data to batches according to the parameter mini_batch_size
            chunk_size = round(n / mini_batch_size) # determine how many sub arrays 
            x_mini_batches = np.array_split(training_data[0], chunk_size)
            y_mini_batches = np.array_split(training_data[1], chunk_size)
            
            # iterate over every mini batch to calculate the gradient and perform backpropagation
            # i = 0
            for x_mini_batch, y_mini_batch in zip(x_mini_batches, y_mini_batches):
                # i += 1
                # print('____Mini batch {0}____'.format(i))
                self.update_mini_batch(x_mini_batch, y_mini_batch, learning_rate, cost_func, is_vectorized)
            
            t = time.time() - t
            if test_data is not None:
                print("Epoch {0}:\t {1} / {2} | Completed in {3} seconds".format(j+1, self.evaluate(test_data), n_test, t))
            
            else:
                print("Epoch {0}\t completed in {1} seconds".format(j+1, t))


    def update_mini_batch(self, x_mini_batch, y_mini_batch, learning_rate, cost_func, is_vectorized):
        '''Update network's weight and biases by applying gradient descent using backpropagation 
            to a single mini batch
        
        Arguments:
            mini_batch {list} -- list of tuples (x, y)
            learning_rate {int} -- learning rate
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        if is_vectorized:
            nabla_b, nabla_w = self.back_propogation(x_mini_batch, y_mini_batch, cost_func, is_vectorized)

        else:
            for x, y in zip(x_mini_batch, y_mini_batch):
                delta_nabla_b, delta_nabla_w = self.back_propogation(x, y, cost_func, is_vectorized)
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # updating the biases and weights after running the packpropagation
        self.weights = [w-(learning_rate/x_mini_batch.shape[0])*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/x_mini_batch.shape[0])*nb for b, nb in zip(self.biases, nabla_b)]

    def back_propogation(self, x, y, cost_func, is_vectorized):
                
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward pass
        # print('Forward pass...')
        activation = x
        activations = [x] #list of activations per layer
        zs = [] # list of weighted inputs per layer
        # i = 0
        for w, b in zip(self.weights, self.biases): #loop over each layer
            # i += 1
            # print('----Layer {0}---- w:{1}\t a:{2}\t b:{3}'.format(i, w.shape, activation.shape, b.shape))
            if is_vectorized:
                z = np.matmul(w, activation) + b
            else:
                z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation(z)
            activations.append(activation)

        #backward pass
        # print('Backward pass...')

        output_error = cost_funcs.get(cost_func)(activations[-1], y) * self.activation(zs[-1], is_derivative=True)
        nabla_b[-1] = output_error
        if is_vectorized:
            nabla_w[-1] = np.matmul(output_error, activations[-2].transpose((0,2,1)))
            # nabla_b[-1] = np.sum(nabla_b[-1], axis=0)
            # nabla_w[-1] = np.sum(nabla_w[-1], axis=0)
        else:
            nabla_w[-1] = np.dot(output_error, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            if is_vectorized:
                output_error = np.matmul(self.weights[-l+1].transpose(), output_error) * self.activation(z, is_derivative=True)
                nabla_b[-l] = output_error
                nabla_w[-l] = np.matmul(output_error, activations[-l-1].transpose((0,2,1)))
                
                # summing over all inputs in mini batch
                # nabla_b[-l] = np.sum(nabla_b[-l], axis=0)
                # nabla_w[-l] = np.sum(nabla_w[-l], axis=0)
                # print('nabla_b[{0}] shape ==> {1}'.format(l, np.array(nabla_b[-l]).shape))
                # print('nabla_w[{0}] shape ==> {1}'.format(l, nabla_w[-l].shape))
            else:
                output_error = np.dot(self.weights[-l+1].transpose(), output_error) * self.activation(z, is_derivative=True)
                nabla_b[-l] = output_error
                nabla_w[-l] = np.dot(output_error, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        # test_result = [(np.argmax(self.feed_forward(test_data[0][i])), test_data[1][i]) for i in range(test_data[0].shape[0])]
        test_result = [(np.argmax(self.feed_forward(x)), y) for (x, y) in zip(test_data[0], test_data[1])]
        return sum(int(x == y) for (x, y) in test_result)

    def fit(self):
        #TODO implement fit method
        pass

    def predict(self):
        #TODO implement predict method
        pass
