import pickle
import gzip

import numpy as np


def load_data(file):
    '''Utility function to load MNIST data into training, validation and test datasets

    Arguments:
        file {str} -- string represent path to the file

    Returns:
         -- [description]
    '''

    f = gzip.open(file, 'rb')
    training_data, validation_data, test_data = pickle.load(
        f, encoding='latin1')
    f.close()

    return (training_data, validation_data, test_data)


def vectorize_y(num_classes, j):
    '''Return one hot encoding vector of Y results of the dataset of the form 0 or 1

    Arguments:
        num_classes {int} -- number of classes to classify. In handwritten digit recognition example, there will be 10 classes (from 0 or 9)
        j {int} -- position of Y in the orignal vector

    Returns:
        Numpy Array -- Numpy vector of the Y result
    '''
    y = np.zeros((num_classes, 1))
    y[j] = 1
    return y


def load_data_wrapper(file, image_size, num_classes):

    tr_d, va_d, te_d = load_data(file)
    # reformating training data
    training_inputs = [np.reshape(x, (image_size, 1)) for x in tr_d[0]]
    training_results = [vectorize_y(num_classes, y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    # reformating validation data
    validation_inputs = [np.reshape(x, (image_size, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    # reformating test data
    test_inputs = [np.reshape(x, (image_size, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)
