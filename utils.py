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
    x_training = np.array([np.reshape(x, (image_size, 1)) for x in tr_d[0]])
    y_training = np.array([vectorize_y(num_classes, y) for y in tr_d[1]])
    # reformating validation data
    x_validation = np.array([np.reshape(x, (image_size, 1)) for x in va_d[0]])
    y_validation =  va_d[1]
    # reformating test data
    x_test = np.array([np.reshape(x, (image_size, 1)) for x in te_d[0]])
    y_test = te_d[1]
    
    
    return (x_training, y_training), (x_validation, y_validation), (x_test, y_test)
