"""Built-in activation functions
"""
import numpy as np
import six

def sigmoid(z, is_derivative=False):
    if is_derivative:
        return sigmoid(z) * (1-sigmoid(z))
    else:
        return 1.0 / 1.0 + np.exp(-z)


def deserialize(name):
    return getattr(locals(), name)


def get(identifier):
    if identifier == None:
        return sigmoid

    if isinstance(identifier, six.string_types):
        return deserialize(str(identifier))