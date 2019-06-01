"""Built-in activation functions
"""
import numpy as np
import six

class Sigmiod():

    @staticmethod
    def fn(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    @staticmethod
    def prime(z):
        return Sigmiod.fn(z) * (1-Sigmiod.fn(z))


def deserialize(name):
    module = __import__('activations')
    return getattr(module, name)


def get(identifier):
    if identifier == None:
        return Sigmiod

    if isinstance(identifier, six.string_types):
        return deserialize(str(identifier))