"""Built-in cost functions
"""
import six
import numpy as np
from activations import Sigmiod

class QuadraticCost():

    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a-y) **2

    @staticmethod
    def prime(z, a, y):
        return (a-y) * Sigmiod.prime(z)
        

def deserialize(name):
    module = __import__('cost_funcs')
    return getattr(module, name)


def get(identifier):
    if identifier == None:
        return QuadraticCost

    if isinstance(identifier, six.string_types):
        return deserialize(str(identifier))