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
    def delta(z, a, y):
        return (a-y) * Sigmiod.prime(z)


class CrossEntropy():
    
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num((-y * np.log(a)) - ((1 - y) * np.log(1 - a))))

    @staticmethod
    def delta(z, a, y):
        return (a - y)


def deserialize(name):
    module = __import__('cost_funcs')
    return getattr(module, name)


def get(identifier):
    if identifier == None:
        return CrossEntropy

    if isinstance(identifier, six.string_types):
        return deserialize(str(identifier))