"""Built-in cost functions
"""
import six

def quadratic_cost_prime(output_activations, y):
    return (output_activations - y)


def deserialize(name):
    return getattr(locals(), name)


def get(identifier):
    if identifier == None:
        return quadratic_cost_prime

    if isinstance(identifier, six.string_types):
        return deserialize(str(identifier))