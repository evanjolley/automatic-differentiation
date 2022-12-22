"""
A module that defines  helper functions for performing additional operations with the Node class as defined in node.py
"""
import numpy as np
import math
try:
    from autodiff_package.node import Node
except:
    from node import Node

def sin(self):
    """Returns sine of the given node, integer, or float value"""
    if not isinstance(self, (int, float, Node)):
        raise TypeError(f"Unsupported type `{type(self)}`")
    if isinstance(self,(int,float)):
        return np.sin(self)
    else:
        local_grads = [(self, np.cos(self.real))]
        return Node(np.sin(self.real), np.cos(self.real) * self.dual, child = local_grads)

def cos(self):
    """Returns cosine of the given node, integer, or float value"""
    if not isinstance(self, (int, float, Node)):
        raise TypeError(f"Unsupported type `{type(self)}`")
    if isinstance(self,(int,float)):
        return np.cos(self)
    local_grads = [(self, -np.sin(self.real))]
    return Node(np.cos(self.real), -1 * np.sin(self.real) * self.dual, child = local_grads)

def tan(self):
    """Returns tangent at the given node, integer, or float"""
    if not isinstance(self, (int, float, Node)):
        raise TypeError(f"Unsupported type `{type(self)}`")
    if isinstance(self,(int,float)):
        return np.tan(self)
    local_grads = [(self, (1/np.cos(self.real))**2)]
    return Node(np.tan(self.real), self.dual/np.cos(self.real)** 2, child = local_grads)

def exp(self):
    """Returns the value of e raised to the power of the given node, integer, or float"""
    if not isinstance(self, (int, float, Node)):
        raise TypeError(f"Unsupported type `{type(self)}`")
    if isinstance(self,(int,float)):
        return np.exp(self)
    return self.__rpow__(math.e)

def log(self):
    """Returns the natural logarithm of the given node, integer, or float"""
    if not isinstance(self, (int,float, Node)):
        raise TypeError(f"Unsupported type `{type(self)}`")
    if isinstance(self,Node):
        if self.real == 0:
            raise ValueError("Cannot take log of 0")
        local_grads = [(self, 1/self.real)]
        return Node(np.log(self.real), 1/self.real * self.dual, child = local_grads)
    if self == 0:
        raise ValueError("Cannot take log of 0")
    return np.log(self)


def logbase(self, other):
    """Returns the log of a given node, integer, or float with a base of other, an integer or float"""
    if not isinstance(other, (int,float)):
        raise TypeError(f"Unsupported type `{type(other)}`")
    if not isinstance(self, (int,float,Node)):
        raise TypeError(f"Unsupported type `{type(self)}`")
    if isinstance(self,(int,float)):
        return np.log(self) / np.log(other)
    local_grads = [(self, 1/self.real/np.log(other))]
    return Node(np.log(self.real) / np.log(other), 1 / self.real / np.log(other) * self.dual, child = local_grads)

def logistic(self):
    """Returns the logistic function of the given node, integer, or float"""
    if not isinstance(self, (Node,int,float)):
        raise TypeError(f"Unsupported type `{type(self)}`")
    if isinstance(self, (int, float)):
        return 1/(1 + np.exp(-self))
    negated = -self.real
    local_grads = [(self, exp(self.real)/((1+exp(self.real))**2))]
    return Node(1/(1 + exp(negated)), exp(self.real)/((1+exp(self.real))**2), child = local_grads)


def arcsin(x):
    """Returns the inverse sine of the given node, integer, or float"""
    if not isinstance(x, (int, float, Node)):
        raise TypeError(f"Unsupported type `{type(x)}`")
    if isinstance(x,(int,float)):
        return np.arcsin(x)
    local_grads = [(x, 1/np.sqrt(1-x.real**2))]
    return Node(np.arcsin(x.real), x.dual/np.sqrt(1-(x.real**2)), child = local_grads)

def arccos(x):
    """Returns the inverse of the cosine of the given node, integer, or float"""
    if not isinstance(x, (int, float, Node)):
        raise TypeError(f"Unsupported type `{type(x)}`")
    if isinstance(x,(int,float)):
        return np.arccos(x)
    local_grads = [(x, -1/np.sqrt(1-x.real**2))]
    return Node(np.arccos(x.real), -x.dual/np.sqrt(1-x.real**2), child = local_grads)

def arctan(x):
    """Returns the inverse tangent of the given node, integer, or float"""
    if not isinstance(x, (int, float, Node)):
        raise TypeError(f"Unsupported type `{type(x)}`")
    if isinstance(x,(int,float)):
        return np.arctan(x)
    local_grads = [x, 1/(1+x.real**2)]
    return Node(np.arctan(x.real), x.dual/(1+x.real**2), child = local_grads)

def sinh(x):
    """Returns the hyperbolic sine of the given node, integer, or float"""
    if not isinstance(x, (int, float, Node)):
        raise TypeError(f"Unsupported type `{type(x)}`")
    if isinstance(x,(int,float)):
        return np.sinh(x)
    local_grads = [(x, 1*np.cosh(x.real))]
    return Node(np.sinh(x.real), x.dual*np.cosh(x.real), child = local_grads)

def cosh(x):
    """Returns the hyperbolic cosine of the given node, integer, or float"""
    if not isinstance(x, (int, float, Node)):
        raise TypeError(f"Unsupported type `{type(x)}`")
    if isinstance(x,(int,float)):
        return np.cosh(x)
    local_grads = [x, np.sinh(x.real)]
    return Node(np.cosh(x.real), x.dual*np.sinh(x.real), child = local_grads)

def tanh(x):
    """Returns the hyperbolic tangent of the given node, integer, or float"""
    if not isinstance(x, (int, float, Node)):
        raise TypeError(f"Unsupported type `{type(x)}`")
    if isinstance(x,(int,float)):
        return np.tanh(x)
    local_grads = 1/(np.cosh(x.real)**2)
    return Node(np.tanh(x.real), x.dual/(np.cosh(x.real)**2), child = local_grads)

def sqrt(x):
    """Returns the square root of the given node, integer, or float"""
    if not isinstance(x, (int, float, Node)):
        raise TypeError(f"Unsupported type `{type(x)}`")
    if isinstance(x,(int,float)):
        return np.sqrt(x)
    local_grads = [(x, 1/(2*np.sqrt(x.real)))]
    return Node(np.sqrt(x.real), x.dual/(2*np.sqrt(x.real)), child = local_grads)


if __name__=='__main__': # pragma: no cover
    pass
