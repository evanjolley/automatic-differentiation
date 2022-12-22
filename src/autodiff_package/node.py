"""
A module to define the Node data structure class and it's associated dunder methods. 
"""
import numpy as np

class Node:
    """
    A data structure containing a real value, a dual value, a reference to child nodes, and an id value. 
    
    This is used for both forward and reverse mode differentiation, 
    where forward uses self.real and self. dual, and reverse uses self.real, self.child, and self.id. 
    The module also has defined dunder methods that specify how to perform basic operations with Node data structures. 
    """
    
    def __init__(self, real, dual=1, child=None, id=None):
        """Initializes a new Node with an inputted real value, dual value of 1, and a child and id of None"""
        self.real = real
        self.dual = dual
        self.child = child
        self.id = id

    def __add__(self, other):
        """Adds a node with another node, integer, or a float"""
        if not isinstance(other, (int, float, Node)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        
        if isinstance(other, (int, float)):
            local_grads = [(self, 1)]
            return Node(other + self.real, self.dual, child = local_grads)
        else:
            local_grads = [(self, 1), (other, 1)]
            return Node(self.real + other.real, self.dual + other.dual, child = local_grads)


    def __mul__(self, other):
        """"Multiplies a node with another node, integer, or a float"""
        if not isinstance(other, (int, float, Node)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        if isinstance(other, (int, float)):
            local_grads = [(self, other)]
            return Node(other * self.real, other * self.dual, child = local_grads)
        else:
            local_grads = [(self, other.real), (other, self.real)]
            return Node(
                self.real * other.real,
                self.real * other.dual + self.dual * other.real,
                child = local_grads)

    def __sub__(self, other):
        """Subtracts a node, integer, or a float from a node"""
        if not isinstance(other, (int, float, Node)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        if isinstance(other, (int, float)):
            local_grads = [(self, 1)]
            return Node(self.real - other, self.dual, child = local_grads)
        else:
            local_grads = [(self, 1), (other, -1)]
            return Node(self.real - other.real, self.dual - other.dual, child = local_grads)

    def __pow__(self, other):
        """Raises a node to the power of another node, integer, or a float"""
        if not isinstance(other, (int, float, Node)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        if isinstance(other, (int, float)):
            local_grads = [(self, other*(self.real ** (other - 1)))]
            return Node(self.real ** other, other * self.real ** (other - 1) * self.dual, child = local_grads)
        else:
            local_grads = [(self, other.real*(self.real ** (other.real - 1))), (other, (self.real ** other.real) * np.log(self.real))]
            return Node(self.real ** other.real, (self.real * other.real)**(other.real - 1) * self.dual + np.log(self.real) * self.real ** other.real * other.dual, child = local_grads)
    
    def __truediv__(self, other):
        """Divides a node by another node, integer, or float"""
        if not isinstance(other, (int, float, Node)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            local_grads = [(self, 1/other)]
            return Node(self.real / other, self.dual / other, child = local_grads)
        else:
            if other.real == 0:
                raise ZeroDivisionError("division by zero")
            local_grads = [(self, 1/other.real),(other, (-self.real)/(other.real ** 2))]
            return Node(self.real / other.real, (self.dual*other.real - self.real*other.dual)/(other.real ** 2), child = local_grads)
    
    def __rpow__(self, other):
        """Raises an integer or float to the power of a node"""
        if not isinstance(other, (int, float, Node)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        local_grads = [(self, (other ** self.real) * np.log(other))]
        return Node(other ** self.real, other ** self.real * np.log(other) * self.dual, child = local_grads)

    def __rtruediv__(self, other):
        """Divides an integer or float by a node"""
        if not isinstance(other, (int, float)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        local_grads = [(self, -other/(self.real**2))]
        return Node(other/self.real, other/self.dual, child = local_grads)
    
    def __rsub__(self, other):
        """Substracts a node from an integer or float"""
        if not isinstance(other, (int, float)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        local_grads = [(self, -1)]
        return Node(other - self.real, -self.dual, child = local_grads)
    
    def __radd__(self, other):
        """Adds a node to an integer or float"""
        return self.__add__(other)
    
    def __rmul__(self, other):
        """Multiplies an integer or float by a node"""
        return self.__mul__(other)
    
    def __neg__(self):
        """Sets a node to have a negative value"""
        local_grads = [(self, -1)]
        return Node(-self.real, -self.dual, child = local_grads)

    def __gt__(self,other):
        if not isinstance(other,(Node,int,float)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        if isinstance(self,(int,float)):
            return self.real > other
        return self.real > other.real

    def __lt__(self,other):
        if not isinstance(other,(Node,int,float)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        if isinstance(self,(int,float)):
            return self.real < other
        return self.real < other.real

    def __ge__(self,other):
        if not isinstance(other,(Node,int,float)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        if isinstance(self,(int,float)):
            return self.real >= other
        return self.real >= other.real

    def __le__(self,other):
        if not isinstance(other,(Node,int,float)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        if isinstance(self,(int,float)):
            return self.real <= other
        return self.real <= other.real

