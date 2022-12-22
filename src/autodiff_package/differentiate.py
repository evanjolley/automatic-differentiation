"""
Calculates the partial derivatives of inputted functions at specified values using either forward or reverse mode automatic differentiation.

The grad function uses forward mode automatic differentiation, while the reverse function uses reverse mode automatic differentiation. 
Each takes one or multiple function arguments and one or multiple values to evaluate each function at. 
Both return both the partial derivates and the values of each function.

"""
import numpy as np
from collections import defaultdict
try:
    from autodiff_package.node import Node
except:
    from node import Node

def grad(fun, val, seed=None):
    """
    Calculate the derivative of a function evaluated at a specific input, with an optional seed
    
    Parameters
    ----------
    fun :
        The function to be differentiated, must be inputted using Python's lambda syntax
        Can be either a list of lambda functions (a vector function), or, a single lambda function
    val :
        Value to evaluate the derivative at. Either a list or a single value
    seed : 
        The weights of each derivative. Must be the same dimension as the val parameter
    
    Returns
    -------
    If a single vector/variable function:
    Node: node
        A real value for the function, and a dual value.

    If a multivector function: 
    Jacobian: array
        An array of lists of dual numbers, with the real portion being the real output of the function, 
        and the dual param being the partial derivative with respect to that value
    
    Examples
    --------
    >>> x = [0, 1, 2]
    >>> f = lambda x: x[1] + x[0]
    >>> y = lambda x: x[1] * x[2]

    >>> ans, jacobian = grad([f, y], x)
    >>> print("The values of the functions are", ans)
    1, 1, 0,  0, 2, 1
    >>> print("The Jacobian of the functions is", jacobian)
    1, 2

    """ 

    # Make sure every entry is a Node and has real and dual parts
    # (this allows it to handle vector functions with partial derivatives equal to 0)
    def check_ans(e):
        """ Checks that the input is a Node, and if it is not, one is created with a real value of the input and a dual value of 0 """
        if isinstance(e, Node):
            return e
        return Node(e,0)

    # Function and value to lists
    is_vallist = True
    if not isinstance(fun,list):
        fun = [fun]
    if not isinstance(val, list):
        val = [val]
        is_vallist = False

    # check input and seed are of the same shape
    seed_shape = np.shape(seed)
    val_shape = np.shape(val)
    if seed is not None and seed_shape != val_shape:
        raise ValueError('Value and seed are not of same dimension')

    ans = []
    jacobian = []
    for f in fun:

        # Not a list of values per function. Answer could have multiple dimensions, jacobian only has one dimension
        if not is_vallist:
            z = Node(val[0])
            ans.append(f(z).real)
            jacobian.append(f(z).dual)

        else:
            # Real value for the function
            ans.append(f(val))

            # Finds the partial derivative at each value in the vector
            partial_list = []
            for i in range(len(val)):

                partial_seed = val.copy()
                partial_seed[i] = Node(val[i])
                ans_partial = check_ans(f(partial_seed))

                partial_list.append(ans_partial.dual)

            # Appends a list of dual nums: 
                # real output for function at val, and partial derivative w/ respect to that variable at val
            if seed is not None:
                partial_list = np.dot(partial_list,seed)
            jacobian.append(partial_list)

    # Return the whole jacobian, unless it was a single vector function, in which only return that function's partial derivatives
    # Right now, answer is a list of answers, jacobian is potentially a list of lists of jacobians
  
    return ans, jacobian

def reverse(function, val, seed=None):
    """
    Calculate the derivative of function(s) evaluated at specific input(s) using reverse mode automatic differentiation
    Parameters
    ----------
    function :
        The function to be differentiated, must be inputted using Python's lambda syntax
        Can be either a list of lambda functions (a vector function), or a single lambda function
    val :
        Value to evaluate the derivative at. Either a list or a single value
    seed :
        Optional seed vector. The weights of each derivative. Must be the same dimension as the val parameter

    Returns
    -------
    Jacobian: array
        An array of lists of partial derivatives, with each list representing the partial derivatives of each inputted function
        and each value within the list representing the partial derivatives in terms of each inputted value.

        If a seed vector was input, then the directional derivative vector of each function is returned as a list instead.
    Function values: array
        A list of each inputted function evaluated at the inputted values 
    
    Examples
    --------
    >>> x = [0, 1, 2]
    >>> f = lambda x: x[1] + x[0]
    >>> y = lambda x: x[1] * x[2]

    >>> ans, jacobian = reverse([f, y], x)
    >>> print("The values of the functions are", ans)
    1, 1, 0,  0, 2, 1
    >>> print("The Jacobian of the functions is", jacobian)
    1, 2
    """ 
    
    # If values are given in a list, a new list is created with each value as a Node
    if isinstance(val, list): 
        seed_shape = np.shape(seed)
        val_shape = np.shape(val)
        if seed is not None and seed_shape != val_shape:
            raise ValueError('Value and seed are not of same dimension')
        node_vals = list(np.full(val_shape, Node(0,0)))
        for i in range(len(val)):
            node_vals[i] = Node(val[i], id=0)

    # If a single value is given, one Node is created with that value 
    else:
        node_vals = Node(val, id=0)

    # Makes function into a list if only a single function is provided
    if not isinstance(function,list):
        function = [function]

    Jacobian = []
    function_vals = []
    for f in function:
        partial = []

        # Adds value of each function to list of all function values
        root_node = f(node_vals)
        function_vals.append(root_node.real)

        gradients = defaultdict(lambda: Node(0))
        
        def compute_gradients(root_node, path_value):
            if root_node.child:
                for child_node, loc_grad in root_node.child:

                    # Calculates the partial derivative of the function based on the elementary function used
                    derivative = loc_grad * path_value  
                    gradients[child_node] += derivative

                    # Updates the id of the node to be the partial derivative value and continues traversing
                    child_node.id = gradients[child_node].real
                    compute_gradients(child_node, derivative)

        # Traverses down the computational graph of the function starting from the root node
        compute_gradients(root_node, path_value=1)  
    
        # Stores the partial derivatives of a function and resets each id of the node
        if isinstance(val, list):
            for i in range(len(val)):
                partial.append(node_vals[i].id)
                node_vals[i].id = 0
        else:
            partial = node_vals.id
            node_vals.id = 0

        # If a seed vector is provided, the directional derivative is found for each function by taking dot product with partial derivatives
        if seed is not None:
            partial = np.dot(partial, seed)
            
        # Creates Jacobian that contains the partial derivatives of all inputted functions
        Jacobian.append(partial)
    
    # Return the Jacobian and function values of all inputted functions
    return function_vals, Jacobian, 

if __name__ == "__main__": # pragma: no cover
    pass