# Milestone 1

## Introduction

Every computer program, at it's most elementary form, can be boiled down to a series of basic arithmetic operations and functions. From the definition of the derivative, we know that we can __always__ calculate it by applying the chain rule repeatedly across the function. Therefore, we can find the derivative of every single function by applying automatic differentiation.

Automatic differentiation is necessary, because other forms of differentiation (namely, symbolic and numerical) have difficulties with converting a computer program into a single full mathematical expression, which automatic differentiation does not struggle with.

## Background

The whole point of automatic differentiation is to break down complex problems into digestable chunks, but there are a few mathematical concepts you will need to be familiar with to understand our implementation.

### <u>Chain Rule:</u>
The Chain Rule is essential when calcluating derivative and is taught in introductory calculus classes around the world. The chain rule is as follows:
$$\frac{dy}{dx}=\frac{dy}{du}*\frac{du}{dx}$$
Essentially, when you take the derivative of a function, you must take into consideration the derivatives of its inputs. Giving an example with real numbers:
$$y=e^{x^2}$$
$$\frac{dy}{dx}=e^{x^2}2x$$

### <u>Graph Structure of Calculations:</u>
Using a graph structure is a helpful way to visualize a complicated function and the steps that are required to evaluate it. Constructing a graph is simple, and it will be helpful to look at the example from lecture:
![](graph.png)

We start by initalizing the first nodes to our inputs. As we trace across edges between nodes, we evaluate the given functions using the one or more nodes that are the origin(s) of a specific edge. As we make our way across the graph, we will complete all steps needed to evaluate the complicated function and eventually be left with its overall output in the graph's final node.


### <u>Elementary Functions:</u>
1. A background in functions that are common in mathematics is also necessary to understand this project. Of course, the four elementary functions $+, -, *,$ and $/$ will be present throughout.
2. Trigonometric functions are also important to know and understand, these being $sin(x)$ and $cos(x)$. These functions oscilate between $-1$ and $1$ in sinusoidal curves. It might be important to note that:
$$\frac{d}{dx} sin(x) = cos(x)$$
$$\frac{d}{dx} cos(x) = -sin(x)$$
3. Raising a number to a power is another operations we will need in this project. That includes raising a number to negative powers or a power between $0$ and $1$. Note that variables can exist in either the base (rational powers of $x$) or in the exponent (exponential functions).
4. Finally, logaritms might be present in our project. Logrithms have a base and an input. If $b$ is your base, $n$ is your input, and $x$ is your output then:
$$x = log_b n$$
$$b^x = n$$

### <u>Constants:</u>
The constant $e$, Euler's number, might come up as well. $e\approx 2.72$ and appears in many functions. For example, $e$ is the base of a natural-log.\\
\\
The constant $\pi$ is also very important. $\pi \approx 3.14$

## How to Use autodiff_package Package

In order to interact with our package, users will begin by installing the package using:
    python -m pip install autodiff_package

After being installed, the user will then import the autodiff_package in order to make use of the auto differentiation abilities of the package. They would do so using the following:
    import autodiff_package as ad

In order to instantiate AD objects, they have to input their function. However, in order for the automatic differentiation to occur, they must use the overloaded functions from our package when defining the function, not numpy or any other methods. The autodiff_package would contain its own versions of elementary functions to use. An example is shown below for if they want to define a function that takes the sine of x: 
    f = lambda x: ad.sin(x)

Next, the user would also have to provide the value that their function should be evaluated at, as shown, where x1_value is the numerical value of the variable: 
    x1 = x1_value

Then, to finally instantiate an AD object, they would have to provide the function and values to evaluate at to the function grad(), which will be the function that finally conducts automatic differentiation. 
    D = ad.grad(f, x1)

The instantiated object would capture the result of the automatic differentiation, including both the value of the function evaluated at the provided values and the value of derivative of the function at that point as well.


## Software Organization
Our current organization in our team38 repository (which will eventually be the root directory of the package) is set up to be:
```
team38
├── LICENSE
├── README.md
├── docs
│   ├── graph.png
│   └── milestone1.md
├── pyproject.toml
├── setup.cfg
├── autodiff_package
│   ├── __init__.py
│   ├── __main__.py
│   ├── differentiate.py
│   ├── dualnums.py
│   └── node.py
└── tests
    ├── run_tests.sh
    ├── test_differentiate.py
    ├── test_dualnums.py
    └── test_node.py
```

The autodiff_package directory is where the actual packge logic will end up happening, which includes the node and dualnums modules, as well as the differentiate file which will allow us to step through each part of the graph. We also add \__init__.py and \__main__.py which is where we will run any functions that need to happen when the module's name is called. 

Additionally, we have a tests directory which 1:1 mirrors the package, which allows us to take advantage of python's unit testing as well.


## Implementation

We plan on implementing AD by breaking up the function we are given into a mathematical graph of the function, where each node $v$ is a step in the differentiation (i.e. $v_0$ is the initial function, $v_1$ is after the first step, etc.) 

### <u>Core Data Structures/Classes</u>

The "Node" will be our outermost data structure, being a node in the graph as explained above. The function of the node is to store the primal trace and the edges of the graph. 

We will also have a class for a dual number, which allows us to store both the regular number and the epsilon portion of each number.

### <u>Functions/Methods</u>

Each elementary function will need to be overloaded such that it functions with our dual numbers as well. This includes simple algorithmic operations like \__add__, \__sub__, etc. 

Additionally, we will in a seperate module define the derivative for each of the elementary functions cos, sin, power, log, exp, etc, which would return the derivative of that function applied to the value we have saved at that node.

### <u>Graph Integration</u>

Computational graphs can be helpful in understanding how variables feed into operations and sequence through the automatic differentiation process, but are only necessary for reverse mode. In forward mode, our dual number data structure should suffice to implement the differentiation, so if our group decides to implement reverse mode as the extension, we will keep reference to each node such that the graph is stored accurately in memory. 

### <u>Handling Cases of Functions that Project onto a different number of outputs </u>

In order to handle cases of different numbers of inputs (m) and objective functions (n), we will have to be able to still compute the Jacobian matrix that computes the gradient. To do this, we will create a class called "F()" that has method "grad()" to compute the Jacobian for an undefined set of functions f(). 

We will iterate through each objective function, compute it's partial derivatives using automatic differentiation, and append the results to a list. Continuing for each objective function, we will finally produce a matrix of size n by m that is the entire Jacobian. Some pseudo-code is below with the basic structure we might employ.

```c
class f():

    def grad(functions_list):
        jacobian = []
        for function in functions_list:
            compute the partial derivatives using forward mode AD
            jacobian.append(list_of_partial_derivatives)

    return Jacobian
```
This will depend on the Dual Number class, a two-factor data structure that contains both a real component and a secondary component. Using this implementation, we can simulate derivatives as in pset 4. To do this, we must define how to perform operations with these data structures. Implementation will follow the structure below. 

```c
class Dual:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual
   
    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass
    
    def __radd__(self, other):
        pass
    
    def __rmul__(self, other):
        pass
```

### <u>External Dependencies</u>

We will use ```numpy``` in order to take advantage of it's linear algebra functions, matrix multiplication, and all other functionalities it provides.


<!--

This is what happens in AD:
- start with a function
- return the computational graph of the function (this is the primal trace)
- then return the derivative of each (tangent trace)

github apparently 



jacobian gives all partial derivatives
    J_ij(x) = df_i()/df_j()

dual number is the form a + be, such that e^2 = 0

When you evaluate a function of a dual number, it returns the value of the function + the first derivative

node
dual number module
elementary function module

-->




## Licensing
We chose the MIT license because it is "simple and permissive." What we're going to be developing through this project has been done before and will be done by many others in this class. Thus, we don't mind if outsiders use this code. Like the helper to choose a license says, anyone can do almost anything with our project, and we don't mind.

## Feedback
### <u>Milestone 1</u>
Our feedback on Milestone 1 was simple, and we implemented our changes fairly quickly.

First, there was some issues to clean up in terms of formatting, including choosing a new photo to desmonstrate computational graphs and removing a floating bullet point that we forgot to get rid of before submission. Additionally, some of our markdown syntax was incorrect in the __Implementation__ section, and we took care of that as well.

Now to more concrete changes. We edited our __How to Use__ section given that 'autodiff_package' will be the name of our package rather than 'team38' as well as more clearly defining the definition and typing of our functions.

Finally, in __Implementation__ we added code scaffolds with pseudocode to better explain the functions we will write and fixed incorrect explanations we gave regarding classes in terms of our graphs and nodes.
