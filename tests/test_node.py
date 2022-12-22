"""
This module contains tests for the methods in the Node class.
"""

import pytest
import numpy as np

import sys
sys.path.append('../src/autodiff_package/')

from node import Node


class TestNode:
    """These are test methods for the dunder methods defined in the Node class."""
    def test_init(self):
        """This is the test for the __init__ method."""
        d0 = Node(0,0)
        d1 = Node(2)
        
        assert d0.real == 0
        assert d0.dual == 0
        assert d0.child == None
        assert d0.id == None
        assert d1.real == 2
        assert d1.dual == 1
        assert d1.child == None
        assert d1.id == None

    def test_add(self):
        """This is the test for the __add__ method."""
        d0 = Node(1,2)
        d1 = Node(-4,5)
        i0 = 10
        s0 = "string"

        d2 = d0 + d1
        d3 = d1 + i0

        assert d2.real == -3
        assert d2.dual == 7
        assert d2.child == [(d0, 1), (d1, 1)]
        assert d3.child == [(d1, 1)]
        assert d3.real == 6
        assert d3.dual == 5
        with pytest.raises(TypeError):
            d1 + s0

    def test_mul(self):
        """This is the test for the __mult__ method."""
        d0 = Node(1,2)
        d1 = Node(-4,5)
        i0 = 10
        s0 = "string"

        d2 = d0 * d1
        d3 = d1 * i0

        assert d2.real == -4
        assert d2.dual == -3
        assert d2.child == [(d0, -4), (d1, 1)]
        assert d3.child == [(d1, 10)]
        assert d3.real == -40
        assert d3.dual == 50
        with pytest.raises(TypeError):
            d1 * s0
    
    def test_sub(self):
        """This is the test for the __sub__ method."""
        d0 = Node(1,2)
        d1 = Node(-4,5)
        i0 = 10
        s0 = "string"

        d2 = d0 - d1
        d3 = d1 - i0

        assert d2.real == 5
        assert d2.dual == -3
        assert d2.child == [(d0, 1), (d1, -1)]
        assert d3.child == [(d1, 1)]
        assert d3.real == -14
        assert d3.dual == 5
        with pytest.raises(TypeError):
            d1 - s0

    def test_pow(self):
        """This is the test for the __pow__ method."""
        d0 = Node(2,3)
        d1 = Node(4,5)
        i0 = 10
        s0 = "string"

        d2 = d0 ** d1
        d3 = d0 ** i0

        assert d2.real == 16
        assert d2.dual == 1536 + np.log(2)*80
        assert d2.child == [(d0, 32), (d1, 16 * np.log(2))]
        assert d3.child == [(d0, 5120)]
        assert d3.real == 1024
        assert d3.dual == 15360

        d4 = Node(2)
        d5 = d4 ** d4

        assert d5.real == 4
        assert d5.dual == pytest.approx(6.7725887)

        with pytest.raises(TypeError):
            d1 ** s0

    def test_truediv(self):
        """This is the test for the __truediv__ method."""
        d0 = Node(1,2)
        d1 = Node(-4,5)
        d3 = Node(0,4)
        i0 = 10
        i1 = 0
        s0 = "string"

        d4 = d0 / d1
        d5 = d1 / i0

        assert d4.real == -0.25
        assert d4.dual == -0.8125
        assert d4.child == [(d0, -0.25),(d1, -0.0625)]
        assert d5.child == [(d1, 0.1)]
        assert d5.real == -0.4
        assert d5.dual == 0.5
        with pytest.raises(ZeroDivisionError):
            d1 / i1
        with pytest.raises(ZeroDivisionError):
            d1 / d3
        with pytest.raises(TypeError):
            d1 / s0
    

    def test_rpow(self):
        """This is the test for the __rpow__ method."""
        d0 = Node(1)
        i0 = 5
        s0 = "string"

        d1 = i0 ** d0
        d2 = i0 ** (2 * d0)

        assert d1.real == 5
        assert d1.dual == pytest.approx(8.0471895)
        assert d1.child == [(d0, 5 * np.log(5))]
       # assert d2.child[0][0] == [(2*d0, 25 * np.log(5))]

        assert d2.real == 25
        assert d2.dual == pytest.approx(80.471895)

        with pytest.raises(TypeError):
            s0 ** d1

    def test_rtruediv(self):
        """This is the test for the __rtruediv__ method."""
        d0 = Node(2,3)
        i0 = 4
        s0 = "string"

        d1 = i0 / d0

        assert d1.real == 2
        assert d1.dual == 4/3
        assert d1.child == [(d0, -1)]
        with pytest.raises(TypeError):
            s0 / d0

    def test_rsub(self):
        """This is the test for the __rsub__ method."""
        d0 = Node(2,3)
        i0 = 4
        s0 = "string"

        d1 = i0 - d0

        assert d1.real == 2
        assert d1.dual == -3
        assert d1.child == [(d0, -1)]
        with pytest.raises(TypeError):
            s0 - d0

    def test_radd(self):
        """This is the test for the __radd__ method."""
        d0 = Node(2,3)
        i0 = 4
        s0 = "string"

        d1 = i0 + d0

        assert d1.real == 6
        assert d1.dual == 3
        assert d1.child == [(d0, 1)]
        with pytest.raises(TypeError):
            s0 + d0

    def test_rmul(self):
        """This is the test for the __rmult__ method."""
        d0 = Node(2,3)
        i0 = 4
        s0 = "string"

        d1 = i0 * d0

        assert d1.real == 8
        assert d1.dual == 12
        assert d1.child == [(d0, 4)]
        with pytest.raises(TypeError):
            s0 * d0

    def test_neg(self):
        """This is the test for the __neg__ method."""
        d0 = Node(2,3)

        d1 = -d0

        assert d1.real == -2
        assert d1.dual == -3
        assert d1.child == [(d0, -1)]

    def test_lt(self):
        """This is the test for the __lt__ method."""
        i0 = 0
        i1 = 1
        d0 = Node(i0)
        d1 = Node(i1)

        b0 = d0 < d1
        b1 = d1 < d0
        b2 = i0 < d1
        assert b0 == True 
        assert b1 == False 
        assert b2 == True

    def test_gt(self):
        """This is the test for the __gt__ method."""
        i0 = 0
        i1 = 1
        d0 = Node(i0)
        d1 = Node(i1)
        
        b0 = d1 > d0
        b1 = d0 > d1
        b2 = i1 > d0

        assert b0 == True
        assert b1 == False
        assert b2 == True

    def test_ge(self):
        """This is the test for the __ge__ method."""
        i0 = 0
        d0 = Node(0)
        d1 = Node(0)
        d2 = Node(1)

        b0 = d0 >= d1
        b1 = d1 >= d0
        b2 = d2 >= d0
        b3 = d0 >= d2
        b4 = i0 >= d0 

        assert b0 == True
        assert b1 == True
        assert b2 == True
        assert b3 == False
        assert b4 == True

    def test_le(self):
        """This is the test for the __le__ method."""
        i0 = 0
        d0 = Node(0)
        d1 = Node(0)
        d2 = Node(1)

        b0 = d0 <= d1
        b1 = d1 <= d0
        b2 = d0 <= d2
        b3 = d2 <= d0
        b4 = i0 <= d0

        assert b0 == True
        assert b1 == True
        assert b2 == True
        assert b3 == False
        assert b4 == True

    # def test_eq(self):
    #     i0 = 0
    #     d0 = Node(0)
    #     d2 = Node(0)
    #     d1 = Node(1)

    #     b0 = i0 == d0
    #     b1 = d0 == d1
    #     b2 = d0 == d2

    #     assert b0 == True 
    #     assert b1 == False
    #     assert b2 == True

    # def test_ne(self):
    #     i0 = 0
    #     d0 = Node(0)
    #     d2 = Node(0)
    #     d1 = Node(1)

    #     b0 = i0 != d0
    #     b1 = d0 != d1
    #     b2 = d0 != d2

    #     assert b0 == False
    #     assert b1 == True
    #     assert b2 == False
