import pytest
import numpy as np
import math
import sys
sys.path.append('../src/autodiff_package/')


from node import Node
import functions as fun

class TestFunctions:
    def test_sin(self):
        d0 = Node(-4,5)
        d1 = fun.sin(d0)
        s0 = "string"
        f0 = 42.4
        f1 = fun.sin(f0)
        i0 = 39
        i1 = fun.sin(i0)

        assert d1.real == np.sin(-4)
        assert d1.dual == np.cos(-4) * 5
        assert d1.child == [(d0, np.cos(-4))]
        assert f1 == np.sin(42.4)
        assert i1 == np.sin(39)
        with pytest.raises(TypeError):
            fun.sin(s0)

    def test_cos(self):
        d0 = Node(-4,5)
        d1 = fun.cos(d0)
        s0 = "string"
        f0 = 42.4
        f1 = fun.cos(f0)

        assert d1.real == np.cos(-4)
        assert d1.dual == -1 * np.sin(-4) * 5
        assert d1.child == [(d0, -np.sin(-4))]
        assert f1 == np.cos(42.4)
        with pytest.raises(TypeError):
            fun.cos(s0)

    def test_tan(self):
        d0 = Node(-4,5)
        d1 = fun.tan(d0)
        s0 = "string"
        f0 = 42.4
        f1 = fun.tan(f0)

        assert d1.real == np.tan(-4)
        assert d1.dual == 5 / np.cos(-4) ** 2
        assert d1.child == [(d0, (1/np.cos(-4))**2)]
        assert f1 == np.tan(42.4)
        with pytest.raises(TypeError):
            fun.tan(s0)

    def text_exp(self):
        d0 = Node(2,3)
        i0 = 4
        s0 = "string"
        d1 = fun.exp(d0)
        i1 = fun.exp(i0)

        assert d1.real == np.exp(2)
        assert d1.dual == 3 * np.exp(2)
        assert d1.child == [(d0, 0.5)]
        assert i1 == np.exp(4)
        with pytest.raises(TypeError):
            fun.exp(s0)

    def test_log(self):
        d0 = Node(4,2)
        d1 = fun.log(d0)
        d2 = Node(0,1)
        i0 = 0
        s0 = "string"
        f0 = 4.0
        f1 = fun.log(f0)

        assert d1.real == np.log(4)
        assert d1.dual == 0.5
        assert d1.child == [(d0, 0.25)]
        assert f1 == np.log(4.0)
        with pytest.raises(TypeError):
            fun.log(s0)
        with pytest.raises(ValueError):
            fun.log(i0)
        with pytest.raises(ValueError):
            fun.log(d2)


    def test_logbase(self):
        d0 = Node(2,3)
        i0 = 4
        i1 = 16
        s0 = "StRiNg!"
        d1 = fun.logbase(d0,i0)
        i3 = fun.logbase(i0,i1)

        assert d1.real == np.log(2) / np.log(4)
        assert d1.dual == 1 / 2 / np.log(4) * 3
        assert d1.child == [(d0, 1/2/np.log(4))]
        assert i3 == 0.5
        with pytest.raises(TypeError):
            fun.logbase(s0,d0)
        with pytest.raises(TypeError):
            fun.logbase(i0,s0)

    def test_logistic(self):
        d0 = Node(1)
        d1 = fun.logistic(d0)
        i0 = 4
        i1 = fun.logistic(i0)
        s0 = 'string!'
        assert d1.real == pytest.approx(0.7310585786)
        assert d1.dual == pytest.approx(0.19661193)
        assert d1.child == [(d0, np.exp(1)/((1+np.exp(1))**2))]
        assert i1 == pytest.approx(0.98201379003)
        with pytest.raises(TypeError):
            fun.logistic(s0)
            
    def test_arcsin(x):
        d0 = Node(0.5)
        d1 = fun.arcsin(d0)
        i0 = 0.5
        i1 = fun.arcsin(i0)
        
        assert i1 == pytest.approx(0.52359877)
        assert d1.real == pytest.approx(0.52359877)
        assert d1.dual == pytest.approx(1.154700)

    def test_arccos(x):
        d0 = Node(0.5)
        d1 = fun.arccos(d0)
        i0 = 0.5
        i1 = fun.arccos(i0)
        
        assert i1 == pytest.approx(1.04719755)
        assert d1.real == pytest.approx(1.04719755)
        assert d1.dual == pytest.approx(-1.154700)

    def test_arctan(x):
        d0 = Node(2)
        d1 = fun.arctan(d0)
        i0 = 2
        i1 = fun.arctan(i0)
        
        assert i1 == pytest.approx(1.107148718)
        assert d1.real == pytest.approx(1.107148718)
        assert d1.dual == 0.2

    def test_sinh(x):
        d0 = Node(4)
        d1 = fun.sinh(d0)
        i0 = 4
        i1 = fun.sinh(i0)
        
        assert i1 == pytest.approx(27.2899172)
        assert d1.real == pytest.approx(27.2899172)
        assert d1.dual == pytest.approx(27.308232)
    
    def test_cosh(x):
        d0 = Node(4)
        d1 = fun.cosh(d0)
        i0 = 4
        i1 = fun.cosh(i0)

        assert d1.real == pytest.approx(27.308232836)
        assert d1.dual == pytest.approx(27.2899172)
        assert d1.child == [d0, np.sinh(4)]
        assert i1 == pytest.approx(27.308232836)
    
    def test_tanh(x):
        d0 = Node(4)
        d1 = fun.tanh(d0)
        i0 = 4
        i1 = fun.tanh(i0)

        assert d1.real == pytest.approx(0.9993292997)
        assert d1.dual == pytest.approx(0.00134095)

    def test_sqrt(x):
        d0 = Node(9)
        d1 = fun.sqrt(d0)
        i0 = 4
        i1 = fun.sqrt(i0)

        assert d1.real == 3
        assert d1.dual == pytest.approx(.16666666)
        assert i1 == 2

if __name__ in "__main__": # pragma: no cover
    print("running tests!")
    d0 = Node(1)
    d1 = fun.logistic(d0)
    
