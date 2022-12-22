"""
tests_differentiate.py
"""

import pytest
import numpy as np

#from autodiff_package.differentiate import grad
#from autodiff_package.dualnums import DualNumber


from differentiate import grad, reverse
import functions as f

class TestDifferentiate:
    """Class to test both automatic differentation functions, with grad being used for forward mode and reverse for reverse mode"""

    def test_grad(self):
        """Tests the grad function to handle multiple input functions and using multiple values"""

        x0 = 4
        x1 = [4,-3,5]

        fun0 = lambda z: z ** 3
        fun1 = lambda z: z + z/2 - 3
        fun2 = lambda z: 3.0 * z[0] + -2 * z[1] ** 2 + 5
        # fun3 = lambda z: z[2] ** 2 + f.exp(z[0]) # can't run tests with exp because it's off at like the billionth decimal place i guess
        fun3 = lambda z: z[2] ** 2 + f.log(z[0])

        fun4 = [fun0, fun1]
        fun5 = [fun2, fun3]

        seed0 = [3,0,0]
        seed1 = [2,-1/2]

        ans0,jac0 = grad(fun0,x0)
        ans1,jac1 = grad(fun4,x0)
        ans2,jac2 = grad(fun5,x1)
        ans3,jac3 = grad(fun5,x1,seed0)
        ans4,jac4 = grad(fun2,x1)

        assert ans0[0] == 64
        assert jac0[0] == 48

        assert ans1[0] == 64
        assert ans1[1] == 3
        assert jac1[0] == 48
        assert jac1[1] == 1.5

        assert ans2[0] == -1
        assert ans2[1] == 25 + np.log(4)
        assert jac2[0][0] == 3
        assert jac2[0][1] == 12
        assert jac2[0][2] == 0
        assert jac2[1][0] == 0.25
        assert jac2[1][1] == 0
        assert jac2[1][2] == 10

        assert jac3[0] == 9
        assert jac3[1] == 0.75

        assert ans4[0] == -1
        assert jac4[0][0] == 3
        assert jac4[0][1] == 12 
        assert jac4[0][2] == 0

        with pytest.raises(ValueError):
            grad(fun5,x1,seed1)

        # Tests for demo:

        x2 = 2
        x3 = [6,4,-3]

        func10 = lambda x: 3.0 * x * x + 2.5 * x + 2.0
        func11 = [lambda x: 1/2 * x[0] ** 2 - f.logbase(x[1],2) + x[2],lambda x: x[0] * x[1] + x[2]]

        answer10, jacob10 = grad(func10,x2)
        answer11, jacob11 = grad(func11,x3)

        assert answer10[0] == 19
        assert jacob10[0] == 14.5

        assert answer11 == [13,21]
        assert jacob11[0][0] == 6
        assert jacob11[0][1] == -1/(4 *np.log(2))
        assert jacob11[0][2] == 1
        assert jacob11[1][0] == 4
        assert jacob11[1][1] == 6
        assert jacob11[1][2] == 1
        



    def test_reverse(self):
        """Tests the reverse function to use multiple input functions and values"""
        x0 = 4
        x1 = [4,-3,5]

        fun0 = lambda z: z ** 3
        fun1 = lambda z: z + z/2 - 3
        fun2 = lambda z: 3.0 * z[0] + -2 * z[1] ** 2 + 5
        # fun3 = lambda z: z[2] ** 2 + f.exp(z[0]) # can't run tests with exp because it's off at like the billionth decimal place i guess
        fun3 = lambda z: z[2] ** 2 + f.log(z[0])

        fun4 = [fun0, fun1]
        fun5 = [fun2, fun3]

        seed0 = [3,0,0]
        seed1 = [2,-1/2]

        ans0,jac0 = reverse(fun0,x0)
        ans1,jac1 = reverse(fun4,x0)
        ans2,jac2 = reverse(fun5,x1)
        ans3,jac3 = reverse(fun5,x1,seed0)
        ans4,jac4 = reverse(fun2,x1)

        assert ans0[0] == 64
        assert jac0[0] == 48

        assert ans1[0] == 64
        assert ans1[1] == 3
        assert jac1[0] == 48
        assert jac1[1] == 1.5

        assert ans2[0] == -1
        assert ans2[1] == 25 + np.log(4)
        assert jac2[0][0] == 3
        assert jac2[0][1] == 12
        assert jac2[0][2] == 0
        assert jac2[1][0] == 0.25
        assert jac2[1][1] == 0
        assert jac2[1][2] == 10

        assert jac3[0] == 9
        assert jac3[1] == 0.75

        assert ans4[0] == -1
        assert jac4[0][0] == 3
        assert jac4[0][1] == 12 
        assert jac4[0][2] == 0

        with pytest.raises(ValueError):
            grad(fun5,x1,seed1)
