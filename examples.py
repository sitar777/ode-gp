"""Examples to solve"""

import numpy as np
from utils import ODE


class ODE1(ODE):
    """First example"""

    initial_condition=[0, 0]
    x_range=[0, 8]
    unary_operators=['sin', 'cos']

    def function(self, x, y):
        return np.array(
            [
                y[1],
                -y[1] - y[0] - 1.6 + 0.4*x + 0.2*x**2 - 8*np.sin(4*x) - 30*np.cos(4*x),
            ]
        )

    def exact_solution(self, x):
        return 0.2*x**2 + 2*np.cos(4*x) - 2


ode_1 = ODE1()


class ODE2(ODE):
    """Second example"""

    initial_condition=[1, 5.4]
    x_range=[0, 8]
    unary_operators=['sin', 'cos', 'exp']

    def function(self, x, y):
        return np.array(
            [
                y[1],
                -0.3*y[1] - 25*y[0] + 25.12 + 10*x - 1.5*np.cos(5*x)*np.exp(-0.3*x),
            ]
        )

    def exact_solution(self, x):
        return 1 + 0.4*x + np.sin(5*x)*np.exp(-0.3*x)


ode_2 = ODE2()


class ODE3(ODE):
    """Third example"""

    initial_condition=[1, 5.4]
    x_range=[0, 8]
    unary_operators=['sin', 'cos', 'exp']

    def function(self, x, y):
        return np.array(
            [
                y[1],
                (6*np.sin(np.log(x)) - x*y[1] - 4*y[0])/x**2,
            ]
        )

    def exact_solution(self, x):
        return 3*np.cos(2*np.log(x)) + 2*np.sin(np.log(x))


ode_3 = ODE3()
