"""Examples to solve"""

import numpy as np
from utils import ODE, PDE


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

    initial_condition=[3, 2]
    x_range=[1, 8]
    unary_operators=['sin', 'cos', 'log']

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


class MalthusModel(ODE):
    """Malthus model"""

    initial_condition = [2]
    x_range = [0, 3] # big interval for exponential function is bad
    unary_operators = ["exp"]
    binary_operators = ["+", "*"]

    def __init__(self, r=3) -> None:
        super().__init__()
        self.r = r # constant growth rate

    def function(self, x, y):
        return np.array(
            y[0] * self.r,
        )

    def exact_solution(self, x):
        return self.initial_condition[0]*np.exp(self.r*x)

malthus_model = MalthusModel()


class HeatEquation1D(PDE):

    x_range = [0, 1.0]
    t_range = [0, 2.0]
    points_number_x = 100
    points_number_t = 4000
    unary_operators = []
    alpha = 0.1

    @property
    def dx(self):
        return (self.x_range[1] - self.x_range[0]) / (self.points_number_x - 1)

    @property
    def dt(self):
        return (self.t_range[1] - self.t_range[0]) / (self.points_number_t - 1)

    @property
    def r(self):
        return self.alpha * self.dt / self.dx**2

    def initial_condition(self, x):
        return np.sin(np.pi * x / self.x_range[1])

    def exact_solution(self, x, t):
        return np.sin(np.pi * x / self.x_range[1]) * np.exp(-self.alpha * (np.pi / self.x_range[1])**2 * t)

    @property
    def numerical_solution(self):
        u = np.zeros((self.points_number_t, self.points_number_x))
        u[0, :] = self.initial_condition(self.x_vals)

        for n in range(self.points_number_t - 1):
            for i in range(1, self.points_number_x - 1):
                u[n + 1, i] = u[n, i] + self.r * (u[n, i - 1] - 2 * u[n, i] + u[n, i + 1])

        return u

heat = HeatEquation1D()
