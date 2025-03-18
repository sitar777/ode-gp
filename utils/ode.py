"""ODE container class"""

import numpy as np


class ODE:
    """Container class for ODE"""

    initial_condition: list
    x_range: list
    unary_operators: list[str]
    points_number: int | None = 100

    @property
    def x_vals(self):
        """Returns x values for range"""
        return np.linspace(self.x_range[0], self.x_range[1], self.points_number)

    def function(self, x, y):
        """ODE to solve"""
        raise NotImplementedError

    def exact_solution(self, x):
        """Exact solution of ODE"""
        raise NotImplementedError
